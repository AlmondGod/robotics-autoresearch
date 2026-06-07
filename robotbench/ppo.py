from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field
import os
from pathlib import Path
from typing import Any

import numpy as np

from robotbench.config import TaskConfig
from robotbench.envs import make_env
from robotbench.train import TrainBudget


@dataclass
class TorchPpoPolicy:
    obs_dim: int
    act_dim: int
    hidden_dim: int
    state_dict: dict[str, np.ndarray]
    architecture: str = "mlp"
    image_shape: tuple[int, int] | None = None
    proprio_dim: int = 0
    training_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        torch, nn, _ = _torch_modules()
        self.torch = torch
        self.net = ActorCritic(
            self.obs_dim,
            self.act_dim,
            self.hidden_dim,
            architecture=self.architecture,
            image_shape=self.image_shape,
            proprio_dim=self.proprio_dim,
        )
        self.net.load_state_dict({k: torch.as_tensor(v) for k, v in self.state_dict.items()})
        self.net.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            obs_t = self.torch.as_tensor(obs, dtype=self.torch.float32).reshape(1, -1)
            mean, _, _ = self.net(obs_t)
            return self.torch.tanh(mean)[0].cpu().numpy()

    def serializable_state(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}


def train_ppo_policy(
    task: TaskConfig,
    budget: TrainBudget,
    seed: int,
    backend: str,
) -> TorchPpoPolicy:
    torch, nn, optim = _torch_modules()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    probe_env = make_env(task=task, world="train", seed=seed, backend=backend)
    obs_dim = int(probe_env.obs_dim)
    act_dim = int(probe_env.act_dim)
    hidden_dim = 128
    image_shape = _env_image_shape(probe_env)
    proprio_dim = int(getattr(probe_env, "proprio_dim", 0))
    architecture = "cnn" if image_shape else "mlp"
    net = ActorCritic(
        obs_dim,
        act_dim,
        hidden_dim,
        architecture=architecture,
        image_shape=image_shape,
        proprio_dim=proprio_dim,
    )
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    _behavior_clone_from_default_demos(
        net=net,
        optimizer=optimizer,
        backend=backend,
        obs_dim=obs_dim,
        act_dim=act_dim,
        torch=torch,
        nn=nn,
    )
    rollout_steps = max(32, min(int(budget.rollouts_per_iteration) * 8, 256))
    num_envs = max(1, min(4, rollout_steps // 16))
    steps_per_env = max(8, rollout_steps // num_envs)
    batch_size = num_envs * steps_per_env
    minibatch_size = min(128, batch_size)
    update_epochs = 4
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    entropy_coef = 0.002
    value_coef = 0.5
    max_grad_norm = 0.5
    target_kl = 0.03
    diagnostics = []

    envs = [make_env(task=task, world="train", seed=seed * 1000 + idx, backend=backend) for idx in range(num_envs)]
    observations = [env.reset() for env in envs]
    start = time.monotonic()
    updates = 0
    while updates < budget.max_iterations and time.monotonic() - start < budget.seconds:
        batch = _collect_vectorized_rollout(
            envs=envs,
            net=net,
            observations=observations,
            steps_per_env=steps_per_env,
            gamma=gamma,
            gae_lambda=gae_lambda,
            torch=torch,
        )
        observations = batch.pop("last_observations")
        rollout_avg_reward = float(batch.pop("rollout_avg_reward"))
        rollout_done_rate = float(batch.pop("rollout_done_rate"))
        advantages = batch["advantages"]
        batch["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(batch_size)
        update_policy_losses = []
        update_value_losses = []
        update_entropies = []
        update_kls = []
        update_clip_fractions = []
        stopped_for_kl = False
        for _ in range(update_epochs):
            rng.shuffle(indices)
            for start_idx in range(0, batch_size, minibatch_size):
                mb = indices[start_idx : start_idx + minibatch_size]
                mean, value, log_std = net(batch["obs"][mb])
                dist = torch.distributions.Normal(mean, log_std.exp())
                z = batch["pre_tanh_action"][mb]
                action = torch.tanh(z)
                log_prob = _tanh_log_prob(dist, z, action).sum(axis=-1)
                ratio = torch.exp(log_prob - batch["log_prob"][mb])
                adv = batch["advantages"][mb]
                policy_loss = -torch.minimum(
                    ratio * adv,
                    torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv,
                ).mean()
                value_loss = (value.squeeze(-1) - batch["returns"][mb]).square().mean()
                entropy = dist.entropy().sum(axis=-1).mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                with torch.no_grad():
                    approx_kl = (batch["log_prob"][mb] - log_prob).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
                update_policy_losses.append(float(policy_loss.detach().cpu()))
                update_value_losses.append(float(value_loss.detach().cpu()))
                update_entropies.append(float(entropy.detach().cpu()))
                update_kls.append(float(approx_kl.detach().cpu()))
                update_clip_fractions.append(float(clip_fraction.detach().cpu()))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()
                if float(approx_kl.detach().cpu()) > 1.5 * target_kl:
                    stopped_for_kl = True
                    break
            if stopped_for_kl:
                break
        diagnostics.append(
            {
                "update": updates,
                "policy_loss": _mean_or_zero(update_policy_losses),
                "value_loss": _mean_or_zero(update_value_losses),
                "entropy": _mean_or_zero(update_entropies),
                "approx_kl": _mean_or_zero(update_kls),
                "clip_fraction": _mean_or_zero(update_clip_fractions),
                "rollout_avg_reward": rollout_avg_reward,
                "rollout_done_rate": rollout_done_rate,
                "stopped_for_kl": stopped_for_kl,
            }
        )
        updates += 1

    final_diagnostics = diagnostics[-1] if diagnostics else {}
    return TorchPpoPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        state_dict={k: v.detach().cpu().numpy() for k, v in net.state_dict().items()},
        architecture=architecture,
        image_shape=image_shape,
        proprio_dim=proprio_dim,
        training_info={
            "updates": updates,
            "rollout_steps": rollout_steps,
            "num_envs": num_envs,
            "steps_per_env": steps_per_env,
            "minibatch_size": minibatch_size,
            "update_epochs": update_epochs,
            "target_kl": target_kl,
            "final_diagnostics": final_diagnostics,
            "diagnostics": diagnostics,
        },
    )


class ActorCritic:
    def __new__(
        cls,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        architecture: str = "mlp",
        image_shape: tuple[int, int] | None = None,
        proprio_dim: int = 0,
    ):
        torch, nn, _ = _torch_modules()

        class _ActorCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.architecture = architecture
                self.image_shape = image_shape
                self.proprio_dim = proprio_dim
                if architecture == "cnn":
                    if image_shape is None:
                        raise ValueError("cnn PPO policy requires image_shape")
                    image_h, image_w = image_shape
                    image_dim = image_h * image_w
                    if image_dim + proprio_dim != obs_dim:
                        raise ValueError(
                            f"obs_dim {obs_dim} does not match image/proprio dims "
                            f"{image_dim}+{proprio_dim}"
                        )
                    self.encoder = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=3, stride=2),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, kernel_size=3, stride=2),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                    with torch.no_grad():
                        encoded_dim = int(self.encoder(torch.zeros(1, 1, image_h, image_w)).shape[1])
                    self.body = nn.Sequential(
                        nn.Linear(encoded_dim + proprio_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                    )
                else:
                    self.encoder = None
                    self.body = nn.Sequential(
                        nn.Linear(obs_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                    )
                self.actor = nn.Linear(hidden_dim, act_dim)
                self.critic = nn.Linear(hidden_dim, 1)
                self.log_std = nn.Parameter(torch.full((act_dim,), -1.2))

            def forward(self, obs):
                if self.architecture == "cnn":
                    image_h, image_w = self.image_shape
                    image_dim = image_h * image_w
                    image = obs[:, :image_dim].reshape(-1, 1, image_h, image_w)
                    encoded = self.encoder(image)
                    if self.proprio_dim:
                        encoded = torch.cat([encoded, obs[:, image_dim:]], dim=-1)
                    features = self.body(encoded)
                else:
                    features = self.body(obs)
                return self.actor(features), self.critic(features), self.log_std

        return _ActorCritic()


def _collect_vectorized_rollout(
    envs: list[Any],
    net: Any,
    observations: list[np.ndarray],
    steps_per_env: int,
    gamma: float,
    gae_lambda: float,
    torch: Any,
) -> dict[str, Any]:
    obs_rows = []
    pre_tanh_actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []
    num_envs = len(envs)

    for _ in range(steps_per_env):
        obs_t = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
        with torch.no_grad():
            mean, value, log_std = net(obs_t)
            dist = torch.distributions.Normal(mean, log_std.exp())
            z = dist.sample()
            action = torch.tanh(z)
            log_prob = _tanh_log_prob(dist, z, action).sum(axis=-1)
        step_rewards = []
        step_dones = []
        obs_rows.append(np.asarray(observations).copy())
        pre_tanh_actions.append(z.cpu().numpy())
        log_probs.append(log_prob.cpu().numpy())
        values.append(value.squeeze(-1).cpu().numpy())
        for env_idx, env in enumerate(envs):
            result = env.step(action[env_idx].cpu().numpy())
            done = bool(result.terminated or result.truncated)
            step_rewards.append(float(result.reward))
            step_dones.append(done)
            observations[env_idx] = env.reset() if done else result.obs
        rewards.append(step_rewards)
        dones.append(step_dones)

    with torch.no_grad():
        last_obs_t = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
        _, last_value, _ = net(last_obs_t)
    values_np = np.asarray(values + [last_value.squeeze(-1).cpu().numpy()], dtype=np.float32)
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros((steps_per_env, num_envs), dtype=np.float32)
    last_gae = np.zeros(num_envs, dtype=np.float32)
    for step in reversed(range(steps_per_env)):
        nonterminal = 1.0 - dones_np[step]
        delta = rewards_np[step] + gamma * values_np[step + 1] * nonterminal - values_np[step]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values_np[:-1]

    return {
        "obs": torch.as_tensor(np.asarray(obs_rows).reshape(steps_per_env * num_envs, -1), dtype=torch.float32),
        "pre_tanh_action": torch.as_tensor(
            np.asarray(pre_tanh_actions).reshape(steps_per_env * num_envs, -1),
            dtype=torch.float32,
        ),
        "log_prob": torch.as_tensor(np.asarray(log_probs).reshape(-1), dtype=torch.float32),
        "advantages": torch.as_tensor(advantages.reshape(-1), dtype=torch.float32),
        "returns": torch.as_tensor(returns.reshape(-1), dtype=torch.float32),
        "last_observations": observations,
        "rollout_avg_reward": float(rewards_np.mean()),
        "rollout_done_rate": float(dones_np.mean()),
    }


def _tanh_log_prob(dist: Any, z: Any, action: Any) -> Any:
    return dist.log_prob(z) - torch_log_1m_action_sq(action, z)


def torch_log_1m_action_sq(action: Any, z: Any) -> Any:
    del z
    return (1.0 - action.square() + 1e-6).log()


def _torch_modules():
    try:
        import torch
        from torch import nn, optim
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PPO training requires PyTorch. Install with: pip install -e .[ppo]"
        ) from exc
    return torch, nn, optim


def _env_image_shape(env: Any) -> tuple[int, int] | None:
    height = getattr(env, "camera_height", None)
    width = getattr(env, "camera_width", None)
    image_obs_dim = getattr(env, "image_obs_dim", None)
    if height is None or width is None or image_obs_dim is None:
        return None
    if int(height) * int(width) != int(image_obs_dim):
        raise ValueError("env camera dimensions do not match image_obs_dim")
    return int(height), int(width)


def _behavior_clone_from_default_demos(
    net: Any,
    optimizer: Any,
    backend: str,
    obs_dim: int,
    act_dim: int,
    torch: Any,
    nn: Any,
) -> None:
    demo_path_env = os.environ.get("ROBOTBENCH_BC_DEMOS", "")
    if not demo_path_env:
        return
    demo_path = Path(demo_path_env)
    if not demo_path.exists():
        return
    data = np.load(demo_path)
    obs = np.asarray(data["obs"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if obs.ndim != 2 or actions.ndim != 2 or obs.shape[1] != obs_dim or actions.shape[1] != act_dim:
        return
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    action_t = torch.as_tensor(np.clip(actions, -0.999, 0.999), dtype=torch.float32)
    target_mean = torch.atanh(action_t)
    batch_size = min(128, len(obs))
    if batch_size <= 0:
        return
    indices = np.arange(len(obs))
    rng = np.random.default_rng(0)
    epochs = 8
    for _ in range(epochs):
        rng.shuffle(indices)
        for start_idx in range(0, len(indices), batch_size):
            mb = indices[start_idx : start_idx + batch_size]
            mean, value, _ = net(obs_t[mb])
            actor_loss = nn.functional.mse_loss(mean, target_mean[mb])
            value_loss = 0.001 * value.square().mean()
            optimizer.zero_grad()
            (actor_loss + value_loss).backward()
            optimizer.step()


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
