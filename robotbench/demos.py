from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robotbench.config import TaskConfig
from robotbench.envs import make_env


def arx_l5_oracle_action(env: Any, gain: float = 3.5, damping: float = 1e-3) -> np.ndarray:
    """Privileged Jacobian controller used only for demonstration generation."""
    if not all(hasattr(env, name) for name in ["model", "data", "mujoco", "target_body"]):
        raise TypeError("arx_l5_oracle_action requires the ARX L5 MuJoCo env")

    ctrl0 = env.current_ctrl.copy()
    gripper0 = env._gripper_pos()
    target = env.model.body_pos[env.target_body].copy()
    error = target - gripper0

    jac_left = np.zeros((3, env.model.nv), dtype=np.float64)
    jac_right = np.zeros((3, env.model.nv), dtype=np.float64)
    jac_rot = np.zeros((3, env.model.nv), dtype=np.float64)
    env.mujoco.mj_jacBodyCom(env.model, env.data, jac_left, jac_rot, env.left_finger)
    env.mujoco.mj_jacBodyCom(env.model, env.data, jac_right, jac_rot, env.right_finger)
    jacobian = 0.5 * (jac_left[:, : env.act_dim] + jac_right[:, : env.act_dim])

    jt_j = jacobian @ jacobian.T
    desired_delta = jacobian.T @ np.linalg.solve(jt_j + damping * np.eye(3), gain * error)
    desired_delta = desired_delta[: env.act_dim]
    lo = env.ctrlrange[:, 0]
    hi = env.ctrlrange[:, 1]
    desired_ctrl = np.clip(ctrl0 + desired_delta, lo, hi)
    desired_delta = desired_ctrl - ctrl0
    train_scale = float(env.params.get("control_scale", 0.05))
    action = desired_delta / max(train_scale, 1e-6)
    action[6] = 0.0
    return np.clip(action, -env.task.action_limit, env.task.action_limit)


def record_arx_l5_demos(
    task: TaskConfig,
    episodes: int,
    seed: int,
    out_path: Path,
    include_video: bool = True,
) -> dict[str, Any]:
    obs_rows = []
    action_rows = []
    reward_rows = []
    done_rows = []
    episode_rows = []
    step_rows = []
    frames = []
    returns = []
    successes = []
    distances = []

    for episode_idx in range(episodes):
        env = make_env(task=task, world="train", seed=seed * 10_000 + episode_idx, backend="arx_l5")
        obs = env.reset()
        total_return = 0.0
        final_info: dict[str, Any] = {}
        for step_idx in range(task.horizon):
            action = arx_l5_oracle_action(env)
            result = env.step(action)
            obs_rows.append(obs.astype(np.float32))
            action_rows.append(action.astype(np.float32))
            reward_rows.append(float(result.reward))
            done = bool(result.terminated or result.truncated)
            done_rows.append(done)
            episode_rows.append(episode_idx)
            step_rows.append(step_idx)
            if include_video:
                frames.append(env.render_rgb(width=320, height=240))
            obs = result.obs
            total_return += float(result.reward)
            final_info = result.info
            if done:
                break
        returns.append(total_return)
        successes.append(bool(final_info.get("success", False)))
        distances.append(float(final_info.get("distance", 999.0)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": np.asarray("arx_l5"),
        "task": np.asarray(task.name),
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "actions": np.asarray(action_rows, dtype=np.float32),
        "rewards": np.asarray(reward_rows, dtype=np.float32),
        "dones": np.asarray(done_rows, dtype=np.bool_),
        "episode": np.asarray(episode_rows, dtype=np.int32),
        "step": np.asarray(step_rows, dtype=np.int32),
        "returns": np.asarray(returns, dtype=np.float32),
        "successes": np.asarray(successes, dtype=np.bool_),
        "final_distances": np.asarray(distances, dtype=np.float32),
    }
    if include_video:
        payload["video_frames"] = np.asarray(frames, dtype=np.uint8)
    np.savez_compressed(out_path, **payload)
    return {
        "path": str(out_path),
        "episodes": episodes,
        "transitions": len(obs_rows),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_final_distance": float(np.mean(distances)) if distances else 999.0,
    }
