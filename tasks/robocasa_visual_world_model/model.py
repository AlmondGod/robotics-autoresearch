from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tasks.robocasa_world_model.model import RoboCasaWorldModel


class ImageVAE(nn.Module):
    """Small RGB VAE used to make visual prediction latent-space aware."""

    def __init__(self, *, image_size: int, latent_dim: int = 64, width: int = 256, dropout: float = 0.05) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, 2 * self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(width), int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Linear(int(width), 3 * self.image_size * self.image_size),
        )

    def encode(self, image: torch.Tensor, *, sample: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.encoder(image)
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 8.0)
        if sample and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z, mu, logvar

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        image = torch.sigmoid(self.decoder(latent))
        return image.reshape(-1, 3, self.image_size, self.image_size)

    def forward(self, image: torch.Tensor, *, sample: bool = False) -> dict[str, torch.Tensor]:
        z, mu, logvar = self.encode(image, sample=sample)
        return {"latent": z, "mu": mu, "logvar": logvar, "reconstruction": self.decode(z)}


class FlowHead(nn.Module):
    """Conditional rectified-flow vector field for latent or state-delta targets."""

    def __init__(self, *, condition_dim: int, sample_dim: int, width: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.sample_dim = int(sample_dim)
        self.net = nn.Sequential(
            nn.Linear(int(condition_dim) + int(sample_dim) + 1, int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(width), int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(sample_dim)),
        )

    def forward(self, condition: torch.Tensor, sample: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([condition, sample, t.float()], dim=-1))


class VisualRoboCasaWorldModel(nn.Module):
    """State/action dynamics model with image-VAE latents and flow-matching heads."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        task_count: int,
        image_size: int = 32,
        width: int = 512,
        depth: int = 4,
        task_dim: int = 64,
        latent_dim: int = 64,
        visual_latent_dim: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.visual_latent_dim = int(visual_latent_dim)
        self.dynamics = RoboCasaWorldModel(
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            task_count=int(task_count),
            width=int(width),
            depth=int(depth),
            task_dim=int(task_dim),
            latent_dim=int(latent_dim),
            dropout=float(dropout),
        )
        self.image_vae = ImageVAE(
            image_size=int(image_size),
            latent_dim=int(visual_latent_dim),
            width=max(128, int(width) // 2),
            dropout=float(dropout),
        )
        self.next_visual_latent = nn.Sequential(
            nn.Linear(int(width), int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(width), int(visual_latent_dim)),
        )
        self.visual_flow = FlowHead(
            condition_dim=int(width),
            sample_dim=int(visual_latent_dim),
            width=int(width),
            dropout=float(dropout),
        )
        self.state_delta_flow = FlowHead(
            condition_dim=int(width),
            sample_dim=int(state_dim),
            width=int(width),
            dropout=float(dropout),
        )

    @property
    def state_dim(self) -> int:
        return int(self.dynamics.state_dim)

    @property
    def action_dim(self) -> int:
        return int(self.dynamics.action_dim)

    @property
    def task_count(self) -> int:
        return int(self.dynamics.task_count)

    @property
    def latent_dim(self) -> int:
        return int(self.dynamics.latent_dim)

    def transition_hidden(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
        progress: torch.Tensor,
        *,
        sample_latent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.dynamics.encode_state(state, sample=sample_latent)
        if progress.ndim == 1:
            progress = progress[:, None]
        h = torch.cat([z, action, self.dynamics.task(task_id.long()), progress.float()], dim=-1)
        return self.dynamics.trunk(h), z, mu, logvar

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
        progress: torch.Tensor,
        *,
        sample_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden, z, mu, logvar = self.transition_hidden(
            state,
            action,
            task_id,
            progress,
            sample_latent=sample_latent,
        )
        next_z = z + self.dynamics.delta(hidden)
        next_state = self.dynamics.decode_state(next_z)
        pred_visual_latent = self.next_visual_latent(hidden)
        next_rgb = self.image_vae.decode(pred_visual_latent)
        return {
            "next_state": next_state,
            "next_latent": next_z,
            "next_progress": torch.sigmoid(self.dynamics.progress(hidden)),
            "reward": self.dynamics.reward(hidden),
            "success_logit": self.dynamics.success(hidden),
            "next_rgb": next_rgb,
            "next_visual_latent": pred_visual_latent,
            "hidden": hidden,
            "latent_mu": mu,
            "latent_logvar": logvar,
        }

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        state_weight: float = 1.0,
        progress_weight: float = 0.25,
        reward_weight: float = 0.25,
        success_weight: float = 0.25,
        visual_weight: float = 1.0,
        visual_delta_weight: float = 0.25,
        visual_latent_weight: float = 0.25,
        image_vae_weight: float = 0.25,
        visual_flow_weight: float = 0.5,
        state_flow_weight: float = 0.25,
        kl_weight: float = 1e-4,
        visual_kl_weight: float = 1e-5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self(
            batch["state"],
            batch["action"],
            batch["task_id"],
            batch["progress"],
            sample_latent=True,
        )
        state_loss = F.mse_loss(out["next_state"], batch["next_state"])
        progress_loss = F.mse_loss(out["next_progress"], batch["next_progress"])
        reward_loss = F.mse_loss(out["reward"], batch["reward"])
        success_loss = F.binary_cross_entropy_with_logits(out["success_logit"], batch["success"])
        rgb_loss = F.mse_loss(out["next_rgb"], batch["next_rgb"])
        pred_delta = out["next_rgb"] - batch["rgb"]
        true_delta = batch["next_rgb"] - batch["rgb"]
        rgb_delta_loss = F.mse_loss(pred_delta, true_delta)

        next_visual, next_visual_mu, next_visual_logvar = self.image_vae.encode(batch["next_rgb"], sample=False)
        current_visual, current_visual_mu, current_visual_logvar = self.image_vae.encode(batch["rgb"], sample=False)
        visual_latent_loss = F.mse_loss(out["next_visual_latent"], next_visual.detach())
        current_recon = self.image_vae.decode(current_visual)
        next_recon = self.image_vae.decode(next_visual)
        image_vae_loss = 0.5 * (F.mse_loss(current_recon, batch["rgb"]) + F.mse_loss(next_recon, batch["next_rgb"]))
        visual_kl = 0.5 * (_kl(current_visual_mu, current_visual_logvar) + _kl(next_visual_mu, next_visual_logvar))
        visual_flow_loss = _flow_matching_loss(
            self.visual_flow,
            condition=out["hidden"],
            target=next_visual.detach(),
        )
        state_delta = (batch["next_state"] - batch["state"]).detach()
        state_flow_loss = _flow_matching_loss(
            self.state_delta_flow,
            condition=out["hidden"],
            target=state_delta,
        )

        if self.latent_dim > 0:
            kl = _kl(out["latent_mu"], out["latent_logvar"])
        else:
            kl = torch.zeros((), dtype=state_loss.dtype, device=state_loss.device)
        total = (
            float(state_weight) * state_loss
            + float(progress_weight) * progress_loss
            + float(reward_weight) * reward_loss
            + float(success_weight) * success_loss
            + float(visual_weight) * rgb_loss
            + float(visual_delta_weight) * rgb_delta_loss
            + float(visual_latent_weight) * visual_latent_loss
            + float(image_vae_weight) * image_vae_loss
            + float(visual_flow_weight) * visual_flow_loss
            + float(state_flow_weight) * state_flow_loss
            + float(kl_weight) * kl
            + float(visual_kl_weight) * visual_kl
        )
        return total, {
            "loss": total.detach(),
            "state_mse": state_loss.detach(),
            "progress_mse": progress_loss.detach(),
            "reward_mse": reward_loss.detach(),
            "success_bce": success_loss.detach(),
            "rgb_mse": rgb_loss.detach(),
            "rgb_delta_mse": rgb_delta_loss.detach(),
            "visual_latent_mse": visual_latent_loss.detach(),
            "image_vae_mse": image_vae_loss.detach(),
            "visual_flow_mse": visual_flow_loss.detach(),
            "state_flow_mse": state_flow_loss.detach(),
            "kl": kl.detach(),
            "visual_kl": visual_kl.detach(),
        }


def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())


def _flow_matching_loss(flow: FlowHead, *, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(target)
    t = torch.rand((target.shape[0], 1), dtype=target.dtype, device=target.device)
    sample = (1.0 - t) * noise + t * target
    velocity = target - noise
    pred = flow(condition, sample, t)
    return F.mse_loss(pred, velocity)
