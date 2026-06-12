from __future__ import annotations

import torch
from torch import nn


class RoboCasaRGBRefiner(nn.Module):
    """Conditional U-Net-style refiner for VAE-predicted next RGB frames."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        task_count: int,
        task_dim: int = 32,
        cond_dim: int = 256,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.task = nn.Embedding(task_count, task_dim)
        self.cond = nn.Sequential(
            nn.Linear(latent_dim + action_dim + task_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.cond_to_channels = nn.Linear(cond_dim, base_channels)
        self.enc1 = _block(6 + base_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.enc2 = _block(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.mid = _block(base_channels * 4, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = _block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec1 = _block(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, 6, 3, padding=1)

    def forward(
        self,
        prior_rgb: torch.Tensor,
        latent: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.cond(torch.cat([latent, action, self.task(task_id)], dim=-1))
        cond_map = self.cond_to_channels(cond)[:, :, None, None].expand(-1, -1, prior_rgb.shape[-2], prior_rgb.shape[-1])
        x = torch.cat([prior_rgb, cond_map], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        d2 = self.dec2(torch.cat([self.up2(mid), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        residual = 0.35 * torch.tanh(self.out(d1))
        return (prior_rgb + residual).clamp(0.0, 1.0)


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )
