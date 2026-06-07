from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TinyVQTokenizer(nn.Module):
    """Tiny image tokenizer: 64x64 RGB -> 8x8 discrete tokens."""

    def __init__(self, codebook_size: int = 128, embed_dim: int = 64):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, 4, stride=2, padding=1),
        )
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def encode_indices(self, images: torch.Tensor) -> torch.Tensor:
        z = self.encode_features(images)
        flat = z.permute(0, 2, 3, 1).reshape(-1, self.embed_dim)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.square().sum(dim=1)
        )
        return distances.argmin(dim=1).reshape(z.shape[0], z.shape[2], z.shape[3])

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.codebook(indices).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode_features(images)
        indices = self.encode_indices(images)
        z_q = self.codebook(indices).permute(0, 3, 1, 2).contiguous()
        z_st = z + (z_q - z).detach()
        recon = self.decoder(z_st)
        recon_loss = F.mse_loss(recon, images)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        loss = recon_loss + codebook_loss + 0.25 * commitment_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "indices": indices,
            "recon": recon,
        }


def images_to_tensor(images) -> torch.Tensor:
    tensor = torch.as_tensor(images, dtype=torch.float32)
    if tensor.ndim == 5:
        tensor = tensor[:, -1]
    if tensor.ndim != 4:
        raise ValueError(f"expected NHWC images, got {tuple(tensor.shape)}")
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor.permute(0, 3, 1, 2).contiguous()
