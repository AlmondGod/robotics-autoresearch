from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TinyInverseDynamics(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128,
        task_count: int = 5,
        action_dim: int = 7,
        proprio_dim: int = 1,
        n_embd: int = 128,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.task_emb = nn.Embedding(task_count, n_embd)
        self.proprio = nn.Linear(proprio_dim, n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(4 * n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, action_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        z_tp1: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        actions: torch.Tensor | None = None,
    ):
        h_t = self.token_emb(z_t).mean(dim=1)
        h_tp1 = self.token_emb(z_tp1).mean(dim=1)
        h_task = self.task_emb(task_id)
        h_prop = self.proprio(proprio)
        pred = self.mlp(torch.cat([h_t, h_tp1, h_prop, h_task], dim=-1))
        loss = None if actions is None else F.mse_loss(pred, actions)
        return pred, loss
