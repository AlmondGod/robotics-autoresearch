from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class NanoVideoGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128,
        task_count: int = 5,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        block_size: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.task_emb = nn.Embedding(task_count, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, tokens: torch.Tensor, task_id: torch.Tensor, targets: torch.Tensor | None = None):
        tokens = tokens[:, : self.block_size]
        x = self.token_emb(tokens)
        x = x + self.task_emb(task_id).unsqueeze(1)
        x = x + self.pos_emb[:, : x.shape[1]]
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1).bool()
        x = self.blocks(x, mask=mask)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            targets = targets[:, : logits.shape[1]]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss
