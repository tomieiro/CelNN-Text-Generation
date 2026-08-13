"""Character language model driven by causal CelNN dynamics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from celllm.cell import CelNNCell
from celllm.config import ModelConfig


class CelNNLanguageModel(nn.Module):
    """Embed characters, evolve a cellular field, and decode once."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d)
        self.cell = CelNNCell(cfg, causal=True)
        self.readout = nn.Linear(cfg.d, cfg.vocab_size, bias=True)
        self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return next-character logits for each sequence position."""
        embedding = self.embed(tokens)
        cell_input = self.cell.control_input(embedding)
        state = torch.zeros_like(embedding)
        for _ in range(self.cfg.k):
            state = self.cell.step(state, cell_input)
        return self.readout(state)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return mean next-character cross-entropy in nats."""
        logits = self(tokens)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
