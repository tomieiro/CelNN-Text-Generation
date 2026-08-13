"""Conventional gated causal convolutional language-model control."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from celllm.config import ModelConfig


class GatedConvBlock(nn.Module):
    """Apply a residual gated convolution with causal left padding."""

    def __init__(self, d: int, kernel: int = 3) -> None:
        super().__init__()
        self.pad = kernel - 1
        self.conv = nn.Conv1d(d, 2 * d, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.pad(x.transpose(1, 2), (self.pad, 0))
        hidden = self.conv(hidden).transpose(1, 2)
        return x + F.glu(hidden, dim=-1)


class GatedConvLM(nn.Module):
    """Rung H control without ODE semantics or depth-wise weight sharing."""

    def __init__(self, cfg: ModelConfig, layers: int = 4) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d)
        self.blocks = nn.ModuleList(GatedConvBlock(cfg.d) for _ in range(layers))
        self.readout = nn.Linear(cfg.d, cfg.vocab_size, bias=True)
        self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    @property
    def cell(self) -> nn.Module:
        """Expose convolution blocks as the core for parameter accounting."""
        return self.blocks

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self.readout(hidden)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return mean next-character cross-entropy in nats."""
        logits = self(tokens)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
