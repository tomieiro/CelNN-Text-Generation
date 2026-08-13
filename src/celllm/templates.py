"""Structured spatial coupling for the CellLM dynamics."""

from __future__ import annotations

import torch
from torch import nn

from celllm.stencil import aggregate


class ScalarTemplate(nn.Module):
    """Use one scalar per offset, shared by every state channel."""

    def __init__(self, r: int, d: int, causal: bool = True) -> None:
        super().__init__()
        self.r = r
        self.causal = causal
        n_offsets = r + 1 if causal else 2 * r + 1
        self.weights = nn.Parameter(torch.randn(n_offsets, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return aggregate(x, self.weights, self.r, self.causal)


class DiagonalTemplate(nn.Module):
    """Use one scalar per offset and state channel."""

    def __init__(self, r: int, d: int, causal: bool = True) -> None:
        super().__init__()
        self.r = r
        self.causal = causal
        n_offsets = r + 1 if causal else 2 * r + 1
        self.weights = nn.Parameter(torch.randn(n_offsets, d) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return aggregate(x, self.weights, self.r, self.causal)


def build_template(
    kind: str,
    r: int,
    d: int,
    causal: bool = True,
) -> nn.Module:
    """Build a spatial template by its configuration name."""
    if kind == "scalar":
        return ScalarTemplate(r, d, causal)
    if kind == "diagonal":
        return DiagonalTemplate(r, d, causal)
    raise ValueError(f"unknown template kind {kind!r}")
