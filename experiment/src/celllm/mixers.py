"""Pointwise channel mixing for the Experiment 0 capacity ladder."""

from __future__ import annotations

import torch
from torch import nn

_RANKS = {"rank4": 4, "rank8": 8, "rank16": 16, "rank32": 32}


class NoMixer(nn.Module):
    """Return no channel contribution and introduce no parameters."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class RankQMixer(nn.Module):
    """Mix channels through a bias-free rank-q bottleneck."""

    def __init__(self, d: int, q: int) -> None:
        super().__init__()
        self.down = nn.Linear(d, q, bias=False)
        self.up = nn.Linear(q, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class DenseMixer(nn.Module):
    """Apply the dense pointwise upper-bound control."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def build_mixer(name: str, d: int) -> nn.Module:
    """Build a channel mixer by its capacity-ladder name."""
    if name == "none":
        return NoMixer(d)
    if name == "dense":
        return DenseMixer(d)
    if name in _RANKS:
        return RankQMixer(d, _RANKS[name])
    raise ValueError(
        f"unknown mixer {name!r}; expected none, rank{{4,8,16,32}} or dense"
    )
