"""Pointwise channel mixing for the Experiment 0 capacity ladder."""

from __future__ import annotations

import torch
from torch import nn

from celnn import HebbianRule, OjaRule, Plasticity, PlasticityState

from celllm.config import PlasticityConfig

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


class PlasticDenseMixer(nn.Module):
    """Dense slow mixing plus caller-owned Hebbian fast weights."""

    def __init__(self, d: int, config: PlasticityConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        rule = (
            HebbianRule(config.learning_rate, config.decay)
            if config.rule == "hebbian"
            else OjaRule(config.learning_rate, config.decay)
        )
        self.plasticity = Plasticity(
            rule,
            alpha=config.alpha,
            learnable_alpha=config.learnable_alpha,
            detach_updates=config.detach_updates,
            memory_limit=config.memory_limit,
        )

    def new_state(self, batch_size: int) -> PlasticityState:
        """Return empty per-sequence memory on the mixer's device."""
        return self.plasticity.new_state(batch_size, self.proj.weight)

    def forward(
        self, x: torch.Tensor, state: PlasticityState
    ) -> torch.Tensor:
        """Mix channels using fixed memory for the current causal block."""
        weight = self.plasticity.effective_weight(self.proj.weight, state)
        return torch.einsum("b...i,boi->b...o", x, weight)

    def observe(
        self, state: PlasticityState, activity: torch.Tensor
    ) -> PlasticityState:
        """Publish an auto-associative update for the next causal block."""
        return self.plasticity.update(state, activity, activity)


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
