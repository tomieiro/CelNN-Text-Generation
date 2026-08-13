"""Configuration objects for Experiment 0."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Shape and dynamics of a CellLM capacity-gate model.

    The receptive-field constraint ``k * r + 1 >= n`` keeps locality out of
    Experiment 0: with a causal neighbourhood, the reach after ``k`` Euler
    steps is ``k * r + 1``. A failure can therefore be attributed to capacity
    rather than distance.
    """

    n: int = 64
    d: int = 128
    r: int = 2
    k: int = 32
    eta: float = 0.5
    vocab_size: int = 27
    spatial: str = "diagonal"
    mixer: str = "none"
    bound_drive: bool = True

    def __post_init__(self) -> None:
        if self.k * self.r + 1 < self.n:
            minimum_k = -(-(self.n - 1) // self.r)
            raise ValueError(
                f"receptive field k*r+1={self.k * self.r + 1} does not cover "
                f"n={self.n}; raise k to at least {minimum_k}"
            )
        if self.spatial not in {"scalar", "diagonal"}:
            raise ValueError(f"unknown spatial template {self.spatial!r}")

    @property
    def offsets(self) -> tuple[int, ...]:
        """Causal neighbourhood offsets, from ``-r`` through zero."""
        return tuple(range(-self.r, 1))

    @property
    def n_offsets(self) -> int:
        """Number of offsets in the causal neighbourhood."""
        return self.r + 1


@dataclass(frozen=True)
class TrainConfig:
    """Training defaults for a capacity-ladder run."""

    steps: int = 20_000
    batch_size: int = 128
    lr: float = 3e-3
    warmup: int = 500
    grad_clip: float = 1.0
    seed: int = 42
    eval_every: int = 1_000
    eval_batches: int = 50
