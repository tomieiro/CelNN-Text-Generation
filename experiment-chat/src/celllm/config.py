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


@dataclass(frozen=True)
class PlasticityConfig:
    """Fast-weight memory settings for a plastic CellLM session."""

    rule: str = "oja"
    learning_rate: float = 0.01
    decay: float = 0.99
    alpha: float = 0.1
    learnable_alpha: bool = True
    detach_updates: bool = True
    memory_limit: float | None = 1.0
    chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.rule not in {"hebbian", "oja"}:
            raise ValueError("plasticity rule must be 'hebbian' or 'oja'")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")


@dataclass(frozen=True)
class HebbianAttentionConfig:
    """Global Delta-Hebbian attention baseline settings."""

    key_size: int = 32
    value_size: int = 32
    learning_rate: float = 0.1
    min_retention: float = 0.95
    retrieval_scale: float = 0.1
    learnable_retrieval_scale: bool = True
    detach_updates: bool = False
    memory_limit: float | None = 1.0
    chunk_size: int = 16

    def __post_init__(self) -> None:
        if self.key_size < 1 or self.value_size < 1:
            raise ValueError("attention key and value sizes must be positive")
        if self.learning_rate < 0:
            raise ValueError("attention learning rate must be non-negative")
        if not 0 <= self.min_retention <= 1:
            raise ValueError("minimum retention must be between zero and one")
        if self.retrieval_scale < 0:
            raise ValueError("retrieval scale must be non-negative")
        if self.memory_limit is not None and self.memory_limit <= 0:
            raise ValueError("memory limit must be positive or None")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")


@dataclass(frozen=True)
class CYHFAConfig:
    """Chua--Yang Hebbian Field Attention settings.

    The associative state has one normalized key--value memory per lattice
    cell. ``diffusion_rate`` is the total causal coupling distributed over
    ``diffusion_radius`` preceding neighbours on every refinement step.
    """

    key_size: int = 32
    value_size: int = 32
    learning_rate: float = 0.1
    min_retention: float = 0.99
    diffusion_rate: float = 0.1
    max_diffusion: float = 0.25
    diffusion_radius: int = 1
    learnable_diffusion: bool = True
    retrieval_scale: float = 0.1
    learnable_retrieval_scale: bool = True
    detach_updates: bool = False
    memory_limit: float | None = 1.0
    epsilon: float = 1e-6
    chunk_size: int = 16

    def __post_init__(self) -> None:
        if self.key_size < 1 or self.value_size < 1:
            raise ValueError("field key and value sizes must be positive")
        if self.learning_rate < 0:
            raise ValueError("field learning rate must be non-negative")
        if not 0 <= self.min_retention <= 1:
            raise ValueError("minimum retention must be between zero and one")
        if not 0 <= self.diffusion_rate <= self.max_diffusion:
            raise ValueError(
                "diffusion rate must be between zero and max_diffusion"
            )
        if not 0 < self.max_diffusion <= 1:
            raise ValueError("max diffusion must be in (0, 1]")
        if self.diffusion_radius < 1:
            raise ValueError("diffusion radius must be positive")
        if self.retrieval_scale < 0:
            raise ValueError("retrieval scale must be non-negative")
        if self.memory_limit is not None and self.memory_limit <= 0:
            raise ValueError("memory limit must be positive or None")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
