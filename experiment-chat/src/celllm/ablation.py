"""Explicit, testable inference interventions for CellLM fast memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class AblationCondition(str, Enum):
    """One causal intervention applied to an otherwise frozen checkpoint."""

    NORMAL = "normal"
    NO_RETRIEVAL = "no_retrieval"
    ZERO_HISTORY = "zero_history"
    NO_CARRY = "no_carry"
    NO_WRITE = "no_write"
    NO_DIFFUSION = "no_diffusion"


@dataclass(frozen=True)
class AblationConfig:
    """Select one intervention without modifying model parameters.

    ``ZERO_HISTORY`` is a boundary intervention orchestrated by the evaluator.
    All other conditions map directly to one path in the associative model.
    """

    condition: AblationCondition = AblationCondition.NORMAL

    @classmethod
    def named(cls, value: str | AblationCondition) -> "AblationConfig":
        return cls(AblationCondition(value))

    @property
    def retrieval_enabled(self) -> bool:
        return self.condition is not AblationCondition.NO_RETRIEVAL

    @property
    def carry_enabled(self) -> bool:
        return self.condition is not AblationCondition.NO_CARRY

    @property
    def write_enabled(self) -> bool:
        return self.condition is not AblationCondition.NO_WRITE

    @property
    def diffusion_enabled(self) -> bool:
        return self.condition is not AblationCondition.NO_DIFFUSION

    @property
    def resets_history(self) -> bool:
        return self.condition is AblationCondition.ZERO_HISTORY

    def validate(self, *, is_field: bool) -> None:
        if self.condition is AblationCondition.NO_DIFFUSION and not is_field:
            raise ValueError("no_diffusion is only defined for CY-HFA")


@dataclass
class TensorSummary:
    """Streaming magnitude summary that never retains evaluation tensors."""

    calls: int = 0
    elements: int = 0
    sum_squares: float = 0.0
    max_abs: float = 0.0

    def record(self, value: torch.Tensor) -> None:
        detached = value.detach().float()
        self.calls += 1
        self.elements += detached.numel()
        self.sum_squares += float(detached.square().sum().cpu())
        if detached.numel():
            self.max_abs = max(
                self.max_abs, float(detached.abs().max().cpu())
            )

    @property
    def rms(self) -> float:
        if not self.elements:
            return 0.0
        return (self.sum_squares / self.elements) ** 0.5

    def to_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "elements": self.elements,
            "rms": self.rms,
            "max_abs": self.max_abs,
        }


@dataclass
class AblationTrace:
    """Aggregate the actual causal terms produced after intervention."""

    terms: dict[str, TensorSummary] = field(default_factory=dict)

    def record(self, name: str, value: torch.Tensor) -> None:
        self.terms.setdefault(name, TensorSummary()).record(value)

    def record_delta(
        self, name: str, before: torch.Tensor, after: torch.Tensor
    ) -> None:
        self.record(name, after - before)

    def maximum(self, name: str) -> float:
        summary = self.terms.get(name)
        return 0.0 if summary is None else summary.max_abs

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        return {
            name: summary.to_dict()
            for name, summary in sorted(self.terms.items())
        }


NORMAL_ABLATION = AblationConfig()
