"""CellLM — Cellular Language Model."""

from typing import Any

from celllm.config import ModelConfig, TrainConfig
from celllm.controls import GatedConvLM
from celllm.model import CelNNLanguageModel


def __getattr__(name: str) -> Any:
    """Load ladder exports lazily so ``python -m celllm.ladder`` stays clean."""
    if name in {"RUNGS", "build_rung"}:
        from celllm import ladder

        return getattr(ladder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RUNGS",
    "CelNNLanguageModel",
    "GatedConvLM",
    "ModelConfig",
    "TrainConfig",
    "build_rung",
]
