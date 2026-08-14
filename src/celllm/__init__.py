"""CellLM — Cellular Language Model."""

from typing import Any

from celllm.config import ModelConfig, PlasticityConfig, TrainConfig
from celllm.controls import GatedConvLM
from celllm.model import CelNNLanguageModel, PlasticCelNNLanguageModel
from celllm.plastic_checkpoint import (
    load_plastic_checkpoint,
    save_plastic_checkpoint,
)


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
    "PlasticCelNNLanguageModel",
    "PlasticityConfig",
    "TrainConfig",
    "build_rung",
    "load_plastic_checkpoint",
    "save_plastic_checkpoint",
]
