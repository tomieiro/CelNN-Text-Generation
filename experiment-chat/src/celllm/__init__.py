"""CellLM — Cellular Language Model."""

from typing import Any

from celllm.attention import (
    CausalFieldPropagation,
    ChuaYangHebbianFieldAttention,
    DeltaHebbianAttention,
    InterBlockFieldCarry,
    StateMatchedGlobalBankAttention,
)
from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    ModelConfig,
    PlasticityConfig,
    StateMatchedBankConfig,
    TrainConfig,
)
from celllm.controls import GatedConvLM
from celllm.model import (
    CYHFACelNNLanguageModel,
    CelNNLanguageModel,
    HebbianAttentionCelNNLanguageModel,
    PlasticCelNNLanguageModel,
    StateMatchedBankCelNNLanguageModel,
)
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
    "CYHFAConfig",
    "CYHFACelNNLanguageModel",
    "CausalFieldPropagation",
    "CelNNLanguageModel",
    "DeltaHebbianAttention",
    "GatedConvLM",
    "HebbianAttentionCelNNLanguageModel",
    "HebbianAttentionConfig",
    "ChuaYangHebbianFieldAttention",
    "InterBlockFieldCarry",
    "ModelConfig",
    "PlasticCelNNLanguageModel",
    "PlasticityConfig",
    "StateMatchedBankCelNNLanguageModel",
    "StateMatchedBankConfig",
    "StateMatchedGlobalBankAttention",
    "TrainConfig",
    "build_rung",
    "load_plastic_checkpoint",
    "save_plastic_checkpoint",
]
