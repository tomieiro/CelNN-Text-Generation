"""Persistence for plastic CellLM slow weights and plasticity configuration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from celllm.config import ModelConfig, PlasticityConfig
from celllm.model import PlasticCelNNLanguageModel


def save_plastic_checkpoint(
    path: str | Path,
    model: PlasticCelNNLanguageModel,
    *,
    result: dict | None = None,
) -> None:
    """Atomically save slow weights; transient session memory is excluded."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "architecture": "plastic-celnn-language-model",
            "model_config": asdict(model.cfg),
            "plasticity_config": asdict(model.plasticity_config),
            "model_state": model.state_dict(),
            "result": result or {},
        },
        temporary,
    )
    temporary.replace(destination)


def load_plastic_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[PlasticCelNNLanguageModel, dict]:
    """Reconstruct a plastic model with reset conversation memory."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "plastic-celnn-language-model":
        raise ValueError("checkpoint is not a plastic CellLM model")
    model = PlasticCelNNLanguageModel(
        ModelConfig(**checkpoint["model_config"]),
        PlasticityConfig(**checkpoint["plasticity_config"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
