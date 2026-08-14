"""Self-contained model metadata for training and serving CellLM chat."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from celllm.chat_model import CellLMChatModel
from celllm.config import ModelConfig, PlasticityConfig


def save_chat_checkpoint(
    path: str | Path,
    model: CellLMChatModel,
    *,
    step: int,
    metrics: dict | None = None,
    optimizer_state: dict | None = None,
) -> None:
    """Atomically save trainable state without transient conversations."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "architecture": "celllm-chat",
            "step": step,
            "model_config": asdict(model.cfg),
            "plasticity_config": asdict(model.plasticity_config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer_state,
            "metrics": metrics or {},
        },
        temporary,
    )
    temporary.replace(destination)


def load_chat_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[CellLMChatModel, dict]:
    """Reconstruct a chat model with empty session memory."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "celllm-chat":
        raise ValueError("checkpoint is not a CellLM chat model")
    model = CellLMChatModel(
        ModelConfig(**checkpoint["model_config"]),
        PlasticityConfig(**checkpoint["plasticity_config"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
