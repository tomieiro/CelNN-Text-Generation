"""Self-contained model metadata for training and serving CellLM chat."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from celllm.chat_model import CellLMChatModel
from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    ModelConfig,
    PlasticityConfig,
)


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
    is_field = model.field_config is not None
    is_global = model.memory_config is not None
    if is_field:
        configuration = {"field_config": asdict(model.field_config)}
        format_version = 3
        architecture = "celllm-chat-cy-hfa"
    elif is_global:
        configuration = {"memory_config": asdict(model.memory_config)}
        format_version = 2
        architecture = "celllm-chat-delta-hebb"
    else:
        configuration = {"plasticity_config": asdict(model.plasticity_config)}
        format_version = 1
        architecture = "celllm-chat"
    torch.save(
        {
            "format_version": format_version,
            "architecture": architecture,
            "step": step,
            "model_config": asdict(model.cfg),
            **configuration,
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
    architecture = checkpoint.get("architecture")
    if architecture not in {
        "celllm-chat",
        "celllm-chat-delta-hebb",
        "celllm-chat-cy-hfa",
    }:
        raise ValueError("checkpoint is not a CellLM chat model")
    if architecture == "celllm-chat-cy-hfa":
        model = CellLMChatModel(
            ModelConfig(**checkpoint["model_config"]),
            field=CYHFAConfig(**checkpoint["field_config"]),
        )
    elif architecture == "celllm-chat-delta-hebb":
        model = CellLMChatModel(
            ModelConfig(**checkpoint["model_config"]),
            memory=HebbianAttentionConfig(**checkpoint["memory_config"]),
        )
    else:
        model = CellLMChatModel(
            ModelConfig(**checkpoint["model_config"]),
            PlasticityConfig(**checkpoint["plasticity_config"]),
        )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
