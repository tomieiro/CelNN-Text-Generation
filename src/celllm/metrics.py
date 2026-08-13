"""Quality, size, and analytic cost metrics for the capacity gate."""

from __future__ import annotations

import math

from torch import nn

from celllm.config import ModelConfig

_RANKS = {"rank4": 4, "rank8": 8, "rank16": 16, "rank32": 32}


def bits_per_character(nats: float) -> float:
    """Convert mean cross-entropy in nats to bits per character."""
    return nats / math.log(2)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Split cellular-core, embedding, and total parameter counts."""
    embedding = sum(parameter.numel() for parameter in model.embed.parameters())
    core = sum(parameter.numel() for parameter in model.cell.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"core": core, "embedding": embedding, "total": total}


def analytic_flops(cfg: ModelConfig) -> dict[str, int]:
    """Count multiply-accumulates for one sequence forward pass."""
    per_template = cfg.n * cfg.d * cfg.n_offsets
    spatial = per_template * cfg.k + per_template

    if cfg.mixer == "none":
        channel = 0
    elif cfg.mixer == "dense":
        channel = cfg.n * cfg.d * cfg.d * cfg.k
    else:
        q = _RANKS[cfg.mixer]
        channel = 2 * cfg.n * cfg.d * q * cfg.k

    readout = cfg.n * cfg.d * cfg.vocab_size
    return {
        "spatial": spatial,
        "channel": channel,
        "readout": readout,
        "total": spatial + channel + readout,
    }


def gated_conv_flops(cfg: ModelConfig, layers: int = 4, kernel: int = 3) -> int:
    """Count multiply-accumulates for the rung-H gated convolutional control."""
    convolution = layers * cfg.n * kernel * cfg.d * (2 * cfg.d)
    readout = cfg.n * cfg.d * cfg.vocab_size
    return convolution + readout
