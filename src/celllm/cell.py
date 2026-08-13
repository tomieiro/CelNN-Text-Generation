"""Differentiable Euler integration of the Chua-Yang CelNN ODE."""

from __future__ import annotations

import torch
from torch import nn

from celllm.config import ModelConfig
from celllm.mixers import build_mixer
from celllm.templates import build_template


def piecewise_linear(x: torch.Tensor) -> torch.Tensor:
    """Apply the classical CelNN output saturation bounded to [-1, 1]."""
    return 0.5 * ((x + 1.0).abs() - (x - 1.0).abs())


class CelNNCell(nn.Module):
    """Combine feedback, control, bias, and channel mixing in one Euler step."""

    def __init__(self, cfg: ModelConfig, causal: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.eta = cfg.eta
        self.a = build_template(cfg.spatial, cfg.r, cfg.d, causal)
        self.b = build_template(cfg.spatial, cfg.r, cfg.d, causal)
        self.z = nn.Parameter(torch.zeros(cfg.d))
        self.mixer = build_mixer(cfg.mixer, cfg.d)

    def control_drive(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute the state-independent B*E+z contribution once per field."""
        cell_input = (
            torch.tanh(embedding) if self.cfg.bound_drive else embedding
        )
        return self.b(cell_input) + self.z

    def step(self, x: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        """Advance the cellular field by one explicit Euler step."""
        derivative = (
            -x + self.a(piecewise_linear(x)) + self.mixer(x) + drive
        )
        return x + self.eta * derivative
