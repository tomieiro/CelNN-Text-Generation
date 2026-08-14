"""CellLM adapter around the canonical libPyCelNN differentiable dynamics."""

from __future__ import annotations

import torch
from torch import nn

from celnn import DifferentiableCellularNetwork
from celnn import piecewise_linear as _piecewise_linear

from celllm.config import ModelConfig, PlasticityConfig
from celllm.mixers import PlasticDenseMixer, build_mixer


def piecewise_linear(x: torch.Tensor) -> torch.Tensor:
    """Expose the libPyCelNN canonical saturation for CellLM callers."""
    return _piecewise_linear(x)


class CelNNCell(nn.Module):
    """Compose libPyCelNN dynamics with caller-owned channel mixing."""

    def __init__(self, cfg: ModelConfig, causal: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.dynamics = DifferentiableCellularNetwork(
            radius=cfg.r,
            channels=cfg.d,
            activation="piecewise_linear",
            boundary="constant",
            dt=cfg.eta,
            steps=1,
            method="euler",
            causal=causal,
            shared_channels=cfg.spatial == "scalar",
        )
        with torch.no_grad():
            self.dynamics.feedback.normal_(std=0.1)
            self.dynamics.control.normal_(std=0.1)
        self.mixer = build_mixer(cfg.mixer, cfg.d)

    def control_input(self, embedding: torch.Tensor) -> torch.Tensor:
        """Bound learned embeddings to the classical CelNN input range."""
        return torch.tanh(embedding) if self.cfg.bound_drive else embedding

    def step(self, x: torch.Tensor, cell_input: torch.Tensor) -> torch.Tensor:
        """Advance through libPyCelNN with an external channel-mixing drive."""
        return self.dynamics.step(
            x,
            cell_input,
            extra_drive=self.mixer(x),
        )


class PlasticCelNNCell(CelNNCell):
    """CelNN cell whose dense channel drive has explicit fast weights."""

    def __init__(
        self, cfg: ModelConfig, plasticity: PlasticityConfig, causal: bool = True
    ) -> None:
        super().__init__(cfg, causal=causal)
        self.mixer = PlasticDenseMixer(cfg.d, plasticity)

    def new_plastic_state(self, batch_size: int):
        return self.mixer.new_state(batch_size)

    def step_with_memory(
        self, x: torch.Tensor, cell_input: torch.Tensor, plastic_state
    ) -> torch.Tensor:
        """Advance without modifying memory inside the current block."""
        return self.dynamics.step(
            x,
            cell_input,
            extra_drive=self.mixer(x, plastic_state),
        )

    def observe(self, plastic_state, activity: torch.Tensor):
        """Update memory from bounded Chua--Yang cell outputs."""
        output = piecewise_linear(activity)
        return self.mixer.observe(plastic_state, output)
