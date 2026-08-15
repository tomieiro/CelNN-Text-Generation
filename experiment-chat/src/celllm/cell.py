"""CellLM adapter around the canonical libPyCelNN differentiable dynamics."""

from __future__ import annotations

import torch
from torch import nn

from celnn import DifferentiableCellularNetwork
from celnn import piecewise_linear as _piecewise_linear

from celllm.ablation import AblationTrace
from celllm.config import ModelConfig, PlasticityConfig
from celllm.attention import (
    ChuaYangHebbianFieldAttention,
    DeltaHebbianAttention,
    StateMatchedGlobalBankAttention,
)
from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    StateMatchedBankConfig,
)
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


class HebbianAttentionCelNNCell(CelNNCell):
    """CelNN dynamics coupled to explicit Delta-Hebbian fast attention."""

    def __init__(
        self,
        cfg: ModelConfig,
        memory: HebbianAttentionConfig,
        causal: bool = True,
    ) -> None:
        super().__init__(cfg, causal=causal)
        self.attention = DeltaHebbianAttention(cfg.d, memory)

    def new_memory_state(self, batch_size: int):
        return self.attention.new_state(batch_size)

    def step_with_memory(
        self,
        x: torch.Tensor,
        cell_input: torch.Tensor,
        memory_state,
        *,
        retrieval_enabled: bool = True,
        trace: AblationTrace | None = None,
    ) -> torch.Tensor:
        """Read fixed fast weights while advancing one cellular step."""
        output = piecewise_linear(x)
        memory_drive = self.attention.retrieve(
            output,
            memory_state,
            enabled=retrieval_enabled,
            trace=trace,
        )
        return self.dynamics.step(
            x,
            cell_input,
            extra_drive=self.mixer(x) + memory_drive,
        )

    def observe(self, memory_state, activity: torch.Tensor, mask=None):
        """Write bounded outputs only after a causal block is complete."""
        return self.attention.write(
            memory_state, piecewise_linear(activity), mask=mask
        )


class StateMatchedBankCelNNCell(CelNNCell):
    """CelNN dynamics reading a state-matched non-spatial global bank."""

    def __init__(
        self,
        cfg: ModelConfig,
        bank: StateMatchedBankConfig,
        causal: bool = True,
    ) -> None:
        super().__init__(cfg, causal=causal)
        self.attention = StateMatchedGlobalBankAttention(cfg.d, bank)

    def new_memory_state(self, batch_size: int):
        return self.attention.new_state(batch_size)

    def step_with_memory(
        self,
        x: torch.Tensor,
        cell_input: torch.Tensor,
        memory_state,
        *,
        retrieval_enabled: bool = True,
        trace: AblationTrace | None = None,
    ) -> torch.Tensor:
        output = piecewise_linear(x)
        memory_drive = self.attention.retrieve(
            output,
            memory_state,
            enabled=retrieval_enabled,
            trace=trace,
        )
        return self.dynamics.step(
            x,
            cell_input,
            extra_drive=self.mixer(x) + memory_drive,
        )

    def observe(
        self,
        memory_state,
        activity: torch.Tensor,
        mask=None,
        *,
        write_enabled: bool = True,
        trace: AblationTrace | None = None,
    ):
        updated = (
            self.attention.write(
                memory_state, piecewise_linear(activity), mask=mask
            )
            if write_enabled
            else memory_state
        )
        if trace is not None:
            trace.record_delta(
                "write_memory_delta",
                memory_state.memory,
                updated.memory,
            )
            trace.record_delta(
                "write_normalizer_delta",
                memory_state.normalizer,
                updated.normalizer,
            )
        return updated


class CYHFACelNNCell(CelNNCell):
    """Evolve Chua--Yang neural and associative field states together."""

    def __init__(
        self, cfg: ModelConfig, field: CYHFAConfig, causal: bool = True
    ) -> None:
        if not causal:
            raise ValueError("CY-HFA language modeling requires causal dynamics")
        super().__init__(cfg, causal=True)
        if field.diffusion_radius > cfg.r:
            raise ValueError(
                "field diffusion radius cannot exceed the CelNN radius"
            )
        self.attention = ChuaYangHebbianFieldAttention(cfg.d, field)

    def new_field_state(self, batch_size: int):
        return self.attention.new_state(batch_size, self.cfg.n)

    def step_with_field(
        self,
        x: torch.Tensor,
        cell_input: torch.Tensor,
        field_state,
        *,
        write_mask: torch.Tensor | None = None,
        retrieval_enabled: bool = True,
        write_enabled: bool = True,
        diffusion_enabled: bool = True,
        trace: AblationTrace | None = None,
    ):
        """Advance ``(x, M, s)`` by one coupled causal refinement step."""
        output = piecewise_linear(x)
        memory_drive = self.attention.retrieve(
            output,
            field_state,
            enabled=retrieval_enabled,
            trace=trace,
        )
        next_x = self.dynamics.step(
            x,
            cell_input,
            extra_drive=self.mixer(x) + memory_drive,
        )
        next_field = self.attention.advance(
            field_state,
            piecewise_linear(next_x),
            mask=write_mask,
            write_enabled=write_enabled,
            diffusion_enabled=diffusion_enabled,
            trace=trace,
        )
        return next_x, next_field
