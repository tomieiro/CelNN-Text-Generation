"""Character language model driven by causal CelNN dynamics."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from celllm.ablation import (
    NORMAL_ABLATION,
    AblationConfig,
    AblationTrace,
)
from celllm.cell import (
    CYHFACelNNCell,
    CelNNCell,
    HebbianAttentionCelNNCell,
    PlasticCelNNCell,
    StateMatchedBankCelNNCell,
)
from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    ModelConfig,
    PlasticityConfig,
    StateMatchedBankConfig,
)


class CelNNLanguageModel(nn.Module):
    """Embed characters, evolve a cellular field, and decode once."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d)
        self.cell = CelNNCell(cfg, causal=True)
        self.readout = nn.Linear(cfg.d, cfg.vocab_size, bias=True)
        self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return next-character logits for each sequence position."""
        embedding = self.embed(tokens)
        cell_input = self.cell.control_input(embedding)
        state = torch.zeros_like(embedding)
        for _ in range(self.cfg.k):
            state = self.cell.step(state, cell_input)
        return self.readout(state)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return mean next-character cross-entropy in nats."""
        logits = self(tokens)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )


class PlasticCelNNLanguageModel(nn.Module):
    """CellLM with explicit Hebbian memory carried between causal blocks."""

    def __init__(
        self,
        cfg: ModelConfig,
        plasticity: PlasticityConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = replace(cfg, mixer="dense")
        self.plasticity_config = plasticity or PlasticityConfig()
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d)
        self.cell = PlasticCelNNCell(
            self.cfg, self.plasticity_config, causal=True
        )
        self.readout = nn.Linear(
            self.cfg.d, self.cfg.vocab_size, bias=True
        )
        self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def new_plastic_state(self, batch_size: int):
        """Start independent sequences or conversations with empty memory."""
        return self.cell.new_plastic_state(batch_size)

    def forward_with_state(
        self,
        tokens: torch.Tensor,
        plastic_state=None,
        *,
        update_plasticity: bool = True,
    ):
        """Return logits and the next explicit session memory."""
        if plastic_state is None:
            plastic_state = self.new_plastic_state(tokens.shape[0])
        embedding = self.embed(tokens)
        cell_input = self.cell.control_input(embedding)
        state = torch.zeros_like(embedding)
        for _ in range(self.cfg.k):
            state = self.cell.step_with_memory(
                state, cell_input, plastic_state
            )
        next_plastic_state = (
            self.cell.observe(plastic_state, state)
            if update_plasticity
            else plastic_state
        )
        return self.readout(state), next_plastic_state

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Evaluate one block from reset memory for standard model APIs."""
        logits, _ = self.forward_with_state(tokens)
        return logits

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Train causally across chunks while carrying fast-weight memory."""
        chunks = tokens.split(self.plasticity_config.chunk_size, dim=1)
        memory = self.new_plastic_state(tokens.shape[0])
        logits = []
        for chunk in chunks:
            chunk_logits, memory = self.forward_with_state(chunk, memory)
            logits.append(chunk_logits)
        joined = torch.cat(logits, dim=1)
        return F.cross_entropy(
            joined[:, :-1].reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )


class HebbianAttentionCelNNLanguageModel(nn.Module):
    """Causal CellLM with a global Delta-Hebbian attention baseline."""

    def __init__(
        self,
        cfg: ModelConfig,
        memory: HebbianAttentionConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = replace(cfg, mixer="dense")
        self.memory_config = memory or HebbianAttentionConfig(
            chunk_size=self.cfg.n
        )
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d)
        self.cell = HebbianAttentionCelNNCell(
            self.cfg, self.memory_config, causal=True
        )
        self.readout = nn.Linear(
            self.cfg.d, self.cfg.vocab_size, bias=True
        )
        self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def new_memory_state(self, batch_size: int):
        """Start independent sequences or conversations with empty memory."""
        return self.cell.new_memory_state(batch_size)

    def forward_with_state(
        self,
        tokens: torch.Tensor,
        memory_state=None,
        *,
        update_memory: bool = True,
        write_mask: torch.Tensor | None = None,
        ablation: AblationConfig = NORMAL_ABLATION,
        trace: AblationTrace | None = None,
    ):
        """Return logits and next explicit key--value memory."""
        ablation.validate(is_field=False)
        if memory_state is None:
            memory_state = self.new_memory_state(tokens.shape[0])
        if not ablation.carry_enabled:
            memory_state = memory_state.reset()
        embedding = self.embed(tokens)
        cell_input = self.cell.control_input(embedding)
        state = torch.zeros_like(embedding)
        for _ in range(self.cfg.k):
            state = self.cell.step_with_memory(
                state,
                cell_input,
                memory_state,
                retrieval_enabled=ablation.retrieval_enabled,
                trace=trace,
            )
        next_memory = (
            self.cell.observe(memory_state, state, mask=write_mask)
            if update_memory
            else memory_state
        )
        return self.readout(state), next_memory

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_state(tokens)
        return logits

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        chunks = tokens.split(self.memory_config.chunk_size, dim=1)
        memory = self.new_memory_state(tokens.shape[0])
        logits = []
        for chunk in chunks:
            chunk_logits, memory = self.forward_with_state(chunk, memory)
            logits.append(chunk_logits)
        joined = torch.cat(logits, dim=1)
        return F.cross_entropy(
            joined[:, :-1].reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )


class StateMatchedBankCelNNLanguageModel(nn.Module):
    """CelNN control with a global bank matching CY-HFA recurrent state."""

    def __init__(
        self,
        cfg: ModelConfig,
        bank: StateMatchedBankConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = replace(cfg, mixer="dense")
        self.bank_config = bank or StateMatchedBankConfig(
            slots=self.cfg.n,
            key_size=max(32, self.cfg.n),
            chunk_size=self.cfg.n,
        )
        if self.bank_config.slots != self.cfg.n:
            raise ValueError("state-matched bank slots must equal lattice size")
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d)
        self.cell = StateMatchedBankCelNNCell(
            self.cfg, self.bank_config, causal=True
        )
        self.readout = nn.Linear(
            self.cfg.d, self.cfg.vocab_size, bias=True
        )
        self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def new_memory_state(self, batch_size: int):
        return self.cell.new_memory_state(batch_size)

    def forward_with_state(
        self,
        tokens: torch.Tensor,
        memory_state=None,
        *,
        update_memory: bool = True,
        write_mask: torch.Tensor | None = None,
        ablation: AblationConfig = NORMAL_ABLATION,
        trace: AblationTrace | None = None,
    ):
        ablation.validate(is_field=False)
        if memory_state is None:
            memory_state = self.new_memory_state(tokens.shape[0])
        if not ablation.carry_enabled:
            memory_state = memory_state.reset()
        if trace is not None:
            trace.record("block_input_memory", memory_state.memory)
            trace.record(
                "block_input_normalizer", memory_state.normalizer
            )
        embedding = self.embed(tokens)
        cell_input = self.cell.control_input(embedding)
        state = torch.zeros_like(embedding)
        for _ in range(self.cfg.k):
            state = self.cell.step_with_memory(
                state,
                cell_input,
                memory_state,
                retrieval_enabled=ablation.retrieval_enabled,
                trace=trace,
            )
        next_memory = (
            self.cell.observe(
                memory_state,
                state,
                mask=write_mask,
                write_enabled=ablation.write_enabled,
                trace=trace,
            )
            if update_memory
            else memory_state
        )
        return self.readout(state), next_memory

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_state(tokens)
        return logits

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        chunks = tokens.split(self.bank_config.chunk_size, dim=1)
        memory = self.new_memory_state(tokens.shape[0])
        logits = []
        for chunk in chunks:
            chunk_logits, memory = self.forward_with_state(chunk, memory)
            logits.append(chunk_logits)
        joined = torch.cat(logits, dim=1)
        return F.cross_entropy(
            joined[:, :-1].reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )


class CYHFACelNNLanguageModel(nn.Module):
    """CellLM with coupled Chua--Yang Hebbian Field Attention dynamics."""

    def __init__(
        self,
        cfg: ModelConfig,
        field: CYHFAConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = replace(cfg, mixer="dense")
        self.field_config = field or CYHFAConfig(chunk_size=self.cfg.n)
        if self.field_config.chunk_size > self.cfg.n:
            raise ValueError("field chunk size cannot exceed lattice size")
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d)
        self.cell = CYHFACelNNCell(
            self.cfg, self.field_config, causal=True
        )
        self.readout = nn.Linear(
            self.cfg.d, self.cfg.vocab_size, bias=True
        )
        self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def new_field_state(self, batch_size: int):
        """Create one normalized associative memory per lattice cell."""
        return self.cell.new_field_state(batch_size)

    def _pad_block(
        self,
        tokens: torch.Tensor,
        write_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        length = tokens.shape[1]
        if not 0 < length <= self.cfg.n:
            raise ValueError("sequence length must be in [1, lattice size]")
        if write_mask is None:
            write_mask = torch.ones_like(tokens, dtype=torch.bool)
        elif write_mask.shape != tokens.shape:
            raise ValueError("write mask must match token shape")
        padding = self.cfg.n - length
        if padding:
            tokens = F.pad(tokens, (0, padding), value=0)
            write_mask = F.pad(write_mask, (0, padding), value=False)
        return tokens, write_mask, length

    def forward_with_state(
        self,
        tokens: torch.Tensor,
        field_state=None,
        *,
        update_memory: bool = True,
        write_mask: torch.Tensor | None = None,
        ablation: AblationConfig = NORMAL_ABLATION,
        trace: AblationTrace | None = None,
    ):
        """Refine neural and memory fields, returning explicit session state."""
        ablation.validate(is_field=True)
        tokens, write_mask, length = self._pad_block(tokens, write_mask)
        if field_state is None:
            field_state = self.new_field_state(tokens.shape[0])
        embedding = self.embed(tokens)
        cell_input = self.cell.control_input(embedding)
        state = torch.zeros_like(embedding)
        working_field = self.cell.attention.begin_block(
            field_state,
            carry_enabled=ablation.carry_enabled,
            trace=trace,
        )
        for _ in range(self.cfg.k):
            state, working_field = self.cell.step_with_field(
                state,
                cell_input,
                working_field,
                write_mask=write_mask,
                retrieval_enabled=ablation.retrieval_enabled,
                write_enabled=ablation.write_enabled,
                diffusion_enabled=ablation.diffusion_enabled,
                trace=trace,
            )
        next_field = working_field if update_memory else field_state
        return self.readout(state[:, :length]), next_field

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_state(tokens)
        return logits

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        chunks = tokens.split(self.field_config.chunk_size, dim=1)
        memory = self.new_field_state(tokens.shape[0])
        logits = []
        for chunk in chunks:
            chunk_logits, memory = self.forward_with_state(chunk, memory)
            logits.append(chunk_logits)
        joined = torch.cat(logits, dim=1)
        return F.cross_entropy(
            joined[:, :-1].reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
