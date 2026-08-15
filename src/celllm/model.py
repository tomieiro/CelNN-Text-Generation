"""Character language model driven by causal CelNN dynamics."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from celllm.cell import (
    CelNNCell,
    HebbianAttentionCelNNCell,
    PlasticCelNNCell,
)
from celllm.config import (
    HebbianAttentionConfig,
    ModelConfig,
    PlasticityConfig,
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
    """Causal CellLM with constant-size Delta-Hebbian key--value attention."""

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
    ):
        """Return logits and next explicit key--value memory."""
        if memory_state is None:
            memory_state = self.new_memory_state(tokens.shape[0])
        embedding = self.embed(tokens)
        cell_input = self.cell.control_input(embedding)
        state = torch.zeros_like(embedding)
        for _ in range(self.cfg.k):
            state = self.cell.step_with_memory(
                state, cell_input, memory_state
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
