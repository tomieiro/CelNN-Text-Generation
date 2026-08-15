"""Delta-Hebbian key--value attention for bounded CelNN activity."""

from __future__ import annotations

import torch
from celnn import DeltaHebbianMemory, DeltaHebbianRule
from torch import nn

from celllm.config import HebbianAttentionConfig


class DeltaHebbianAttention(nn.Module):
    """Learn what to query, store, retrieve, forget, and write.

    The projections and gates are slow learned parameters. The matrix returned
    by :meth:`new_state` is transient fast memory owned by one conversation.
    Reading never mutates it; writing uses a local Delta-Hebbian correction.
    """

    def __init__(self, dimensions: int, config: HebbianAttentionConfig) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.config = config
        self.query = nn.Linear(dimensions, config.key_size, bias=False)
        self.key = nn.Linear(dimensions, config.key_size, bias=False)
        self.value = nn.Linear(dimensions, config.value_size, bias=False)
        self.output = nn.Linear(config.value_size, dimensions, bias=False)
        self.write_controls = nn.Linear(dimensions, 2)
        nn.init.zeros_(self.write_controls.weight)
        with torch.no_grad():
            self.write_controls.bias.copy_(torch.tensor([-1.0, 2.0]))

        scale = torch.tensor(float(config.retrieval_scale))
        if config.learnable_retrieval_scale:
            self.retrieval_scale = nn.Parameter(scale)
        else:
            self.register_buffer("retrieval_scale", scale)
        self.memory = DeltaHebbianMemory(
            config.key_size,
            config.value_size,
            DeltaHebbianRule(
                learning_rate=config.learning_rate,
                retention=config.min_retention,
            ),
            detach_updates=config.detach_updates,
            memory_limit=config.memory_limit,
        )

    def new_state(self, batch_size: int):
        """Create empty fast weights on the projection parameters' device."""
        return self.memory.new_state(batch_size, like=self.query.weight)

    def retrieve(self, activity: torch.Tensor, state) -> torch.Tensor:
        """Return a dense CelNN drive retrieved without changing memory."""
        query = self.query(activity)
        retrieved = self.memory.read(state, query)
        return self.retrieval_scale * self.output(retrieved)

    def associations(
        self, activity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose projected associations and gates for tests/diagnostics."""
        key = self.key(activity)
        value = torch.tanh(self.value(activity))
        controls = torch.sigmoid(self.write_controls(activity))
        learning_rate = self.config.learning_rate * controls[..., 0]
        retention = self.config.min_retention + (
            1.0 - self.config.min_retention
        ) * controls[..., 1]
        return key, value, learning_rate, retention

    def write(self, state, activity: torch.Tensor, mask=None):
        """Write a sequence causally, one local association at a time."""
        if activity.ndim != 3:
            raise ValueError("activity must have shape (batch, sequence, features)")
        if mask is not None and mask.shape != activity.shape[:2]:
            raise ValueError("write mask must match batch and sequence axes")
        keys, values, rates, retentions = self.associations(activity)
        next_state = state
        for index in range(activity.shape[1]):
            rate = rates[:, index]
            retention = retentions[:, index]
            if mask is not None:
                active = mask[:, index].to(rate.dtype)
                rate = rate * active
                retention = retention * active + (1.0 - active)
            next_state = self.memory.write(
                next_state,
                keys[:, index],
                values[:, index],
                learning_rate=rate,
                retention=retention,
            )
        return next_state
