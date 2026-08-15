"""Delta-Hebbian key--value attention for bounded CelNN activity."""

from __future__ import annotations

import math

import torch
from celnn import (
    AssociativeFieldState,
    DeltaHebbianMemory,
    DeltaHebbianRule,
    NormalizedDeltaHebbianField,
)
from torch import nn

from celllm.config import CYHFAConfig, HebbianAttentionConfig


class DeltaHebbianAttention(nn.Module):
    """Global fast-memory baseline for comparison with CY-HFA.

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


class CausalFieldPropagation(nn.Module):
    """Diffuse a field only from preceding cells with a stable convex step.

    The operator accepts arbitrary feature axes after ``(batch, cells)``. Its
    non-negative coupling weights sum to at most ``max_rate``, which keeps one
    explicit propagation step bounded and preserves a positive normalizer.
    """

    def __init__(
        self,
        radius: int,
        rate: float,
        *,
        max_rate: float = 0.25,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError("propagation radius must be positive")
        if not 0 < max_rate <= 1:
            raise ValueError("maximum propagation rate must be in (0, 1]")
        if not 0 <= rate <= max_rate:
            raise ValueError("propagation rate must be in [0, max_rate]")
        self.radius = int(radius)
        self.max_rate = float(max_rate)
        fraction = min(max(rate / max_rate, 1e-6), 1.0 - 1e-6)
        initial = torch.tensor(math.log(fraction / (1.0 - fraction)))
        offsets = torch.zeros(radius)
        if learnable:
            self.raw_rate = nn.Parameter(initial)
            self.offset_logits = nn.Parameter(offsets)
        else:
            self.register_buffer("raw_rate", initial)
            self.register_buffer("offset_logits", offsets)

    @property
    def rate(self) -> torch.Tensor:
        """Return the bounded total diffusion applied in one field step."""
        return self.max_rate * torch.sigmoid(self.raw_rate)

    def weights(self) -> torch.Tensor:
        """Return one non-negative coupling for each preceding offset."""
        return self.rate * torch.softmax(self.offset_logits, dim=0)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim < 3:
            raise ValueError("field must have shape (batch, cells, ...)")
        cells = field.shape[1]
        if cells <= self.radius:
            raise ValueError("field must contain more cells than the radius")
        propagated = field
        positions = torch.arange(cells, device=field.device)
        for distance, weight in enumerate(self.weights(), start=1):
            source = (positions - distance).clamp_min(0)
            neighbour = field.index_select(1, source)
            propagated = propagated + weight * (neighbour - field)
        return propagated


class ChuaYangHebbianFieldAttention(nn.Module):
    """Normalized associative attention whose memory is a cellular field.

    Every lattice cell owns ``M_i`` and ``s_i``. Slow projections produce
    local queries, keys, values, and write/forget gates. Both fast fields
    propagate causally before a local normalized Delta-Hebbian write.
    """

    def __init__(self, dimensions: int, config: CYHFAConfig) -> None:
        super().__init__()
        self.dimensions = int(dimensions)
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
        self.propagation = CausalFieldPropagation(
            config.diffusion_radius,
            config.diffusion_rate,
            max_rate=config.max_diffusion,
            learnable=config.learnable_diffusion,
        )
        self.memory = NormalizedDeltaHebbianField(
            config.key_size,
            config.value_size,
            learning_rate=config.learning_rate,
            retention=config.min_retention,
            epsilon=config.epsilon,
            detach_updates=config.detach_updates,
            memory_limit=config.memory_limit,
        )

    def new_state(self, batch_size: int, cells: int) -> AssociativeFieldState:
        """Create empty local memories on the projection parameter device."""
        return self.memory.new_state(
            batch_size, cells, like=self.query.weight
        )

    def retrieve(
        self, activity: torch.Tensor, state: AssociativeFieldState
    ) -> torch.Tensor:
        """Query each cell's current propagated associative memory."""
        retrieved = self.memory.read(state, self.query(activity))
        return self.retrieval_scale * self.output(retrieved)

    def associations(
        self, activity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose projected associations and bounded local controls."""
        key = self.key(activity)
        value = torch.tanh(self.value(activity))
        controls = torch.sigmoid(self.write_controls(activity))
        rate = self.config.learning_rate * controls[..., 0]
        retention = self.config.min_retention + (
            1.0 - self.config.min_retention
        ) * controls[..., 1]
        return key, value, rate, retention

    def propagate(
        self, state: AssociativeFieldState
    ) -> AssociativeFieldState:
        """Apply the same causal cellular operator to ``M_i`` and ``s_i``."""
        return AssociativeFieldState(
            self.propagation(state.memory),
            self.propagation(state.normalizer),
            state.updates,
        )

    @staticmethod
    def begin_block(
        state: AssociativeFieldState,
    ) -> AssociativeFieldState:
        """Use the preceding block's terminal field as causal boundary.

        The final cell is the only cell downstream of the full causal lattice.
        Broadcasting it initializes every position of the next block with
        past-only information before current-token writes begin.
        """
        memory = state.memory[:, -1:].expand_as(state.memory)
        normalizer = state.normalizer[:, -1:].expand_as(state.normalizer)
        return AssociativeFieldState(memory, normalizer, state.updates)

    def advance(
        self,
        state: AssociativeFieldState,
        activity: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> AssociativeFieldState:
        """Propagate the field, then make one gated local Delta-Hebb write."""
        propagated = self.propagate(state)
        key, value, rate, retention = self.associations(activity)
        return self.memory.write(
            propagated,
            key,
            value,
            learning_rate=rate,
            retention=retention,
            mask=mask,
        )
