"""Delta-Hebbian key--value attention for bounded CelNN activity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch
from celnn import (
    AssociativeFieldState,
    DeltaHebbianMemory,
    DeltaHebbianRule,
    NormalizedDeltaHebbianField,
)
from torch import nn

from celllm.ablation import AblationTrace
from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    LocalAssociativeConfig,
    StateMatchedBankConfig,
)


class BankWriteAction(IntEnum):
    """One controlled intervention on a global-bank write candidate."""

    FULL = 0
    NO_ACTION = 1
    DECAY_ONLY = 2
    WRITE_NO_DECAY = 3


@dataclass(frozen=True)
class BankWriteCandidates:
    """Projected BANK associations, controls, routing, and pre-write error."""

    keys: torch.Tensor
    values: torch.Tensor
    rates: torch.Tensor
    retentions: torch.Tensor
    routing: torch.Tensor
    associative_error: torch.Tensor | None

    @property
    def shape(self) -> torch.Size:
        """Return the common batch/sequence candidate axes."""
        return self.rates.shape


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

    def retrieve(
        self,
        activity: torch.Tensor,
        state,
        *,
        enabled: bool = True,
        trace: AblationTrace | None = None,
    ) -> torch.Tensor:
        """Return a dense CelNN drive retrieved without changing memory."""
        query = self.query(activity)
        retrieved = self.memory.read(state, query)
        drive = self.retrieval_scale * self.output(retrieved)
        if not enabled:
            drive = torch.zeros_like(drive)
        if trace is not None:
            trace.record("retrieval_drive", drive)
        return drive

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


class StateMatchedGlobalBankAttention(nn.Module):
    """Fixed-size global memory bank without cellular spatial propagation.

    The bank owns the same ``slots * (value * key + key)`` recurrent scalars
    as an equally sized CY-HFA field. Fixed deterministic slot addresses avoid
    trainable parameter inflation; two learned routing temperatures match the
    two radius-one diffusion parameters used by CY-HFA.
    """

    def __init__(
        self, dimensions: int, config: StateMatchedBankConfig
    ) -> None:
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
        self.raw_read_temperature = nn.Parameter(
            torch.tensor(math.log(config.read_temperature))
        )
        self.raw_write_temperature = nn.Parameter(
            torch.tensor(math.log(config.write_temperature))
        )
        positions = torch.arange(config.slots, dtype=torch.float32)[:, None]
        frequencies = torch.arange(config.key_size, dtype=torch.float32)[None]
        addresses = torch.cos(
            math.pi
            * (positions + 0.5)
            * frequencies
            / float(config.slots)
        )
        addresses = torch.nn.functional.normalize(addresses, dim=-1)
        self.register_buffer("slot_addresses", addresses)
        self.memory = NormalizedDeltaHebbianField(
            config.key_size,
            config.value_size,
            learning_rate=config.learning_rate,
            retention=config.min_retention,
            epsilon=config.epsilon,
            detach_updates=config.detach_updates,
            memory_limit=config.memory_limit,
        )

    @property
    def read_temperature(self) -> torch.Tensor:
        return self.raw_read_temperature.exp()

    @property
    def write_temperature(self) -> torch.Tensor:
        return self.raw_write_temperature.exp()

    def new_state(self, batch_size: int) -> AssociativeFieldState:
        return self.memory.new_state(
            batch_size,
            self.config.slots,
            like=self.query.weight,
            dtype=torch.float32,
        )

    def _routing(
        self, projected: torch.Tensor, temperature: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(
            device_type=projected.device.type, enabled=False
        ):
            scores = torch.einsum(
                "b...k,sk->b...s",
                projected.float(),
                self.slot_addresses.float(),
            )
            return torch.softmax(scores / temperature.float(), dim=-1)

    def retrieve(
        self,
        activity: torch.Tensor,
        state: AssociativeFieldState,
        *,
        enabled: bool = True,
        trace: AblationTrace | None = None,
    ) -> torch.Tensor:
        """Read all past-only slots and route each query without topology."""
        query = self.query(activity)
        slot_values = self.memory.read_all(state, query)
        routing = self._routing(query, self.read_temperature)
        with torch.autocast(
            device_type=activity.device.type, enabled=False
        ):
            retrieved = torch.einsum(
                "b...s,b...sv->b...v", routing, slot_values
            )
        projected = self.output(retrieved.to(activity.dtype))
        drive = (self.retrieval_scale * projected).to(activity.dtype)
        if not enabled:
            drive = torch.zeros_like(drive)
        if trace is not None:
            trace.record("retrieval_drive", drive)
        return drive

    def associations(
        self, activity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.key(activity)
        value = torch.tanh(self.value(activity))
        controls = torch.sigmoid(self.write_controls(activity))
        rate = self.config.learning_rate * controls[..., 0]
        retention = self.config.min_retention + (
            1.0 - self.config.min_retention
        ) * controls[..., 1]
        return key, value, rate, retention

    def prepare_write(
        self,
        state: AssociativeFieldState,
        activity: torch.Tensor,
        *,
        measure_error: bool = True,
    ) -> BankWriteCandidates:
        """Prepare immutable candidates without changing fast memory.

        The weighted error is measured against every routed slot before any
        candidate in the block is committed. This mirrors the BANK rule,
        whose completed block is observed only after its logits are fixed.
        """
        if activity.ndim != 3:
            raise ValueError(
                "activity must have shape (batch, sequence, features)"
            )
        keys, values, rates, retentions = self.associations(activity)
        routing = self._routing(keys, self.write_temperature)
        associative_error = None
        if measure_error:
            predictions = self.memory.read_all(state, keys)
            with torch.autocast(
                device_type=activity.device.type, enabled=False
            ):
                errors = (
                    values.float().unsqueeze(-2) - predictions.float()
                ).square().mean(dim=-1)
                associative_error = (routing.float() * errors).sum(dim=-1)
        return BankWriteCandidates(
            keys=keys,
            values=values,
            rates=rates,
            retentions=retentions,
            routing=routing,
            associative_error=associative_error,
        )

    @staticmethod
    def _select_state(
        selector: torch.Tensor,
        selected: AssociativeFieldState,
        fallback: AssociativeFieldState,
    ) -> AssociativeFieldState:
        """Select one state per batch item while retaining a shared counter."""
        memory_selector = selector[:, None, None, None]
        normalizer_selector = selector[:, None, None]
        return AssociativeFieldState(
            torch.where(
                memory_selector, selected.memory, fallback.memory
            ),
            torch.where(
                normalizer_selector,
                selected.normalizer,
                fallback.normalizer,
            ),
            selected.updates,
        )

    def apply_write(
        self,
        state: AssociativeFieldState,
        candidates: BankWriteCandidates,
        mask: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> AssociativeFieldState:
        """Commit candidates with independently controlled causal actions.

        ``actions`` has the candidate batch/sequence shape and contains
        :class:`BankWriteAction` values. A masked candidate is an exact tensor
        identity. The update counter advances once per candidate to preserve
        the original BANK bookkeeping, including for padding.
        """
        if mask is not None and mask.shape != candidates.shape:
            raise ValueError("write mask must match candidate axes")
        controlled = actions is not None
        if actions is None:
            actions = torch.full(
                candidates.shape,
                int(BankWriteAction.FULL),
                dtype=torch.int64,
                device=candidates.rates.device,
            )
        elif actions.shape != candidates.shape:
            raise ValueError("write actions must match candidate axes")
        if actions.dtype == torch.bool or actions.is_floating_point():
            raise ValueError("write actions must use an integer dtype")
        minimum = int(actions.min().item())
        maximum = int(actions.max().item())
        if minimum < int(BankWriteAction.FULL) or maximum > int(
            BankWriteAction.WRITE_NO_DECAY
        ):
            raise ValueError("unknown BANK write action")

        next_state = state
        slots = self.config.slots
        for index in range(candidates.shape[1]):
            assignment = candidates.routing[:, index]
            rate = candidates.rates[:, index, None].float() * assignment
            retention = 1.0 - assignment * (
                1.0 - candidates.retentions[:, index, None].float()
            )
            key = candidates.keys[:, index, None].expand(-1, slots, -1)
            value = candidates.values[:, index, None].expand(-1, slots, -1)
            full = self.memory.write(
                next_state,
                key,
                value,
                learning_rate=rate,
                retention=retention,
            )
            identity = AssociativeFieldState(
                next_state.memory,
                next_state.normalizer,
                next_state.updates + 1,
            )
            if not controlled:
                updated = full
                if mask is not None:
                    updated = self._select_state(
                        ~mask[:, index].bool(), identity, updated
                    )
                next_state = updated
                continue
            no_decay = self.memory.write(
                next_state,
                key,
                value,
                learning_rate=rate,
                retention=torch.ones_like(retention),
            )
            decay = AssociativeFieldState(
                retention[..., None, None] * next_state.memory,
                retention[..., None] * next_state.normalizer,
                next_state.updates + 1,
            )
            action = actions[:, index]
            updated = full
            updated = self._select_state(
                action == int(BankWriteAction.NO_ACTION), identity, updated
            )
            updated = self._select_state(
                action == int(BankWriteAction.DECAY_ONLY), decay, updated
            )
            updated = self._select_state(
                action == int(BankWriteAction.WRITE_NO_DECAY),
                no_decay,
                updated,
            )
            if mask is not None:
                updated = self._select_state(
                    ~mask[:, index].bool(), identity, updated
                )
            next_state = updated
        return next_state

    def write(
        self,
        state: AssociativeFieldState,
        activity: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> AssociativeFieldState:
        """Route completed-block associations into independent global slots."""
        candidates = self.prepare_write(
            state, activity, measure_error=False
        )
        return self.apply_write(state, candidates, mask=mask)


class LocalAssociativeMessagePassing(nn.Module):
    """Stateless causal kernel attention over preceding lattice cells.

    The module constructs the normalized Hebbian numerator/normalizer
    implicitly at every refinement. It owns no recurrent fast state and cannot
    carry information across blocks; that remains the global BANK's role.
    """

    def __init__(self, dimensions: int, config: LocalAssociativeConfig) -> None:
        super().__init__()
        self.dimensions = int(dimensions)
        self.config = config
        self.query = nn.Linear(dimensions, config.key_size, bias=False)
        self.key = nn.Linear(dimensions, config.key_size, bias=False)
        self.value = nn.Linear(dimensions, config.value_size, bias=False)
        self.output = nn.Linear(config.value_size, dimensions, bias=False)
        self.gate = nn.Linear(4 * dimensions, 1)
        self.offset_bias = nn.Parameter(torch.zeros(config.radius))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        scale = torch.tensor(float(config.retrieval_scale))
        if config.learnable_retrieval_scale:
            self.retrieval_scale = nn.Parameter(scale)
        else:
            self.register_buffer("retrieval_scale", scale)

    @staticmethod
    def _features(projected: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(projected.float()) + 1.0

    def forward(
        self,
        activity: torch.Tensor,
        global_drive: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return local drive and optionally normalized causal edge weights."""
        if activity.ndim != 3:
            raise ValueError("activity must have shape (batch, cells, features)")
        if global_drive.shape != activity.shape:
            raise ValueError("global drive must match activity")
        if mask is not None and mask.shape != activity.shape[:2]:
            raise ValueError("local mask must match batch and cells")
        batch, cells, _ = activity.shape
        query = self._features(self.query(activity))
        keys = self._features(self.key(activity))
        values = torch.tanh(self.value(activity)).float()
        numerator = activity.new_zeros(
            (batch, cells, self.config.value_size), dtype=torch.float32
        )
        denominator = activity.new_zeros(
            (batch, cells), dtype=torch.float32
        )
        edge_weights = activity.new_zeros(
            (batch, cells, self.config.radius), dtype=torch.float32
        )
        positions = torch.arange(cells, device=activity.device)
        for offset in range(1, self.config.radius + 1):
            source_positions = (positions - offset).clamp_min(0)
            source_activity = activity.index_select(1, source_positions)
            source_keys = keys.index_select(1, source_positions)
            source_values = values.index_select(1, source_positions)
            valid = positions.ge(offset)[None].expand(batch, -1)
            if mask is not None:
                receiver_valid = mask.bool()
                source_valid = mask.index_select(1, source_positions).bool()
                valid = valid & receiver_valid & source_valid
            kernel = (query * source_keys).sum(dim=-1) / self.config.key_size
            if self.config.gated:
                gate_input = torch.cat(
                    (
                        activity,
                        source_activity,
                        activity - source_activity,
                        global_drive,
                    ),
                    dim=-1,
                )
                gate = torch.sigmoid(
                    self.gate(gate_input).squeeze(-1)
                    + self.offset_bias[offset - 1]
                ).float()
            else:
                gate = torch.ones_like(kernel, dtype=torch.float32)
            weight = gate * kernel.float() * valid.to(torch.float32)
            numerator = numerator + weight[..., None] * source_values
            denominator = denominator + weight
            edge_weights[..., offset - 1] = weight
        retrieved = numerator / (denominator[..., None] + self.config.epsilon)
        drive = self.retrieval_scale * self.output(retrieved.to(activity.dtype))
        if mask is not None:
            drive = drive * mask[..., None].to(drive.dtype)
        if return_weights:
            normalized = edge_weights / (
                denominator[..., None] + self.config.epsilon
            )
            return drive, normalized
        return drive


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


class InterBlockFieldCarry(nn.Module):
    """Reduce a completed local field to a past-only block boundary.

    Numerator and normalizer are averaged with identical non-negative weights.
    Broadcasting that boundary into the next lattice separates long-range
    block transport from deliberately local intra-block diffusion without
    adding recurrent state.
    """

    def forward(
        self, state: AssociativeFieldState
    ) -> AssociativeFieldState:
        memory = state.memory.mean(dim=1, keepdim=True)
        normalizer = state.normalizer.mean(dim=1, keepdim=True)
        return AssociativeFieldState(
            memory.expand_as(state.memory),
            normalizer.expand_as(state.normalizer),
            state.updates,
        )


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
        self.carry = InterBlockFieldCarry()
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
        """Create FP32 accumulators on the projection parameter device."""
        return self.memory.new_state(
            batch_size,
            cells,
            like=self.query.weight,
            dtype=torch.float32,
        )

    def retrieve(
        self,
        activity: torch.Tensor,
        state: AssociativeFieldState,
        *,
        enabled: bool = True,
        trace: AblationTrace | None = None,
    ) -> torch.Tensor:
        """Query each cell's current propagated associative memory."""
        retrieved = self.memory.read(state, self.query(activity))
        projected = self.output(retrieved.to(activity.dtype))
        drive = (self.retrieval_scale * projected).to(activity.dtype)
        if not enabled:
            drive = torch.zeros_like(drive)
        if trace is not None:
            trace.record("retrieval_drive", drive)
        return drive

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

    def begin_block(
        self,
        state: AssociativeFieldState,
        *,
        carry_enabled: bool = True,
        trace: AblationTrace | None = None,
    ) -> AssociativeFieldState:
        """Reduce the preceding block to a causal boundary for the next.

        This operation happens only between completed blocks. Current-block
        future cells therefore never feed back into their own prefix.
        """
        carried = self.carry(state) if carry_enabled else state.reset()
        if trace is not None:
            trace.record("block_input_memory", carried.memory)
            trace.record("block_input_normalizer", carried.normalizer)
        return carried

    def advance(
        self,
        state: AssociativeFieldState,
        activity: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        write_enabled: bool = True,
        diffusion_enabled: bool = True,
        trace: AblationTrace | None = None,
    ) -> AssociativeFieldState:
        """Propagate the field, then make one gated local Delta-Hebb write."""
        propagated = self.propagate(state) if diffusion_enabled else state
        if trace is not None:
            trace.record_delta(
                "diffusion_memory_delta", state.memory, propagated.memory
            )
            trace.record_delta(
                "diffusion_normalizer_delta",
                state.normalizer,
                propagated.normalizer,
            )
        if write_enabled:
            key, value, rate, retention = self.associations(activity)
            written = self.memory.write(
                propagated,
                key,
                value,
                learning_rate=rate,
                retention=retention,
                mask=mask,
            )
        else:
            written = propagated
        if trace is not None:
            trace.record_delta(
                "write_memory_delta", propagated.memory, written.memory
            )
            trace.record_delta(
                "write_normalizer_delta",
                propagated.normalizer,
                written.normalizer,
            )
        return written
