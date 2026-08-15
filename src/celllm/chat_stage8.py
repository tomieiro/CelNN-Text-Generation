"""Post-training diagnostics for BANK trajectory energy and write utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from celnn import AssociativeFieldState

from celllm.attention import (
    BankWriteAction,
    BankWriteCandidates,
)
from celllm.cell import piecewise_linear
from celllm.chat_model import CellLMChatModel


@dataclass(frozen=True)
class BankTrajectoryMetrics:
    """FP32 diagnostics for every token in one native BANK block."""

    trajectory_energy: torch.Tensor
    tail_energy: torch.Tensor
    displacement: torch.Tensor
    terminal_residual: torch.Tensor
    memory_trajectory_energy: torch.Tensor
    memory_final_displacement: torch.Tensor
    retrieval_drive_energy: torch.Tensor


@dataclass(frozen=True)
class BankBlockDiagnostics:
    """A normal BANK block plus non-mutating trajectory observations."""

    logits: torch.Tensor
    activity: torch.Tensor
    next_memory: AssociativeFieldState
    candidates: BankWriteCandidates
    metrics: BankTrajectoryMetrics


@dataclass(frozen=True)
class CandidateUtility:
    """Features and causal future-loss effects for one write candidate."""

    dialogue_id: int
    block_index: int
    block_position: int
    sequence_position: int
    token_id: int
    target_role: str
    learning_rate: float
    retention: float
    routing_entropy: float
    routing_maximum: float
    associative_error: float
    trajectory_energy: float
    tail_energy: float
    displacement: float
    terminal_residual: float
    memory_trajectory_energy: float
    memory_final_displacement: float
    retrieval_drive_energy: float
    utilities: dict[str, dict[int, float | None]]
    evaluated_tokens: dict[int, int]


@dataclass(frozen=True)
class DialogueUtilityResult:
    """Normal sufficient statistics and candidate rows for one dialogue."""

    dialogue_id: int
    normal_loss_sum: float
    normal_token_count: int
    candidates: tuple[CandidateUtility, ...]

    @property
    def normal_nll(self) -> float:
        return self.normal_loss_sum / self.normal_token_count


def diagnose_bank_block(
    model: CellLMChatModel,
    tokens: torch.Tensor,
    memory: AssociativeFieldState,
    *,
    write_mask: torch.Tensor | None = None,
) -> BankBlockDiagnostics:
    """Run one native block and a retrieval-free shadow trajectory.

    The main path uses the checkpoint's exact refinement and write modules.
    The shadow begins from the same neural state but cannot retrieve or write.
    All reported reductions are explicitly accumulated in FP32.
    """
    if model.bank_config is None:
        raise ValueError("trajectory diagnostics require a BANK model")
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    if not 0 < tokens.shape[1] <= model.chunk_size:
        raise ValueError("tokens must contain one non-empty native block")
    if write_mask is None:
        write_mask = tokens.ne(0)
    elif write_mask.shape != tokens.shape:
        raise ValueError("write mask must match tokens")

    core = model.core
    embedding = core.embed(tokens)
    cell_input = core.cell.control_input(embedding)
    state = torch.zeros_like(embedding)
    shadow = torch.zeros_like(embedding)
    initial_output = piecewise_linear(state).float()
    previous_output = initial_output
    final_previous = initial_output
    energy_sum = torch.zeros(tokens.shape, device=tokens.device)
    tail_sum = torch.zeros_like(energy_sum)
    memory_energy_sum = torch.zeros_like(energy_sum)
    retrieval_energy_sum = torch.zeros_like(energy_sum)
    eta = float(core.cfg.eta)

    for step in range(core.cfg.k):
        next_state, drive = core.cell.refine_with_memory(
            state, cell_input, memory, retrieval_enabled=True
        )
        next_shadow, _ = core.cell.refine_with_memory(
            shadow, cell_input, memory, retrieval_enabled=False
        )
        output = piecewise_linear(next_state).float()
        shadow_output = piecewise_linear(next_shadow).float()
        transition = ((output - previous_output) / eta).square().mean(-1)
        energy_sum = energy_sum + transition
        if step > 0:
            tail_sum = tail_sum + transition
        memory_energy_sum = memory_energy_sum + (
            output - shadow_output
        ).square().mean(-1)
        retrieval_energy_sum = retrieval_energy_sum + drive.float().square().sum(
            dim=-1
        )
        final_previous = previous_output
        previous_output = output
        state = next_state
        shadow = next_shadow

    steps = float(core.cfg.k)
    tail_steps = float(max(core.cfg.k - 1, 1))
    metrics = BankTrajectoryMetrics(
        trajectory_energy=energy_sum / steps,
        tail_energy=tail_sum / tail_steps,
        displacement=(previous_output - initial_output).square().mean(-1),
        terminal_residual=(previous_output - final_previous).square().mean(-1),
        memory_trajectory_energy=memory_energy_sum / steps,
        memory_final_displacement=(
            previous_output - piecewise_linear(shadow).float()
        ).square().mean(-1),
        retrieval_drive_energy=retrieval_energy_sum / steps,
    )
    activity = piecewise_linear(state)
    candidates = core.cell.attention.prepare_write(memory, activity)
    if candidates.associative_error is None:
        raise RuntimeError("diagnostic candidate error was not measured")
    next_memory = core.cell.attention.apply_write(
        memory, candidates, mask=write_mask
    )
    return BankBlockDiagnostics(
        logits=core.readout(state),
        activity=activity,
        next_memory=next_memory,
        candidates=candidates,
        metrics=metrics,
    )


def _loss_sum(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    assistant_mask: torch.Tensor,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, int]:
    """Score next-token predictions at positions in ``[start, stop)``."""
    stop = min(stop, tokens.shape[1] - 1)
    if stop <= start:
        return logits.new_zeros((), dtype=torch.float32), 0
    target_mask = assistant_mask[:, start + 1 : stop + 1]
    count = int(target_mask.sum().item())
    if count == 0:
        return logits.new_zeros((), dtype=torch.float32), 0
    losses = F.cross_entropy(
        logits[:, start:stop].float().reshape(-1, logits.shape[-1]),
        tokens[:, start + 1 : stop + 1].reshape(-1),
        reduction="none",
    ).reshape_as(target_mask)
    return (losses * target_mask).sum(), count


@torch.inference_mode()
def evaluate_bank_write_utilities(
    model: CellLMChatModel,
    token_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    *,
    dialogue_id: int,
    horizons: Iterable[int] = (1, 4),
    actions: Iterable[BankWriteAction] = (
        BankWriteAction.NO_ACTION,
    ),
    max_candidates: int | None = None,
) -> DialogueUtilityResult:
    """Measure leave-one-write-out utility on future native blocks.

    A candidate action is changed only after its current block has completed,
    so current-block logits are shared exactly. Future blocks then evolve and
    write normally from the counterfactual state, measuring the total causal
    effect at each requested horizon.
    """
    if model.bank_config is None:
        raise ValueError("write utility requires a BANK model")
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    if assistant_mask.ndim == 1:
        assistant_mask = assistant_mask.unsqueeze(0)
    if token_ids.shape != assistant_mask.shape or token_ids.shape[0] != 1:
        raise ValueError("utility evaluation expects one aligned dialogue")
    requested_horizons = tuple(sorted(set(int(item) for item in horizons)))
    if not requested_horizons or requested_horizons[0] < 1:
        raise ValueError("horizons must be positive")
    requested_actions = tuple(actions)
    if not requested_actions:
        raise ValueError("at least one counterfactual action is required")
    if any(item == BankWriteAction.FULL for item in requested_actions):
        raise ValueError("FULL is the reference, not a counterfactual action")

    chunks = token_ids.split(model.chunk_size, dim=1)
    mask_chunks = token_ids.ne(0).split(model.chunk_size, dim=1)
    memory = model.new_plastic_state(1)
    diagnostics: list[BankBlockDiagnostics] = []
    input_memories: list[AssociativeFieldState] = []
    for chunk, write_mask in zip(chunks, mask_chunks):
        input_memories.append(memory)
        observed = diagnose_bank_block(
            model, chunk, memory, write_mask=write_mask
        )
        diagnostics.append(observed)
        memory = observed.next_memory
    normal_logits = torch.cat([item.logits for item in diagnostics], dim=1)
    normal_loss, normal_count = _loss_sum(
        normal_logits, token_ids, assistant_mask, 0, token_ids.shape[1]
    )
    if normal_count == 0:
        raise ValueError("dialogue has no assistant targets")

    rows: list[CandidateUtility] = []
    maximum_horizon = requested_horizons[-1]
    scope_counts: dict[int, dict[int, int]] = {}
    coordinates: list[tuple[int, int]] = []
    for block_index, (chunk, write_mask) in enumerate(
        zip(chunks[:-1], mask_chunks[:-1])
    ):
        block_start = (block_index + 1) * model.chunk_size
        eligible_horizons = {}
        for horizon in requested_horizons:
            block_stop = min(
                block_start + horizon * model.chunk_size,
                token_ids.shape[1],
            )
            _, count = _loss_sum(
                normal_logits,
                token_ids,
                assistant_mask,
                block_start,
                block_stop,
            )
            eligible_horizons[horizon] = (
                count
                if block_index + horizon < len(chunks)
                else 0
            )
        if not any(eligible_horizons.values()):
            continue
        scope_counts[block_index] = eligible_horizons
        for position in range(chunk.shape[1]):
            if bool(write_mask[0, position]):
                coordinates.append((block_index, position))

    if max_candidates is not None and len(coordinates) > max_candidates:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive or None")
        if max_candidates == 1:
            coordinates = [coordinates[len(coordinates) // 2]]
        else:
            last = len(coordinates) - 1
            indices = [
                round(index * last / (max_candidates - 1))
                for index in range(max_candidates)
            ]
            coordinates = [coordinates[index] for index in indices]

    for block_index, position in coordinates:
        chunk = chunks[block_index]
        write_mask = mask_chunks[block_index]
        observed = diagnostics[block_index]
        block_start = (block_index + 1) * model.chunk_size
        utilities: dict[str, dict[int, float | None]] = {}
        counts: dict[int, int] = {}
        for action in requested_actions:
            action_map = torch.full(
                chunk.shape,
                int(BankWriteAction.FULL),
                dtype=torch.int64,
                device=chunk.device,
            )
            action_map[0, position] = int(action)
            counter_memory = model.core.cell.attention.apply_write(
                input_memories[block_index],
                observed.candidates,
                mask=write_mask,
                actions=action_map,
            )
            counter_logits = []
            for future_index in range(
                block_index + 1,
                min(len(chunks), block_index + 1 + maximum_horizon),
            ):
                logits, counter_memory = model.forward_with_state(
                    chunks[future_index],
                    counter_memory,
                    write_mask=mask_chunks[future_index],
                )
                counter_logits.append(logits)
            action_utilities: dict[int, float | None] = {}
            for horizon in requested_horizons:
                available = min(horizon, len(counter_logits))
                start = block_start
                stop = min(
                    start + available * model.chunk_size,
                    token_ids.shape[1],
                )
                normal_sum, count = _loss_sum(
                    normal_logits,
                    token_ids,
                    assistant_mask,
                    start,
                    stop,
                )
                if count == 0 or available < horizon:
                    action_utilities[horizon] = None
                else:
                    joined = torch.cat(counter_logits[:horizon], dim=1)
                    counter_sum, counter_count = _loss_sum(
                        joined,
                        token_ids[:, start : start + joined.shape[1] + 1],
                        assistant_mask[
                            :, start : start + joined.shape[1] + 1
                        ],
                        0,
                        joined.shape[1],
                    )
                    if counter_count != count:
                        raise RuntimeError("counterfactual token scope changed")
                    action_utilities[horizon] = float(
                        ((counter_sum - normal_sum) / count).item()
                    )
                counts[horizon] = count if available == horizon else 0
            utilities[action.name.lower()] = action_utilities

        routing = observed.candidates.routing[0, position].float()
        metrics = observed.metrics
        sequence_position = block_index * model.chunk_size + position
        associative_error = observed.candidates.associative_error
        if associative_error is None:
            raise RuntimeError("candidate error missing from diagnostic path")
        rows.append(
            CandidateUtility(
                dialogue_id=dialogue_id,
                block_index=block_index,
                block_position=position,
                sequence_position=sequence_position,
                token_id=int(chunk[0, position].item()),
                target_role=(
                    "assistant"
                    if bool(assistant_mask[0, sequence_position])
                    else "context"
                ),
                learning_rate=float(
                    observed.candidates.rates[0, position].item()
                ),
                retention=float(
                    observed.candidates.retentions[0, position].item()
                ),
                routing_entropy=float(
                    -(routing * routing.clamp_min(1e-12).log()).sum().item()
                ),
                routing_maximum=float(routing.max().item()),
                associative_error=float(
                    associative_error[0, position].item()
                ),
                trajectory_energy=float(
                    metrics.trajectory_energy[0, position].item()
                ),
                tail_energy=float(metrics.tail_energy[0, position].item()),
                displacement=float(metrics.displacement[0, position].item()),
                terminal_residual=float(
                    metrics.terminal_residual[0, position].item()
                ),
                memory_trajectory_energy=float(
                    metrics.memory_trajectory_energy[0, position].item()
                ),
                memory_final_displacement=float(
                    metrics.memory_final_displacement[0, position].item()
                ),
                retrieval_drive_energy=float(
                    metrics.retrieval_drive_energy[0, position].item()
                ),
                utilities=utilities,
                evaluated_tokens=counts,
            )
        )
    return DialogueUtilityResult(
        dialogue_id=dialogue_id,
        normal_loss_sum=float(normal_loss.item()),
        normal_token_count=normal_count,
        candidates=tuple(rows),
    )
