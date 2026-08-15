"""Stage-7 causal evaluation and paired uncertainty for frozen chat models."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from celnn import AssociativeFieldState
from torch.utils.data import DataLoader

from celllm.ablation import (
    NORMAL_ABLATION,
    AblationCondition,
    AblationConfig,
    AblationTrace,
)
from celllm.chat_data import (
    Conversation,
    ConversationDataset,
    EncodedConversation,
    collate_conversations,
    load_jsonl,
)
from celllm.chat_model import CellLMChatModel


@dataclass(frozen=True)
class DialogueLoss:
    """Sufficient statistics for one independent bootstrap unit."""

    dialogue_id: int
    loss_sum: float
    token_count: int

    @property
    def nll(self) -> float:
        return self.loss_sum / self.token_count


@dataclass(frozen=True)
class BoundaryProbe:
    """One assistant response suffix aligned to the model's native chunks."""

    dialogue_id: int
    encoded: EncodedConversation
    response_start: int
    boundary: int
    response_end: int

    @property
    def boundary_to_end(self) -> int:
        return self.response_end - self.boundary


@dataclass(frozen=True)
class EvaluationResult:
    """One condition evaluated on a declared token scope."""

    condition: str
    scope: str
    dialogues: tuple[DialogueLoss, ...]
    response_count: int
    boundary_to_end: tuple[int, ...] = ()
    trace: dict | None = None

    @property
    def dialogue_count(self) -> int:
        return len(self.dialogues)

    @property
    def token_count(self) -> int:
        return sum(item.token_count for item in self.dialogues)

    @property
    def loss_sum(self) -> float:
        return sum(item.loss_sum for item in self.dialogues)

    @property
    def nll(self) -> float:
        return self.loss_sum / self.token_count

    @property
    def perplexity(self) -> float:
        return math.exp(self.nll)

    def to_dict(self, *, include_dialogues: bool = True) -> dict:
        report = {
            "condition": self.condition,
            "scope": self.scope,
            "nll": self.nll,
            "perplexity": self.perplexity,
            "dialogue_count": self.dialogue_count,
            "response_count": self.response_count,
            "token_count": self.token_count,
            "boundary_to_end": {
                "minimum": min(self.boundary_to_end, default=None),
                "median": (
                    float(np.median(self.boundary_to_end))
                    if self.boundary_to_end
                    else None
                ),
                "maximum": max(self.boundary_to_end, default=None),
            },
            "trace": self.trace,
        }
        if include_dialogues:
            report["dialogues"] = [asdict(item) for item in self.dialogues]
        return report


def validation_conversations(
    paths: Sequence[str | Path], *, seed: int = 7
) -> list[Conversation]:
    """Reproduce the deterministic per-source validation split from training."""
    validation = []
    for path in paths:
        unique = {}
        for conversation in load_jsonl(path):
            key = tuple(
                (message.role, message.content)
                for message in conversation.messages
            )
            unique[key] = conversation
        source = list(unique.values())
        random.Random(seed).shuffle(source)
        split = max(1, int(len(source) * 0.95))
        validation.extend(source[split:] or source[-1:])
    return validation


def _dialogue_rows(
    loss_sums: Iterable[float],
    token_counts: Iterable[int],
    dialogue_ids: Iterable[int],
) -> tuple[DialogueLoss, ...]:
    aggregated: dict[int, list[float | int]] = {}
    for dialogue_id, loss_sum, token_count in zip(
        dialogue_ids, loss_sums, token_counts
    ):
        current = aggregated.setdefault(int(dialogue_id), [0.0, 0])
        current[0] = float(current[0]) + float(loss_sum)
        current[1] = int(current[1]) + int(token_count)
    return tuple(
        DialogueLoss(index, float(values[0]), int(values[1]))
        for index, values in sorted(aggregated.items())
        if int(values[1]) > 0
    )


@torch.inference_mode()
def evaluate_full_validation(
    model: CellLMChatModel,
    dataset: ConversationDataset,
    *,
    condition: AblationCondition = AblationCondition.NORMAL,
    device: str,
    batch_size: int = 128,
    trace: AblationTrace | None = None,
) -> EvaluationResult:
    """Evaluate every assistant token without changing native chunking."""
    config = AblationConfig(condition)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_conversations,
    )
    sums: list[float] = []
    counts: list[int] = []
    dialogue_ids: list[int] = []
    offset = 0
    for tokens, assistant_mask in loader:
        batch_sums, batch_counts = model.loss_statistics(
            tokens.to(device),
            assistant_mask.to(device),
            ablation=config,
            trace=trace,
        )
        size = tokens.shape[0]
        sums.extend(batch_sums.cpu().tolist())
        counts.extend(batch_counts.cpu().tolist())
        dialogue_ids.extend(range(offset, offset + size))
        offset += size
    rows = _dialogue_rows(sums, counts, dialogue_ids)
    return EvaluationResult(
        condition.value,
        "full_validation",
        rows,
        response_count=sum(
            _assistant_span_count(item.assistant_mask[: item.length])
            for item in dataset.examples
        ),
        trace=None if trace is None else trace.to_dict(),
    )


def _assistant_spans(mask: torch.Tensor) -> list[tuple[int, int]]:
    values = mask.bool().tolist()
    spans = []
    start = None
    for index, active in enumerate([*values, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index))
            start = None
    return spans


def _assistant_span_count(mask: torch.Tensor) -> int:
    return len(_assistant_spans(mask))


def native_boundary_probes(
    dataset: ConversationDataset, *, chunk_size: int
) -> tuple[BoundaryProbe, ...]:
    """Select response suffixes that cross a real post-prefix block boundary."""
    probes = []
    for dialogue_id, encoded in enumerate(dataset.examples):
        mask = encoded.assistant_mask[: encoded.length]
        for response_start, response_end in _assistant_spans(mask):
            boundary = (
                (response_start + chunk_size - 1) // chunk_size
            ) * chunk_size
            if boundary == 0 or boundary + 1 >= response_end:
                continue
            probes.append(
                BoundaryProbe(
                    dialogue_id,
                    encoded,
                    response_start,
                    boundary,
                    response_end,
                )
            )
    if not probes:
        raise ValueError("validation data has no eligible native-boundary probes")
    return tuple(probes)


def _reset_rows(
    state: AssociativeFieldState,
    rows: torch.Tensor,
    *,
    trace: AblationTrace | None = None,
) -> AssociativeFieldState:
    """Reset selected conversations once while preserving the batch layout."""
    if rows.ndim != 1 or rows.shape[0] != state.memory.shape[0]:
        raise ValueError("reset rows must match the state batch axis")
    if trace is not None and torch.any(rows):
        trace.record("history_before_reset_memory", state.memory[rows])
        trace.record(
            "history_before_reset_normalizer", state.normalizer[rows]
        )
    memory_mask = rows.reshape((-1,) + (1,) * (state.memory.ndim - 1))
    normalizer_mask = rows.reshape(
        (-1,) + (1,) * (state.normalizer.ndim - 1)
    )
    reset = AssociativeFieldState(
        torch.where(memory_mask, torch.zeros_like(state.memory), state.memory),
        torch.where(
            normalizer_mask,
            torch.zeros_like(state.normalizer),
            state.normalizer,
        ),
        state.updates,
    )
    if trace is not None and torch.any(rows):
        trace.record("history_after_reset_memory", reset.memory[rows])
        trace.record(
            "history_after_reset_normalizer", reset.normalizer[rows]
        )
    return reset


def _collate_probes(
    probes: Sequence[BoundaryProbe], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(item.encoded.length for item in probes)
    tokens = []
    score_masks = []
    boundaries = []
    dialogue_ids = []
    for probe in probes:
        encoded = probe.encoded
        padding = length - encoded.length
        tokens.append(F.pad(encoded.token_ids[: encoded.length], (0, padding), value=pad_id))
        score = torch.zeros(length, dtype=torch.bool)
        score[probe.boundary + 1 : probe.response_end] = True
        score &= F.pad(
            encoded.assistant_mask[: encoded.length], (0, padding), value=False
        )
        score_masks.append(score)
        boundaries.append(probe.boundary)
        dialogue_ids.append(probe.dialogue_id)
    return (
        torch.stack(tokens),
        torch.stack(score_masks),
        torch.tensor(boundaries, dtype=torch.long),
        torch.tensor(dialogue_ids, dtype=torch.long),
    )


@torch.inference_mode()
def evaluate_native_boundary(
    model: CellLMChatModel,
    probes: Sequence[BoundaryProbe],
    *,
    condition: AblationCondition,
    pad_id: int,
    device: str,
    batch_size: int = 128,
    trace: AblationTrace | None = None,
) -> EvaluationResult:
    """Evaluate matched response suffixes with one native-boundary intervention."""
    if condition not in {
        AblationCondition.NORMAL,
        AblationCondition.ZERO_HISTORY,
        AblationCondition.NO_WRITE,
    }:
        raise ValueError("native-boundary regime supports normal, zero_history, no_write")
    all_sums = []
    all_counts = []
    all_ids = []
    model.eval()
    for offset in range(0, len(probes), batch_size):
        batch = probes[offset : offset + batch_size]
        tokens, score_mask, boundaries, dialogue_ids = _collate_probes(
            batch, pad_id
        )
        tokens = tokens.to(device)
        score_mask = score_mask.to(device)
        boundaries = boundaries.to(device)
        memory = model.new_plastic_state(tokens.shape[0])
        logits = []
        for chunk_start, chunk in enumerate(
            tokens.split(model.chunk_size, dim=1)
        ):
            absolute_start = chunk_start * model.chunk_size
            at_boundary = boundaries.eq(absolute_start)
            if condition is AblationCondition.ZERO_HISTORY:
                memory = _reset_rows(memory, at_boundary, trace=trace)
            write_mask = chunk.ne(pad_id)
            if condition is AblationCondition.NO_WRITE:
                frozen = boundaries.le(absolute_start)
                write_mask = write_mask & ~frozen[:, None]
            chunk_logits, memory = model.forward_with_state(
                chunk,
                memory,
                write_mask=write_mask,
                ablation=NORMAL_ABLATION,
                trace=trace,
            )
            if (
                condition is AblationCondition.ZERO_HISTORY
                and torch.any(at_boundary)
                and trace is not None
            ):
                trace.record(
                    "history_after_rewrite_memory", memory.memory[at_boundary]
                )
                trace.record(
                    "history_after_rewrite_normalizer",
                    memory.normalizer[at_boundary],
                )
            logits.append(chunk_logits)
        predictions = torch.cat(logits, dim=1)[:, :-1]
        targets = tokens[:, 1:]
        target_mask = score_mask[:, 1:]
        losses = F.cross_entropy(
            predictions.reshape(-1, model.cfg.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        sums = (losses * target_mask).sum(dim=1)
        counts = target_mask.sum(dim=1)
        all_sums.extend(sums.cpu().tolist())
        all_counts.extend(counts.cpu().tolist())
        all_ids.extend(dialogue_ids.tolist())
    rows = _dialogue_rows(all_sums, all_counts, all_ids)
    return EvaluationResult(
        condition.value,
        "native_boundary_probes",
        rows,
        response_count=len(probes),
        boundary_to_end=tuple(item.boundary_to_end for item in probes),
        trace=None if trace is None else trace.to_dict(),
    )


def paired_bootstrap(
    normal: EvaluationResult,
    intervention: EvaluationResult,
    *,
    samples: int = 10_000,
    seed: int = 7,
) -> dict[str, float | int | list[float]]:
    """Bootstrap paired token-weighted NLL differences by dialogue."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    normal_map = {item.dialogue_id: item for item in normal.dialogues}
    intervention_map = {
        item.dialogue_id: item for item in intervention.dialogues
    }
    if normal_map.keys() != intervention_map.keys():
        raise ValueError("paired results must contain identical dialogues")
    ids = sorted(normal_map)
    normal_sums = np.array([normal_map[index].loss_sum for index in ids])
    intervention_sums = np.array(
        [intervention_map[index].loss_sum for index in ids]
    )
    normal_counts = np.array(
        [normal_map[index].token_count for index in ids]
    )
    intervention_counts = np.array(
        [intervention_map[index].token_count for index in ids]
    )
    if not np.array_equal(normal_counts, intervention_counts):
        raise ValueError("paired results must score identical token counts")

    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected = rng.integers(0, len(ids), size=len(ids))
        denominator = normal_counts[selected].sum()
        normal_nll = normal_sums[selected].sum() / denominator
        intervention_nll = intervention_sums[selected].sum() / denominator
        deltas[sample] = intervention_nll - normal_nll

    delta = intervention.nll - normal.nll
    normal_ppl = normal.perplexity
    intervention_ppl = intervention.perplexity
    return {
        "dialogue_count": len(ids),
        "bootstrap_samples": samples,
        "seed": seed,
        "normal_nll": normal.nll,
        "intervention_nll": intervention.nll,
        "delta_nll": delta,
        "delta_nll_percent": 100.0 * delta / normal.nll,
        "normal_perplexity": normal_ppl,
        "intervention_perplexity": intervention_ppl,
        "delta_perplexity_percent": 100.0
        * (intervention_ppl - normal_ppl)
        / normal_ppl,
        "bootstrap_standard_error": float(deltas.std(ddof=1)),
        "ci95_delta_nll": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "bootstrap_fraction_delta_positive": float((deltas > 0).mean()),
    }
