#!/usr/bin/env python3
"""Evaluate the frozen BANK checkpoint on the Stage-8.2 MQAR gate."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from celnn import AssociativeFieldState

from celllm.chat_checkpoint import load_chat_checkpoint
from celllm.chat_generation import SamplingConfig, sample_token
from celllm.chat_tokenizer import ChatTokenizer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def answer_batch(
    tokenizer: ChatTokenizer,
    prompt: str,
    candidates: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = [
        tokenizer.special_id("<bos>"),
        tokenizer.special_id("<user>"),
        *tokenizer.encode(prompt),
        tokenizer.eos_id,
        tokenizer.special_id("<assistant>"),
    ]
    sequences = []
    masks = []
    for candidate in candidates:
        answer = tokenizer.encode(candidate)
        sequences.append([*prefix, *answer, tokenizer.eos_id])
        masks.append([False] * len(prefix) + [True] * len(answer) + [False])
    width = max(len(item) for item in sequences)
    tokens = torch.full(
        (len(sequences), width),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    assistant = torch.zeros_like(tokens, dtype=torch.bool)
    for row, (sequence, mask) in enumerate(zip(sequences, masks)):
        tokens[row, : len(sequence)] = torch.tensor(sequence, device=device)
        assistant[row, : len(mask)] = torch.tensor(mask, device=device)
    return tokens, assistant


def _repeat_state(
    state: AssociativeFieldState, copies: int
) -> AssociativeFieldState:
    return AssociativeFieldState(
        state.memory.expand(copies, *state.memory.shape[1:]),
        state.normalizer.expand(copies, *state.normalizer.shape[1:]),
        state.updates,
    )


@torch.inference_mode()
def prepare_prefix(
    model,
    tokenizer: ChatTokenizer,
    prompt: str,
    device: str,
) -> tuple[list[int], AssociativeFieldState]:
    """Commit only complete blocks preceding the final prefix block."""
    prefix = [
        tokenizer.special_id("<bos>"),
        tokenizer.special_id("<user>"),
        *tokenizer.encode(prompt),
        tokenizer.eos_id,
        tokenizer.special_id("<assistant>"),
    ]
    tail_start = ((len(prefix) - 1) // model.chunk_size) * model.chunk_size
    memory = model.new_plastic_state(1)
    committed = torch.tensor(prefix[:tail_start], device=device).unsqueeze(0)
    for chunk in committed.split(model.chunk_size, dim=1):
        if chunk.shape[1] == 0:
            continue
        _, memory = model.forward_with_state(
            chunk, memory, write_mask=torch.ones_like(chunk, dtype=torch.bool)
        )
    return prefix[tail_start:], memory


@torch.inference_mode()
def candidate_losses(
    model,
    tokenizer: ChatTokenizer,
    prompt: str,
    candidates: list[str],
    device: str,
    *,
    prepared: tuple[list[int], AssociativeFieldState] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score answer sequences after sharing all complete prefix blocks."""
    tail, memory = prepared or prepare_prefix(
        model, tokenizer, prompt, device
    )

    answers = [tokenizer.encode(candidate) for candidate in candidates]
    sequences = [[*tail, *answer] for answer in answers]
    width = max(len(item) for item in sequences)
    tokens = torch.full(
        (len(sequences), width),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = torch.tensor(sequence, device=device)
    memory = _repeat_state(memory, len(sequences))
    logits = []
    for chunk in tokens.split(model.chunk_size, dim=1):
        chunk_logits, memory = model.forward_with_state(
            chunk,
            memory,
            write_mask=chunk.ne(tokenizer.pad_id),
        )
        logits.append(chunk_logits)
    joined = torch.cat(logits, dim=1)
    sums = joined.new_zeros(len(sequences), dtype=torch.float32)
    counts = torch.tensor(
        [len(answer) for answer in answers], device=device, dtype=torch.long
    )
    for row, answer in enumerate(answers):
        start = len(tail) - 1
        stop = start + len(answer)
        targets = torch.tensor(answer, device=device)
        sums[row] = F.cross_entropy(
            joined[row, start:stop].float(), targets, reduction="sum"
        )
    return sums, counts


@torch.inference_mode()
def greedy_answer(
    model,
    tokenizer: ChatTokenizer,
    tail: list[int],
    memory: AssociativeFieldState,
    device: str,
    *,
    max_new_tokens: int = 8,
) -> str:
    """Generate from a shared native-boundary prefix without replaying it."""
    generated: list[int] = []
    assistant_id = tokenizer.special_id("<assistant>")
    forbidden = {
        tokenizer.pad_id,
        tokenizer.special_id("<unk>"),
        tokenizer.special_id("<bos>"),
        tokenizer.special_id("<user>"),
        assistant_id,
    }
    sampling = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    for _ in range(max_new_tokens):
        sequence = torch.tensor(
            [*tail, *generated], dtype=torch.long, device=device
        ).unsqueeze(0)
        branch_memory = memory
        logits = []
        for chunk in sequence.split(model.chunk_size, dim=1):
            chunk_logits, branch_memory = model.forward_with_state(
                chunk,
                branch_memory,
                write_mask=torch.ones_like(chunk, dtype=torch.bool),
            )
            logits.append(chunk_logits)
        token = sample_token(
            torch.cat(logits, dim=1)[0, -1],
            sampling,
            generated=generated,
            forbidden=forbidden,
        )
        if token == tokenizer.eos_id:
            break
        generated.append(token)
    return tokenizer.decode(generated).strip()


def wilson_interval(successes: int, total: int, z: float = 1.95996398454) -> tuple[float, float]:
    if total < 1:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    top1 = sum(item["rank"] == 1 for item in rows)
    exact = sum(bool(item["exact_match"]) for item in rows if item["exact_match"] is not None)
    generated = sum(item["exact_match"] is not None for item in rows)
    low, high = wilson_interval(top1, total)
    return {
        "count": total,
        "top1_accuracy": top1 / total,
        "top1_wilson95": [low, high],
        "mean_rank": sum(item["rank"] for item in rows) / total,
        "mean_reciprocal_rank": sum(1.0 / item["rank"] for item in rows) / total,
        "correct_answer_nll": sum(item["correct_nll"] for item in rows) / total,
        "exact_match": None if generated == 0 else exact / generated,
        "generated_count": generated,
    }


@torch.inference_mode()
def main() -> None:
    args = arguments()
    data_path = Path(args.data).resolve()
    records = load_jsonl(data_path)
    if args.max_records is not None:
        records = records[: args.max_records]
    checkpoint_path = Path(args.bank).resolve()
    model, checkpoint = load_chat_checkpoint(checkpoint_path, args.device)
    if model.bank_config is None:
        raise SystemExit("MQAR gate requires a BANK checkpoint")
    tokenizer = ChatTokenizer.load(checkpoint_path.parent / "tokenizer.json")
    rows = []
    for index, record in enumerate(records):
        candidates = record["rank_candidates"]
        prepared = prepare_prefix(
            model, tokenizer, record["prompt"], args.device
        )
        loss_sums, counts = candidate_losses(
            model,
            tokenizer,
            record["prompt"],
            candidates,
            args.device,
            prepared=prepared,
        )
        average_nll = loss_sums / counts
        correct_index = candidates.index(record["answer"])
        correct_score = float(loss_sums[correct_index].item())
        correct_average = float(average_nll[correct_index].item())
        rank = int(
            1
            + torch.count_nonzero(
                loss_sums < loss_sums[correct_index]
            ).item()
        )
        generated = None
        exact = None
        if not args.skip_generation:
            generated = greedy_answer(
                model,
                tokenizer,
                prepared[0],
                prepared[1],
                args.device,
            )
            exact = generated.strip() == record["answer"]
        rows.append(
            {
                "id": record["id"],
                "kind": record["kind"],
                "load": record["load"],
                "distance": record["effective_distance"],
                "distractors": record["requested_distractors"],
                "correct_nll": correct_score,
                "correct_nll_per_token": correct_average,
                "answer_token_count": int(counts[correct_index].item()),
                "rank": rank,
                "generated": generated,
                "exact_match": exact,
            }
        )
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"evaluated={index + 1}/{len(records)}", flush=True)

    grouped = {}
    dimensions = {
        "kind": lambda row: row["kind"],
        "load": lambda row: str(row["load"]),
        "distance": lambda row: str(row["distance"]),
        "distractors": lambda row: str(row["distractors"]),
    }
    for name, key_fn in dimensions.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)
        grouped[name] = {
            key: summarize(items) for key, items in sorted(groups.items())
        }
    overall = summarize(rows)
    chance = 1.0 / len(records[0]["rank_candidates"])
    gate_passed = overall["top1_wilson95"][0] > chance
    report = {
        "protocol": "celllm-stage8.2-mqar-gate-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "data": str(data_path),
        "device": args.device,
        "record_limit": args.max_records,
        "rank_candidate_count": len(records[0]["rank_candidates"]),
        "chance_top1": chance,
        "gate_rule": "overall top1 Wilson95 lower bound > chance",
        "gate_passed": gate_passed,
        "decision": (
            "continue_to_causal_utility"
            if gate_passed
            else "inconclusive_checkpoint_at_chance"
        ),
        "overall": overall,
        "groups": grouped,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("gate_passed", "decision", "overall")}, indent=2))


if __name__ == "__main__":
    main()
