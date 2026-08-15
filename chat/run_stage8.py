#!/usr/bin/env python3
"""Collect BANK trajectory features and causal write utilities."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from celllm.attention import BankWriteAction
from celllm.chat_ablation import validation_conversations
from celllm.chat_checkpoint import load_chat_checkpoint
from celllm.chat_data import ConversationDataset
from celllm.chat_stage8 import evaluate_bank_write_utilities
from celllm.chat_tokenizer import ChatTokenizer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, help="BANK best.pt")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", default="stage8-pilot.json")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--max-dialogues", type=int, default=None)
    parser.add_argument("--max-candidates-per-dialogue", type=int, default=None)
    parser.add_argument(
        "--decompose-write",
        action="store_true",
        help="also measure decay_only and write_no_decay",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    checkpoint_path = Path(args.bank).resolve()
    model, checkpoint = load_chat_checkpoint(checkpoint_path, args.device)
    if model.bank_config is None:
        raise SystemExit("Stage 8.0 currently requires a BANK checkpoint")
    tokenizer = ChatTokenizer.load(checkpoint_path.parent / "tokenizer.json")
    conversations = validation_conversations(args.data, seed=args.seed)
    dataset = ConversationDataset(
        conversations,
        tokenizer,
        max_length=args.sequence_length,
    )
    examples = dataset.examples
    if args.max_dialogues is not None:
        examples = examples[: args.max_dialogues]

    results = []
    actions = [BankWriteAction.NO_ACTION]
    if args.decompose_write:
        actions.extend(
            [BankWriteAction.DECAY_ONLY, BankWriteAction.WRITE_NO_DECAY]
        )
    for dialogue_id, encoded in enumerate(examples):
        result = evaluate_bank_write_utilities(
            model,
            encoded.token_ids[: encoded.length].to(args.device),
            encoded.assistant_mask[: encoded.length].to(args.device),
            dialogue_id=dialogue_id,
            horizons=args.horizons,
            actions=actions,
            max_candidates=args.max_candidates_per_dialogue,
        )
        results.append(asdict(result))
        print(
            f"dialogue={dialogue_id} candidates={len(result.candidates)} "
            f"normal_nll={result.normal_nll:.6f}",
            flush=True,
        )

    total_loss = sum(item["normal_loss_sum"] for item in results)
    total_tokens = sum(item["normal_token_count"] for item in results)
    report = {
        "protocol": "celllm-stage8-trajectory-utility-v1",
        "checkpoint": str(checkpoint_path),
        "architecture": checkpoint["architecture"],
        "step": checkpoint["step"],
        "expected_validation_nll": checkpoint["metrics"].get("valid_loss"),
        "device": args.device,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "horizons": sorted(set(args.horizons)),
        "exploratory": True,
        "write_decomposition": args.decompose_write,
        "dialogue_count": len(results),
        "candidate_count": sum(len(item["candidates"]) for item in results),
        "normal_nll": total_loss / total_tokens,
        "dialogues": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved={output} candidates={report['candidate_count']}")


if __name__ == "__main__":
    main()
