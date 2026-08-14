#!/usr/bin/env python3
"""Evaluate a CellLM checkpoint on simple English chat behaviors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from celllm.chat_checkpoint import load_chat_checkpoint
from celllm.chat_evaluation import evaluate_simple_chat
from celllm.chat_generation import ChatSession, SamplingConfig
from celllm.chat_tokenizer import ChatTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--require-score", type=float)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else (
        checkpoint_path.parent / "tokenizer.json"
    )
    model, metadata = load_chat_checkpoint(checkpoint_path, args.device)
    tokenizer = ChatTokenizer.load(tokenizer_path)
    session = ChatSession(
        model,
        tokenizer,
        sampling=SamplingConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0,
            top_k=0,
            top_p=1,
        ),
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "step": metadata["step"],
        **evaluate_simple_chat(session),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    if args.require_score is not None and report["score"] < args.require_score:
        raise SystemExit(
            f"score {report['score']:.3f} is below {args.require_score:.3f}"
        )


if __name__ == "__main__":
    main()
