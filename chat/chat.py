#!/usr/bin/env python3
"""Talk to a trained CellLM chat checkpoint."""

from __future__ import annotations

import argparse

import torch

from celllm.chat_checkpoint import load_chat_checkpoint
from celllm.chat_generation import ChatSession, SamplingConfig
from celllm.chat_tokenizer import ChatTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="chat/outputs/tokenizer.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    model, metadata = load_chat_checkpoint(args.checkpoint, args.device)
    tokenizer = ChatTokenizer.load(args.tokenizer)
    session = ChatSession(
        model,
        tokenizer,
        sampling=SamplingConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ),
    )
    print(f"CellLM chat step {metadata['step']} on {args.device}")
    print("Use /reset to clear Hebbian memory and /quit to leave.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text == "/quit":
            break
        if text == "/reset":
            session.reset()
            print("memory reset")
            continue
        if text:
            print(f"celllm> {session.reply(text)}\n")


if __name__ == "__main__":
    main()
