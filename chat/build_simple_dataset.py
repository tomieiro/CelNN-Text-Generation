#!/usr/bin/env python3
"""Write the reproducible simple-English conversation curriculum as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from celllm.simple_dialogues import build_simple_dialogues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="chat/data/simple-dialogues.jsonl")
    parser.add_argument("--repeats", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    conversations = build_simple_dialogues(
        seed=args.seed, repeats=args.repeats
    )
    unique = {}
    for conversation in conversations:
        key = tuple(
            (message.role, message.content)
            for message in conversation.messages
        )
        unique[key] = conversation
    conversations = list(unique.values())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            record = {
                "messages": [
                    {"role": item.role, "content": item.content}
                    for item in conversation.messages
                ]
            }
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(conversations)} conversations to {output}")


if __name__ == "__main__":
    main()
