#!/usr/bin/env python3
"""Download a bounded SmolTalk everyday-conversation subset as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="experiment-chat/data/smoltalk-everyday.jsonl"
    )
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--max-messages", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=160)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "Install the preparation dependency with `pip install datasets`."
        ) from error

    stream = load_dataset(
        "HuggingFaceTB/smoltalk",
        "everyday-conversations",
        split="train",
        streaming=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in stream:
            messages = []
            expected = "user"
            for item in record["messages"][: args.max_messages]:
                role = str(item["role"])
                content = " ".join(str(item["content"]).split())
                if role != expected or not content or len(content) > args.max_chars:
                    break
                messages.append({"role": role, "content": content})
                expected = "assistant" if expected == "user" else "user"
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                continue
            handle.write(json.dumps({"messages": messages}) + "\n")
            written += 1
            if written == args.limit:
                break
    print(f"wrote {written} SmolTalk conversations to {output}")


if __name__ == "__main__":
    main()
