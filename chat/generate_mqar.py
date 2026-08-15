#!/usr/bin/env python3
"""Generate and freeze the Stage-8.2 synthetic MQAR dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from celllm.chat_tokenizer import ChatTokenizer
from celllm.mqar import MQARConfig, configuration_dict, generate_mqar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-cell", type=int, default=16)
    parser.add_argument("--seed", type=int, default=82017)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer).resolve()
    tokenizer = ChatTokenizer.load(tokenizer_path)
    config = MQARConfig(
        seed=args.seed, samples_per_cell=args.samples_per_cell
    )
    records = generate_mqar(tokenizer, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(item, sort_keys=True) + "\n" for item in records)
    output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    manifest = {
        "protocol": "celllm-stage8.2-mqar-v1",
        "dataset": str(output.resolve()),
        "sha256": digest,
        "record_count": len(records),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "configuration": configuration_dict(config),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
