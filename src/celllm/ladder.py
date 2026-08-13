"""Experiment 0 capacity ladder and command-line runner."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from celllm.bench import measure_latency
from celllm.config import ModelConfig, TrainConfig
from celllm.controls import GatedConvLM
from celllm.data import Batcher, load_text8, split_text8
from celllm.metrics import analytic_flops, count_parameters, gated_conv_flops
from celllm.model import CelNNLanguageModel
from celllm.train import set_seed, train

RUNGS: dict[str, dict[str, str | bool]] = {
    "A": {"spatial": "scalar", "mixer": "none", "control": False},
    "B": {"spatial": "diagonal", "mixer": "none", "control": False},
    "C": {"spatial": "diagonal", "mixer": "rank4", "control": False},
    "D": {"spatial": "diagonal", "mixer": "rank8", "control": False},
    "E": {"spatial": "diagonal", "mixer": "rank16", "control": False},
    "F": {"spatial": "diagonal", "mixer": "rank32", "control": False},
    "G": {"spatial": "diagonal", "mixer": "dense", "control": False},
    "H": {"spatial": "diagonal", "mixer": "none", "control": True},
}


def _rung_config(name: str, base: ModelConfig) -> ModelConfig:
    try:
        spec = RUNGS[name]
    except KeyError as error:
        raise ValueError(f"unknown rung {name!r}") from error
    return replace(base, spatial=str(spec["spatial"]), mixer=str(spec["mixer"]))


def build_rung(name: str, base: ModelConfig) -> nn.Module:
    """Construct one capacity-ladder model while holding other settings fixed."""
    config = _rung_config(name, base)
    if RUNGS[name]["control"]:
        return GatedConvLM(config, layers=4)
    return CelNNLanguageModel(config)


def run_rung(
    name: str,
    base: ModelConfig,
    train_config: TrainConfig,
    train_ids,
    valid_ids,
    seeds: tuple[int, ...] = (42, 1337, 2024),
    device: str = "cpu",
    checkpoint_dir: str | Path | None = "checkpoints",
) -> dict[str, float | int | str]:
    """Train one rung across seeds and collect quality and cost measurements."""
    scores: list[float] = []
    model: nn.Module | None = None
    for seed in seeds:
        checkpoint_path = (
            Path(checkpoint_dir) / f"rung-{name}-seed-{seed}.pt"
            if checkpoint_dir is not None
            else None
        )
        if checkpoint_path is not None and checkpoint_path.exists():
            model, checkpoint = load_checkpoint(checkpoint_path, device=device)
            result = checkpoint["result"]
            print(f"resumed {checkpoint_path}", flush=True)
        else:
            set_seed(seed)
            model = build_rung(name, base).to(device)
            result = train(
                model,
                Batcher(train_ids, base.n, train_config.batch_size, seed),
                Batcher(valid_ids, base.n, train_config.batch_size, seed + 1),
                replace(train_config, seed=seed),
            )
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    name,
                    base,
                    train_config,
                    seed,
                    model,
                    result,
                )
                print(f"saved {checkpoint_path}", flush=True)
        scores.append(float(result["final_bpc"]))

    assert model is not None
    counts = count_parameters(model)
    config = _rung_config(name, base)
    tokens = torch.randint(0, base.vocab_size, (train_config.batch_size, base.n))
    flops = (
        gated_conv_flops(config)
        if RUNGS[name]["control"]
        else analytic_flops(config)["total"]
    )
    return {
        "rung": name,
        "bpc_mean": statistics.mean(scores),
        "bpc_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "core": counts["core"],
        "total": counts["total"],
        "flops": flops,
        "latency_ms": measure_latency(model, tokens) * 1000.0,
    }


def save_checkpoint(
    path: str | Path,
    rung: str,
    base: ModelConfig,
    train_config: TrainConfig,
    seed: int,
    model: nn.Module,
    result: dict[str, object],
) -> None:
    """Persist trained weights and enough metadata to reconstruct the model."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "rung": rung,
            "seed": seed,
            "model_config": asdict(base),
            "train_config": asdict(train_config),
            "model_state": model.state_dict(),
            "result": result,
        },
        temporary,
    )
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, object]]:
    """Reconstruct a trained rung for evaluation or later inference."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_rung(str(checkpoint["rung"]), ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def format_table(results: list[dict]) -> str:
    """Format ladder results as a compact fixed-width table."""
    header = (
        f"{'rung':<6}{'BPC':>8}{'±':>7}{'core':>10}"
        f"{'total':>10}{'MFLOPs':>10}{'ms':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        lines.append(
            f"{row['rung']:<6}{row['bpc_mean']:>8.3f}"
            f"{row['bpc_std']:>7.3f}{row['core']:>10,}{row['total']:>10,}"
            f"{row['flops'] / 1e6:>10.1f}{row['latency_ms']:>9.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Run selected capacity rungs against an extracted Text8 file."""
    parser = argparse.ArgumentParser(description="Experiment 0 capacity gate")
    parser.add_argument("--data", required=True, help="path to the text8 file")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--rungs", default="ABCDEFGH")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    train_ids, valid_ids, _ = split_text8(load_text8(args.data))
    base = ModelConfig(n=64, d=128, r=2, k=32, vocab_size=27)
    train_config = TrainConfig(steps=args.steps)
    results = []
    for name in args.rungs:
        print(f"=== rung {name} ===", flush=True)
        row = run_rung(
            name,
            base,
            train_config,
            train_ids,
            valid_ids,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        )
        results.append(row)
        print(json.dumps(row), flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + format_table(results))


if __name__ == "__main__":
    main()
