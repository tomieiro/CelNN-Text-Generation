"""Training and evaluation utilities for the capacity gate."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from celllm.config import TrainConfig
from celllm.data import Batcher
from celllm.metrics import bits_per_character


class LanguageModel(Protocol):
    """Minimal interface shared by CellLM and its architectural control."""

    training: bool

    def parameters(self): ...

    def train(self, mode: bool = True): ...

    def eval(self): ...

    def loss(self, tokens: torch.Tensor) -> torch.Tensor: ...


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: LanguageModel, batcher: Batcher, n_batches: int) -> float:
    """Return mean bits per character over sampled windows."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    mean_nats = sum(
        model.loss(batcher.next().to(device)).item() for _ in range(n_batches)
    ) / n_batches
    model.train(was_training)
    return bits_per_character(mean_nats)


def _learning_rate(step: int, config: TrainConfig) -> float:
    if step < config.warmup:
        return config.lr * (step + 1) / config.warmup
    progress = (step - config.warmup) / max(1, config.steps - config.warmup)
    return config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    model: LanguageModel,
    train_batcher: Batcher,
    valid_batcher: Batcher,
    config: TrainConfig,
    progress_path: str | Path | None = None,
    checkpoint_every: int = 1_000,
) -> dict[str, object]:
    """Train a language model and return its validation BPC history."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[tuple[int, float]] = []
    best_bpc = float("inf")
    start_step = 0

    progress = Path(progress_path) if progress_path is not None else None
    if progress is not None and progress.exists():
        saved = torch.load(progress, map_location=device, weights_only=True)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        train_batcher.load_state_dict(saved["train_batcher_state"])
        valid_batcher.load_state_dict(saved["valid_batcher_state"])
        history = [tuple(item) for item in saved["history"]]
        best_bpc = float(saved["best_bpc"])
        start_step = int(saved["step"])
        print(f"resumed {progress} at step {start_step}", flush=True)

    model.train()
    for step in range(start_step, config.steps):
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(step, config)

        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(train_batcher.next().to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if (step + 1) % config.eval_every == 0 or step + 1 == config.steps:
            bpc = evaluate(model, valid_batcher, config.eval_batches)
            history.append((step + 1, bpc))
            best_bpc = min(best_bpc, bpc)

        if (
            progress is not None
            and checkpoint_every > 0
            and (step + 1) % checkpoint_every == 0
            and step + 1 < config.steps
        ):
            _save_progress(
                progress,
                step + 1,
                model,
                optimizer,
                train_batcher,
                valid_batcher,
                history,
                best_bpc,
            )

    return {
        "final_bpc": history[-1][1],
        "best_bpc": best_bpc,
        "history": history,
    }


def _save_progress(
    path: Path,
    step: int,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    train_batcher: Batcher,
    valid_batcher: Batcher,
    history: list[tuple[int, float]],
    best_bpc: float,
) -> None:
    """Atomically overwrite the single in-progress checkpoint for a seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_batcher_state": train_batcher.state_dict(),
            "valid_batcher_state": valid_batcher.state_dict(),
            "history": history,
            "best_bpc": best_bpc,
        },
        temporary,
    )
    temporary.replace(path)
