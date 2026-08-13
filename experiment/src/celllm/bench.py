"""Wall-clock latency and accelerator-memory measurements."""

from __future__ import annotations

import statistics
import time

import torch
from torch import nn


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    tokens: torch.Tensor,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """Return median seconds per forward pass."""
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    model.eval()

    for _ in range(warmup):
        model(tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        model(tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def measure_peak_memory(model: nn.Module, tokens: torch.Tensor) -> int:
    """Return peak CUDA allocation for one training step, or zero on CPU."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return 0
    torch.cuda.reset_peak_memory_stats(device)
    model.loss(tokens.to(device)).backward()
    model.zero_grad(set_to_none=True)
    return torch.cuda.max_memory_allocated(device)
