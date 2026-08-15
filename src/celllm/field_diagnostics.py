"""Quantitative diagnostics for causal associative-field transport."""

from __future__ import annotations

import torch

from celllm.attention import CausalFieldPropagation


@torch.no_grad()
def impulse_trajectory(
    propagation: CausalFieldPropagation,
    *,
    cells: int,
    steps: int,
    source: int = 0,
    write_after_step: int = 0,
) -> torch.Tensor:
    """Return field magnitude after every real propagate-then-write step.

    ``write_after_step=0`` models a new association produced by the first
    Chua--Yang refinement: step one propagates the old field, then writes the
    impulse. The returned shape is ``(steps + 1, cells)`` and includes the
    empty field before the first step.
    """
    if cells < 2:
        raise ValueError("cells must be at least two")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0 <= source < cells:
        raise ValueError("source must identify a lattice cell")
    if not 0 <= write_after_step < steps:
        raise ValueError("write_after_step must identify a refinement step")

    like = propagation.raw_rate
    field = torch.zeros(1, cells, 1, dtype=like.dtype, device=like.device)
    impulse = torch.zeros_like(field)
    impulse[:, source] = 1.0
    history = [field[0, :, 0].clone()]
    for step in range(steps):
        field = propagation(field)
        if step == write_after_step:
            field = field + impulse
        history.append(field[0, :, 0].clone())
    return torch.stack(history)


def furthest_reached(
    trajectory: torch.Tensor, *, threshold: float = 0.0
) -> int | None:
    """Return the furthest cell above ``threshold`` in the final step."""
    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape (steps, cells)")
    reached = torch.nonzero(trajectory[-1].abs() > threshold).flatten()
    return int(reached[-1]) if reached.numel() else None
