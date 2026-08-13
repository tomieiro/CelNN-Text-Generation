"""Empirical settling diagnostics for the causal cellular field."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def settling_trace(model: nn.Module, tokens: torch.Tensor) -> dict[str, list[float]]:
    """Measure mean state and consecutive-update norms through all steps."""
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    model.eval()

    embedding = model.embed(tokens)
    drive = model.cell.control_drive(embedding)
    state = torch.zeros_like(embedding)
    state_norm: list[float] = []
    delta_norm: list[float] = []

    for step in range(model.cfg.k):
        previous = state
        state = model.cell.step(state, drive)
        state_norm.append(state.norm(dim=-1).mean().item())
        if step > 0:
            delta_norm.append((state - previous).norm(dim=-1).mean().item())

    return {"state_norm": state_norm, "delta_norm": delta_norm}
