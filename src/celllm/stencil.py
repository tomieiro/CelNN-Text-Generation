"""Local neighbourhood aggregation for CelNN template application.

Offset index ``j`` maps to spatial offset ``j - r``. Causal mode restricts
the neighbourhood to offsets from ``-r`` through zero; symmetric mode spans
``-r`` through ``+r``. Both modes use zero boundary padding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate(
    x: torch.Tensor,
    weights: torch.Tensor,
    r: int,
    causal: bool = True,
) -> torch.Tensor:
    """Apply a channel-wise local template to a ``(batch, n, d)`` field."""
    n = x.shape[1]
    left, right = (r, 0) if causal else (r, r)
    padded = F.pad(x, (0, 0, left, right))

    out = torch.zeros_like(x)
    for offset_index in range(weights.shape[0]):
        out = out + weights[offset_index] * padded[
            :, offset_index : offset_index + n, :
        ]
    return out
