"""Text8 character-level data pipeline.

Text8 contains cleaned lowercase Wikipedia text reduced to ``a-z`` and space.
Keeping the vocabulary at 27 symbols makes Experiment 0 measure the cellular
core rather than the vocabulary projection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

CHARS = " abcdefghijklmnopqrstuvwxyz"
_STOI = {character: index for index, character in enumerate(CHARS)}


def encode(text: str) -> np.ndarray:
    """Map a Text8-style string to an array of int64 token IDs."""
    return np.fromiter(
        (_STOI[character] for character in text),
        dtype=np.int64,
        count=len(text),
    )


def decode(ids: np.ndarray) -> str:
    """Map token IDs back to a string."""
    return "".join(CHARS[int(token_id)] for token_id in ids)


def load_text8(path: str | Path) -> np.ndarray:
    """Read and encode an extracted Text8 file."""
    return encode(Path(path).read_text(encoding="ascii"))


def split_text8(
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the conventional 90M/5M/5M Text8 partitions."""
    return ids[:90_000_000], ids[90_000_000:95_000_000], ids[95_000_000:]


class Batcher:
    """Sample deterministic random contiguous windows of length ``n``."""

    def __init__(
        self,
        ids: np.ndarray,
        n: int,
        batch_size: int,
        seed: int,
    ) -> None:
        if len(ids) < n:
            raise ValueError(f"need at least {n} tokens, got {len(ids)}")

        self._ids = ids
        self._n = n
        self._batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    def next(self) -> torch.Tensor:
        """Return the next batch as an int64 CPU tensor."""
        starts = self._rng.integers(
            0,
            len(self._ids) - self._n + 1,
            size=self._batch_size,
        )
        windows = np.stack(
            [self._ids[start : start + self._n] for start in starts]
        )
        return torch.from_numpy(windows)
