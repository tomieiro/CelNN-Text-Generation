import numpy as np
import torch

from celllm.data import CHARS, Batcher, decode, encode, split_text8


def test_vocabulary_is_27_characters():
    assert len(CHARS) == 27
    assert CHARS[0] == " "
    assert set(CHARS[1:]) == set("abcdefghijklmnopqrstuvwxyz")


def test_encode_decode_roundtrip():
    text = "the quick brown fox"
    assert decode(encode(text)) == text


def test_split_uses_conventional_90_5_5():
    ids = np.zeros(100_000_000, dtype=np.int64)
    train, valid, test = split_text8(ids)
    assert len(train) == 90_000_000
    assert len(valid) == 5_000_000
    assert len(test) == 5_000_000


def test_batcher_shape_and_dtype():
    ids = np.arange(10_000, dtype=np.int64) % 27
    batcher = Batcher(ids, n=64, batch_size=8, seed=0)
    batch = batcher.next()
    assert batch.shape == (8, 64)
    assert batch.dtype == torch.int64
    assert batch.max().item() < 27


def test_batcher_is_deterministic_under_seed():
    ids = np.arange(10_000, dtype=np.int64) % 27
    a = Batcher(ids, n=64, batch_size=8, seed=7).next()
    b = Batcher(ids, n=64, batch_size=8, seed=7).next()
    assert torch.equal(a, b)
