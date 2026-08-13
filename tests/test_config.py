import pytest

from celllm.config import ModelConfig


def test_default_config_satisfies_receptive_field():
    cfg = ModelConfig()
    assert cfg.k * cfg.r + 1 >= cfg.n


def test_insufficient_depth_is_rejected():
    with pytest.raises(ValueError, match="receptive field"):
        ModelConfig(n=64, r=2, k=16)


def test_offsets_are_causal():
    cfg = ModelConfig(r=2)
    assert cfg.offsets == (-2, -1, 0)
    assert cfg.n_offsets == 3


def test_drive_is_bounded_by_default():
    """Chua-Yang eq. 2.1e constrains |v_u| <= 1; embeddings are unbounded."""
    assert ModelConfig().bound_drive is True
