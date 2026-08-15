import pytest

from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    ModelConfig,
    PlasticityConfig,
    StateMatchedBankConfig,
)


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


def test_plasticity_rule_and_chunk_size_are_validated():
    assert PlasticityConfig().rule == "oja"
    with pytest.raises(ValueError, match="rule"):
        PlasticityConfig(rule="unknown")
    with pytest.raises(ValueError, match="chunk_size"):
        PlasticityConfig(chunk_size=0)


def test_hebbian_attention_configuration_is_validated():
    assert HebbianAttentionConfig().key_size == 32
    with pytest.raises(ValueError, match="sizes"):
        HebbianAttentionConfig(key_size=0)
    with pytest.raises(ValueError, match="retention"):
        HebbianAttentionConfig(min_retention=1.1)
    with pytest.raises(ValueError, match="chunk_size"):
        HebbianAttentionConfig(chunk_size=0)


def test_cy_hfa_configuration_bounds_local_field_dynamics():
    config = CYHFAConfig()
    assert config.key_size == 32
    assert config.diffusion_rate <= config.max_diffusion <= 1
    with pytest.raises(ValueError, match="diffusion rate"):
        CYHFAConfig(diffusion_rate=0.3, max_diffusion=0.2)
    with pytest.raises(ValueError, match="radius"):
        CYHFAConfig(diffusion_radius=0)
    with pytest.raises(ValueError, match="epsilon"):
        CYHFAConfig(epsilon=0)


def test_state_matched_bank_configuration_is_validated():
    config = StateMatchedBankConfig()
    assert config.slots == config.chunk_size == 16
    with pytest.raises(ValueError, match="slots"):
        StateMatchedBankConfig(slots=0)
    with pytest.raises(ValueError, match="temperatures"):
        StateMatchedBankConfig(read_temperature=0)
