import math

from celllm.config import ModelConfig
from celllm.metrics import (
    analytic_flops,
    bits_per_character,
    count_parameters,
    gated_conv_flops,
)
from celllm.model import CelNNLanguageModel


def test_bpc_of_uniform_predictions_over_27_symbols():
    assert abs(bits_per_character(math.log(27)) - math.log2(27)) < 1e-9
    assert abs(bits_per_character(math.log(27)) - 4.7549) < 1e-3


def test_core_parameter_count_matches_the_ladder_table():
    cfg = ModelConfig(d=128, r=2, k=32, n=64, mixer="rank8")
    counts = count_parameters(CelNNLanguageModel(cfg))
    assert counts["core"] == 2_944
    assert counts["embedding"] == 27 * 128


def test_rung_b_core_is_896():
    cfg = ModelConfig(d=128, r=2, k=32, n=64, mixer="none")
    assert count_parameters(CelNNLanguageModel(cfg))["core"] == 896


def test_spatial_flops_scale_as_n_d_k_offsets():
    cfg = ModelConfig(n=64, d=128, r=2, k=32, mixer="none")
    flops = analytic_flops(cfg)
    assert flops["spatial"] == 64 * 128 * 3 * 32 + 64 * 128 * 3
    assert flops["channel"] == 0


def test_rank_q_channel_flops_are_2ndqk():
    cfg = ModelConfig(n=64, d=128, r=2, k=32, mixer="rank8")
    assert analytic_flops(cfg)["channel"] == 2 * 64 * 128 * 8 * 32


def test_gated_control_counts_every_convolutional_layer():
    cfg = ModelConfig(n=64, d=128, r=2, k=32)
    expected = 4 * 64 * 3 * 128 * 256 + 64 * 128 * 27
    assert gated_conv_flops(cfg) == expected
