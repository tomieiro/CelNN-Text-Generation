import pytest
import torch

from celllm.mixers import DenseMixer, NoMixer, RankQMixer, build_mixer


def _n_params(module):
    return sum(parameter.numel() for parameter in module.parameters())


def test_no_mixer_is_zero_and_parameterless():
    x = torch.randn(2, 6, 8)
    mixer = NoMixer(d=8)
    torch.testing.assert_close(mixer(x), torch.zeros_like(x))
    assert _n_params(mixer) == 0


@pytest.mark.parametrize(
    ("q", "expected"), [(4, 1024), (8, 2048), (16, 4096), (32, 8192)]
)
def test_rank_q_mixer_has_2dq_parameters(q, expected):
    assert _n_params(RankQMixer(d=128, q=q)) == expected


def test_dense_mixer_has_d_squared_parameters():
    assert _n_params(DenseMixer(d=128)) == 16_384


def test_rank_q_mixer_actually_mixes_channels():
    torch.manual_seed(0)
    mixer = RankQMixer(d=8, q=4)
    x = torch.zeros(1, 1, 8)
    base = mixer(x)
    x[0, 0, 0] = 1.0
    after = mixer(x)
    changed = (~torch.isclose(base[0, 0], after[0, 0])).sum().item()
    assert changed > 1


def test_factory_names():
    assert isinstance(build_mixer("none", 8), NoMixer)
    assert isinstance(build_mixer("rank8", 128), RankQMixer)
    assert isinstance(build_mixer("dense", 8), DenseMixer)
    with pytest.raises(ValueError):
        build_mixer("rank7", 8)
