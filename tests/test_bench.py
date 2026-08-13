import torch

from celllm.bench import measure_latency, measure_peak_memory
from celllm.config import ModelConfig
from celllm.model import CelNNLanguageModel


def test_latency_is_positive_and_finite():
    cfg = ModelConfig(n=64, d=32, vocab_size=27, mixer="rank8")
    model = CelNNLanguageModel(cfg)
    tokens = torch.randint(0, 27, (8, 64))
    latency = measure_latency(model, tokens, warmup=2, iters=3)
    assert 0 < latency < 60


def test_dense_mixer_is_not_faster_than_no_mixer():
    tokens = torch.randint(0, 27, (8, 64))
    light = CelNNLanguageModel(ModelConfig(n=64, d=128, mixer="none"))
    heavy = CelNNLanguageModel(ModelConfig(n=64, d=128, mixer="dense"))
    light_latency = measure_latency(light, tokens, warmup=2, iters=5)
    heavy_latency = measure_latency(heavy, tokens, warmup=2, iters=5)
    assert heavy_latency >= light_latency * 0.8


def test_peak_memory_is_zero_on_cpu():
    cfg = ModelConfig(n=64, d=32, vocab_size=27)
    model = CelNNLanguageModel(cfg)
    tokens = torch.randint(0, 27, (2, 64))
    assert measure_peak_memory(model, tokens) == 0
