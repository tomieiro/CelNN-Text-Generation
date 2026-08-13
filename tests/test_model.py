import math

import torch

from celllm.config import ModelConfig
from celllm.model import CelNNLanguageModel


def test_forward_shape():
    cfg = ModelConfig(n=64, d=32, r=2, k=32, vocab_size=27, mixer="rank8")
    model = CelNNLanguageModel(cfg)
    tokens = torch.randint(0, 27, (4, 64))
    assert model(tokens).shape == (4, 64, 27)


def test_end_to_end_causality():
    torch.manual_seed(0)
    cfg = ModelConfig(n=64, d=32, r=2, k=32, vocab_size=27, mixer="rank8")
    model = CelNNLanguageModel(cfg).eval()
    tokens = torch.randint(0, 27, (1, 64))
    with torch.no_grad():
        base = model(tokens)
        altered = tokens.clone()
        altered[0, 40] = (altered[0, 40] + 1) % 27
        after = model(altered)
    torch.testing.assert_close(base[:, :40], after[:, :40])
    assert not torch.allclose(base[:, 40], after[:, 40])


def test_readout_is_tied_to_embedding():
    model = CelNNLanguageModel(ModelConfig(n=64, d=32, vocab_size=27))
    assert model.readout.weight is model.embed.weight


def test_untrained_loss_is_near_uniform():
    torch.manual_seed(0)
    cfg = ModelConfig(n=64, d=32, vocab_size=27, mixer="rank8")
    model = CelNNLanguageModel(cfg)
    tokens = torch.randint(0, 27, (16, 64))
    loss = model.loss(tokens).item()
    assert abs(loss - math.log(27)) < 0.5


def test_can_overfit_a_single_batch():
    torch.manual_seed(0)
    cfg = ModelConfig(n=64, d=64, vocab_size=27, mixer="rank8")
    model = CelNNLanguageModel(cfg)
    tokens = torch.randint(0, 27, (2, 64))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = model.loss(tokens).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = model.loss(tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    assert loss.item() < first - 0.3, f"{first:.3f} -> {loss.item():.3f}"
