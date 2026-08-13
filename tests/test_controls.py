import torch

from celllm.config import ModelConfig
from celllm.controls import GatedConvLM


def test_forward_shape():
    cfg = ModelConfig(n=64, d=32, vocab_size=27)
    model = GatedConvLM(cfg, layers=4)
    assert model(torch.randint(0, 27, (4, 64))).shape == (4, 64, 27)


def test_control_is_causal():
    torch.manual_seed(0)
    cfg = ModelConfig(n=64, d=32, vocab_size=27)
    model = GatedConvLM(cfg, layers=4).eval()
    tokens = torch.randint(0, 27, (1, 64))
    with torch.no_grad():
        base = model(tokens)
        altered = tokens.clone()
        altered[0, 40] = (altered[0, 40] + 1) % 27
        after = model(altered)
    torch.testing.assert_close(base[:, :40], after[:, :40])


def test_layers_do_not_share_weights():
    cfg = ModelConfig(n=64, d=32, vocab_size=27)
    model = GatedConvLM(cfg, layers=4)
    assert model.blocks[0].conv.weight is not model.blocks[1].conv.weight
