import torch

from celllm.config import ModelConfig
from celllm.diagnostics import settling_trace
from celllm.model import CelNNLanguageModel


def test_trace_lengths_match_the_step_count():
    cfg = ModelConfig(n=64, d=32, r=2, k=32, mixer="rank8")
    trace = settling_trace(
        CelNNLanguageModel(cfg), torch.randint(0, 27, (2, 64))
    )
    assert len(trace["state_norm"]) == 32
    assert len(trace["delta_norm"]) == 31


def test_zero_templates_keep_zero_state():
    cfg = ModelConfig(n=64, d=32, r=2, k=32, eta=0.5, mixer="none")
    model = CelNNLanguageModel(cfg)
    with torch.no_grad():
        model.cell.dynamics.feedback.zero_()
        model.cell.dynamics.control.zero_()
        model.cell.dynamics.bias.zero_()
    trace = settling_trace(model, torch.randint(0, 27, (2, 64)))
    assert max(trace["delta_norm"]) < 1e-6


def test_contracting_dynamics_show_shrinking_deltas():
    cfg = ModelConfig(n=64, d=32, r=2, k=32, eta=0.5, mixer="none")
    model = CelNNLanguageModel(cfg)
    with torch.no_grad():
        model.cell.dynamics.feedback.mul_(0.01)
    deltas = settling_trace(model, torch.randint(0, 27, (2, 64)))[
        "delta_norm"
    ]
    assert deltas[-1] < deltas[0]
