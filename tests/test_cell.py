import torch

from celllm.cell import CelNNCell, piecewise_linear
from celllm.config import ModelConfig


def test_piecewise_linear_is_the_chua_saturation():
    x = torch.tensor([-3.0, -1.0, 0.0, 0.5, 1.0, 3.0])
    expected = torch.tensor([-1.0, -1.0, 0.0, 0.5, 1.0, 1.0])
    torch.testing.assert_close(piecewise_linear(x), expected)


def test_leak_term_decays_state_when_templates_are_zero():
    cfg = ModelConfig(d=8, r=2, k=32, n=64, eta=0.5)
    cell = CelNNCell(cfg)
    with torch.no_grad():
        cell.a.weights.zero_()
        cell.b.weights.zero_()
        cell.z.zero_()
    x = torch.randn(2, 64, 8)
    torch.testing.assert_close(cell.step(x, torch.zeros_like(x)), 0.5 * x)


def test_gradients_reach_templates_and_bias():
    cell = CelNNCell(ModelConfig(d=8, r=2, k=32, n=64))
    embedding = torch.randn(2, 64, 8)
    drive = cell.control_drive(embedding)
    x = cell.step(torch.zeros_like(embedding), drive)
    x = cell.step(x, drive)
    x.sum().backward()
    assert cell.a.weights.grad is not None
    assert cell.b.weights.grad is not None
    assert cell.z.grad is not None
    assert cell.a.weights.grad.abs().sum() > 0


def test_control_drive_is_independent_of_state():
    cell = CelNNCell(ModelConfig(d=8, r=2, k=32, n=64))
    embedding = torch.randn(1, 64, 8)
    torch.testing.assert_close(
        cell.control_drive(embedding), cell.control_drive(embedding)
    )


def test_bounded_drive_respects_the_input_constraint():
    def identity_control(bound: bool):
        cell = CelNNCell(
            ModelConfig(d=8, r=2, k=32, n=64, bound_drive=bound)
        )
        with torch.no_grad():
            cell.b.weights.zero_()
            cell.b.weights[-1] = 1.0
            cell.z.zero_()
        return cell

    huge = torch.full((1, 64, 8), 50.0)
    assert identity_control(True).control_drive(huge).abs().max() <= 1.0
    assert identity_control(False).control_drive(huge).abs().max() > 1.0
