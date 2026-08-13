import torch

from celllm.stencil import aggregate


def test_output_shape_matches_input():
    x = torch.randn(4, 16, 8)
    weights = torch.randn(3, 8)
    assert aggregate(x, weights, r=2, causal=True).shape == (4, 16, 8)


def test_identity_weights_select_current_position():
    x = torch.randn(2, 10, 4)
    weights = torch.zeros(3, 4)
    weights[2] = 1.0
    torch.testing.assert_close(aggregate(x, weights, r=2, causal=True), x)


def test_shift_weights_select_previous_position():
    x = torch.randn(2, 10, 4)
    weights = torch.zeros(3, 4)
    weights[1] = 1.0
    out = aggregate(x, weights, r=2, causal=True)
    torch.testing.assert_close(out[:, 1:], x[:, :-1])
    torch.testing.assert_close(out[:, 0], torch.zeros(2, 4))


def test_causality_future_cannot_influence_present():
    """Perturbing position j must not change any output at position i < j."""
    torch.manual_seed(0)
    x = torch.randn(1, 12, 4)
    weights = torch.randn(3, 4)
    base = aggregate(x, weights, r=2, causal=True)

    perturbed = x.clone()
    perturbed[0, 7] += 100.0
    after = aggregate(perturbed, weights, r=2, causal=True)

    torch.testing.assert_close(base[:, :7], after[:, :7])
    assert not torch.allclose(base[:, 7], after[:, 7])


def test_symmetric_mode_uses_both_sides():
    x = torch.randn(1, 10, 4)
    weights = torch.zeros(5, 4)
    weights[4] = 1.0
    out = aggregate(x, weights, r=2, causal=False)
    torch.testing.assert_close(out[:, :-2], x[:, 2:])
