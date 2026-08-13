import torch

from celllm.templates import DiagonalTemplate, ScalarTemplate, build_template


def _n_params(module):
    return sum(parameter.numel() for parameter in module.parameters())


def test_scalar_template_has_one_parameter_per_offset():
    assert _n_params(ScalarTemplate(r=2, d=128, causal=True)) == 3


def test_diagonal_template_has_d_parameters_per_offset():
    assert _n_params(DiagonalTemplate(r=2, d=128, causal=True)) == 384


def test_scalar_equals_diagonal_with_uniform_channels():
    torch.manual_seed(0)
    x = torch.randn(2, 12, 8)
    scalar = ScalarTemplate(r=2, d=8, causal=True)
    diagonal = DiagonalTemplate(r=2, d=8, causal=True)
    with torch.no_grad():
        scalar.weights.copy_(torch.tensor([[0.3], [-0.5], [1.1]]))
        diagonal.weights.copy_(
            torch.tensor([0.3, -0.5, 1.1]).unsqueeze(1).expand(3, 8)
        )
    torch.testing.assert_close(scalar(x), diagonal(x))


def test_factory_dispatches_by_name():
    assert isinstance(build_template("scalar", 2, 8, True), ScalarTemplate)
    assert isinstance(build_template("diagonal", 2, 8, True), DiagonalTemplate)
