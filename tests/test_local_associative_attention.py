"""Causal invariants for local associative message passing."""

from __future__ import annotations

import torch

from celllm.attention import LocalAssociativeMessagePassing
from celllm.chat_model import CellLMChatModel
from celllm.config import (
    LocalAssociativeConfig,
    ModelConfig,
    StateMatchedBankConfig,
)


def local_config(**overrides) -> LocalAssociativeConfig:
    values = {
        "radius": 2,
        "key_size": 3,
        "value_size": 3,
        "retrieval_scale": 1.0,
    }
    values.update(overrides)
    return LocalAssociativeConfig(**values)


def test_local_messages_are_strictly_causal_and_bounded_by_radius():
    torch.manual_seed(4)
    module = LocalAssociativeMessagePassing(4, local_config(gated=True))
    activity = torch.randn(2, 6, 4)
    global_drive = torch.randn_like(activity)
    changed = activity.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 20

    original, weights = module(
        activity, global_drive, return_weights=True
    )
    perturbed = module(changed, global_drive)

    torch.testing.assert_close(original[:, :4], perturbed[:, :4])
    torch.testing.assert_close(original[:, 0], torch.zeros_like(original[:, 0]))
    assert weights.shape == (2, 6, 2)
    assert torch.count_nonzero(weights[:, 0]) == 0
    assert torch.count_nonzero(weights[:, 1, 1]) == 0


def test_ungated_messages_are_normalized_local_value_averages():
    module = LocalAssociativeMessagePassing(
        2,
        LocalAssociativeConfig(
            radius=2,
            key_size=2,
            value_size=2,
            gated=False,
            retrieval_scale=1.0,
            learnable_retrieval_scale=False,
            epsilon=1e-8,
        ),
    )
    with torch.no_grad():
        module.query.weight.zero_()
        module.key.weight.zero_()
        module.value.weight.copy_(torch.eye(2))
        module.output.weight.copy_(torch.eye(2))
    activity = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]])

    drive, weights = module(
        activity, torch.zeros_like(activity), return_weights=True
    )

    torch.testing.assert_close(drive[:, 0], torch.zeros(1, 2))
    torch.testing.assert_close(drive[:, 1], torch.tanh(activity[:, 0]))
    expected = torch.tanh(activity[:, :2]).mean(dim=1)
    torch.testing.assert_close(drive[:, 2], expected)
    torch.testing.assert_close(weights[:, 2].sum(dim=-1), torch.ones(1))


def test_mask_removes_sources_and_receivers():
    module = LocalAssociativeMessagePassing(3, local_config(gated=False))
    activity = torch.randn(1, 5, 3)
    mask = torch.tensor([[True, False, True, True, False]])

    drive, weights = module(
        activity,
        torch.zeros_like(activity),
        mask=mask,
        return_weights=True,
    )

    assert torch.count_nonzero(drive[:, 1]) == 0
    assert torch.count_nonzero(drive[:, 4]) == 0
    assert weights[0, 2, 0] == 0
    assert weights[0, 3, 1] == 0


def test_gate_has_gradient_and_variants_have_equal_parameter_count():
    gated = LocalAssociativeMessagePassing(4, local_config(gated=True))
    ungated = LocalAssociativeMessagePassing(4, local_config(gated=False))
    assert sum(p.numel() for p in gated.parameters()) == sum(
        p.numel() for p in ungated.parameters()
    )
    activity = torch.randn(2, 5, 4, requires_grad=True)
    gated(activity, torch.randn_like(activity)).square().mean().backward()
    assert gated.gate.weight.grad is not None
    assert torch.isfinite(gated.gate.weight.grad).all()


def test_bank_without_local_is_unchanged_and_local_requires_bank():
    cfg = ModelConfig(n=4, d=8, r=2, k=2, vocab_size=32)
    bank = StateMatchedBankConfig(
        slots=4, key_size=4, value_size=4, chunk_size=4
    )
    baseline = CellLMChatModel(cfg, bank=bank)
    assert baseline.local_config is None
    assert baseline.core.cell.local_attention is None

    try:
        CellLMChatModel(cfg, local=local_config())
    except ValueError as error:
        assert "require BANK" in str(error)
    else:
        raise AssertionError("local messages without BANK must fail")
