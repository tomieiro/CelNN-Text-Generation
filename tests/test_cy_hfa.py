"""CY-HFA is a causal normalized associative field, not global attention."""

from __future__ import annotations

import torch

from celnn import AssociativeFieldState
from celllm.attention import (
    CausalFieldPropagation,
    ChuaYangHebbianFieldAttention,
)
from celllm.chat_model import CellLMChatModel
from celllm.config import CYHFAConfig, ModelConfig


def field_config(**overrides):
    values = {
        "key_size": 4,
        "value_size": 3,
        "diffusion_radius": 1,
        "diffusion_rate": 0.1,
        "max_diffusion": 0.25,
        "chunk_size": 4,
        "detach_updates": False,
    }
    values.update(overrides)
    return CYHFAConfig(**values)


def make_model(**field_overrides):
    return CellLMChatModel(
        ModelConfig(n=4, d=8, r=1, k=3, vocab_size=300),
        field=field_config(**field_overrides),
    )


def test_field_state_has_one_memory_and_normalizer_per_cell():
    model = make_model()
    state = model.new_plastic_state(2)

    assert state.memory.shape == (2, 4, 3, 4)
    assert state.normalizer.shape == (2, 4, 4)
    assert state.updates == 0
    assert model.memory_config is None
    assert model.field_config.key_size == 4


def test_empty_field_retrieval_is_zero_and_non_mutating():
    attention = ChuaYangHebbianFieldAttention(8, field_config())
    state = attention.new_state(2, 4)
    activity = torch.randn(2, 4, 8)

    retrieved = attention.retrieve(activity, state)

    torch.testing.assert_close(retrieved, torch.zeros_like(activity))
    assert state.updates == 0


def test_propagation_moves_memory_forward_but_never_backward():
    propagation = CausalFieldPropagation(
        radius=1, rate=0.2, max_rate=0.2, learnable=False
    )
    baseline = torch.zeros(1, 4, 1)
    baseline[:, 2] = 1
    changed_future = baseline.clone()
    changed_future[:, 3] = 9

    first = propagation(baseline)
    second = propagation(changed_future)

    torch.testing.assert_close(first[:, :3], second[:, :3])
    assert first[0, 3, 0] > 0
    assert first[0, 1, 0] == 0


def test_memory_and_normalizer_share_the_same_cellular_propagator():
    attention = ChuaYangHebbianFieldAttention(
        8,
        field_config(
            diffusion_rate=0.2,
            max_diffusion=0.2,
            learnable_diffusion=False,
        ),
    )
    memory = torch.zeros(1, 4, 3, 4)
    normalizer = torch.zeros(1, 4, 4)
    memory[:, 0] = 1
    normalizer[:, 0] = 2
    state = AssociativeFieldState(memory, normalizer)

    propagated = attention.advance(
        state,
        torch.zeros(1, 4, 8),
        mask=torch.zeros(1, 4, dtype=torch.bool),
    )

    torch.testing.assert_close(
        propagated.memory[:, 1], 0.2 * memory[:, 0]
    )
    torch.testing.assert_close(
        propagated.normalizer[:, 1], 0.2 * normalizer[:, 0]
    )
    assert torch.all(propagated.normalizer >= 0)


def test_next_block_starts_from_previous_terminal_causal_field():
    attention = ChuaYangHebbianFieldAttention(8, field_config())
    memory = torch.randn(1, 4, 3, 4)
    normalizer = torch.rand(1, 4, 4)
    state = AssociativeFieldState(memory, normalizer, updates=3)

    carried = attention.begin_block(state)

    for position in range(4):
        torch.testing.assert_close(
            carried.memory[:, position], memory[:, -1]
        )
        torch.testing.assert_close(
            carried.normalizer[:, position], normalizer[:, -1]
        )
    assert carried.updates == 3


def test_future_tokens_cannot_change_prefix_logits():
    torch.manual_seed(7)
    model = make_model().eval()
    first = torch.tensor([[11, 12, 13, 14]])
    second = torch.tensor([[11, 12, 77, 88]])

    first_logits = model(first)
    second_logits = model(second)

    torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2])


def test_partial_block_is_padded_without_changing_public_length():
    model = make_model()
    state = model.new_plastic_state(1)
    tokens = torch.tensor([[11, 12]])

    logits, updated = model.forward_with_state(tokens, state)

    assert logits.shape == (1, 2, 300)
    assert updated.memory.shape == state.memory.shape
    assert updated.updates == model.cfg.k


def test_speculative_forward_evolves_field_but_does_not_commit_it():
    model = make_model()
    state = model.new_plastic_state(1)
    tokens = torch.tensor([[11, 12, 13]])

    _, returned = model.forward_with_state(
        tokens, state, update_plasticity=False
    )

    assert returned is state
    assert state.updates == 0
    assert torch.count_nonzero(state.memory) == 0


def test_chat_loss_reaches_all_associative_and_diffusion_parameters():
    torch.manual_seed(4)
    model = make_model(value_size=4)
    tokens = torch.randint(6, 300, (2, 8))
    assistant = torch.ones_like(tokens, dtype=torch.bool)

    loss = model.loss(tokens, assistant)
    loss.backward()

    module = model.core.cell.attention
    for parameter in (
        module.query.weight,
        module.key.weight,
        module.value.weight,
        module.output.weight,
        module.write_controls.weight,
        module.retrieval_scale,
        module.propagation.raw_rate,
        module.propagation.offset_logits,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
