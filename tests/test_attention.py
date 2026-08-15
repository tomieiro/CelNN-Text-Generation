"""Delta-Hebbian attention remains explicit, causal, and differentiable."""

from __future__ import annotations

import torch

from celllm.attention import DeltaHebbianAttention
from celllm.chat_model import CellLMChatModel
from celllm.config import HebbianAttentionConfig, ModelConfig


def attention_config(**overrides):
    values = {
        "key_size": 4,
        "value_size": 3,
        "chunk_size": 4,
        "detach_updates": False,
    }
    values.update(overrides)
    return HebbianAttentionConfig(**values)


def test_attention_state_is_small_explicit_and_resettable():
    attention = DeltaHebbianAttention(8, attention_config())
    state = attention.new_state(2)

    assert state.memory.shape == (2, 3, 4)
    assert state.updates == 0
    assert torch.count_nonzero(state.memory) == 0
    assert state.reset().updates == 0


def test_read_is_non_mutating_and_empty_memory_returns_zero_drive():
    attention = DeltaHebbianAttention(8, attention_config())
    state = attention.new_state(2)
    activity = torch.randn(2, 5, 8)

    drive = attention.retrieve(activity, state)

    torch.testing.assert_close(drive, torch.zeros_like(activity))
    assert state.updates == 0


def test_write_gates_have_bounded_interpretable_ranges():
    config = attention_config(learning_rate=0.2, min_retention=0.9)
    attention = DeltaHebbianAttention(8, config)
    _, values, rates, retentions = attention.associations(
        torch.randn(2, 5, 8)
    )

    assert values.abs().max() <= 1
    assert torch.all((0 <= rates) & (rates <= 0.2))
    assert torch.all((0.9 <= retentions) & (retentions <= 1))


def test_masked_positions_do_not_change_fast_memory():
    attention = DeltaHebbianAttention(8, attention_config())
    state = attention.new_state(1)
    activity = torch.randn(1, 3, 8)

    masked = attention.write(
        state, activity, mask=torch.tensor([[True, False, False]])
    )
    first_only = attention.write(state, activity[:, :1])

    torch.testing.assert_close(masked.memory, first_only.memory)


def test_chat_loss_assigns_credit_through_query_key_value_and_gates():
    torch.manual_seed(4)
    model = CellLMChatModel(
        ModelConfig(n=4, d=8, r=1, k=3, vocab_size=300),
        memory=attention_config(value_size=4),
    )
    tokens = torch.randint(6, 300, (2, 12))
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
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_attention_memory_size_does_not_depend_on_sequence_length():
    model = CellLMChatModel(
        ModelConfig(n=4, d=8, r=1, k=3, vocab_size=300),
        memory=attention_config(),
    )
    before = model.new_plastic_state(1)
    _, after = model.forward_with_state(
        torch.randint(6, 300, (1, 4)), before
    )

    assert before.memory.shape == after.memory.shape == (1, 3, 4)
    assert after.updates == 4


def test_block_logits_are_computed_before_the_block_is_written():
    torch.manual_seed(8)
    model = CellLMChatModel(
        ModelConfig(n=4, d=8, r=1, k=3, vocab_size=300),
        memory=attention_config(
            value_size=4, learning_rate=1.0, retrieval_scale=1.0
        ),
    ).eval()
    first = torch.randint(6, 300, (1, 4))
    second = torch.randint(6, 300, (1, 4))
    empty = model.new_plastic_state(1)

    logits_with_write, written = model.forward_with_state(first, empty)
    logits_without_write, unchanged = model.forward_with_state(
        first, empty, update_plasticity=False
    )
    next_with_memory, _ = model.forward_with_state(second, written)
    next_without_memory, _ = model.forward_with_state(second, unchanged)

    torch.testing.assert_close(logits_with_write, logits_without_write)
    assert torch.count_nonzero(written.memory) > 0
    assert not torch.allclose(next_with_memory, next_without_memory)
