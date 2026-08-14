"""Hebbian CellLM memory remains explicit, causal, and session-local."""

from __future__ import annotations

import torch

from celllm.config import ModelConfig, PlasticityConfig
from celllm.model import PlasticCelNNLanguageModel


def make_model(**plasticity_overrides):
    config = ModelConfig(n=16, d=8, r=1, k=15, vocab_size=27)
    plasticity = PlasticityConfig(
        chunk_size=4,
        **plasticity_overrides,
    )
    return PlasticCelNNLanguageModel(config, plasticity)


def test_forward_shape_and_explicit_state():
    model = make_model()
    tokens = torch.randint(0, 27, (2, 16))

    logits, state = model.forward_with_state(tokens)

    assert logits.shape == (2, 16, 27)
    assert state.memory.shape == (2, 8, 8)
    assert state.updates == 1


def test_current_block_cannot_use_the_memory_it_writes():
    torch.manual_seed(0)
    model = make_model().eval()
    tokens = torch.randint(0, 27, (1, 16))
    altered = tokens.clone()
    altered[:, 10] = (altered[:, 10] + 1) % 27

    base, _ = model.forward_with_state(tokens)
    after, _ = model.forward_with_state(altered)

    torch.testing.assert_close(base[:, :10], after[:, :10])
    assert not torch.allclose(base[:, 10], after[:, 10])


def test_memory_changes_only_when_update_is_requested():
    model = make_model()
    tokens = torch.randint(0, 27, (1, 16))
    initial = model.new_plastic_state(1)

    _, unchanged = model.forward_with_state(
        tokens, initial, update_plasticity=False
    )
    _, learned = model.forward_with_state(tokens, initial)

    assert unchanged is initial
    assert torch.count_nonzero(initial.memory) == 0
    assert torch.count_nonzero(learned.memory) > 0


def test_memory_from_one_block_changes_the_next_block():
    torch.manual_seed(1)
    model = make_model(
        alpha=10.0, learning_rate=1.0, memory_limit=None
    ).eval()
    context = torch.randint(0, 27, (1, 16))
    query = torch.randint(0, 27, (1, 16))
    empty = model.new_plastic_state(1)

    _, learned = model.forward_with_state(context, empty)
    baseline, _ = model.forward_with_state(
        query, empty, update_plasticity=False
    )
    recalled, _ = model.forward_with_state(
        query, learned, update_plasticity=False
    )

    assert not torch.allclose(recalled, baseline)


def test_reset_prevents_memory_leaking_between_sessions():
    model = make_model()
    tokens = torch.randint(0, 27, (1, 16))
    empty = model.new_plastic_state(1)
    _, learned = model.forward_with_state(tokens, empty)

    reset = learned.reset()

    assert learned.updates == 1
    assert reset.updates == 0
    assert torch.count_nonzero(reset.memory) == 0


def test_chunked_loss_reaches_slow_weights_and_alpha():
    torch.manual_seed(2)
    model = make_model(detach_updates=True, learnable_alpha=True)
    tokens = torch.randint(0, 27, (2, 16))

    loss = model.loss(tokens)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.cell.mixer.proj.weight.grad is not None
    assert model.cell.mixer.plasticity.alpha.grad is not None
    assert model.embed.weight.grad is not None


def test_hebbian_and_oja_rules_are_swappable_configuration():
    hebbian = make_model(rule="hebbian")
    oja = make_model(rule="oja")
    tokens = torch.randint(0, 27, (1, 16))

    _, hebbian_state = hebbian.forward_with_state(tokens)
    _, oja_state = oja.forward_with_state(tokens)

    assert hebbian_state.memory.shape == oja_state.memory.shape
