from __future__ import annotations

import pytest
import torch

from celllm.chat_model import CellLMChatModel
from celllm.config import ModelConfig, PlasticityConfig


def make_model():
    return CellLMChatModel(
        ModelConfig(n=8, d=4, r=1, k=7, vocab_size=300),
        PlasticityConfig(chunk_size=8, alpha=1.0),
    )


def test_chat_loss_uses_assistant_targets_and_reaches_plastic_alpha():
    torch.manual_seed(0)
    model = make_model()
    tokens = torch.randint(0, 300, (2, 16))
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, 5:15] = True

    loss = model.loss(tokens, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.core.cell.mixer.plasticity.alpha.grad is not None
    assert model.core.embed.weight.grad is not None


def test_chat_loss_rejects_batches_without_assistant_targets():
    model = make_model()
    tokens = torch.randint(0, 300, (1, 8))
    with pytest.raises(ValueError, match="no assistant"):
        model.loss(tokens, torch.zeros_like(tokens, dtype=torch.bool))


def test_chat_model_exposes_explicit_session_state():
    model = make_model()
    state = model.new_plastic_state(3)
    assert state.memory.shape == (3, 4, 4)
