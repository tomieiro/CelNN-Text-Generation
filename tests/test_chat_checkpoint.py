from __future__ import annotations

import torch

from celllm.chat_checkpoint import load_chat_checkpoint, save_chat_checkpoint
from celllm.chat_model import CellLMChatModel
from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    ModelConfig,
    PlasticityConfig,
)


def test_chat_checkpoint_round_trip(tmp_path):
    model = CellLMChatModel(
        ModelConfig(n=8, d=4, r=1, k=7, vocab_size=300),
        PlasticityConfig(chunk_size=8),
    ).eval()
    tokens = torch.randint(0, 300, (1, 8))
    expected = model(tokens)
    path = tmp_path / "chat.pt"

    save_chat_checkpoint(path, model, step=12, metrics={"loss": 1.5})
    restored, metadata = load_chat_checkpoint(path)

    torch.testing.assert_close(restored(tokens), expected)
    assert metadata["step"] == 12
    assert metadata["metrics"] == {"loss": 1.5}
    assert all("memory" not in key for key in metadata["model_state"])


def test_delta_hebb_checkpoint_reconstructs_attention_model(tmp_path):
    model = CellLMChatModel(
        ModelConfig(n=4, d=8, r=1, k=3, vocab_size=300),
        memory=HebbianAttentionConfig(
            key_size=4, value_size=3, chunk_size=4
        ),
    )
    path = tmp_path / "attention.pt"

    save_chat_checkpoint(path, model, step=9)
    restored, metadata = load_chat_checkpoint(path)

    assert metadata["architecture"] == "celllm-chat-delta-hebb"
    assert metadata["format_version"] == 2
    assert restored.memory_config.key_size == 4
    assert restored.new_plastic_state(2).memory.shape == (2, 3, 4)


def test_cy_hfa_checkpoint_reconstructs_local_field_model(tmp_path):
    model = CellLMChatModel(
        ModelConfig(n=4, d=8, r=1, k=3, vocab_size=300),
        field=CYHFAConfig(
            key_size=4,
            value_size=3,
            diffusion_radius=1,
            chunk_size=4,
        ),
    )
    path = tmp_path / "field.pt"

    save_chat_checkpoint(path, model, step=17)
    restored, metadata = load_chat_checkpoint(path)

    assert metadata["architecture"] == "celllm-chat-cy-hfa"
    assert metadata["format_version"] == 3
    assert restored.field_config.key_size == 4
    state = restored.new_plastic_state(2)
    assert state.memory.shape == (2, 4, 3, 4)
    assert state.normalizer.shape == (2, 4, 4)
