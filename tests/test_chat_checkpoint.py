from __future__ import annotations

import torch

from celllm.chat_checkpoint import load_chat_checkpoint, save_chat_checkpoint
from celllm.chat_model import CellLMChatModel
from celllm.config import ModelConfig, PlasticityConfig


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
