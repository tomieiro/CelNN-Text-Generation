from __future__ import annotations

import torch

from celllm.config import ModelConfig, PlasticityConfig
from celllm.model import PlasticCelNNLanguageModel
from celllm.plastic_checkpoint import (
    load_plastic_checkpoint,
    save_plastic_checkpoint,
)


def test_checkpoint_round_trip_excludes_transient_memory(tmp_path):
    torch.manual_seed(0)
    model = PlasticCelNNLanguageModel(
        ModelConfig(n=8, d=4, r=1, k=7, vocab_size=27),
        PlasticityConfig(rule="hebbian", chunk_size=4),
    ).eval()
    tokens = torch.randint(0, 27, (1, 8))
    expected, memory = model.forward_with_state(tokens)
    assert torch.count_nonzero(memory.memory) > 0
    path = tmp_path / "plastic.pt"

    save_plastic_checkpoint(path, model, result={"bpc": 3.0})
    restored, metadata = load_plastic_checkpoint(path)
    actual, restored_memory = restored.forward_with_state(tokens)

    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(restored_memory.memory) > 0
    assert metadata["plasticity_config"]["rule"] == "hebbian"
    assert metadata["result"] == {"bpc": 3.0}
    assert all("memory" not in key for key in metadata["model_state"])
