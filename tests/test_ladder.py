from celllm.config import ModelConfig
from celllm.controls import GatedConvLM
import torch

from celllm.config import TrainConfig
from celllm.ladder import (
    RUNGS,
    build_rung,
    format_table,
    load_checkpoint,
    save_checkpoint,
)
from celllm.metrics import count_parameters
from celllm.model import CelNNLanguageModel


def test_all_eight_rungs_are_defined():
    assert sorted(RUNGS) == ["A", "B", "C", "D", "E", "F", "G", "H"]


def test_rung_core_sizes_match_the_plan_table():
    base = ModelConfig(n=64, d=128, r=2, k=32, vocab_size=27)
    expected = {
        "A": 7,
        "B": 896,
        "C": 1_920,
        "D": 2_944,
        "E": 4_992,
        "F": 9_088,
        "G": 17_280,
    }
    for name, wanted in expected.items():
        actual = count_parameters(build_rung(name, base))["core"]
        assert actual == wanted, f"rung {name}: expected {wanted}, got {actual}"


def test_rung_h_is_the_convolutional_control():
    base = ModelConfig(n=64, d=128, r=2, k=32, vocab_size=27)
    assert isinstance(build_rung("H", base), GatedConvLM)
    assert isinstance(build_rung("D", base), CelNNLanguageModel)


def test_format_table_contains_rung_bpc_and_core_size():
    results = [
        {
            "rung": "B",
            "bpc_mean": 3.21,
            "bpc_std": 0.02,
            "core": 896,
            "total": 4_379,
            "flops": 123,
            "latency_ms": 1.5,
        }
    ]
    table = format_table(results)
    assert "B" in table and "3.21" in table and "896" in table


def test_checkpoint_reconstructs_trained_rung(tmp_path):
    config = ModelConfig(n=64, d=16, r=2, k=32, vocab_size=27, mixer="rank4")
    model = build_rung("C", config).eval()
    tokens = torch.randint(0, 27, (2, 64))
    expected = model(tokens)
    path = tmp_path / "rung-C-seed-42.pt"

    save_checkpoint(
        path,
        "C",
        config,
        TrainConfig(steps=1),
        42,
        model,
        {"final_bpc": 4.0, "best_bpc": 4.0, "history": [(1, 4.0)]},
    )
    restored, metadata = load_checkpoint(path)

    torch.testing.assert_close(restored(tokens), expected)
    assert metadata["rung"] == "C"
    assert metadata["seed"] == 42
