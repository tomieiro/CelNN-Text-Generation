from celllm.config import ModelConfig
from celllm.controls import GatedConvLM
from celllm.ladder import RUNGS, build_rung, format_table
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
