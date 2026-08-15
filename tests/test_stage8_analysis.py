"""Tests for grouped incremental prediction in Stage 8.1."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_PATH = Path(__file__).parents[1] / "chat" / "analyze_stage8.py"
_SPEC = importlib.util.spec_from_file_location("stage8_analysis", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
Table = _MODULE.Table
auc = _MODULE.auc
bootstrap_r2_delta = _MODULE.bootstrap_r2_delta
group_folds = _MODULE.group_folds
metrics = _MODULE.metrics
ridge_oof = _MODULE.ridge_oof
stratified_shuffle = _MODULE.stratified_shuffle


def test_group_folds_never_split_a_dialogue():
    dialogue = np.repeat(np.arange(12), 3)
    folds = group_folds(dialogue, 5, seed=7)

    for item in np.unique(dialogue):
        assert np.unique(folds[dialogue == item]).size == 1


def test_ridge_oof_detects_an_incremental_signal():
    rng = np.random.default_rng(4)
    dialogue = np.repeat(np.arange(20), 5)
    baseline = rng.normal(size=(100, 2))
    signal = rng.normal(size=100)
    target = 2.0 * signal + rng.normal(scale=0.1, size=100)
    folds = group_folds(dialogue, 5, seed=3)

    base_prediction = ridge_oof(baseline, target, folds, alpha=1.0)
    augmented_prediction = ridge_oof(
        np.column_stack((baseline, signal)), target, folds, alpha=1.0
    )

    assert metrics(target, augmented_prediction)["r2"] > 0.98
    assert metrics(target, augmented_prediction)["r2"] > metrics(
        target, base_prediction
    )["r2"]


def test_bootstrap_is_clustered_and_reports_positive_delta():
    target = np.linspace(-1, 1, 40)
    dialogue = np.repeat(np.arange(10), 4)
    rows = tuple({} for _ in range(40))
    baseline = np.zeros_like(target)
    augmented = target + 0.01
    result = bootstrap_r2_delta(
        Table(rows, dialogue, target),
        baseline,
        augmented,
        samples=500,
        seed=8,
    )

    assert result["mean_delta_r2"] > 0
    assert result["fraction_delta_positive"] == 1.0


def test_stratified_shuffle_preserves_values_and_exchanges_dialogues():
    rows = tuple(
        {
            "block_position": index % 2,
            "target_role": "assistant",
        }
        for index in range(16)
    )
    dialogue = np.repeat(np.arange(4), 4)
    table = Table(rows, dialogue, np.zeros(16))
    matrix = np.column_stack((np.arange(16), np.arange(16) + 100.0))
    folds = np.zeros(16, dtype=np.int64)
    shuffled = stratified_shuffle(
        matrix,
        ("base", "energy"),
        ("energy",),
        table,
        folds,
        np.random.default_rng(9),
    )

    np.testing.assert_array_equal(shuffled[:, 0], matrix[:, 0])
    np.testing.assert_array_equal(
        np.sort(shuffled[:, 1]), np.sort(matrix[:, 1])
    )
    source_by_value = (shuffled[:, 1] - 100).astype(int)
    assert np.all(dialogue[source_by_value] != dialogue)


def test_shuffle_can_exchange_predictors_across_folds_without_moving_targets():
    rows = tuple(
        {"block_position": 0, "target_role": "assistant"}
        for _ in range(8)
    )
    dialogue = np.repeat(np.arange(4), 2)
    table = Table(rows, dialogue, np.arange(8.0))
    matrix = np.arange(8.0)[:, None]
    folds = np.repeat(np.arange(2), 4)
    shuffled = stratified_shuffle(
        matrix,
        ("energy",),
        ("energy",),
        table,
        folds,
        np.random.default_rng(3),
    )

    np.testing.assert_array_equal(table.target, np.arange(8.0))
    np.testing.assert_array_equal(np.sort(shuffled[:, 0]), np.arange(8.0))


def test_auc_and_ranking_metrics_have_expected_orientation():
    target = np.asarray([-2.0, -1.0, 1.0, 2.0])

    assert auc(target > 0, target) == 1.0
    assert metrics(target, target)["spearman"] == 1.0
