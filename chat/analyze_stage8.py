#!/usr/bin/env python3
"""Analyze incremental predictive value of Stage-8 trajectory features."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


BASELINE = (
    "learning_rate",
    "retention",
    "associative_error",
    "displacement",
    "terminal_residual",
    "sequence_position",
    "target_role",
    "routing_entropy",
    "routing_maximum",
    "retrieval_drive_energy",
)
VARIANTS = {
    "baseline": (),
    "trajectory_energy": ("trajectory_energy",),
    "tail_energy": ("tail_energy",),
    "memory_trajectory": (
        "memory_trajectory_energy",
        "memory_final_displacement",
    ),
}


@dataclass(frozen=True)
class Table:
    rows: tuple[dict, ...]
    dialogue: np.ndarray
    target: np.ndarray


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="stage8-analysis.json")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=1_000)
    return parser.parse_args()


def load_table(report: dict, horizon: int) -> Table:
    rows = []
    targets = []
    dialogues = []
    key = str(horizon)
    for dialogue in report["dialogues"]:
        for candidate in dialogue["candidates"]:
            utility = candidate["utilities"]["no_action"].get(key)
            if utility is None:
                continue
            rows.append(candidate)
            targets.append(float(utility))
            dialogues.append(int(dialogue["dialogue_id"]))
    return Table(
        tuple(rows),
        np.asarray(dialogues, dtype=np.int64),
        np.asarray(targets, dtype=np.float64),
    )


def feature_matrix(rows: tuple[dict, ...], names: tuple[str, ...]) -> np.ndarray:
    columns = []
    for name in names:
        if name == "target_role":
            values = [item[name] == "assistant" for item in rows]
        else:
            values = [item[name] for item in rows]
        columns.append(np.asarray(values, dtype=np.float64))
    return np.column_stack(columns)


def group_folds(dialogue: np.ndarray, folds: int, seed: int) -> np.ndarray:
    unique = np.unique(dialogue)
    if folds < 2 or folds > unique.size:
        raise ValueError("folds must be between 2 and the dialogue count")
    shuffled = np.random.default_rng(seed).permutation(unique)
    assignment = {int(item): index % folds for index, item in enumerate(shuffled)}
    return np.asarray([assignment[int(item)] for item in dialogue])


def ridge_oof(
    matrix: np.ndarray,
    target: np.ndarray,
    fold_ids: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    predictions = np.empty_like(target)
    for fold in np.unique(fold_ids):
        test = fold_ids == fold
        train = ~test
        mean = matrix[train].mean(axis=0)
        scale = matrix[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        train_x = (matrix[train] - mean) / scale
        test_x = (matrix[test] - mean) / scale
        target_mean = target[train].mean()
        centered = target[train] - target_mean
        gram = train_x.T @ train_x
        coefficients = np.linalg.solve(
            gram + alpha * np.eye(gram.shape[0]), train_x.T @ centered
        )
        predictions[test] = target_mean + test_x @ coefficients
    return predictions


def auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target.astype(bool)
    positives = int(positive.sum())
    negatives = target.size - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(score, method="average")
    rank_sum = ranks[positive].sum()
    return float(
        (rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    residual = target - prediction
    total = np.square(target - target.mean()).sum()
    threshold = np.quantile(prediction, 0.75)
    selected = prediction >= threshold
    utility_threshold = np.quantile(target, 0.75)
    return {
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1.0 - np.square(residual).sum() / total),
        "spearman": float(
            np.corrcoef(rankdata(target), rankdata(prediction))[0, 1]
        ),
        "top25_mean_utility": float(target[selected].mean()),
        "top25_uplift": float(target[selected].mean() - target.mean()),
        "auc_utility_positive": auc(target > 0, prediction),
        "auc_utility_top_quartile": auc(target >= utility_threshold, prediction),
    }


def bootstrap_r2_delta(
    table: Table,
    baseline: np.ndarray,
    augmented: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict:
    unique = np.unique(table.dialogue)
    sufficient = []
    for dialogue in unique:
        mask = table.dialogue == dialogue
        target = table.target[mask]
        sufficient.append(
            (
                target.size,
                target.sum(),
                np.square(target).sum(),
                np.square(target - baseline[mask]).sum(),
                np.square(target - augmented[mask]).sum(),
            )
        )
    stats = np.asarray(sufficient, dtype=np.float64)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        unique.size,
        np.full(unique.size, 1.0 / unique.size),
        size=samples,
    )
    totals = counts @ stats
    n, sum_y, sum_y2, base_sse, augmented_sse = totals.T
    sst = sum_y2 - np.square(sum_y) / n
    delta = (1.0 - augmented_sse / sst) - (1.0 - base_sse / sst)
    return {
        "samples": samples,
        "mean_delta_r2": float(delta.mean()),
        "standard_error": float(delta.std(ddof=1)),
        "ci95": [float(item) for item in np.quantile(delta, [0.025, 0.975])],
        "fraction_delta_positive": float((delta > 0).mean()),
    }


def stratified_shuffle(
    matrix: np.ndarray,
    names: tuple[str, ...],
    added: tuple[str, ...],
    table: Table,
    _fold_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = matrix.copy()
    columns = [names.index(name) for name in added]
    primary: dict[tuple[int, str], list[int]] = {}
    for index, row in enumerate(table.rows):
        key = (
            int(row["block_position"]) // 4,
            str(row["target_role"]),
        )
        primary.setdefault(key, []).append(index)
    fallback_bins = set()
    for (position_bin, _), indices in primary.items():
        counts = np.unique(
            table.dialogue[np.asarray(indices)], return_counts=True
        )[1]
        if counts.size < 2 or int(counts.max()) * 2 > len(indices):
            fallback_bins.add(position_bin)

    strata: dict[tuple[int, str], list[int]] = {}
    for index, row in enumerate(table.rows):
        position_bin = int(row["block_position"]) // 4
        key = (
            position_bin,
            (
                "all_roles"
                if position_bin in fallback_bins
                else str(row["target_role"])
            ),
        )
        strata.setdefault(key, []).append(index)
    for indices in strata.values():
        source = np.asarray(indices)
        groups = []
        for dialogue in rng.permutation(np.unique(table.dialogue[source])):
            members = source[table.dialogue[source] == dialogue]
            groups.append(rng.permutation(members))
        recipients = np.concatenate(groups)
        largest_group = max(len(group) for group in groups)
        donors = np.roll(recipients, -largest_group)
        donor_by_recipient = dict(zip(recipients.tolist(), donors.tolist()))
        order = np.asarray([donor_by_recipient[int(item)] for item in source])
        if not np.all(table.dialogue[order] != table.dialogue[source]):
            raise RuntimeError(
                "permutation stratum cannot exchange every row across dialogues"
            )
        shuffled[np.ix_(source, columns)] = matrix[np.ix_(order, columns)]
    return shuffled


def permutation_control(
    table: Table,
    names: tuple[str, ...],
    matrix: np.ndarray,
    added: tuple[str, ...],
    fold_ids: np.ndarray,
    baseline_r2: float,
    real_delta: float,
    *,
    alpha: float,
    permutations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    deltas = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = stratified_shuffle(
            matrix, names, added, table, fold_ids, rng
        )
        prediction = ridge_oof(
            shuffled, table.target, fold_ids, alpha=alpha
        )
        deltas[index] = metrics(table.target, prediction)["r2"] - baseline_r2
    return {
        "permutations": permutations,
        "real_delta_r2": real_delta,
        "null_mean_delta_r2": float(deltas.mean()),
        "null_ci95": [
            float(item) for item in np.quantile(deltas, [0.025, 0.975])
        ],
        "fraction_null_at_least_real": float((deltas >= real_delta).mean()),
    }


def analyze_horizon(table: Table, args: argparse.Namespace) -> dict:
    fold_ids = group_folds(table.dialogue, args.folds, args.seed)
    predictions = {}
    reports = {}
    matrices = {}
    names_by_variant = {}
    for name, added in VARIANTS.items():
        names = BASELINE + added
        matrix = feature_matrix(table.rows, names)
        prediction = ridge_oof(
            matrix, table.target, fold_ids, alpha=args.alpha
        )
        predictions[name] = prediction
        matrices[name] = matrix
        names_by_variant[name] = names
        reports[name] = {
            "features": list(names),
            "metrics": metrics(table.target, prediction),
        }

    baseline_r2 = reports["baseline"]["metrics"]["r2"]
    comparisons = {}
    for offset, (name, added) in enumerate(list(VARIANTS.items())[1:], 1):
        augmented_r2 = reports[name]["metrics"]["r2"]
        delta = augmented_r2 - baseline_r2
        comparisons[name] = {
            "delta_metrics": {
                key: reports[name]["metrics"][key]
                - reports["baseline"]["metrics"][key]
                for key in reports[name]["metrics"]
            },
            "bootstrap_delta_r2": bootstrap_r2_delta(
                table,
                predictions["baseline"],
                predictions[name],
                samples=args.bootstrap_samples,
                seed=args.seed + offset,
            ),
            "permutation_delta_r2": permutation_control(
                table,
                names_by_variant[name],
                matrices[name],
                added,
                fold_ids,
                baseline_r2,
                delta,
                alpha=args.alpha,
                permutations=args.permutations,
                seed=args.seed + 100 + offset,
            ),
        }
    return {
        "candidate_count": len(table.rows),
        "dialogue_count": int(np.unique(table.dialogue).size),
        "target": {
            "mean": float(table.target.mean()),
            "standard_deviation": float(table.target.std(ddof=1)),
            "positive_fraction": float((table.target > 0).mean()),
        },
        "models": reports,
        "comparisons_to_baseline": comparisons,
    }


def main() -> None:
    args = arguments()
    source = Path(args.input)
    report = json.loads(source.read_text(encoding="utf-8"))
    results = {
        "protocol": "celllm-stage8.1-incremental-prediction-v1",
        "source": str(source.resolve()),
        "source_protocol": report["protocol"],
        "exploratory": True,
        "ridge_alpha": args.alpha,
        "folds": args.folds,
        "split_unit": "dialogue",
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "horizons": {},
    }
    for horizon in sorted(set(args.horizons)):
        print(f"analyzing horizon={horizon}", flush=True)
        results["horizons"][str(horizon)] = analyze_horizon(
            load_table(report, horizon), args
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
