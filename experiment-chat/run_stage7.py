#!/usr/bin/env python3
"""Run frozen-checkpoint causal ablations with paired dialogue bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from celllm.ablation import AblationCondition, AblationConfig
from celllm.chat_ablation import (
    evaluate_full_validation,
    evaluate_native_boundary,
    native_boundary_probes,
    paired_bootstrap,
    validation_conversations,
)
from celllm.chat_checkpoint import load_chat_checkpoint
from celllm.chat_data import ConversationDataset
from celllm.chat_evaluation import evaluate_simple_chat
from celllm.chat_generation import ChatSession, SamplingConfig
from celllm.chat_tokenizer import ChatTokenizer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True, help="CY-HFA best.pt")
    parser.add_argument("--bank", required=True, help="state-matched bank best.pt")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", default="stage7-results.json")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--sanity-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--normal-only",
        action="store_true",
        help="stop after checkpoint-loss reproduction",
    )
    parser.add_argument(
        "--behavior",
        action="store_true",
        help="also run corrected deterministic chat behavior on Regime A",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _behavior(model, tokenizer, condition: AblationCondition) -> dict:
    session = ChatSession(
        model,
        tokenizer,
        sampling=SamplingConfig(
            max_new_tokens=40,
            temperature=0,
            top_k=0,
            top_p=1,
        ),
        ablation=AblationConfig(condition),
    )
    return evaluate_simple_chat(session)


def _evaluate_model(
    name: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> dict:
    model, checkpoint = load_chat_checkpoint(checkpoint_path, args.device)
    tokenizer = ChatTokenizer.load(checkpoint_path.parent / "tokenizer.json")
    conversations = validation_conversations(args.data, seed=args.seed)
    dataset = ConversationDataset(
        conversations,
        tokenizer,
        max_length=args.sequence_length,
    )
    normal = evaluate_full_validation(
        model,
        dataset,
        condition=AblationCondition.NORMAL,
        device=args.device,
        batch_size=args.batch_size,
    )
    expected = float(checkpoint["metrics"]["valid_loss"])
    sanity_delta = normal.nll - expected
    report = {
        "model": name,
        "checkpoint": str(checkpoint_path),
        "architecture": checkpoint["architecture"],
        "step": checkpoint["step"],
        "expected_validation_nll": expected,
        "sanity_delta": sanity_delta,
        "sanity_tolerance": args.sanity_tolerance,
        "sanity_passed": abs(sanity_delta) < args.sanity_tolerance,
        "full_validation": {
            "normal": {
                **normal.to_dict(),
                "behavioral": (
                    _behavior(model, tokenizer, AblationCondition.NORMAL)
                    if args.behavior and not args.normal_only
                    else None
                ),
            },
            "ablations": {},
        },
        "native_boundary_probes": None,
    }
    if not report["sanity_passed"] or args.normal_only:
        return report

    full_conditions = [
        AblationCondition.NO_RETRIEVAL,
        AblationCondition.NO_CARRY,
    ]
    if model.field_config is not None:
        full_conditions.append(AblationCondition.NO_DIFFUSION)
    for condition in full_conditions:
        result = evaluate_full_validation(
            model,
            dataset,
            condition=condition,
            device=args.device,
            batch_size=args.batch_size,
        )
        report["full_validation"]["ablations"][condition.value] = {
            "result": {
                **result.to_dict(),
                "behavioral": (
                    _behavior(model, tokenizer, condition)
                    if args.behavior
                    else None
                ),
            },
            "paired_bootstrap": paired_bootstrap(
                normal,
                result,
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
        }

    probes = native_boundary_probes(dataset, chunk_size=model.chunk_size)
    matched_normal = evaluate_native_boundary(
        model,
        probes,
        condition=AblationCondition.NORMAL,
        pad_id=tokenizer.pad_id,
        device=args.device,
        batch_size=args.batch_size,
    )
    probe_report = {
        "matched_normal": matched_normal.to_dict(),
        "ablations": {},
    }
    for condition in (
        AblationCondition.ZERO_HISTORY,
        AblationCondition.NO_WRITE,
    ):
        result = evaluate_native_boundary(
            model,
            probes,
            condition=condition,
            pad_id=tokenizer.pad_id,
            device=args.device,
            batch_size=args.batch_size,
        )
        probe_report["ablations"][condition.value] = {
            "result": result.to_dict(),
            "paired_bootstrap": paired_bootstrap(
                matched_normal,
                result,
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
        }
    report["native_boundary_probes"] = probe_report
    return report


def _markdown(report: dict) -> str:
    def behavior_cell(result: dict) -> str:
        behavior = result.get("behavioral")
        return "—" if behavior is None else f"{behavior['passed']}/{behavior['total']}"

    lines = [
        "# Etapa 7 — resultados de ablação causal",
        "",
        "## Reprodução do validation original",
        "",
        "| Modelo | Step | NLL original | NLL reproduzida | Δ | Sanidade |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for model in report["models"]:
        normal = model["full_validation"]["normal"]
        lines.append(
            f"| {model['model']} | {model['step']} | "
            f"{model['expected_validation_nll']:.6f} | "
            f"{normal['nll']:.6f} | {model['sanity_delta']:+.2e} | "
            f"{'✓' if model['sanity_passed'] else 'FALHOU'} |"
        )
    if report["normal_only"] or not all(
        model["sanity_passed"] for model in report["models"]
    ):
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "",
            "## Regime A — validation completo",
            "",
            "| Modelo | Condição | NLL | ΔNLL | ΔNLL % | PPL | ΔPPL % | CI95% ΔNLL | P(Δ>0) | Behavior |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for model in report["models"]:
        normal = model["full_validation"]["normal"]
        lines.append(
            f"| {model['model']} | normal | {normal['nll']:.6f} | — | — | "
            f"{normal['perplexity']:.3f} | — | — | — | {behavior_cell(normal)} |"
        )
        for condition, entry in model["full_validation"]["ablations"].items():
            result = entry["result"]
            stats = entry["paired_bootstrap"]
            low, high = stats["ci95_delta_nll"]
            lines.append(
                f"| {model['model']} | {condition} | {result['nll']:.6f} | "
                f"{stats['delta_nll']:+.6f} | {stats['delta_nll_percent']:+.3f}% | "
                f"{result['perplexity']:.3f} | {stats['delta_perplexity_percent']:+.3f}% | "
                f"[{low:+.6f}, {high:+.6f}] | "
                f"{stats['bootstrap_fraction_delta_positive']:.3f} | "
                f"{behavior_cell(result)} |"
            )

    lines.extend(
        [
            "",
            "## Regime B — probes de fronteira nativa",
            "",
            "Os deltas abaixo usam seu próprio replay normal pareado e não devem ser comparados diretamente em magnitude ao validation completo.",
            "",
            "As intervenções ocorrem em fronteiras nativas; comportamento livre não é atribuído a este regime.",
            "",
            "| Modelo | Condição | Diálogos | Respostas | Tokens | NLL | ΔNLL | ΔNLL % | PPL | CI95% ΔNLL | P(Δ>0) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for model in report["models"]:
        probe = model["native_boundary_probes"]
        normal = probe["matched_normal"]
        lines.append(
            f"| {model['model']} | matched_normal | {normal['dialogue_count']} | "
            f"{normal['response_count']} | {normal['token_count']} | "
            f"{normal['nll']:.6f} | — | — | {normal['perplexity']:.3f} | — | — |"
        )
        for condition, entry in probe["ablations"].items():
            result = entry["result"]
            stats = entry["paired_bootstrap"]
            low, high = stats["ci95_delta_nll"]
            lines.append(
                f"| {model['model']} | {condition} | {result['dialogue_count']} | "
                f"{result['response_count']} | {result['token_count']} | "
                f"{result['nll']:.6f} | {stats['delta_nll']:+.6f} | "
                f"{stats['delta_nll_percent']:+.3f}% | {result['perplexity']:.3f} | "
                f"[{low:+.6f}, {high:+.6f}] | "
                f"{stats['bootstrap_fraction_delta_positive']:.3f} |"
            )
    lines.extend(["", "### Cobertura dos probes", ""])
    for model in report["models"]:
        normal = model["native_boundary_probes"]["matched_normal"]
        distance = normal["boundary_to_end"]
        lines.append(
            f"- {model['model']}: {normal['dialogue_count']} diálogos, "
            f"{normal['response_count']} respostas e {normal['token_count']} "
            "tokens; distância fronteira→fim "
            f"min/mediana/máx = {distance['minimum']}/"
            f"{distance['median']:.0f}/{distance['maximum']} tokens."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = arguments()
    output = Path(args.output)
    report = {
        "protocol": "celllm-stage7-v1",
        "device": args.device,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "normal_only": args.normal_only,
        "behavior": args.behavior,
        "models": [
            _evaluate_model("FIELD", Path(args.field), args),
            _evaluate_model("BANK", Path(args.bank), args),
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = _markdown(report)
    output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    if not all(model["sanity_passed"] for model in report["models"]):
        raise SystemExit("normal reproduction failed; ablations were not run")


if __name__ == "__main__":
    main()
