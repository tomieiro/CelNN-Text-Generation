"""Tests for MQAR sequence scoring and the explicit chance gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

from celllm.chat_tokenizer import ChatTokenizer
from celllm.chat_model import CellLMChatModel
from celllm.chat_generation import ChatSession, SamplingConfig
from celllm.config import ModelConfig, StateMatchedBankConfig

_PATH = Path(__file__).parents[1] / "chat" / "evaluate_mqar_gate.py"
_SPEC = importlib.util.spec_from_file_location("mqar_gate", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
answer_batch = _MODULE.answer_batch
candidate_losses = _MODULE.candidate_losses
greedy_answer = _MODULE.greedy_answer
prepare_prefix = _MODULE.prepare_prefix
summarize = _MODULE.summarize
wilson_interval = _MODULE.wilson_interval


def tokenizer() -> ChatTokenizer:
    return ChatTokenizer.train(
        ["Question: KAV? 731 284", "Answer with only the value."],
        vocab_size=384,
        min_frequency=1,
    )


def test_answer_batch_marks_only_candidate_tokens():
    item = tokenizer()
    tokens, mask = answer_batch(item, "Question: KAV?", ["731", "284"], "cpu")

    assert tokens.shape == mask.shape
    assert mask.sum(dim=1).tolist() == [
        len(item.encode("731")),
        len(item.encode("284")),
    ]
    assert not torch.any(mask[tokens == item.pad_id])


def test_wilson_and_summary_use_sequence_level_top1():
    rows = [
        {"rank": 1, "correct_nll": 1.0, "exact_match": True},
        {"rank": 2, "correct_nll": 3.0, "exact_match": False},
    ]
    report = summarize(rows)
    low, high = wilson_interval(1, 2)

    assert report["top1_accuracy"] == 0.5
    assert report["top1_wilson95"] == [low, high]
    assert report["mean_rank"] == 1.5
    assert report["correct_answer_nll"] == 2.0
    assert report["exact_match"] == 0.5


def test_shared_prefix_candidate_loss_matches_full_sequence_forward():
    torch.manual_seed(21)
    item = tokenizer()
    model = CellLMChatModel(
        ModelConfig(n=4, d=8, r=1, k=3, vocab_size=item.vocab_size),
        bank=StateMatchedBankConfig(
            slots=4,
            key_size=4,
            value_size=4,
            chunk_size=4,
            detach_updates=False,
        ),
    ).eval()
    prompt = "Question: KAV?"
    candidates = ["731", "284"]
    tokens, mask = answer_batch(item, prompt, candidates, "cpu")
    expected, expected_counts = model.loss_statistics(tokens, mask)

    observed, observed_counts = candidate_losses(
        model, item, prompt, candidates, "cpu"
    )

    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(observed_counts, expected_counts)

    prepared = prepare_prefix(model, item, prompt, "cpu")
    observed_text = greedy_answer(
        model, item, prepared[0], prepared[1], "cpu", max_new_tokens=8
    )
    session = ChatSession(
        model,
        item,
        sampling=SamplingConfig(
            max_new_tokens=8,
            temperature=0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
        ),
    )
    assert observed_text == session.reply(prompt)
