"""Tests for deterministic, token-aligned MQAR generation."""

from __future__ import annotations

from celllm.chat_tokenizer import ChatTokenizer
from celllm.mqar import MQARConfig, generate_mqar


def tokenizer() -> ChatTokenizer:
    texts = [
        "Remember these pairs. KAV maps to 731. Background: the and a.",
        "Question: What value maps to KAV? Answer with only the value.",
        "MUP maps to 284. ZEL maps to 915." + " the" * 64,
    ]
    return ChatTokenizer.train(texts, vocab_size=512, min_frequency=1)


def test_offsets_cover_the_original_text():
    item = tokenizer()
    text = "KAV maps to 731."
    ids, offsets = item.encode_with_offsets(text)

    assert ids == item.encode(text)
    assert offsets[0][0] == 0
    assert offsets[-1][1] == len(text)


def test_mqar_generation_is_deterministic_and_distance_exact():
    item = tokenizer()
    config = MQARConfig(
        seed=12,
        samples_per_cell=1,
        loads=(2,),
        distances=(16, 32),
        distractors=(0, 8),
        overwrite_load=2,
        rank_candidates=8,
    )

    first = generate_mqar(item, config)
    second = generate_mqar(item, config)

    assert first == second
    assert len(first) == 12
    assert {record["kind"] for record in first} == {
        "new_association",
        "overwrite",
        "preservation",
    }
    for record in first:
        assert record["effective_distance"] == record["requested_distance"]
        assert len(record["rank_candidates"]) == 8
        assert len(set(record["rank_candidates"])) == 8
        assert record["answer"] in record["rank_candidates"]
        assert record["prompt_token_count"] > record["query_start_token"]


def test_association_keys_and_values_are_unique_except_overwrite():
    item = tokenizer()
    records = generate_mqar(
        item,
        MQARConfig(
            seed=13,
            samples_per_cell=1,
            loads=(4,),
            distances=(16,),
            distractors=(0,),
            overwrite_load=4,
            rank_candidates=8,
        ),
    )

    for record in records:
        pairs = record["associations"]
        values = [value for _, value in pairs]
        assert len(values) == len(set(values))
        keys = [key for key, _ in pairs]
        duplicates = len(keys) - len(set(keys))
        assert duplicates == (
            1 if record["kind"] in {"overwrite", "preservation"} else 0
        )
