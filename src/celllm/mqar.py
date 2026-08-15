"""Deterministic synthetic MQAR records aligned to native CellLM blocks."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from celllm.chat_tokenizer import ChatTokenizer


@dataclass(frozen=True)
class MQARConfig:
    seed: int = 82017
    samples_per_cell: int = 16
    loads: tuple[int, ...] = (2, 4, 8, 16, 32)
    distances: tuple[int, ...] = (16, 32, 64, 128)
    distractors: tuple[int, ...] = (0, 8, 32, 64)
    overwrite_load: int = 8
    rank_candidates: int = 32


def _key_pool() -> list[str]:
    consonants = "BCDFGHJKLMNPRSTVXZ"
    vowels = "AEIOU"
    return [a + b + c for a in consonants for b in vowels for c in consonants]


def _distance(
    tokenizer: ChatTokenizer,
    prompt: str,
    key: str,
    value: str,
) -> tuple[int, int, int]:
    _, offsets = tokenizer.encode_with_offsets(prompt)
    statement = f"{key} maps to {value}."
    statement_start = prompt.rfind(statement)
    if statement_start < 0:
        raise ValueError("target association not found in prompt")
    value_start_character = statement_start + statement.index(value)
    value_end_character = value_start_character + len(value)
    query_start_character = prompt.rfind(key)
    if query_start_character <= value_end_character:
        raise ValueError("query key not found after target write")
    value_tokens = [
        index
        for index, (start, stop) in enumerate(offsets)
        if start < value_end_character and stop > value_start_character
    ]
    query_tokens = [
        index
        for index, (start, stop) in enumerate(offsets)
        if start <= query_start_character < stop
    ]
    if not value_tokens or not query_tokens:
        raise ValueError("token offsets do not cover write/query anchors")
    value_end = value_tokens[-1]
    query_key = query_tokens[0]
    return query_key - value_end - 1, value_end, query_key


def _prompt_with_distance(
    tokenizer: ChatTokenizer,
    prefix: str,
    *,
    key: str,
    value: str,
    distance: int,
) -> tuple[str, int, int, int]:
    suffix = f" Question: What value maps to {key}? Answer with only the value."
    for filler_count in range(distance + 1):
        prompt = prefix + (" the" * filler_count) + suffix
        effective, write_end, query_start = _distance(
            tokenizer, prompt, key, value
        )
        if effective == distance:
            return prompt, effective, write_end, query_start
        if effective > distance:
            break
    raise ValueError(f"cannot realize exact token distance {distance}")


def _distractor_text(count: int) -> str:
    if count == 0:
        return ""
    words = (" the", " and", " a")
    return " Background:" + "".join(words[index % 3] for index in range(count)) + "."


def _rank_values(correct: str, values: list[str], size: int, rng: random.Random) -> list[str]:
    pool = [str(item) for item in range(100, 1000) if str(item) not in values]
    decoys = rng.sample(pool, size - 1)
    candidates = [correct, *decoys]
    rng.shuffle(candidates)
    return candidates


def _record(
    tokenizer: ChatTokenizer,
    *,
    record_id: str,
    kind: str,
    associations: list[tuple[str, str]],
    query_key: str,
    answer: str,
    old_answer: str | None,
    requested_distance: int,
    distractors: int,
    rank_candidates: int,
    rng: random.Random,
) -> dict:
    target_index = max(
        index for index, (key, _) in enumerate(associations) if key == query_key
    )
    before = associations[:target_index]
    target = associations[target_index]
    after = associations[target_index + 1 :]
    if after:
        raise ValueError("queried write must be the final association")
    statements = "Remember these pairs."
    statements += "".join(f" {key} maps to {value}." for key, value in before)
    distractor_text = _distractor_text(distractors)
    statements += distractor_text
    statements += f" {target[0]} maps to {target[1]}."
    prompt, effective, write_end, query_start = _prompt_with_distance(
        tokenizer,
        statements,
        key=query_key,
        value=answer,
        distance=requested_distance,
    )
    values = list(dict.fromkeys(value for _, value in associations))
    return {
        "id": record_id,
        "kind": kind,
        "prompt": prompt,
        "answer": answer,
        "answer_token_ids": tokenizer.encode(answer),
        "query_key": query_key,
        "old_answer": old_answer,
        "associations": [list(item) for item in associations],
        "load": len({key for key, _ in associations}),
        "requested_distance": requested_distance,
        "effective_distance": effective,
        "requested_distractors": distractors,
        "distractor_token_count": len(tokenizer.encode(distractor_text)),
        "write_end_token": write_end,
        "query_start_token": query_start,
        "prompt_token_count": len(tokenizer.encode(prompt)),
        "rank_candidates": _rank_values(
            answer, values, rank_candidates, rng
        ),
    }


def generate_mqar(
    tokenizer: ChatTokenizer, config: MQARConfig = MQARConfig()
) -> tuple[dict, ...]:
    """Generate the frozen factorial plus overwrite/preservation pairs."""
    rng = random.Random(config.seed)
    keys = _key_pool()
    records = []
    serial = 0
    for load in config.loads:
        for distance in config.distances:
            for distractors in config.distractors:
                for _ in range(config.samples_per_cell):
                    chosen_keys = rng.sample(keys, load)
                    values = [str(item) for item in rng.sample(range(100, 1000), load)]
                    associations = list(zip(chosen_keys, values))
                    target = associations.pop(rng.randrange(load))
                    rng.shuffle(associations)
                    associations.append(target)
                    records.append(
                        _record(
                            tokenizer,
                            record_id=f"standard-{serial:06d}",
                            kind="new_association",
                            associations=associations,
                            query_key=target[0],
                            answer=target[1],
                            old_answer=None,
                            requested_distance=distance,
                            distractors=distractors,
                            rank_candidates=config.rank_candidates,
                            rng=rng,
                        )
                    )
                    serial += 1

    for distance in config.distances:
        for distractors in config.distractors:
            for sample in range(config.samples_per_cell):
                chosen_keys = rng.sample(keys, config.overwrite_load)
                values = [
                    str(item)
                    for item in rng.sample(range(100, 1000), config.overwrite_load + 1)
                ]
                original = list(zip(chosen_keys, values[: config.overwrite_load]))
                overwrite_key, old_value = original[0]
                new_value = values[-1]
                others = original[1:]
                rng.shuffle(others)
                overwrite_associations = [
                    (overwrite_key, old_value),
                    *others,
                    (overwrite_key, new_value),
                ]
                pair_id = f"overwrite-{distance}-{distractors}-{sample:04d}"
                records.append(
                    _record(
                        tokenizer,
                        record_id=pair_id + "-target",
                        kind="overwrite",
                        associations=overwrite_associations,
                        query_key=overwrite_key,
                        answer=new_value,
                        old_answer=old_value,
                        requested_distance=distance,
                        distractors=distractors,
                        rank_candidates=config.rank_candidates,
                        rng=rng,
                    )
                )
                preservation_key, preservation_value = others[-1]
                preservation_associations = [
                    (overwrite_key, old_value),
                    *others[:-1],
                    (overwrite_key, new_value),
                    (preservation_key, preservation_value),
                ]
                records.append(
                    _record(
                        tokenizer,
                        record_id=pair_id + "-preservation",
                        kind="preservation",
                        associations=preservation_associations,
                        query_key=preservation_key,
                        answer=preservation_value,
                        old_answer=None,
                        requested_distance=distance,
                        distractors=distractors,
                        rank_candidates=config.rank_candidates,
                        rng=rng,
                    )
                )
    return tuple(records)


def configuration_dict(config: MQARConfig) -> dict:
    return asdict(config)
