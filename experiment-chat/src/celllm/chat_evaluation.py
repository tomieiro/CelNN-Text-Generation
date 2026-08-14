"""Small, reproducible behavioral evaluation for conversational CellLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from celllm.chat_generation import ChatSession


@dataclass(frozen=True)
class EvaluationTurn:
    prompt: str
    expected_any: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    category: str
    turns: tuple[EvaluationTurn, ...]


SIMPLE_CHAT_CASES = (
    EvaluationCase(
        "greeting",
        "conversation",
        (EvaluationTurn("hello", ("hello", "hi")),),
    ),
    EvaluationCase(
        "identity",
        "conversation",
        (EvaluationTurn("what is your name", ("celllm",)),),
    ),
    EvaluationCase(
        "coffee",
        "everyday",
        (EvaluationTurn("do you like coffee", ("coffee",)),),
    ),
    EvaluationCase(
        "sky",
        "knowledge",
        (EvaluationTurn("what color is the sky", ("blue",)),),
    ),
    EvaluationCase(
        "arithmetic",
        "knowledge",
        (EvaluationTurn("what is two plus two", ("four",)),),
    ),
    EvaluationCase(
        "remember-name",
        "memory",
        (
            EvaluationTurn("my name is robin"),
            EvaluationTurn("what is my name", ("robin",)),
        ),
    ),
    EvaluationCase(
        "remember-color",
        "memory",
        (
            EvaluationTurn("my favorite color is purple"),
            EvaluationTurn("what is my favorite color", ("purple",)),
        ),
    ),
)


def repeated_bigram_rate(text: str) -> float:
    """Return the fraction of word bigrams that repeat within one answer."""
    words = text.lower().split()
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


def evaluate_simple_chat(
    session: ChatSession,
    cases: tuple[EvaluationCase, ...] = SIMPLE_CHAT_CASES,
) -> dict:
    """Run isolated cases and report keyword recall and degeneration."""
    results = []
    for case in cases:
        session.reset()
        turns = []
        for turn in case.turns:
            response = session.reply(turn.prompt)
            normalized = response.lower()
            passed = not turn.expected_any or any(
                keyword in normalized for keyword in turn.expected_any
            )
            turns.append(
                {
                    **asdict(turn),
                    "response": response,
                    "passed": passed,
                    "repeated_bigram_rate": repeated_bigram_rate(response),
                }
            )
        scored = [turn for turn in turns if turn["expected_any"]]
        results.append(
            {
                "name": case.name,
                "category": case.category,
                "passed": all(turn["passed"] for turn in scored),
                "turns": turns,
            }
        )

    passed = sum(case["passed"] for case in results)
    scored_responses = [
        turn
        for case in results
        for turn in case["turns"]
        if turn["expected_any"]
    ]
    category_scores = {}
    for category in sorted({case.category for case in cases}):
        matching = [item for item in results if item["category"] == category]
        category_scores[category] = sum(item["passed"] for item in matching) / len(
            matching
        )
    score = passed / len(results)
    mean_repetition = sum(
        turn["repeated_bigram_rate"] for turn in scored_responses
    ) / len(scored_responses)
    accepted = (
        passed >= 5
        and category_scores.get("memory", 0) >= 0.5
        and mean_repetition <= 0.35
    )
    return {
        "accepted": accepted,
        "acceptance_criteria": {
            "minimum_passed_cases": 5,
            "minimum_memory_score": 0.5,
            "maximum_mean_repeated_bigram_rate": 0.35,
        },
        "score": score,
        "passed": passed,
        "total": len(results),
        "category_scores": category_scores,
        "mean_repeated_bigram_rate": mean_repetition,
        "cases": results,
    }
