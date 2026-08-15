from celllm.chat_evaluation import (
    EvaluationCase,
    EvaluationTurn,
    contains_expected,
    evaluate_simple_chat,
    normalized_words,
    repeated_bigram_rate,
)


class StubSession:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def reply(self, prompt):
        return {
            "hello": "hello friend",
            "what is my name": "your name is robin",
        }[prompt]


def test_repeated_bigram_rate_detects_degenerate_loops():
    assert repeated_bigram_rate("the state of the state of the state") > 0
    assert repeated_bigram_rate("hello friend") == 0


def test_expected_words_never_match_inside_other_words():
    for response in ("think", "this", "while", "something"):
        assert not contains_expected(response, "hi")
    assert contains_expected("Hi! How are you?", "hi")
    assert contains_expected("I am CellLM.", "celllm")


def test_expected_phrases_are_normalized_and_contiguous():
    assert normalized_words("Coffee, PLEASE!") == ("coffee", "please")
    assert contains_expected("Coffee, please!", "coffee please")
    assert not contains_expected("coffee is nice please", "coffee please")


def test_evaluation_reports_cases_categories_and_resets():
    cases = (
        EvaluationCase(
            "greeting", "conversation", (EvaluationTurn("hello", ("hello",)),)
        ),
        EvaluationCase(
            "memory",
            "memory",
            (EvaluationTurn("what is my name", ("robin",)),),
        ),
    )
    session = StubSession()
    report = evaluate_simple_chat(session, cases)

    assert not report["accepted"]  # The full gate also requires continuity.
    assert report["score"] == 1
    assert report["category_scores"] == {"conversation": 1, "memory": 1}
    assert report["mean_repeated_bigram_rate"] == 0
    assert session.reset_count == 2
