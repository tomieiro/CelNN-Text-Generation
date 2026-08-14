from celllm.simple_dialogues import build_simple_dialogues


def test_curriculum_is_deterministic_and_contains_memory_turns():
    first = build_simple_dialogues(seed=4, repeats=2)
    second = build_simple_dialogues(seed=4, repeats=2)
    assert first == second
    assert len(first) > 80
    assert any(len(item.messages) == 4 for item in first)
    assert any(
        "favorite color" in message.content
        for item in first
        for message in item.messages
    )
    assert any(
        len(item.messages) == 6
        for item in first
    )
