"""Deterministic curriculum for small, everyday English conversations."""

from __future__ import annotations

import random

from celllm.chat_data import Conversation, Message

NAMES = ("alex", "sam", "jordan", "taylor", "morgan", "casey")
COLORS = ("blue", "green", "red", "yellow", "purple", "orange")
DRINKS = ("coffee", "tea", "water", "milk", "juice")
FOODS = ("pizza", "pasta", "rice", "bread", "apples", "bananas")
ANIMALS = ("cats", "dogs", "birds", "horses", "rabbits")
MOODS = ("happy", "good", "fine", "calm", "tired")


def _conversation(*turns: tuple[str, str]) -> Conversation:
    messages = []
    for user, assistant in turns:
        messages.extend((Message("user", user), Message("assistant", assistant)))
    return Conversation(tuple(messages))


def build_simple_dialogues(
    *, seed: int = 7, repeats: int = 40
) -> list[Conversation]:
    """Build greetings, preferences, facts, and short memory dialogues."""
    randomizer = random.Random(seed)
    dialogues = []
    greetings = (
        ("hello", "hello how are you"),
        ("hi", "hi it is nice to meet you"),
        ("good morning", "good morning how are you today"),
        ("how are you", "i am fine thank you"),
        ("what is your name", "my name is celllm"),
        ("who are you", "i am celllm a small conversational model"),
        ("goodbye", "goodbye have a nice day"),
    )
    for _ in range(repeats):
        for user, assistant in greetings:
            dialogues.append(_conversation((user, assistant)))

        name = randomizer.choice(NAMES)
        color = randomizer.choice(COLORS)
        drink = randomizer.choice(DRINKS)
        food = randomizer.choice(FOODS)
        animal = randomizer.choice(ANIMALS)
        mood = randomizer.choice(MOODS)
        dialogues.extend(
            (
                _conversation(
                    (f"my name is {name}", f"nice to meet you {name}"),
                    ("what is my name", f"your name is {name}"),
                ),
                _conversation(
                    (
                        f"my favorite color is {color}",
                        f"{color} is a nice color",
                    ),
                    (
                        "what is my favorite color",
                        f"your favorite color is {color}",
                    ),
                ),
                _conversation(
                    (f"i like {drink}", f"{drink} is a good drink"),
                    ("what drink do i like", f"you like {drink}"),
                ),
                _conversation(
                    (f"i like {food}", f"{food} can be delicious"),
                    ("what food do i like", f"you like {food}"),
                ),
                _conversation(
                    (f"i like {animal}", f"{animal} are interesting animals"),
                    ("what animals do i like", f"you like {animal}"),
                ),
                _conversation(
                    (f"i feel {mood}", f"thank you for telling me you feel {mood}"),
                    ("how do i feel", f"you said that you feel {mood}"),
                ),
                _conversation(
                    ("what color is the sky", "the sky is usually blue"),
                    ("and grass", "grass is usually green"),
                ),
                _conversation(
                    ("what is two plus two", "two plus two is four"),
                    ("what is three plus two", "three plus two is five"),
                ),
                _conversation(
                    ("do you like coffee", "yes i like coffee"),
                    ("why", "because coffee smells good"),
                ),
                _conversation(
                    ("tell me about cats", "cats are small domestic animals"),
                    ("can they climb", "yes cats can climb very well"),
                ),
            )
        )

    identity_prompts = (
        "tell me about yourself",
        "what kind of model are you",
        "can you introduce yourself",
    )
    for prompt in identity_prompts:
        dialogues.append(
            _conversation(
                (prompt, "i am celllm a small conversational model")
            )
        )

    for animal in ANIMALS:
        dialogues.extend(
            (
                _conversation(
                    (
                        f"what do you think about {animal}",
                        f"i think {animal} are interesting animals",
                    )
                ),
                _conversation(
                    (
                        f"are {animal} interesting",
                        f"yes {animal} are interesting animals",
                    )
                ),
                _conversation(
                    (
                        f"tell me something about {animal}",
                        f"{animal} are interesting animals",
                    )
                ),
            )
        )

    for mood in MOODS:
        for animal in ANIMALS:
            dialogues.append(
                _conversation(
                    ("hi there", "hello how are you"),
                    (
                        f"i am feeling {mood} today",
                        f"i am glad you told me you feel {mood}",
                    ),
                    (
                        f"what do you think about {animal}",
                        f"i think {animal} are interesting animals",
                    ),
                )
            )

    for name in NAMES:
        dialogues.append(
            _conversation(
                (f"please remember my name is {name}", "i will remember that"),
                ("can you tell me my name", f"your name is {name}"),
            )
        )
    for color in COLORS:
        dialogues.append(
            _conversation(
                (
                    f"please remember that i like {color}",
                    f"i will remember that you like {color}",
                ),
                ("which color do i like", f"you like {color}"),
            )
        )

    arithmetic = (
        ("can you add one and one", "one plus one is two"),
        ("what do you get when you add two and three", "two plus three is five"),
        ("please add three and three", "three plus three is six"),
    )
    dialogues.extend(_conversation(turn) for turn in arithmetic)
    randomizer.shuffle(dialogues)
    return dialogues
