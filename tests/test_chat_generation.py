from __future__ import annotations

import torch

from celllm.chat_generation import ChatSession, SamplingConfig, sample_token
from celllm.chat_model import CellLMChatModel
from celllm.chat_tokenizer import ChatTokenizer
from celllm.config import ModelConfig, PlasticityConfig


def tokenizer():
    return ChatTokenizer.train(
        ["hello how are you", "i am fine", "coffee is good"],
        vocab_size=300,
        min_frequency=1,
    )


def model(vocab_size):
    return CellLMChatModel(
        ModelConfig(n=8, d=4, r=1, k=7, vocab_size=vocab_size),
        PlasticityConfig(chunk_size=8),
    )


def test_greedy_sampling_respects_forbidden_and_repetition():
    logits = torch.tensor([1.0, 5.0, 4.0])
    config = SamplingConfig(
        temperature=0, repetition_penalty=2, top_k=0
    )
    assert sample_token(
        logits, config, generated=[1], forbidden={2}
    ) == 1
    assert sample_token(
        logits, config, generated=[], forbidden={1}
    ) == 2


def test_session_commits_full_blocks_and_reset_clears_memory():
    chat_tokenizer = tokenizer()
    chat_model = model(chat_tokenizer.vocab_size)
    session = ChatSession(
        chat_model,
        chat_tokenizer,
        sampling=SamplingConfig(max_new_tokens=3, temperature=0),
    )

    response = session.reply("hello how are you coffee")

    assert isinstance(response, str)
    committed = len(session.history) - len(session.pending)
    assert session.memory.updates == committed // chat_model.cfg.n
    assert 0 < len(session.pending) <= chat_model.cfg.n
    assert torch.count_nonzero(session.memory.memory) > 0
    session.reset()
    assert session.history == [chat_tokenizer.special_id("<bos>")]
    assert session.pending == session.history
    assert session.memory.updates == 0
    assert torch.count_nonzero(session.memory.memory) == 0


def test_stochastic_sampling_is_repeatable_with_a_generator():
    logits = torch.tensor([0.1, 0.2, 0.3])
    config = SamplingConfig(temperature=1, top_k=3, top_p=1)
    first = torch.Generator().manual_seed(4)
    second = torch.Generator().manual_seed(4)
    assert sample_token(
        logits, config, generated=[], forbidden=set(), generator=first
    ) == sample_token(
        logits, config, generated=[], forbidden=set(), generator=second
    )
