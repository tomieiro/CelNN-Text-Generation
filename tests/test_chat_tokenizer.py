from __future__ import annotations

import json

import pytest
import torch

from celllm.chat_data import (
    Conversation,
    ConversationDataset,
    Message,
    collate_conversations,
    encode_conversation,
    load_jsonl,
)
from celllm.chat_tokenizer import ChatTokenizer, SPECIAL_TOKENS


@pytest.fixture
def tokenizer():
    return ChatTokenizer.train(
        [
            "hello how are you",
            "i am fine thank you",
            "what is your favorite color",
            "my favorite color is blue",
        ],
        vocab_size=300,
        min_frequency=1,
    )


def test_byte_bpe_round_trip_and_stable_specials(tokenizer, tmp_path):
    text = "hello, café!"
    recovered = tokenizer.decode(tokenizer.encode(text))
    assert recovered == text
    assert [tokenizer.special_id(token) for token in SPECIAL_TOKENS] == list(
        range(len(SPECIAL_TOKENS))
    )

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = ChatTokenizer.load(path)
    assert restored.encode(text) == tokenizer.encode(text)


def test_conversation_requires_alternating_nonempty_messages():
    with pytest.raises(ValueError, match="alternate"):
        Conversation(
            (Message("user", "hello"), Message("user", "again"))
        )
    with pytest.raises(ValueError, match="empty"):
        Message("assistant", "  ")


def test_only_assistant_tokens_are_marked_as_targets(tokenizer):
    conversation = Conversation(
        (Message("user", "hello"), Message("assistant", "hi there"))
    )
    encoded = encode_conversation(conversation, tokenizer, max_length=32)
    marked = encoded.token_ids[encoded.assistant_mask].tolist()

    assert tokenizer.special_id("<assistant>") in marked
    assert tokenizer.eos_id in marked
    assert tokenizer.special_id("<user>") not in marked
    assert not torch.any(
        encoded.assistant_mask[encoded.token_ids == tokenizer.pad_id]
    )


def test_dataset_collation_and_jsonl_loading(tokenizer, tmp_path):
    record = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello there"},
        ]
    }
    path = tmp_path / "chat.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    conversations = load_jsonl(path)
    dataset = ConversationDataset(
        conversations * 2, tokenizer, max_length=24
    )

    token_ids, masks = collate_conversations([dataset[0], dataset[1]])

    assert token_ids.shape == masks.shape
    assert token_ids.shape[0] == 2
    assert token_ids.shape[1] < 24
    assert masks.dtype == torch.bool
