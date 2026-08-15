"""Conversation records, assistant-only targets, and padded chat batches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from celllm.chat_tokenizer import ChatTokenizer

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError(f"unsupported chat role: {self.role!r}")
        if not self.content.strip():
            raise ValueError("message content must not be empty")


@dataclass(frozen=True)
class Conversation:
    messages: tuple[Message, ...]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("conversation must contain messages")
        expected: Role = "user"
        for message in self.messages:
            if message.role != expected:
                raise ValueError("messages must alternate user and assistant")
            expected = "assistant" if expected == "user" else "user"

    @classmethod
    def from_dict(cls, record: dict) -> "Conversation":
        return cls(
            tuple(
                Message(str(item["role"]), str(item["content"]))
                for item in record["messages"]
            )
        )

    def texts(self) -> list[str]:
        return [message.content for message in self.messages]


@dataclass(frozen=True)
class EncodedConversation:
    token_ids: torch.Tensor
    assistant_mask: torch.Tensor
    length: int


def encode_conversation(
    conversation: Conversation,
    tokenizer: ChatTokenizer,
    *,
    max_length: int,
) -> EncodedConversation:
    """Encode roles and mark only assistant tokens as learning targets."""
    ids = [tokenizer.special_id("<bos>")]
    learn = [False]
    for message in conversation.messages:
        is_assistant = message.role == "assistant"
        ids.append(tokenizer.special_id(f"<{message.role}>"))
        learn.append(is_assistant)
        content = tokenizer.encode(message.content)
        ids.extend(content)
        learn.extend([is_assistant] * len(content))
        ids.append(tokenizer.eos_id)
        learn.append(is_assistant)

    ids = ids[:max_length]
    learn = learn[:max_length]
    padding = max_length - len(ids)
    length = len(ids)
    ids.extend([tokenizer.pad_id] * padding)
    learn.extend([False] * padding)
    return EncodedConversation(
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(learn, dtype=torch.bool),
        length,
    )


class ConversationDataset(Dataset):
    """Pre-encoded fixed-width conversations for deterministic training."""

    def __init__(
        self,
        conversations: list[Conversation],
        tokenizer: ChatTokenizer,
        *,
        max_length: int,
    ) -> None:
        encoded = [
            encode_conversation(item, tokenizer, max_length=max_length)
            for item in conversations
        ]
        self.examples = [
            item for item in encoded if torch.any(item.assistant_mask)
        ]
        if not self.examples:
            raise ValueError(
                "no conversation has assistant targets within max_length"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedConversation:
        return self.examples[index]


def collate_conversations(
    examples: list[EncodedConversation],
) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(item.length for item in examples)
    return (
        torch.stack([item.token_ids[:length] for item in examples]),
        torch.stack([item.assistant_mask[:length] for item in examples]),
    )


def load_jsonl(path: str | Path) -> list[Conversation]:
    """Load `{\"messages\": [{\"role\": ..., \"content\": ...}]}` rows."""
    conversations = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                conversations.append(Conversation.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid conversation at line {line_number}: {error}"
                ) from error
    return conversations
