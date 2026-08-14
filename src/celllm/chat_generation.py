"""Stateful turn-by-turn generation for the plastic CellLM chat model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from celllm.chat_model import CellLMChatModel
from celllm.chat_tokenizer import ChatTokenizer


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 80
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be at least one")


def sample_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    *,
    generated: list[int],
    forbidden: set[int],
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token with top-k/top-p and repetition control."""
    scores = logits.float().clone()
    if generated and config.repetition_penalty > 1:
        repeated = torch.tensor(
            sorted(set(generated)), device=scores.device, dtype=torch.long
        )
        values = scores[repeated]
        scores[repeated] = torch.where(
            values < 0,
            values * config.repetition_penalty,
            values / config.repetition_penalty,
        )
    if forbidden:
        indices = torch.tensor(
            sorted(forbidden), device=scores.device, dtype=torch.long
        )
        scores[indices] = -torch.inf
    if config.temperature == 0:
        return int(scores.argmax().item())

    scores = scores / config.temperature
    if 0 < config.top_k < scores.numel():
        threshold = torch.topk(scores, config.top_k).values[-1]
        scores[scores < threshold] = -torch.inf
    probabilities = torch.softmax(scores, dim=-1)
    if config.top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, descending=True
        )
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= config.top_p
        sorted_probabilities[remove] = 0
        probabilities = torch.zeros_like(probabilities).scatter(
            0, sorted_indices, sorted_probabilities
        )
        probabilities = probabilities / probabilities.sum()
    return int(
        torch.multinomial(probabilities, 1, generator=generator).item()
    )


class ChatSession:
    """Own one conversation's local token history and Hebbian memory."""

    def __init__(
        self,
        model: CellLMChatModel,
        tokenizer: ChatTokenizer,
        *,
        sampling: SamplingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.sampling = sampling or SamplingConfig()
        self.generator = generator
        self.device = next(model.parameters()).device
        self.reset()

    def reset(self) -> None:
        """Clear both visible context and transient plastic memory."""
        self.memory = self.model.new_plastic_state(1)
        self.history = [self.tokenizer.special_id("<bos>")]
        self.pending = self.history.copy()

    def _turn_ids(self, role: str, content: str) -> list[int]:
        return [
            self.tokenizer.special_id(f"<{role}>"),
            *self.tokenizer.encode(content.strip()),
            self.tokenizer.eos_id,
        ]

    @torch.inference_mode()
    def _append(self, token_ids: list[int]) -> None:
        """Append tokens, committing each completed causal block once."""
        self.history.extend(token_ids)
        self.pending.extend(token_ids)
        while len(self.pending) > self.model.cfg.n:
            chunk = torch.tensor(
                self.pending[: self.model.cfg.n],
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            _, self.memory = self.model.forward_with_state(
                chunk, self.memory, update_plasticity=True
            )
            del self.pending[: self.model.cfg.n]

    @torch.inference_mode()
    def reply(self, user_text: str) -> str:
        """Generate one answer while carrying block-aligned fast memory."""
        if not user_text.strip():
            raise ValueError("user message must not be empty")
        user_turn = self._turn_ids("user", user_text)
        self._append(user_turn)

        assistant_id = self.tokenizer.special_id("<assistant>")
        self._append([assistant_id])
        generated: list[int] = []
        forbidden = {
            self.tokenizer.pad_id,
            self.tokenizer.special_id("<unk>"),
            self.tokenizer.special_id("<bos>"),
            self.tokenizer.special_id("<user>"),
            assistant_id,
        }
        for _ in range(self.sampling.max_new_tokens):
            tokens = torch.tensor(
                self.pending, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            logits, _ = self.model.forward_with_state(
                tokens, self.memory, update_plasticity=False
            )
            token = sample_token(
                logits[0, -1],
                self.sampling,
                generated=generated,
                forbidden=forbidden,
                generator=self.generator,
            )
            if token == self.tokenizer.eos_id:
                self._append([token])
                break
            generated.append(token)
            self._append([token])
        else:
            self._append([self.tokenizer.eos_id])
        return self.tokenizer.decode(generated).strip()
