"""Assistant-targeted training on the plastic Chua--Yang CellLM core."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from celllm.config import ModelConfig, PlasticityConfig
from celllm.model import PlasticCelNNLanguageModel


class CellLMChatModel(nn.Module):
    """A small chat LM with explicit memory between causal context blocks."""

    def __init__(
        self,
        model: ModelConfig,
        plasticity: PlasticityConfig | None = None,
    ) -> None:
        super().__init__()
        self.core = PlasticCelNNLanguageModel(model, plasticity)

    @property
    def cfg(self) -> ModelConfig:
        return self.core.cfg

    @property
    def plasticity_config(self) -> PlasticityConfig:
        return self.core.plasticity_config

    def new_plastic_state(self, batch_size: int):
        return self.core.new_plastic_state(batch_size)

    def forward_with_state(self, *args, **kwargs):
        return self.core.forward_with_state(*args, **kwargs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.core(token_ids)

    def loss(
        self, token_ids: torch.Tensor, assistant_mask: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy only where the next token belongs to the assistant."""
        if token_ids.shape != assistant_mask.shape:
            raise ValueError("token_ids and assistant_mask must have same shape")
        memory = self.new_plastic_state(token_ids.shape[0])
        logits = []
        for chunk in token_ids.split(
            self.plasticity_config.chunk_size, dim=1
        ):
            chunk_logits, memory = self.forward_with_state(chunk, memory)
            logits.append(chunk_logits)
        predictions = torch.cat(logits, dim=1)[:, :-1]
        targets = token_ids[:, 1:]
        mask = assistant_mask[:, 1:]
        if not torch.any(mask):
            raise ValueError("batch has no assistant targets")
        losses = F.cross_entropy(
            predictions.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        return losses[mask].mean()
