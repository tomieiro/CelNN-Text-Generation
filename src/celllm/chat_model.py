"""Assistant-targeted training on the plastic Chua--Yang CellLM core."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    ModelConfig,
    PlasticityConfig,
)
from celllm.model import (
    CYHFACelNNLanguageModel,
    HebbianAttentionCelNNLanguageModel,
    PlasticCelNNLanguageModel,
)


class CellLMChatModel(nn.Module):
    """A small chat LM with explicit memory between causal context blocks."""

    def __init__(
        self,
        model: ModelConfig,
        plasticity: PlasticityConfig | None = None,
        *,
        memory: HebbianAttentionConfig | None = None,
        field: CYHFAConfig | None = None,
    ) -> None:
        super().__init__()
        selected = sum(
            option is not None for option in (plasticity, memory, field)
        )
        if selected > 1:
            raise ValueError(
                "choose legacy plasticity, global memory, or CY-HFA"
            )
        if plasticity is not None:
            self.core = PlasticCelNNLanguageModel(model, plasticity)
            self.memory_config = None
            self.field_config = None
        elif memory is not None:
            self.core = HebbianAttentionCelNNLanguageModel(model, memory)
            self.memory_config = self.core.memory_config
            self.field_config = None
        else:
            self.core = CYHFACelNNLanguageModel(model, field)
            self.memory_config = None
            self.field_config = self.core.field_config

    @property
    def cfg(self) -> ModelConfig:
        return self.core.cfg

    @property
    def plasticity_config(self) -> PlasticityConfig:
        if self.uses_associative_state:
            raise AttributeError(
                "model uses associative attention, not legacy plasticity"
            )
        return self.core.plasticity_config

    @property
    def uses_associative_state(self) -> bool:
        return self.memory_config is not None or self.field_config is not None

    @property
    def chunk_size(self) -> int:
        config = (
            self.field_config
            or self.memory_config
            or self.core.plasticity_config
        )
        return config.chunk_size

    @property
    def retrieval_scale(self) -> torch.Tensor | None:
        if not self.uses_associative_state:
            return None
        return self.core.cell.attention.retrieval_scale

    @property
    def diffusion_rate(self) -> torch.Tensor | None:
        if self.field_config is None:
            return None
        return self.core.cell.attention.propagation.rate

    def new_plastic_state(self, batch_size: int):
        if self.field_config is not None:
            return self.core.new_field_state(batch_size)
        if self.memory_config is None:
            return self.core.new_plastic_state(batch_size)
        return self.core.new_memory_state(batch_size)

    def forward_with_state(self, *args, **kwargs):
        if self.uses_associative_state:
            if "update_plasticity" in kwargs:
                kwargs["update_memory"] = kwargs.pop("update_plasticity")
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
        token_chunks = token_ids.split(self.chunk_size, dim=1)
        mask_chunks = token_ids.ne(0).split(self.chunk_size, dim=1)
        for chunk, write_mask in zip(token_chunks, mask_chunks):
            kwargs = (
                {"write_mask": write_mask}
                if self.uses_associative_state
                else {}
            )
            chunk_logits, memory = self.forward_with_state(
                chunk, memory, **kwargs
            )
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
