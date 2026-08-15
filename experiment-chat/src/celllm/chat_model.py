"""Assistant-targeted training on the plastic Chua--Yang CellLM core."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from celllm.ablation import (
    NORMAL_ABLATION,
    AblationConfig,
    AblationTrace,
)
from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    LocalAssociativeConfig,
    ModelConfig,
    PlasticityConfig,
    StateMatchedBankConfig,
)
from celllm.model import (
    CYHFACelNNLanguageModel,
    HebbianAttentionCelNNLanguageModel,
    PlasticCelNNLanguageModel,
    StateMatchedBankCelNNLanguageModel,
)


class CellLMChatModel(nn.Module):
    """A small chat LM with explicit memory between causal context blocks."""

    def __init__(
        self,
        model: ModelConfig,
        plasticity: PlasticityConfig | None = None,
        *,
        memory: HebbianAttentionConfig | None = None,
        bank: StateMatchedBankConfig | None = None,
        local: LocalAssociativeConfig | None = None,
        field: CYHFAConfig | None = None,
    ) -> None:
        super().__init__()
        selected = sum(
            option is not None for option in (plasticity, memory, bank, field)
        )
        if selected > 1:
            raise ValueError("choose legacy plasticity, global memory, bank, or CY-HFA")
        if local is not None and bank is None:
            raise ValueError("local associative messages require BANK")
        if plasticity is not None:
            self.core = PlasticCelNNLanguageModel(model, plasticity)
            self.memory_config = None
            self.bank_config = None
            self.local_config = None
            self.field_config = None
        elif memory is not None:
            self.core = HebbianAttentionCelNNLanguageModel(model, memory)
            self.memory_config = self.core.memory_config
            self.bank_config = None
            self.local_config = None
            self.field_config = None
        elif bank is not None:
            self.core = StateMatchedBankCelNNLanguageModel(model, bank, local=local)
            self.memory_config = None
            self.bank_config = self.core.bank_config
            self.local_config = self.core.local_config
            self.field_config = None
        else:
            self.core = CYHFACelNNLanguageModel(model, field)
            self.memory_config = None
            self.bank_config = None
            self.local_config = None
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
        return any(
            config is not None
            for config in (
                self.memory_config,
                self.bank_config,
                self.field_config,
            )
        )

    @property
    def chunk_size(self) -> int:
        config = (
            self.field_config
            or self.bank_config
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
        if self.bank_config is not None:
            return self.core.new_memory_state(batch_size)
        if self.memory_config is None:
            return self.core.new_plastic_state(batch_size)
        return self.core.new_memory_state(batch_size)

    def forward_with_state(self, *args, **kwargs):
        if self.uses_associative_state:
            if "update_plasticity" in kwargs:
                kwargs["update_memory"] = kwargs.pop("update_plasticity")
        else:
            ablation = kwargs.pop("ablation", NORMAL_ABLATION)
            kwargs.pop("trace", None)
            if ablation != NORMAL_ABLATION:
                raise ValueError("associative ablations require associative state")
        return self.core.forward_with_state(*args, **kwargs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.core(token_ids)

    def sequence_logits(
        self,
        token_ids: torch.Tensor,
        *,
        ablation: AblationConfig = NORMAL_ABLATION,
        trace: AblationTrace | None = None,
    ) -> torch.Tensor:
        """Run native chunks while preserving explicit associative state."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if not self.uses_associative_state and ablation != NORMAL_ABLATION:
            raise ValueError("associative ablations require associative state")
        memory = self.new_plastic_state(token_ids.shape[0])
        logits = []
        token_chunks = token_ids.split(self.chunk_size, dim=1)
        mask_chunks = token_ids.ne(0).split(self.chunk_size, dim=1)
        for chunk, write_mask in zip(token_chunks, mask_chunks):
            kwargs = {"write_mask": write_mask} if self.uses_associative_state else {}
            if self.uses_associative_state:
                kwargs.update({"ablation": ablation, "trace": trace})
            chunk_logits, memory = self.forward_with_state(chunk, memory, **kwargs)
            logits.append(chunk_logits)
        return torch.cat(logits, dim=1)

    def loss_statistics(
        self,
        token_ids: torch.Tensor,
        assistant_mask: torch.Tensor,
        *,
        ablation: AblationConfig = NORMAL_ABLATION,
        trace: AblationTrace | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return assistant NLL sums and token counts for each dialogue."""
        if token_ids.shape != assistant_mask.shape:
            raise ValueError("token_ids and assistant_mask must have same shape")
        predictions = self.sequence_logits(token_ids, ablation=ablation, trace=trace)[
            :, :-1
        ]
        targets = token_ids[:, 1:]
        mask = assistant_mask[:, 1:]
        if not torch.any(mask):
            raise ValueError("batch has no assistant targets")
        losses = F.cross_entropy(
            predictions.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        return (losses * mask).sum(dim=1), mask.sum(dim=1)

    def loss(
        self,
        token_ids: torch.Tensor,
        assistant_mask: torch.Tensor,
        *,
        ablation: AblationConfig = NORMAL_ABLATION,
        trace: AblationTrace | None = None,
    ) -> torch.Tensor:
        """Cross-entropy only where the next token belongs to the assistant."""
        loss_sums, token_counts = self.loss_statistics(
            token_ids,
            assistant_mask,
            ablation=ablation,
            trace=trace,
        )
        return loss_sums.sum() / token_counts.sum()
