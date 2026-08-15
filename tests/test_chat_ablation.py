from __future__ import annotations

import torch
import torch.nn.functional as F
from celnn import AssociativeFieldState

from celllm.ablation import (
    AblationCondition,
    AblationConfig,
    AblationTrace,
)
from celllm.chat_ablation import (
    BoundaryProbe,
    DialogueLoss,
    EvaluationResult,
    evaluate_native_boundary,
    paired_bootstrap,
)
from celllm.chat_data import EncodedConversation
from celllm.chat_model import CellLMChatModel
from celllm.config import CYHFAConfig, ModelConfig, StateMatchedBankConfig


def field_model() -> CellLMChatModel:
    return CellLMChatModel(
        ModelConfig(n=4, d=8, r=1, k=3, vocab_size=300),
        field=CYHFAConfig(
            key_size=4,
            value_size=4,
            chunk_size=4,
            diffusion_rate=0.1,
            detach_updates=False,
        ),
    ).eval()


def bank_model() -> CellLMChatModel:
    return CellLMChatModel(
        ModelConfig(n=4, d=8, r=1, k=3, vocab_size=300),
        bank=StateMatchedBankConfig(
            slots=4,
            key_size=4,
            value_size=4,
            chunk_size=4,
            detach_updates=False,
        ),
    ).eval()


def populated_field(model: CellLMChatModel) -> AssociativeFieldState:
    empty = model.new_plastic_state(1)
    return AssociativeFieldState(
        torch.randn_like(empty.memory),
        torch.rand_like(empty.normalizer) + 0.1,
        updates=2,
    )


def test_no_retrieval_cuts_the_actual_memory_drive():
    model = field_model()
    tokens = torch.tensor([[11, 12, 13, 14]])
    state = populated_field(model)
    normal_trace = AblationTrace()
    cut_trace = AblationTrace()

    normal_logits, _ = model.forward_with_state(
        tokens, state, update_memory=False, trace=normal_trace
    )
    cut_logits, _ = model.forward_with_state(
        tokens,
        state,
        update_memory=False,
        ablation=AblationConfig(AblationCondition.NO_RETRIEVAL),
        trace=cut_trace,
    )

    assert normal_trace.maximum("retrieval_drive") > 0
    assert cut_trace.maximum("retrieval_drive") == 0
    assert not torch.allclose(normal_logits, cut_logits)


def test_field_interventions_zero_only_the_declared_terms():
    model = field_model()
    tokens = torch.tensor([[11, 12, 13, 14]])
    state = populated_field(model)
    activity = torch.randn(1, 4, 8)
    attention = model.core.cell.attention

    no_write = AblationTrace()
    attention.advance(
        state,
        activity,
        write_enabled=False,
        trace=no_write,
    )
    assert no_write.maximum("write_memory_delta") == 0
    assert no_write.maximum("write_normalizer_delta") == 0
    assert no_write.maximum("diffusion_memory_delta") > 0

    no_diffusion = AblationTrace()
    attention.advance(
        state,
        activity,
        diffusion_enabled=False,
        trace=no_diffusion,
    )
    assert no_diffusion.maximum("diffusion_memory_delta") == 0
    assert no_diffusion.maximum("diffusion_normalizer_delta") == 0
    assert no_diffusion.maximum("write_memory_delta") > 0

    no_carry = AblationTrace()
    model.forward_with_state(
        tokens,
        state,
        ablation=AblationConfig(AblationCondition.NO_CARRY),
        trace=no_carry,
    )
    assert no_carry.maximum("block_input_memory") == 0
    assert no_carry.maximum("block_input_normalizer") == 0


def test_bank_no_write_and_no_carry_have_direct_invariants():
    model = bank_model()
    state = populated_field(model)
    tokens = torch.tensor([[11, 12, 13, 14]])
    trace = AblationTrace()

    _, returned = model.forward_with_state(
        tokens,
        state,
        ablation=AblationConfig(AblationCondition.NO_WRITE),
        trace=trace,
    )

    torch.testing.assert_close(returned.memory, state.memory)
    torch.testing.assert_close(returned.normalizer, state.normalizer)
    assert trace.maximum("write_memory_delta") == 0
    assert trace.maximum("write_normalizer_delta") == 0

    with torch.no_grad():
        no_carry = AblationTrace()
        model.forward_with_state(
            tokens,
            state,
            ablation=AblationConfig(AblationCondition.NO_CARRY),
            trace=no_carry,
        )
    assert no_carry.maximum("block_input_memory") == 0


def test_native_zero_history_resets_once_and_rebuilds_memory():
    torch.manual_seed(5)
    model = field_model()
    tokens = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    assistant = torch.zeros(12, dtype=torch.bool)
    assistant[2:12] = True
    encoded = EncodedConversation(tokens, assistant, length=12)
    probe = BoundaryProbe(
        dialogue_id=0,
        encoded=encoded,
        response_start=2,
        boundary=4,
        response_end=12,
    )
    trace = AblationTrace()

    normal = evaluate_native_boundary(
        model,
        [probe],
        condition=AblationCondition.NORMAL,
        pad_id=0,
        device="cpu",
    )
    reset = evaluate_native_boundary(
        model,
        [probe],
        condition=AblationCondition.ZERO_HISTORY,
        pad_id=0,
        device="cpu",
        trace=trace,
    )

    assert normal.token_count == reset.token_count == 7
    logits = model.sequence_logits(tokens.unsqueeze(0))
    target_positions = torch.arange(5, 12)
    expected = F.cross_entropy(
        logits[0, target_positions - 1], tokens[target_positions]
    )
    torch.testing.assert_close(torch.tensor(normal.nll), expected)
    assert trace.maximum("history_before_reset_memory") > 0
    assert trace.maximum("history_after_reset_memory") == 0
    assert trace.maximum("history_after_rewrite_memory") > 0
    assert trace.terms["history_after_reset_memory"].calls == 1


def test_paired_bootstrap_preserves_dialogue_pairing_and_token_weights():
    normal = EvaluationResult(
        "normal",
        "test",
        (
            DialogueLoss(0, 2.0, 1),
            DialogueLoss(1, 18.0, 9),
        ),
        response_count=2,
    )
    intervention = EvaluationResult(
        "cut",
        "test",
        (
            DialogueLoss(0, 3.0, 1),
            DialogueLoss(1, 27.0, 9),
        ),
        response_count=2,
    )

    report = paired_bootstrap(normal, intervention, samples=500, seed=3)

    assert report["delta_nll"] == 1.0
    assert report["bootstrap_fraction_delta_positive"] == 1.0
    low, high = report["ci95_delta_nll"]
    assert low > 0
    assert high > 0
