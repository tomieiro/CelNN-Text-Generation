"""The global bank controls CY-HFA state capacity without spatial topology."""

from __future__ import annotations

import torch
from celnn import AssociativeFieldState

from celllm.attention import BankWriteAction
from celllm.chat_stage8 import (
    diagnose_bank_block,
    evaluate_bank_write_utilities,
)
from celllm.chat_model import CellLMChatModel
from celllm.config import (
    CYHFAConfig,
    ModelConfig,
    StateMatchedBankConfig,
)


def model_config():
    return ModelConfig(n=4, d=8, r=1, k=3, vocab_size=300)


def bank_config(**overrides):
    values = {
        "slots": 4,
        "key_size": 4,
        "value_size": 4,
        "chunk_size": 4,
        "detach_updates": False,
    }
    values.update(overrides)
    return StateMatchedBankConfig(**values)


def field_config(**overrides):
    values = {
        "key_size": 4,
        "value_size": 4,
        "diffusion_radius": 1,
        "chunk_size": 4,
        "detach_updates": False,
    }
    values.update(overrides)
    return CYHFAConfig(**values)


def make_bank(**overrides):
    return CellLMChatModel(model_config(), bank=bank_config(**overrides))


def test_bank_matches_cy_hfa_dynamic_state_and_parameter_count():
    bank = make_bank()
    field = CellLMChatModel(model_config(), field=field_config())
    bank_state = bank.new_plastic_state(2)
    field_state = field.new_plastic_state(2)

    assert bank_state.memory.shape == field_state.memory.shape
    assert bank_state.normalizer.shape == field_state.normalizer.shape
    assert bank_state.memory.numel() == field_state.memory.numel()
    assert bank_state.normalizer.numel() == field_state.normalizer.numel()
    assert sum(p.numel() for p in bank.parameters()) == sum(
        p.numel() for p in field.parameters()
    )


def test_bank_has_fixed_slots_and_no_cellular_propagator():
    model = make_bank()
    attention = model.core.cell.attention

    assert attention.slot_addresses.shape == (4, 4)
    assert not hasattr(attention, "propagation")
    assert "slot_addresses" not in dict(attention.named_parameters())


def test_bank_state_and_accumulation_remain_fp32_under_autocast():
    model = make_bank()
    state = model.new_plastic_state(2)
    tokens = torch.randint(6, 300, (2, 4))

    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, updated = model.forward_with_state(tokens, state)

    assert updated.memory.dtype == torch.float32
    assert updated.normalizer.dtype == torch.float32


def test_bank_does_not_write_current_block_before_its_logits():
    torch.manual_seed(8)
    model = make_bank(learning_rate=1.0, retrieval_scale=1.0).eval()
    first = torch.randint(6, 300, (1, 4))
    second = torch.randint(6, 300, (1, 4))
    empty = model.new_plastic_state(1)

    logits_with_write, written = model.forward_with_state(first, empty)
    logits_without_write, unchanged = model.forward_with_state(
        first, empty, update_plasticity=False
    )
    next_with_memory, _ = model.forward_with_state(second, written)
    next_without_memory, _ = model.forward_with_state(second, unchanged)

    torch.testing.assert_close(logits_with_write, logits_without_write)
    assert torch.count_nonzero(written.memory) > 0
    assert not torch.allclose(next_with_memory, next_without_memory)


def test_future_tokens_cannot_change_bank_prefix_logits():
    torch.manual_seed(7)
    model = make_bank().eval()
    first = torch.tensor([[11, 12, 13, 14]])
    second = torch.tensor([[11, 12, 77, 88]])

    first_logits = model(first)
    second_logits = model(second)

    torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2])


def test_chat_loss_reaches_bank_routing_and_associative_parameters():
    torch.manual_seed(4)
    model = make_bank()
    tokens = torch.randint(6, 300, (2, 8))
    assistant = torch.ones_like(tokens, dtype=torch.bool)

    loss = model.loss(tokens, assistant)
    loss.backward()

    module = model.core.cell.attention
    for parameter in (
        module.query.weight,
        module.key.weight,
        module.value.weight,
        module.output.weight,
        module.write_controls.weight,
        module.retrieval_scale,
        module.raw_read_temperature,
        module.raw_write_temperature,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_prepared_full_write_matches_public_bank_write():
    torch.manual_seed(15)
    model = make_bank().eval()
    attention = model.core.cell.attention
    state = model.new_plastic_state(2)
    activity = torch.randn(2, 4, model.cfg.d).tanh()
    mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )

    candidates = attention.prepare_write(state, activity)
    prepared = attention.apply_write(state, candidates, mask=mask)
    public = attention.write(state, activity, mask=mask)

    torch.testing.assert_close(prepared.memory, public.memory)
    torch.testing.assert_close(prepared.normalizer, public.normalizer)
    assert prepared.updates == public.updates


def test_bank_write_interventions_have_exact_state_semantics():
    torch.manual_seed(16)
    model = make_bank().eval()
    attention = model.core.cell.attention
    base = model.new_plastic_state(1)
    seeded = AssociativeFieldState(
        torch.randn_like(base.memory),
        torch.rand_like(base.normalizer),
        updates=3,
    )
    activity = torch.randn(1, 1, model.cfg.d).tanh()
    candidates = attention.prepare_write(seeded, activity)

    no_action = attention.apply_write(
        seeded,
        candidates,
        actions=torch.tensor([[int(BankWriteAction.NO_ACTION)]]),
    )
    torch.testing.assert_close(no_action.memory, seeded.memory, rtol=0, atol=0)
    torch.testing.assert_close(
        no_action.normalizer, seeded.normalizer, rtol=0, atol=0
    )

    decay_only = attention.apply_write(
        seeded,
        candidates,
        actions=torch.tensor([[int(BankWriteAction.DECAY_ONLY)]]),
    )
    assignment = candidates.routing[:, 0]
    retention = 1.0 - assignment * (
        1.0 - candidates.retentions[:, 0, None].float()
    )
    torch.testing.assert_close(
        decay_only.memory,
        retention[..., None, None] * seeded.memory,
    )
    torch.testing.assert_close(
        decay_only.normalizer,
        retention[..., None] * seeded.normalizer,
    )


def test_bank_diagnostic_path_reproduces_normal_block_and_is_fp32():
    torch.manual_seed(17)
    model = make_bank().eval()
    tokens = torch.randint(6, 300, (2, 4))
    memory = model.new_plastic_state(2)

    expected_logits, expected_memory = model.forward_with_state(tokens, memory)
    observed = diagnose_bank_block(model, tokens, memory)

    torch.testing.assert_close(observed.logits, expected_logits)
    torch.testing.assert_close(
        observed.next_memory.memory, expected_memory.memory
    )
    torch.testing.assert_close(
        observed.next_memory.normalizer, expected_memory.normalizer
    )
    assert observed.metrics.trajectory_energy.dtype == torch.float32
    assert observed.metrics.memory_trajectory_energy.dtype == torch.float32
    assert observed.candidates.associative_error.dtype == torch.float32
    assert torch.isfinite(observed.metrics.trajectory_energy).all()


def test_post_block_write_action_cannot_change_current_logits():
    torch.manual_seed(18)
    model = make_bank().eval()
    tokens = torch.randint(6, 300, (1, 4))
    memory = model.new_plastic_state(1)
    observed = diagnose_bank_block(model, tokens, memory)
    actions = torch.full(
        tokens.shape,
        int(BankWriteAction.FULL),
        dtype=torch.int64,
    )
    actions[:, 2] = int(BankWriteAction.NO_ACTION)
    counterfactual = model.core.cell.attention.apply_write(
        memory, observed.candidates, actions=actions
    )

    torch.testing.assert_close(
        observed.logits,
        model.forward_with_state(tokens, memory, update_memory=False)[0],
    )
    assert not torch.equal(counterfactual.memory, observed.next_memory.memory)


def test_write_utility_scores_only_future_assistant_targets():
    torch.manual_seed(19)
    model = make_bank().eval()
    tokens = torch.randint(6, 300, (12,))
    assistant = torch.zeros(12, dtype=torch.bool)
    assistant[5:] = True

    result = evaluate_bank_write_utilities(
        model,
        tokens,
        assistant,
        dialogue_id=9,
        horizons=(1,),
        actions=(BankWriteAction.NO_ACTION,),
        max_candidates=2,
    )

    assert result.dialogue_id == 9
    assert result.normal_token_count == 7
    assert len(result.candidates) == 2
    for candidate in result.candidates:
        utility = candidate.utilities["no_action"][1]
        assert utility is not None
        assert torch.isfinite(torch.tensor(utility))
        assert candidate.evaluated_tokens[1] > 0
