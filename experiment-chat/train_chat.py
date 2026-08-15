#!/usr/bin/env python3
"""Train the small CellLM with CY-HFA or its global-memory baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from celllm.chat_checkpoint import load_chat_checkpoint, save_chat_checkpoint
from celllm.chat_data import (
    ConversationDataset,
    collate_conversations,
    load_jsonl,
)
from celllm.chat_generation import ChatSession, SamplingConfig
from celllm.chat_evaluation import evaluate_simple_chat
from celllm.chat_model import CellLMChatModel
from celllm.chat_tokenizer import ChatTokenizer
from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    LocalAssociativeConfig,
    ModelConfig,
    StateMatchedBankConfig,
)

SAMPLE_PROMPTS = (
    "hello",
    "what is your name",
    "do you like coffee",
    "what color is the sky",
)


def repeat_loader(loader):
    """Iterate forever while allowing DataLoader to reshuffle each epoch."""
    while True:
        yield from loader


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", nargs="+", default=["chat/data/simple-dialogues.jsonl"]
    )
    parser.add_argument("--output", default="chat/outputs")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--dynamics-steps", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=1_024)
    parser.add_argument(
        "--attention",
        choices=("field", "global", "bank"),
        default="field",
        help=("CY-HFA field, small global Delta-Hebb, or state-matched bank"),
    )
    parser.add_argument("--memory-size", type=int, default=32)
    parser.add_argument("--hebb-rate", type=float, default=0.1)
    parser.add_argument("--min-retention", type=float, default=0.99)
    parser.add_argument("--diffusion-rate", type=float, default=0.1)
    parser.add_argument("--max-diffusion", type=float, default=0.25)
    parser.add_argument("--diffusion-radius", type=int, default=1)
    parser.add_argument("--retrieval-scale", type=float, default=0.1)
    parser.add_argument(
        "--local-associative",
        choices=("none", "ungated", "gated"),
        default="none",
        help="add stateless causal local messages to the BANK baseline",
    )
    parser.add_argument("--local-radius", type=int, default=2)
    parser.add_argument("--local-retrieval-scale", type=float, default=0.1)
    parser.add_argument(
        "--detach-memory",
        action="store_true",
        help="detach Hebbian writes between blocks (credit-assignment ablation)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--curriculum-repeat",
        type=int,
        default=5,
        help="training-only weight for the first --data source",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def evaluate(model, loader, device: str, batches: int = 10) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for index, (tokens, mask) in enumerate(loader):
            if index == batches:
                break
            losses.append(model.loss(tokens.to(device), mask.to(device)).item())
    model.train()
    return sum(losses) / len(losses)


def samples(model, tokenizer, device: str) -> dict[str, str]:
    generated = {}
    for prompt in SAMPLE_PROMPTS:
        generator = torch.Generator(device=device).manual_seed(11)
        session = ChatSession(
            model,
            tokenizer,
            sampling=SamplingConfig(max_new_tokens=40),
            generator=generator,
        )
        generated[prompt] = session.reply(prompt)
    model.train()
    return generated


def chat_evaluation(model, tokenizer, device: str) -> dict:
    """Evaluate the deterministic chat milestone used for model selection."""
    session = ChatSession(
        model,
        tokenizer,
        sampling=SamplingConfig(
            max_new_tokens=40,
            temperature=0,
            top_k=0,
            top_p=1,
        ),
    )
    report = evaluate_simple_chat(session)
    model.train()
    return report


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    for data_path in args.data:
        unique = {}
        for conversation in load_jsonl(data_path):
            key = tuple(
                (message.role, message.content) for message in conversation.messages
            )
            unique[key] = conversation
        source = list(unique.values())
        random.Random(args.seed).shuffle(source)
        split = max(1, int(len(source) * 0.95))
        sources.append((source[:split], source[split:] or source[-1:]))
    train_conversations = [
        item
        for index, (train_source, _) in enumerate(sources)
        for item in train_source * (args.curriculum_repeat if index == 0 else 1)
    ]
    valid_conversations = [item for _, valid_source in sources for item in valid_source]
    tokenizer_conversations = [
        item
        for train_source, valid_source in sources
        for item in (*train_source, *valid_source)
    ]
    random.Random(args.seed).shuffle(train_conversations)
    tokenizer_path = output / "tokenizer.json"

    if args.resume:
        tokenizer = ChatTokenizer.load(tokenizer_path)
        model, checkpoint = load_chat_checkpoint(args.resume, args.device)
        if not model.uses_associative_state:
            raise ValueError(
                "cannot resume a legacy Oja checkpoint into associative attention"
            )
        resumed_attention = (
            "field"
            if model.field_config is not None
            else "bank"
            if model.bank_config is not None
            else "global"
        )
        if args.attention != resumed_attention:
            raise ValueError(
                f"resume checkpoint uses {resumed_attention} attention, "
                f"not {args.attention}"
            )
        resumed_local = (
            "none"
            if model.local_config is None
            else "gated"
            if model.local_config.gated
            else "ungated"
        )
        if args.local_associative != resumed_local:
            raise ValueError(
                f"resume checkpoint uses {resumed_local} local messages, "
                f"not {args.local_associative}"
            )
        start_step = int(checkpoint["step"])
    else:
        if args.local_associative != "none" and args.attention != "bank":
            raise ValueError("local associative messages require --attention bank")
        tokenizer = ChatTokenizer.train(
            (
                text
                for conversation in tokenizer_conversations
                for text in conversation.texts()
            ),
            vocab_size=args.vocab_size,
        )
        tokenizer.save(tokenizer_path)
        model_config = ModelConfig(
            n=args.context,
            d=args.dimensions,
            r=args.radius,
            k=args.dynamics_steps,
            vocab_size=tokenizer.vocab_size,
        )
        if args.attention == "field":
            field = CYHFAConfig(
                key_size=args.memory_size,
                value_size=args.memory_size,
                learning_rate=args.hebb_rate,
                min_retention=args.min_retention,
                diffusion_rate=args.diffusion_rate,
                max_diffusion=args.max_diffusion,
                diffusion_radius=args.diffusion_radius,
                retrieval_scale=args.retrieval_scale,
                detach_updates=args.detach_memory,
                chunk_size=args.context,
            )
            model = CellLMChatModel(model_config, field=field)
        elif args.attention == "global":
            memory = HebbianAttentionConfig(
                key_size=args.memory_size,
                value_size=args.memory_size,
                learning_rate=args.hebb_rate,
                min_retention=args.min_retention,
                retrieval_scale=args.retrieval_scale,
                detach_updates=args.detach_memory,
                chunk_size=args.context,
            )
            model = CellLMChatModel(model_config, memory=memory)
        else:
            bank = StateMatchedBankConfig(
                slots=args.context,
                key_size=args.memory_size,
                value_size=args.memory_size,
                learning_rate=args.hebb_rate,
                min_retention=args.min_retention,
                retrieval_scale=args.retrieval_scale,
                detach_updates=args.detach_memory,
                chunk_size=args.context,
            )
            local = (
                None
                if args.local_associative == "none"
                else LocalAssociativeConfig(
                    radius=args.local_radius,
                    key_size=args.memory_size,
                    value_size=args.memory_size,
                    gated=args.local_associative == "gated",
                    retrieval_scale=args.local_retrieval_scale,
                )
            )
            model = CellLMChatModel(model_config, bank=bank, local=local)
        model.to(args.device)
        checkpoint = {}
        start_step = 0

    train_data = ConversationDataset(
        train_conversations,
        tokenizer,
        max_length=args.sequence_length,
    )
    valid_data = ConversationDataset(
        valid_conversations,
        tokenizer,
        max_length=args.sequence_length,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_conversations,
        pin_memory=args.device.startswith("cuda"),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        collate_fn=collate_conversations,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    model.train()
    iterator = repeat_loader(train_loader)
    best_validation = float(checkpoint.get("metrics", {}).get("valid_loss", "inf"))
    best_chat_rank = tuple(
        checkpoint.get("metrics", {}).get(
            "chat_rank", (-1, float("-inf"), float("-inf"))
        )
    )
    history = []
    for step in range(start_step, args.steps):
        tokens, mask = next(iterator)
        tokens = tokens.to(args.device, non_blocking=True)
        mask = mask.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.device.startswith("cuda"),
        ):
            loss = model.loss(tokens, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        current = step + 1
        if current % 25 == 0:
            print(f"step={current} train_loss={loss.item():.4f}", flush=True)
        if current % args.eval_every == 0 or current == args.steps:
            validation = evaluate(model, valid_loader, args.device)
            generated = samples(model, tokenizer, args.device)
            behavioral = chat_evaluation(model, tokenizer, args.device)
            chat_rank = (
                behavioral["passed"],
                -behavioral["mean_repeated_bigram_rate"],
                -validation,
            )
            metrics = {
                "step": current,
                "attention": (
                    "cy-hfa"
                    if model.field_config is not None
                    else "state-matched-bank"
                    if model.bank_config is not None
                    else "global"
                ),
                "train_loss": loss.item(),
                "valid_loss": validation,
                "retrieval_scale": float(model.retrieval_scale.detach()),
                "diffusion_rate": (
                    float(model.diffusion_rate.detach())
                    if model.diffusion_rate is not None
                    else None
                ),
                "samples": generated,
                "behavioral": behavioral,
                "chat_rank": chat_rank,
            }
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
            if validation < best_validation:
                best_validation = validation
                save_chat_checkpoint(
                    output / "best.pt",
                    model,
                    step=current,
                    metrics=metrics,
                )
            if chat_rank > best_chat_rank:
                best_chat_rank = chat_rank
                save_chat_checkpoint(
                    output / "best-chat.pt",
                    model,
                    step=current,
                    metrics=metrics,
                )
        if current % args.checkpoint_every == 0:
            save_chat_checkpoint(
                output / "progress.pt",
                model,
                step=current,
                metrics={
                    "valid_loss": best_validation,
                    "chat_rank": best_chat_rank,
                },
                optimizer_state=optimizer.state_dict(),
            )

    save_chat_checkpoint(
        output / "final.pt",
        model,
        step=args.steps,
        metrics={"valid_loss": best_validation, "history": history},
    )
    (output / "metrics.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
