# A100 chat experiment

This self-contained bundle trains the first small English CellLM chat model:

- byte-level BPE vocabulary up to 1,024 tokens;
- 256-dimensional embeddings and tied readout;
- 16-token local causal Chua–Yang context;
- dense slow channel mixer plus one normalized 32×32 associative field per
  lattice cell;
- learned Q/K/V projections, write/forget gates, and causal field diffusion;
- coupled `(x, M, s)` refinement with differentiable Delta-Hebbian writes;
- assistant-only loss over conversations capped at 128 tokens;
- BF16 neural computation with FP32 associative accumulators on CUDA;
- rotating progress checkpoint every 250 steps;
- validation and fixed-prompt samples every 500 steps;
- deterministic behavioral evaluation for greeting, identity, everyday facts,
  arithmetic, and recall from transient memory.

The data mixes a deterministic 102-dialogue memory curriculum with 1,398 short
examples filtered from the official SmolTalk
[`everyday-conversations`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
subset. The
generated curriculum has a median length of 38 BPE tokens and a maximum of 58.
A 16-token local context makes roughly 95% of those conversations cross a
causal block boundary; the longer natural dialogues exercise several memory
updates. A larger local context would let the model bypass plastic memory on
most curriculum examples.

The 102 curriculum dialogues are unique on disk. After the train/validation
split, only the training portion is repeated five times to emphasize greetings,
preferences, and recall; validation never contains duplicated training rows.

Submit it with:

```bash
ovpn-job-submitter \
  experiment-chat/experiment-chat-gpu.ipynb \
  /home/tomieiro/Desktop/SSH \
  --include-files \
  --partition arandu \
  --gpus 1 \
  --cpus 12 \
  --time-limit 08:00:00 \
  --async
```

Collect with `ovpn-job-submitter --collect-results <job-id>`.

The state-matched control uses the same submission resources:

```bash
ovpn-job-submitter \
  experiment-chat/experiment-chat-bank-gpu.ipynb \
  /home/tomieiro/Desktop/SSH \
  --include-files \
  --partition arandu \
  --gpus 1 \
  --cpus 12 \
  --time-limit 08:00:00 \
  --async
```

It writes to `outputs/chat-bank`, separately from CY-HFA in `outputs/chat`.

The collected `outputs/chat/evaluation.json` reports the overall keyword
recall, per-category scores, every generated response, and a repeated-bigram
rate to expose degenerate loops. It is an explicit acceptance report, not a
replacement for reading the sample conversations. The expanded gate requires
at least six of eight cases, the full natural three-turn conversation, one of
the two memory cases, and no severe repeated-bigram loop. A passing gate means
the model reached this deliberately small milestone; it does not claim general
language-model competence.

The trainer keeps two independent model-selection artifacts: `best.pt` for
minimum validation loss and `best-chat.pt` for the strongest behavioral score
(using validation loss only as a tie-breaker). `final.pt` always contains the
last requested step. The notebook evaluates `final.pt`, because job 11832
showed that validation loss alone can favor an earlier checkpoint that is less
useful as a chat model.

By default, gradients pass through Delta-Hebbian writes so the representation
can learn to store useful key--value associations. `--detach-memory` is retained
only as an explicit credit-assignment ablation. With 16 lattice cells and
32-dimensional keys and values, the normalized `M_i/s_i` field contains 16,896
scalars per conversation (about 66 KiB in FP32), independent of conversation
length. At batch 128 the recurrent field occupies about 8.25 MiB. No token
`QK^T` matrix is formed.

For a controlled comparison, `--attention bank` selects a global bank with the
same 16,896 recurrent scalars and the same trainable parameter count. Its slots
are routed by content but have no lattice neighbourhood or diffusion.

To resume a timed-out run, copy `progress.pt` and `tokenizer.json` into
`experiment-chat/resume/` before resubmitting. The notebook stages them into
the durable output mount and continues from the saved optimizer step. Only
resume `celllm-chat-cy-hfa` checkpoints for the default `--attention field`
run. Global Delta-Hebb checkpoints require `--attention global`; the Oja
checkpoints from jobs 11832 and 11833 are intentionally rejected. State-matched
bank checkpoints require `--attention bank`.

## Stage 7 frozen-checkpoint ablations

`run_stage7.py` compares the collected `best.pt` checkpoints without training
or changing their parameters. It first reproduces each stored validation NLL
within `1e-4`; a failed reproduction stops the protocol before any ablation.

The full-validation regime evaluates normal, no retrieval, no carry, and (for
CY-HFA) no diffusion. Zero-history and no-write use matched response suffixes
that cross a real 16-token boundary, preserving the chunking seen in training.
The report keeps these regimes separate and bootstraps paired differences by
dialogue.

```bash
python experiment-chat/run_stage7.py \
  --field experiment-chat/.dgx-results/11834/chat/best.pt \
  --bank experiment-chat/.dgx-results/11835/chat-bank/best.pt \
  --data \
    experiment-chat/data/simple-dialogues.jsonl \
    experiment-chat/data/smoltalk-everyday.jsonl \
  --output stage7-results.json \
  --device cuda \
  --behavior
```

The JSON retains per-dialogue `(loss_sum, token_count)` sufficient statistics;
the adjacent Markdown file contains compact result tables. The corrected
behavioral evaluator matches complete normalized words, so `hi` no longer
matches `think`, `this`, or `while`.
