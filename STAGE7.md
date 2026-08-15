# Stage 7 — causal checkpoint ablations

Stage 7 evaluates frozen CY-HFA and state-matched bank checkpoints. It does
not train, update, or rewrite model parameters.

## Conditions

| Condition | Intervention |
| --- | --- |
| `normal` | no path changed |
| `no_retrieval` | returns a zero associative drive while preserving writes and state transport |
| `no_carry` | resets associative state at every native block boundary |
| `no_diffusion` | disables only CY-HFA propagation while preserving retrieval, writes, and carry |
| `zero_history` | resets accumulated state once at a matched native boundary and then allows rebuilding |
| `no_write` | zeros only the Delta-Hebbian write contribution after a matched native boundary |

`zero_history` and `no_write` use response-suffix probes. A response is
eligible when it crosses a native block boundary after memory has formed. Its
matched normal replay uses the same tokens, boundary, chunking, and scored
suffix. Their deltas therefore must not be compared directly with deltas over
the full validation set.

The behavioral evaluator matches normalized complete words or word sequences.
It never treats substrings such as `hi` inside `think` as correct.

## Run

Activate the project environment and provide the two best-validation
checkpoints plus the exact training data sources:

```bash
conda activate ag
python chat/run_stage7.py \
  --field experiment-chat/.dgx-results/11834/chat/best.pt \
  --bank experiment-chat/.dgx-results/11835/chat-bank/best.pt \
  --data \
    experiment-chat/data/simple-dialogues.jsonl \
    experiment-chat/data/smoltalk-everyday.jsonl \
  --output stage7-results.json \
  --device cuda \
  --behavior
```

Use `--normal-only` for the mandatory sanity gate. The runner reads the
original validation NLL from each checkpoint and refuses to interpret
ablations unless the reproduced value differs by less than `1e-4` by default.

The JSON output retains per-dialogue sufficient statistics `(loss_sum,
token_count)`. Its sibling Markdown file contains the compact tables. Paired
bootstrap resamples dialogues 10,000 times by default and reports delta NLL,
relative NLL and perplexity, standard error, percentile CI95%, and the fraction
of bootstrap deltas above zero.
