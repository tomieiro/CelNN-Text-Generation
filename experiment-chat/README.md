# A100 chat experiment

This self-contained bundle trains the first small English CellLM chat model:

- byte-level BPE vocabulary up to 1,024 tokens;
- 256-dimensional embeddings and tied readout;
- 16-token local causal Chua–Yang context;
- dense slow channel mixer plus transient Oja memory;
- assistant-only loss over conversations capped at 128 tokens;
- BF16 training on CUDA;
- rotating progress checkpoint every 250 steps;
- validation and fixed-prompt samples every 500 steps;
- deterministic behavioral evaluation for greeting, identity, everyday facts,
  arithmetic, and recall from transient memory.

The data mixes a deterministic 44-dialogue memory curriculum with 1,398 short
examples filtered from the official SmolTalk
[`everyday-conversations`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
subset. The
generated curriculum has a median length of 24 BPE tokens and a maximum of 30.
A 16-token local context makes roughly 59% of those conversations cross a
causal block boundary; the longer natural dialogues exercise several memory
updates. A larger local context would let the model bypass plastic memory on
most curriculum examples.

The 44 curriculum dialogues are unique on disk. After the train/validation
split, only the training portion is repeated 20 times to emphasize greetings,
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

The collected `outputs/chat/evaluation.json` reports the overall keyword
recall, per-category scores, every generated response, and a repeated-bigram
rate to expose degenerate loops. It is an explicit acceptance report, not a
replacement for reading the sample conversations. The initial gate requires
at least five of seven cases, one of the two memory cases, and no severe
repeated-bigram loop. A passing gate means the model reached this deliberately
small milestone; it does not claim general language-model competence.

To resume a timed-out run, copy `progress.pt` and `tokenizer.json` into
`experiment-chat/resume/` before resubmitting. The notebook stages them into
the durable output mount and continues from the saved optimizer step.
