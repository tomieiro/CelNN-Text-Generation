# Hebbian CellLM

`PlasticCelNNLanguageModel` extends the dense-mixer CelNN architecture used by
rung G with transient fast weights from `libPyCelNN`. The original G model and
its checkpoints remain unchanged.

## Causal memory contract

Memory is explicit and belongs to the caller:

```python
from celllm import ModelConfig, PlasticCelNNLanguageModel, PlasticityConfig

model = PlasticCelNNLanguageModel(
    ModelConfig(),
    PlasticityConfig(rule="oja", chunk_size=64),
)
memory = model.new_plastic_state(batch_size=1)

logits, memory = model.forward_with_state(first_block, memory)
logits, memory = model.forward_with_state(second_block, memory)
memory = memory.reset()
```

Every block reads memory produced only by earlier blocks. The model publishes
its Hebbian/Oja update after all logits for the current block are computed, so
future positions cannot influence earlier predictions through fast weights.

`forward(tokens)` starts from reset memory for compatibility with ordinary
PyTorch APIs. `loss(tokens)` splits the sequence into causal chunks and carries
memory between them, allowing slow parameters and a learnable plasticity scale
to train against the effect of previous chunks.

The default chunk has 64 characters, matching rung G's original context. A
training sample must contain more than one chunk for the loss to exercise memory
written by an earlier block; the Stage 2 experiment must therefore use windows
longer than 64 characters. Smaller chunks are supported but change the spatial
context and must be treated as a separate ablation.

## Slow and fast state

The dense channel mixer uses:

```text
effective mixer = trained dense mixer + alpha * transient memory
```

Only slow parameters and configuration are checkpointed. Conversation memory
is intentionally excluded so loading a model cannot leak data from a previous
session. Use `save_plastic_checkpoint` and `load_plastic_checkpoint` for a
reconstructable model.

## Configuration

`PlasticityConfig` selects:

- `rule`: `hebbian` or `oja`;
- `learning_rate` and `decay`;
- `alpha`, optionally learnable;
- detached online updates or differentiable meta-learning;
- a symmetric memory limit;
- the causal training chunk size.

The first controlled comparison should hold the rung-G model dimensions fixed
and evaluate G, G+Hebb, and G+Oja under matched data, steps, and seeds.
