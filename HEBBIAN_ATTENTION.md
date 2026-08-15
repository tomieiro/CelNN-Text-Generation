# Delta-Hebbian attention in CellLM

The chat model combines a causal Chua--Yang cellular core with a small,
constant-size key--value memory. The components have deliberately separate
responsibilities:

```text
libPyCelNN
  DeltaHebbianRule       local corrective equation
  DeltaHebbianMemory     explicit state plus read/write operations

CellLM
  DeltaHebbianAttention  Q/K/V projections, write/forget gates, output drive
  HebbianAttentionCelNNCell
                         couples retrieval to cellular dynamics
  ChatSession            owns one conversation's tokens and fast memory
```

## Read path

During each Chua--Yang integration step, the bounded cellular output `h`
produces a query:

```text
q = Wq h
r = M q
drive = retrieval_scale * Wo r
```

The retrieved drive is added to the ordinary dense mixer drive inside the
cellular differential equation. Reading is functional: it never changes `M`.
Within one causal block, every position sees only memory written by earlier
blocks.

## Write path

After all logits for a block have been computed, its bounded outputs produce
keys, values, and two learned controls:

```text
k = Wk h
v = tanh(Wv h)
rate = max_rate * sigmoid(write_logit(h))
retention = min_retention + (1 - min_retention) * sigmoid(keep_logit(h))
```

Each token then applies the Delta-Hebbian correction in causal order:

```text
prediction = M k
error = v - prediction
M_next = retention * M + rate * error k^T
```

Unlike an additive Hebbian outer product, this rule changes an association
toward its new target rather than reinforcing it without bound. Padding has a
zero write rate and unit retention, so it cannot alter memory.

## State and training contract

- `M` belongs to the caller and is reset for every independent conversation.
- Checkpoints contain projections and gates, never transient conversation data.
- Writes remain differentiable by default, allowing later losses to teach
  earlier states what to store.
- `--detach-memory` exists only as a controlled credit-assignment ablation.
- In generation, completed blocks are written once; the incomplete block stays
  as the local cellular context.

## Size and FPGA boundary

The default key and value widths are both 32, so one session owns a `32 x 32`
matrix: 1,024 dynamic scalars, or 2 KiB in FP16. Its size does not grow with the
conversation. Q/K/V and output projections are slow weights and can later be
quantized independently from the transient memory.

This is a hybrid architecture: the spatial/recurrent computation remains a
causal Chua--Yang CelNN, while temporal retrieval is a local-rule associative
fast memory. It preserves an explicit hardware boundary but does not by itself
prove an FPGA implementation; fixed-point behavior and synthesis remain future
validation steps.
