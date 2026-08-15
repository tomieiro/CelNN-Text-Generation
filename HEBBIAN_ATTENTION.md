# Chua--Yang Hebbian Field Attention (CY-HFA)

CY-HFA is the primary associative mechanism of the chat CellLM. Attention is
not represented by an `N x N` score matrix. It is a normalized key--value
memory field that evolves locally together with the Chua--Yang neural field.

Two controls remain available. `DeltaHebbianAttention` is the small global
memory baseline. `StateMatchedGlobalBankAttention` owns the same number of
dynamic scalars and trainable parameters as radius-one CY-HFA, but has no
spatial topology. Both write only after a complete block.

## State and equations

Every lattice cell `i` owns three dynamic values:

```text
x_i : Chua--Yang neural state
M_i : value_size x key_size associative numerator
s_i : key_size positive normalizer
```

Bounded cellular activity `y_i = f(x_i)` produces small learned projections:

```text
q_i = Wq y_i
k_i = Wk y_i
v_i = tanh(Wv y_i)
phi(z) = elu(z) + 1
```

The normalized local read is:

```text
r_i = M_i phi(q_i) / (s_i^T phi(q_i) + epsilon)
```

and `retrieval_scale * Wo r_i` is added directly to the ordinary cellular
drive. The state equation therefore has the conceptual form:

```text
dx_i/dt = ChuaYang(x, u)_i + dense_mixer(x_i) + C r_i
```

## Cellular propagation

Before each write, `M_i` and `s_i` pass through the same causal diffusion
operator:

```text
field_i' = field_i + sum_d D_d (field_(i-d) - field_i)
```

Only preceding cells `i-d` are used. Couplings are learned, non-negative, and
their sum is bounded by `max_diffusion`, so each explicit propagation step is
stable and `s_i` remains non-negative. The left edge uses a replicated
boundary.

After several refinement steps, a cell can contain propagated outer products
from earlier positions. Its query retrieves only content compatible with that
cell's key features. Relevance consequently depends on content, lattice
topology, relaxation depth, coupling strength, and fast-memory history.

For the A100 configuration, this transport is intentionally local. An impulse
written at cell 0 after the first propagation reaches at most cell 14 after 15
refinements, never terminal cell 15. Its cell-14 magnitude is `1e-14` for
`D=0.10` and `3.73e-9` for `D=0.25`. The implementation therefore does not use
the terminal cell as a whole-block summary.

## Normalized Delta-Hebbian write

Two learned gates determine the local write rate and retention:

```text
rate_i = max_rate * sigmoid(write_logit(y_i))
keep_i = min_retention + (1 - min_retention) * sigmoid(keep_logit(y_i))
```

The reusable `NormalizedDeltaHebbianField` computes the current normalized
prediction for `k_i`, then moves it by exactly `rate_i` of its error toward
`v_i`. Its outer-product correction accounts for the simultaneous update of
`s_i`; this avoids the failure mode where increasing only the denominator
makes an already-correct association worse.

If a field limit is configured, `M_i` and `s_i` are rescaled together. This
bounds both states without changing their ratio. Padding disables writes and
forgetting, although causal propagation into later padded cells is harmless.

## Coupled refinement

One model refinement step is deliberately explicit:

```text
y       = f(x)
drive   = CYHFA.read(y, field)
x_next  = ChuaYang.step(x, input, dense_drive + drive)
field'  = CYHFA.propagate(field)
field_next = CYHFA.delta_write(field', f(x_next), mask)
```

Thus the block computes `(x_next, M_next, s_next) = F(x, M, s, input)` rather
than holding memory fixed while the neural lattice settles. This is the
architectural distinction behind CY-HFA.

## Causality and session contract

- Neural and memory propagation both use only the current or preceding cell.
- Changing future tokens cannot change prefix logits; this is covered by a
  direct regression test.
- A session owns the transient field and resets it between conversations.
- Checkpoints contain slow projections, gates, and couplings, never a user's
  transient memories.
- During generation, an incomplete block evolves a speculative field for its
  logits but does not commit it. A completed block is committed once.
- At a block boundary, `M_i` and `s_i` from all completed cells are averaged
  with identical non-negative weights. This normalized past-only boundary is
  broadcast into the next lattice. Long-range carry is therefore separate
  from deliberately local intra-block diffusion.
- Writes are differentiable by default. `--detach-memory` is an ablation.

## Mixed precision

The neural state, Q/K/V projections, and other slow computation can use BF16.
The recurrent associative state always uses FP32:

```text
M, s                         FP32
phi(q), phi(k) for memory    cast to FP32
numerator and denominator    FP32 accumulation
Delta-Hebbian correction     FP32
retrieved neural drive       cast back to the activity dtype
```

The memory primitive explicitly disables autocast during reads and writes, so
this contract remains true inside the notebook's BF16 autocast region.

## Modularity

```text
libPyCelNN
  AssociativeFieldState             explicit M_i and s_i
  NormalizedDeltaHebbianField       normalized reads and local writes

CellLM
  CausalFieldPropagation            causal topology and diffusion
  InterBlockFieldCarry              topology-free normalized block boundary
  ChuaYangHebbianFieldAttention     projections, gates, propagation, read/write
  StateMatchedGlobalBankAttention   equal-state non-spatial baseline
  CYHFACelNNCell                    coupled (x, M, s) refinement
  CYHFACelNNLanguageModel           fixed lattice blocks and session state
  CellLMChatModel / ChatSession     training and conversation lifecycle
```

The library primitive knows nothing about text or lattice topology. The
propagator accepts arbitrary trailing feature axes, so the same operator is
used for numerator matrices and normalizer vectors. The global Delta-Hebb and
legacy Oja implementations remain loadable as controlled baselines.

## State-matched baseline

The global bank uses `n` normalized memory slots and fixed deterministic slot
addresses. Queries read all slots; completed-block keys route writes by
content. It has no neighbour relation or diffusion. With radius one, its two
learned routing temperatures replace the field's two diffusion parameters, so
CY-HFA and the bank have equal trainable parameter counts as well as equal
recurrent-state sizes. The small one-matrix global baseline remains available
as a capacity-unmatched reference.

## Cost and hardware boundary

For lattice width `n`, key width `r`, and value width `v`, one conversation
uses `n * (v*r + r)` dynamic scalars. With the A100 configuration
`n=16, r=v=32`, that is 16,896 scalars: about 66 KiB in FP32, or 8.25 MiB for
batch 128. The state is independent of total conversation length and no token
`QK^T` matrix is materialized.

The mechanism uses local stencil reads, outer products, elementwise gates,
positive feature maps, and divisions. This provides a clear FPGA mapping
boundary, but fixed-point error, reciprocal implementation, memory bandwidth,
and synthesis cost still require dedicated validation.
