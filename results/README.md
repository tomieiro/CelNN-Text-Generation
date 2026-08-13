# Experiment 0 results

## Smoke run

The initial pipeline check trained rungs B, D, and H for 200 steps and three
seeds on Text8. These numbers validate execution only; they are not the
20,000-step capacity-gate result and must not be used for a gate verdict.

| Rung | BPC mean | BPC std | Core parameters | Total parameters | MACs | CPU latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B | 3.977 | 0.010 | 896 | 4,379 | 1.03 M | 144.05 ms |
| D | 3.879 | 0.013 | 2,944 | 6,427 | 5.23 M | 154.44 ms |
| H | 3.256 | 0.024 | 394,240 | 397,723 | 25.39 M | 30.97 ms |

All three models improved over the uniform baseline of 4.755 BPC. At this
short horizon, rank-8 channel mixing improved rung D over the independent-
channel rung B, while the conventional gated convolution remained ahead.
CPU latency did not track analytic operation count: the dense convolutional
kernels in H were substantially faster than the small recurrent stencil
operations. This is preliminary evidence of the kernel-overhead risk already
identified in the architecture document.

## Gate policy

Before inspecting the full results, the capacity gate is defined as passed if
at least one C2-compliant rung finishes within 0.25 BPC of rung H under the
matched 20,000-step, three-seed protocol. The knee is the cheapest passing rung
whose next rung improves mean BPC by less than 0.05.

The full ladder has now been completed. The best CelNN rung (G, 2.2802 BPC)
finished 0.4490 BPC behind the convolutional control (H, 1.8312 BPC), so the
pre-registered 0.25-BPC capacity gate did not pass. The complete protocol,
results, caveats, and next-step decision are recorded in
[`findings/etapa-1.md`](../findings/etapa-1.md).
