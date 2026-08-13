# Running Experiment 0 on the A100 cluster

Submit the notebook together with the project files. An eight-hour allocation
is a conservative first request for the full eight-rung, three-seed ladder:

```bash
ovpn-job-submitter \
  experiment-0-a100.ipynb \
  /path/to/vpn-directory \
  --include-files \
  --partition devwork \
  --gpus 1 \
  --cpus 8 \
  --memory 32G \
  --time-limit 08:00:00
```

`vpn-directory` must contain exactly one `.ovpn` file and its referenced
certificates. Time limits accept `HH:MM:SS` or `D-HH:MM:SS`.

The notebook writes all durable artifacts below `/workspace/outputs`, which
the submitter downloads into `.dgx-results/<job-id>/`:

```text
outputs/
├── checkpoints/rung-<name>-seed-<seed>.pt
├── experiment-0.json
└── notebook.executed.ipynb
```

Each completed seed is checkpointed. If the allocation times out, copy the
downloaded `.pt` files into the local `checkpoints/` directory and submit the
same notebook again. It stages those files back into the output mount and
skips completed seeds automatically.

The current checkpoint boundary is the end of a seed. A timeout during a seed
repeats that seed on the next submission; previously completed seeds remain
safe.
