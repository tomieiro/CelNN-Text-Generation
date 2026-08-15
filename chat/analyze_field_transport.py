#!/usr/bin/env python3
"""Plot the real CY-HFA propagate-then-write impulse cone."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from celllm.attention import CausalFieldPropagation
from celllm.field_diagnostics import furthest_reached, impulse_trajectory


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.25])
    parser.add_argument("--output", default="field-transport")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trajectories = []
    for rate in args.rates:
        propagation = CausalFieldPropagation(
            radius=1,
            rate=rate,
            max_rate=max(args.rates),
            learnable=False,
        )
        trajectory = impulse_trajectory(
            propagation, cells=args.cells, steps=args.steps
        ).cpu()
        trajectories.append((rate, trajectory))

    csv_path = destination.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("rate", "step", "cell", "magnitude"))
        for rate, trajectory in trajectories:
            for step, row in enumerate(trajectory):
                for cell, magnitude in enumerate(row):
                    writer.writerow((rate, step, cell, float(magnitude)))

    figure, axes = plt.subplots(
        1, len(trajectories), figsize=(6 * len(trajectories), 4), squeeze=False
    )
    for axis, (rate, trajectory) in zip(axes[0], trajectories):
        image = axis.imshow(
            trajectory.numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_title(
            f"D={rate:g}; furthest={furthest_reached(trajectory)}"
        )
        axis.set_xlabel("lattice cell")
        axis.set_ylabel("refinement step")
        figure.colorbar(image, ax=axis, label="field magnitude")
    figure.suptitle("CY-HFA impulse written after refinement step 1")
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=180)

    for rate, trajectory in trajectories:
        print(
            {
                "rate": rate,
                "furthest_nonzero": furthest_reached(trajectory),
                "terminal": float(trajectory[-1, -1]),
                "penultimate": float(trajectory[-1, -2]),
            }
        )


if __name__ == "__main__":
    main()
