"""Numerical equivalence with the scalar libPyCelNN reference dynamics."""

import celnn
import numpy as np
import torch

from celllm.cell import CelNNCell
from celllm.config import ModelConfig


def test_torch_cell_matches_celnn_reference_at_d1():
    rng = np.random.default_rng(0)
    n, r, dt, steps = 32, 2, 0.1, 20
    cell_input = rng.normal(size=n)
    feedback = np.array([0.2, -0.4, 1.0, -0.4, 0.2])
    control = np.array([0.1, 0.3, 0.8, 0.3, 0.1])
    bias = 0.05

    network = celnn.CellularNetwork(
        input=cell_input,
        feedback=feedback,
        control=control,
        bias=bias,
        activation="piecewise_linear",
        boundary="constant",
        dtype=np.float64,
    )
    reference = network.run(
        celnn.SimulationConfig(t_end=dt * steps, dt=dt, solver="euler")
    ).state

    cfg = ModelConfig(
        n=n,
        d=1,
        r=r,
        k=steps,
        eta=dt,
        spatial="diagonal",
        mixer="none",
        bound_drive=False,
    )
    cell = CelNNCell(cfg, causal=False).double()
    with torch.no_grad():
        cell.a.weights.copy_(torch.from_numpy(feedback).unsqueeze(1))
        cell.b.weights.copy_(torch.from_numpy(control).unsqueeze(1))
        cell.z.fill_(bias)

    embedding = torch.from_numpy(cell_input).reshape(1, n, 1)
    drive = cell.control_drive(embedding)
    state = torch.zeros(1, n, 1, dtype=torch.float64)
    for _ in range(steps):
        state = cell.step(state, drive)

    np.testing.assert_allclose(
        state.squeeze().detach().numpy(), reference, rtol=1e-9, atol=1e-9
    )
