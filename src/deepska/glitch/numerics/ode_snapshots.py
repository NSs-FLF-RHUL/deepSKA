# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Ode solve that only keeps the whole state at a few chosen steps.

So we can keep eg just the crust at every step and the full profile at a few steps,
instead of everything always.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Fn, Num, Steps
from deepska.glitch.numerics.ode import solve_ode


def solve_ode_snapshots(
    rhs: Fn, x0: Num, steps: Steps, snapshot_steps: list[int], keep: Fn | None = None
) -> tuple[Array, Array]:
    """
    Solve an ode keeping the whole state at only a few chosen steps.

    :param rhs: Function giving dx/dt from (x, t).
    :param x0: The state at the start.
    :param steps: Where the grid starts, the step size and the number of steps.
    :param snapshot_steps: The steps to keep the whole state at.
    :param keep: Function picking what to store at every step.
    :returns kept: What keep picked at every step.
    :returns snaps: The whole state at each of snapshot_steps.
    """
    x = x0
    snaps: list[Array] = []
    kept: list[Array] = []
    done = 0
    for s in [*list(snapshot_steps), steps.n]:  # n last for whatever is left
        if s > done:
            chunk = Steps(steps.t0 + done * steps.dt, steps.dt, s - done)
            _, k, _, x, _ = solve_ode(rhs, x, chunk, keep=keep)
            kept.append(k[1:])  # k[0] is the chunks starting value, already counted
            done = s
        if len(snaps) < len(snapshot_steps):  # dont snapshot the extra n at the end
            snaps.append(x)
    return jnp.concatenate(kept), jnp.stack(snaps)
