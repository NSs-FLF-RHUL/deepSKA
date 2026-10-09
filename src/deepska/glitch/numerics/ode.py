# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Rk4 solver for any dx/dt = rhs(x, t) on a fixed grid.

Used for the tov star (stops at the surface where p hits p_min) and the spin equations
(no stop).

So we can keep eg just the crust at every step and the full profile at a few steps,
instead of everything always.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Fn, Num, Steps, Store
from deepska.glitch.numerics.linear_root import linear_root
from deepska.glitch.numerics.rk4_steps import rk4_steps


def solve_ode(
    rhs: Fn,
    x0: Num,
    steps: Steps,
    stop: tuple[int, float] | None = None,
    store: Store | None = None,
) -> tuple[Array, Array, Array, Array, Array | int, Array | None]:
    """
    Solve dx/dt = rhs(x, t) forward with rk4.

    All n steps get run even with a stop so it can be vmapped, n_valid then says how
    many rows are real.

    With store.snapshot_steps the whole state is also kept at those steps, so
    store.keep can store just a small piece every step and the full state only a few
    times.

    :param rhs: Function giving dx/dt from (x, t).
    :param x0: The state at the start.
    :param steps: Where the grid starts, the step size and the number of steps.
    :param stop: Component to watch and the value it stops at, None for no stop.
    :param store: What to store at each step and snapshot steps.
    :returns t_hist: Time of every row.
    :returns x_hist: What was kept at every row, the start included.
    :returns t_end: Time at the stop, or the last time.
    :returns x_end: State at the stop, or the last state.
    :returns n_valid: Number of real rows before the padding.
    :returns snaps: The whole state at each of snapshot_steps, None without them.
    """

    # default, the whole state every step
    def keep_all(x: Num) -> Array:
        return x

    if store is None:
        keep = None
        snapshot_steps = None
    else:
        keep = store.keep
        snapshot_steps = store.snapshot_steps

    if keep is None:
        keep = keep_all
    elif stop is not None:  # cant find the crossing without the full history
        msg = "stopping at a value needs the whole history: leave keep=None"
        raise ValueError(msg)
    if stop is not None and snapshot_steps is not None:
        msg = "stopping at a value cant go with snapshotting: leave snapshot_steps=None"
        raise ValueError(msg)

    wanted = [] if snapshot_steps is None else list(snapshot_steps)
    x = x0
    rows = [keep(x0)[None]]  # start row on the front so row i is t0 + i * dt
    snaps: list[Array] = []
    done = 0
    for end in [*wanted, steps.n]:  # n last for whatever is left
        if end > done:
            chunk = Steps(steps.t0 + done * steps.dt, steps.dt, end - done)
            x, kept = rk4_steps(rhs, x, chunk, keep)
            rows.append(kept)
            done = end
        if len(snaps) < len(wanted):  # dont snapshot the extra n at the end
            snaps.append(x)

    x_end = x
    x_hist = jnp.concatenate(rows)
    t_hist = steps.t0 + steps.dt * jnp.arange(steps.n + 1)
    snapshots = jnp.stack(snaps) if snaps else None

    if stop is None:
        return t_hist, x_hist, t_hist[-1], x_end, steps.n + 1, snapshots

    component, value = stop
    s = x_hist[:, component]
    # rows still on the same side as the start
    before = (s > value) == (s[0] > value)
    n_valid = jnp.sum(before)  # real rows before the crossing, rest is padding
    a = n_valid - 1
    # interpolate for exactly where it crossed
    t_end = linear_root(t_hist[a], t_hist[a + 1], s[a], s[a + 1], value)
    x_end = linear_root(x_hist[a], x_hist[a + 1], s[a], s[a + 1], value)
    return t_hist, x_hist, t_end, x_end, n_valid, None
