# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Rk4 solver for any dx/dt = rhs(x, t) on a fixed grid.

Used for the tov star (stops at the surface where p hits p_min) and the spin equations
(no stop).
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Fn, Num, Steps
from deepska.glitch.numerics.linear_root import linear_root
from deepska.glitch.numerics.rk4_steps import rk4_steps


def solve_ode(
    rhs: Fn,
    x0: Num,
    steps: Steps,
    stop: tuple[int, float] | None = None,
    keep: Fn | None = None,
) -> tuple[Array, Array, Array, Array, Array | int]:
    """
    Solve dx/dt = rhs(x, t) forward with rk4.

    All n steps get run even with a stop so it can be vmapped, n_valid then says how
    many rows are real.

    :param rhs: Function giving dx/dt from (x, t).
    :param x0: The state at the start.
    :param steps: Where the grid starts, the step size and the number of steps.
    :param stop: Component to watch and the value it stops at, None for no stop.
    :param keep: Function picking what to store each step, None for the whole state.
    :returns t_hist: Time of every row.
    :returns x_hist: What was kept at every row, the start included.
    :returns t_end: Time at the stop, or the last time.
    :returns x_end: State at the stop, or the last state.
    :returns n_valid: Number of real rows before the padding.
    """

    # default, the whole state every step
    def keep_all(x: Num) -> Array:
        return x

    if keep is None:
        keep = keep_all
    elif stop is not None:  # cant find the crossing without the full history
        msg = "stopping at a value needs the whole history: leave keep=None"
        raise ValueError(msg)

    x_end, xs = rk4_steps(rhs, x0, steps, keep)
    t_hist = steps.t0 + steps.dt * jnp.arange(steps.n + 1)
    # start row on the front so row i is t0 + i * dt
    x_hist = jnp.concatenate([keep(x0)[None], xs])
    if stop is None:
        return t_hist, x_hist, t_hist[-1], x_end, steps.n + 1

    component, value = stop
    s = x_hist[:, component]
    # rows still on the same side as the start
    before = (s > value) == (s[0] > value)
    n_valid = jnp.sum(before)  # real rows before the crossing, rest is padding
    a = n_valid - 1
    # interpolate for exactly where it crossed
    t_end = linear_root(t_hist[a], t_hist[a + 1], s[a], s[a + 1], value)
    x_end = linear_root(x_hist[a], x_hist[a + 1], s[a], s[a + 1], value)
    return t_hist, x_hist, t_end, x_end, n_valid
