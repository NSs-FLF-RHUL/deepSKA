# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Many rk4 steps in a row, run by lax.scan.

lax.scan is the compiled for loop. The number of steps is fixed when it compiles so n
has to be a plain int, and the state has to keep the same shape and dtype every step. So
to stop early (eg at the surface) all n steps get run anyway and solve_ode works out
which rows are real.
"""

import jax.numpy as jnp
from jax import Array, lax

from deepska.glitch.kinds import Fn, Num, Steps
from deepska.glitch.numerics.rk4_step import rk4_step


def rk4_steps(f: Fn, x0: Num, steps: Steps, keep: Fn) -> tuple[Array, Array]:
    """
    Take many rk4 steps in a row.

    :param f: Function giving dx/dt from (x, t).
    :param x0: The state at the start.
    :param steps: Where the grid starts, the step size and the number of steps.
    :param keep: Function picking what to store from the state each step.
    :returns x_end: The state after the last step.
    :returns kept: What keep picked at every step, x0 itself is NOT included.
    """

    # scan wants (carry, x) in and (carry, output) out
    def step(x: Num, i: Num) -> tuple[Array, Array]:
        x = rk4_step(x, steps.t0 + i * steps.dt, steps.dt, f)
        return x, keep(x)  # (carry to the next step, output kept for this step)

    x_end, kept = lax.scan(step, x0, jnp.arange(steps.n))
    return x_end, kept
