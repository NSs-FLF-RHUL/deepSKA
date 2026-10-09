# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""One rk4 step."""

from jax import Array

from deepska.glitch.kinds import Fn, Num


def rk4_step(x: Num, r: Num, dr: Num, f: Fn) -> Array:
    """
    Take one rk4 step.

    :param x: The state at r.
    :param r: Where the step starts.
    :param dr: Size of the step.
    :param f: Function giving dx/dr from (x, r).
    :returns: The state at r + dr.
    """
    k1 = dr * f(x, r)
    k2 = dr * f(x + 0.5 * k1, r + 0.5 * dr)
    k3 = dr * f(x + 0.5 * k2, r + 0.5 * dr)
    k4 = dr * f(x + k3, r + dr)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0  # middle ones count double
