# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Slope at the pchip knots that arent on an edge."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Arr, Num


def pchip_interior(i: Num, h: Arr, delta: Arr) -> Array:
    """
    Calculate the slope at a knot that isnt on an edge.

    :param i: Which knot, from 1 to n-2.
    :param h: Width of every gap.
    :param delta: Gradient over every gap.
    :returns: The slope at knot i, 0 where the gaps either side slope opposite ways.
    """
    w1 = 2.0 * h[i] + h[i - 1]  # weights, wider gap counts for more
    w2 = h[i] + 2.0 * h[i - 1]
    return jnp.where(
        delta[i - 1] * delta[i] <= 0.0,
        0.0,
        # knots need to be same sign as whole area slope
        (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]),
    )
