# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Which interval between knots a point falls in."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Arr, Num


def pchip_interval(x: Arr, q: Num) -> Array:
    """
    Find which interval between knots a point falls in.

    Clipped so a point past the last knot uses the last interval.

    :param x: Knot positions.
    :param q: The point.
    :returns: The i with x[i] <= q < x[i + 1].
    """
    n = x.shape[0]
    return jnp.clip(jnp.sum(x <= q) - 1, 0, n - 2)
