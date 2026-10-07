# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Pchip spline, used as it keeps the shape of the data.

No overshoot between knots like a normal cubic spline.
"""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Arr, Num
from deepska.glitch.numerics.pchip_edge import pchip_edge
from deepska.glitch.numerics.pchip_gap import pchip_gap
from deepska.glitch.numerics.pchip_interior import pchip_interior

MIN_KNOTS = 3  # two ends and at least one middle point


def pchip_build(
    x: Arr, y: Num, n_valid: Num = None
) -> tuple[Array, Array, Array, Array]:
    """
    Build a pchip spline through the points.

    Gives the params and not a function, as cant return a function and vmap it.

    :param x: Knot positions, at least 3 and always increasing.
    :param y: Value at each knot.
    :param n_valid: Number of real points if the arrays have padding on the end.
    :returns: The params (x, y, h, m), h the gap widths and m the knot slopes.
    """
    n = x.shape[0]
    if n < MIN_KNOTS:  # as needs two end points and middle points to work
        msg = "pchip needs at least 3 points"
        raise ValueError(msg)

    # needs order of array of x to always be increasing, nan if not
    ok = jnp.all(x[1:] > x[:-1])
    y = jnp.where(ok, y, jnp.nan)

    h, delta = jax.vmap(pchip_gap, in_axes=(0, None, None))(jnp.arange(n - 1), x, y)

    # n-2 interior slopes
    m_int = jax.vmap(pchip_interior, in_axes=(0, None, None))(
        jnp.arange(1, n - 1), h, delta
    )

    m_0 = pchip_edge(h[0], h[1], delta[0], delta[1])
    m_n1 = pchip_edge(h[-1], h[-2], delta[-1], delta[-2])
    m = jnp.concatenate([m_0[None], m_int, m_n1[None]])  # one array

    if n_valid is not None:
        k = n_valid - 1  # last real point
        # its really an end point, everything after it is padding
        m = m.at[k].set(pchip_edge(h[k - 1], h[k - 2], delta[k - 1], delta[k - 2]))
    # return param for actual function as cant return a function and vmap it
    return (x, y, h, m)
