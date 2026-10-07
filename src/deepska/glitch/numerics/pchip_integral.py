# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Integral of a pchip between any two points.

Exact as its just cubics.
"""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.pchip_area import pchip_area
from deepska.glitch.numerics.pchip_interval import pchip_interval
from deepska.glitch.numerics.pchip_part import pchip_part


def pchip_integral(params: tuple[Array, ...], a: Num, b: Num) -> Array:
    """
    Integrate the spline between two points.

    Whole intervals in the middle plus the partial bits at each end.

    :param params: Spline params (x, y, h, m) from pchip_build.
    :param a: Lower limit, doesnt need to be on a knot.
    :param b: Upper limit, doesnt need to be on a knot.
    :returns: The integral from a to b.
    """
    x = params[0]
    n = x.shape[0]
    j = pchip_interval(x, a)  # interval a belongs to
    i = pchip_interval(x, b)  # interval b belongs to
    k = jnp.arange(n - 1)
    # whole intervals strictly between
    inside = (k >= j + 1) & (k < i)
    # vmap over all, mask picks j+1 .. i-1
    middle = jnp.sum(jax.vmap(pchip_area, in_axes=(None, 0))(params, k) * inside)
    # interval j minus the bit below a
    a_piece = pchip_area(params, j) - pchip_part(params, j, a)
    b_piece = pchip_part(params, i, b)  # bit of interval i above x_i
    same = j == i
    # same interval would double count
    return jnp.where(
        same,
        pchip_part(params, i, b) - pchip_part(params, j, a),
        a_piece + middle + b_piece,
    )
