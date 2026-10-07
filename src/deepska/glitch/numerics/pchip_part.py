# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Integral of part of a pchip interval."""

from jax import Array

from deepska.glitch.kinds import Num


def pchip_part(params: tuple[Array, ...], i: Num, xq: Num) -> Array:
    """
    Calculate the integral of the spline over part of one interval.

    :param params: Spline params (x, y, h, m) from pchip_build.
    :param i: Which interval.
    :param xq: Where to stop, inside interval i.
    :returns: The area under the spline from knot x[i] up to xq.
    """
    x, y, h, m = params
    t = (xq - x[i]) / h[i]
    return h[i] * (
        y[i] * (t - t**3 + t**4 / 2)
        + y[i + 1] * (t**3 - t**4 / 2)
        + h[i] * m[i] * (t**2 / 2 - 2 * t**3 / 3 + t**4 / 4)
        # hermite cubic integrated 0 to t, times the width
        + h[i] * m[i + 1] * (t**4 / 4 - t**3 / 3)
    )
