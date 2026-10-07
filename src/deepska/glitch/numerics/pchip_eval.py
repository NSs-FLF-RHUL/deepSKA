# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The spline itself, from the params pchip_build gives."""

from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.pchip_interval import pchip_interval


def pchip_eval(params: tuple[Array, ...], xq: Num) -> Array:
    """
    Evaluate the spline at one point.

    :param params: Spline params (x, y, h, m) from pchip_build.
    :param xq: Where to evaluate, one value, vmap it for more.
    :returns: The value of the spline at xq.
    """
    x, y, h, m = params
    i = pchip_interval(x, xq)  # find what range xq falls in
    t = (xq - x[i]) / h[i]
    return (
        y[i] * (1 - 3 * t**2 + 2 * t**3)
        + y[i + 1] * (3 * t**2 - 2 * t**3)
        # hermite cubic, values and slopes at both ends
        + h[i] * m[i] * (t - 2 * t**2 + t**3)
        + h[i] * m[i + 1] * (t**3 - t**2)
    )
