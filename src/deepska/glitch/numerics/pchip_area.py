# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Integral of one whole pchip interval."""

from jax import Array

from deepska.glitch.kinds import Num


def pchip_area(params: tuple[Array, ...], i: Num) -> Array:
    """
    Calculate the integral of the spline over one whole interval.

    :param params: Spline params (x, y, h, m) from pchip_build.
    :param i: Which interval, from 0 to n-2.
    :returns: The area under the spline over interval i.
    """
    _x, y, h, m = params
    return h[i] * (y[i] + y[i + 1]) / 2.0 + h[i] ** 2 * (m[i] - m[i + 1]) / 12.0
