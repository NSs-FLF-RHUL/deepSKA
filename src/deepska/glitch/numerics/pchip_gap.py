# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The gaps between pchip knots."""

from jax import Array

from deepska.glitch.kinds import Arr, Num


def pchip_gap(i: Num, x: Arr, y: Arr) -> tuple[Array, Array]:
    """
    Calculate the width of a gap between knots and the gradient over it.

    :param i: Which gap.
    :param x: Knot positions.
    :param y: Value at each knot.
    :returns h_i: Width of the gap.
    :returns delta_i: Gradient over the entire width.
    """
    h_i = x[i + 1] - x[i]
    delta_i = (y[i + 1] - y[i]) / h_i
    return h_i, delta_i
