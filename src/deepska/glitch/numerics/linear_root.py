# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Where a straight line through two points hits a value."""

from jax import Array

from deepska.glitch.kinds import Num


def linear_root(x0: Num, x1: Num, y0: Num, y1: Num, y: Num) -> Array:
    """
    Find where the straight line through two points reaches a value.

    Finishes a bisection, over the last tiny bracket the function is as good as
    straight. And it moves smoothly with y so the gradient isnt 0.

    :param x0: x of the first point.
    :param x1: x of the second point.
    :param y0: y of the first point.
    :param y1: y of the second point.
    :param y: The value to reach.
    :returns: The x where the line reaches y.
    """
    t = (y - y0) / (y1 - y0)
    return x0 + t * (x1 - x0)
