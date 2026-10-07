# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Inverse of a function that only goes one way, by bisection."""

from jax import Array

from deepska.glitch.kinds import Fn, Num
from deepska.glitch.numerics.bisection import bisection
from deepska.glitch.numerics.linear_root import linear_root


def invert(f: Fn, target: Num, low: Num, high: Num, n: int = 30) -> Array:
    """
    Find the x where f(x) = target.

    :param f: The function to invert, has to go one way only between low and high.
    :param target: The value f has to reach.
    :param low: Lower end of the search.
    :param high: Upper end of the search.
    :param n: Number of halvings, 30 is plenty as the linear finish does the rest.
    :returns: The x where f(x) = target.
    """
    up = f(high) > f(low)  # which way f runs, decided once
    low, high = bisection(f, target, (low, high), up, n)
    # linear interpolation across the final bracket, so the result is smooth in target
    # and gradients arent 0
    return linear_root(low, high, f(low), f(high), target)
