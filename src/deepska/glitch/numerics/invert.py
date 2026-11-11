# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Vmappable inverse of a function that only goes one way, by bisection.

- One step of bisection.
- Full bisection with a fixed number of steps, so it can be vmapped.
- Invert a function by bisection, with a linear finish so the result is smooth in
target and gradients aren't 0.
"""

import jax.numpy as jnp
from jax import Array, lax

from deepska.glitch.kinds import Fn, Num
from deepska.glitch.numerics.linear_root import linear_root


def bis(
    carry: tuple[Array, Array], f: Fn, target: Num, up: Num
) -> tuple[tuple[Array, Array], float]:
    """
    Do one step of bisection.

    :param carry: The bracket (low, high) around the root.
    :param f: The function being inverted.
    :param target: The value f has to reach.
    :param up: True if f goes up with x.
    :returns carry: The half of the bracket that the root is in.
    :returns out: Always 0.0, as scan wants an output per step.
    """
    low, high = carry
    mid = 0.5 * (low + high)
    go_up = (f(mid) < target) == up  # the root lies above mid
    # 0.0 as scan wants an output per step
    return (jnp.where(go_up, mid, low), jnp.where(go_up, high, mid)), 0.0


def bisection(
    f: Fn, target: Num, bracket: tuple[Num, Num], up: Num, n: int
) -> tuple[Array, Array]:
    """
    Shrink a bracket around the x where f(x) = target by bisection.

    :param f: The function being inverted.
    :param target: The value f has to reach.
    :param bracket: The starting (low, high).
    :param up: True if f goes up with x.
    :param n: Number of halvings, 30 gets to 1e-9 of the start width.
    :returns low: Lower end of the final bracket.
    :returns high: Upper end of the final bracket.
    """

    # bis with this f, target and direction filled in, as scan wants (carry, _)
    def step(carry: tuple[Array, Array], _: None) -> tuple[tuple[Array, Array], float]:
        return bis(carry, f, target, up)

    (low, high), _ = lax.scan(step, bracket, None, length=n)
    return low, high


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
