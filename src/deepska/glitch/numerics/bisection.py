# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Bisection with a fixed number of steps, so it can be vmapped."""

from jax import Array, lax

from deepska.glitch.kinds import Fn, Num
from deepska.glitch.numerics.bis import bis


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
