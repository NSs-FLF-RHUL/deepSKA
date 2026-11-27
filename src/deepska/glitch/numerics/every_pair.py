# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Two nested vmaps, for running something over a grid of two inputs."""

import jax
from jax import Array

from deepska.glitch.kinds import Fn, Num, Star


def every_pair(
    run: Fn, inner: Num, outer: Num, args: tuple[object, ...] = ()
) -> Array | tuple[Array, ...]:
    """
    Call a function for every pair from two sets of inputs.

    :param run: Function called as run(i, o, *args).
    :param inner: The values of i.
    :param outer: The values of o.
    :param args: Extra inputs passed to every call.
    :returns: The outputs stacked with outer as the first axis, inner the second.
    """

    # shared inputs filled in, as vmap wants only the two swept things
    def f(i: Num, o: Star) -> Array | tuple[Array, ...]:
        return run(i, o, *args)

    over_inner = jax.vmap(f, in_axes=(0, None))
    # outer ends up as the first axis
    over_outer = jax.vmap(over_inner, in_axes=(None, 0))
    return over_outer(inner, outer)
