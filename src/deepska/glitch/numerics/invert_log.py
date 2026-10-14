# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Invert but in log x and log f, for things that span decades like P(rho)."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Fn, Num
from deepska.glitch.numerics.invert import invert


def invert_log(f: Fn, target: Num, low: Num, high: Num, n: int = 30) -> Array:
    """
    Find the x where f(x) = target, working in log x and log f.

    :param f: The function to invert, has to go one way only between low and high.
    :param target: The value f has to reach.
    :param low: Lower end of the search.
    :param high: Upper end of the search.
    :param n: Number of halvings, 30 is plenty as the linear finish does the rest.
    :returns: The x where f(x) = target.
    """

    def log_f(log_x: Num) -> Array:
        return jnp.log(f(jnp.exp(log_x)))

    log_x = invert(log_f, jnp.log(target), jnp.log(low), jnp.log(high), n=n)
    return jnp.exp(log_x)
