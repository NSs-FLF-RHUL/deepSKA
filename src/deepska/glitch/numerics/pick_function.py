# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Picking one function out of a list by number."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Fn, Num


def pick_function(fs: list[Fn], k: Num, x: Num) -> Array:
    """
    Evaluate one function out of a list, picked by number.

    vmap cant go over a list of functions, only numbers, so all of them get run.

    :param fs: The functions.
    :param k: Which one to take.
    :param x: The input passed to every function.
    :returns: The output of function number k.
    """
    return jnp.stack([f(x) for f in fs])[k]
