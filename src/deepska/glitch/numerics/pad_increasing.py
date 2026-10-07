# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Fixing up padding rows so pchip can take them."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Arr, Num


def pad_increasing(x: Arr, n_valid: Num) -> Array:
    """
    Overwrite the padding rows of x so it keeps increasing.

    The padding rows repeat the last row and pchip cant take repeated x. Cant chop them
    off as jit needs every array the same length.

    :param x: Values that increase up to row n_valid.
    :param n_valid: Number of real rows before the padding.
    :returns: x with the padding rows stepping up.
    """
    i = jnp.arange(x.shape[0])
    last = x[n_valid - 1]
    # padding rows step up by |last| each row
    return jnp.where(i < n_valid, x, last + (i - n_valid + 1) * jnp.abs(last))
