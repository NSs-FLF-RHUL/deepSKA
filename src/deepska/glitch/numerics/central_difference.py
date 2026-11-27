# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Gradient by central difference."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Arr, Num


def central_difference(y: Arr, dx: Num) -> Array:
    """
    Calculate the gradient of y on an evenly spaced grid.

    :param y: Values on the grid.
    :param dx: Spacing of the grid.
    :returns: The gradient, set to 0 at both ends.
    """
    der = (y[2:] - y[:-2]) / (2 * dx)
    # 0 at both ends as theres nothing either side
    return jnp.concatenate([jnp.zeros(1), der, jnp.zeros(1)])
