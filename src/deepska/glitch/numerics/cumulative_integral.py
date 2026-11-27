# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Integral up to each point of an evenly sampled series."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num


def cumulative_integral(y: Num, dt: Num) -> Array:
    """
    Integrate evenly spaced samples up to each point.

    :param y: The samples, along the last axis.
    :param dt: Spacing of the samples.
    :returns: The integral up to each sample.
    """
    return jnp.cumsum(y * dt, axis=-1)
