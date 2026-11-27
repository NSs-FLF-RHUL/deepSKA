# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Bin edges from bin centres."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Arr, Num


def midpoint_edges(centres: Arr, lo: Num, hi: Num) -> Array:
    """
    Make bin edges from bin centres.

    :param centres: Centre of every bin.
    :param lo: The lowest edge.
    :param hi: The highest edge.
    :returns: The edges, halfway between the centres with lo and hi at the ends.
    """
    return jnp.concatenate(
        [jnp.array([lo]), 0.5 * (centres[1:] + centres[:-1]), jnp.array([hi])]
    )
