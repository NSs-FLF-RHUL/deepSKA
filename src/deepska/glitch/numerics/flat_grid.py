# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Every combination of several lists of values."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Grid


def flat_grid(values: Grid) -> tuple[dict[str, Array], tuple[int, ...]]:
    """
    List every combination of several sets of values.

    Eg 3 values of one input and 4 of another give two lists of 12 and the shape (3, 4).

    :param values: The values of each input, a 1d array per name.
    :returns draws: Each input flattened to one value per combination.
    :returns shape: Number of values of each input.
    """
    if not values:
        return {}, ()
    mesh = jnp.meshgrid(*values.values(), indexing="ij")
    return {n: m.ravel() for n, m in zip(values, mesh, strict=False)}, tuple(
        len(v) for v in values.values()
    )
