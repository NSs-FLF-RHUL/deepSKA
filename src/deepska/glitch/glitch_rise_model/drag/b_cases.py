# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""B for all three pinning cases at once."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.drag.b_eval import b_eval
from deepska.glitch.kinds import Num


def b_cases(p_b: tuple[Array, ...], rho: Num) -> Array:
    """
    Calculate the mutual friction coefficient B for the three pinning cases.

    :param p_b: The log-log splines of B against density, one per case.
    :param rho: The mass densities in g/cm**3 to evaluate at.
    :returns: B at each density, one row per case (A, B, C).
    """
    # one row per case
    return jnp.stack([b_eval(p_b, k, rho) for k in range(p_b[0].shape[0])])
