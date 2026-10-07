# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Density at every row of the tov history."""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Fn, Num


def density_profile(p_hist: Num, p_min: Num, rho_of_p: Fn) -> Array:
    """
    Convert the pressure history of the tov solve into density.

    :param p_hist: Pressure at every tov row in erg/cm**3.
    :param p_min: Surface pressure in erg/cm**3, rows below it are padding.
    :param rho_of_p: The inverse eos, rho(P).
    :returns: The mass density at every row in g/cm**3.
    """
    # clip at p_min as the padding sits below it
    return jax.vmap(rho_of_p)(jnp.maximum(p_hist, p_min))
