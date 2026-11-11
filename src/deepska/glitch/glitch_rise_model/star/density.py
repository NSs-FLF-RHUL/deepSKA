# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Density of the star from the tov history.

- Density at every row of the tov history.
- Spline of density against radius.
"""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Fn, Num
from deepska.glitch.numerics.pchip import pchip_build


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


def density_spline(r_hist: Num, rho_hist: Num, n_valid: Num) -> Array:
    """
    Build the spline of density against radius.

    :param r_hist: Radius at every tov row in cm.
    :param rho_hist: Mass density at every tov row in g/cm**3.
    :param n_valid: Number of real rows before the padding.
    :returns: The spline params (x, y, h, m).
    """
    return pchip_build(r_hist, rho_hist, n_valid)
