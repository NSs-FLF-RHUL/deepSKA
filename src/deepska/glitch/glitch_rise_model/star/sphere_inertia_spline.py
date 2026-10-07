# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Spline for the moment of inertia of spherical shells."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.pchip_build import pchip_build


def sphere_inertia_spline(r_hist: Num, rho_hist: Num, n_valid: Num) -> Array:
    """
    Build the spline of dI/dr of spherical shells.

    dI/dr is 8 pi / 3 * rho * r**4.

    :param r_hist: Radius at every tov row in cm.
    :param rho_hist: Mass density at every tov row in g/cm**3.
    :param n_valid: Number of real rows before the padding.
    :returns: The spline params (x, y, h, m).
    """
    return pchip_build(r_hist, 8.0 * jnp.pi / 3.0 * rho_hist * r_hist**4, n_valid)
