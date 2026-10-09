# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Spline for the moment of inertia of cylindrical shells."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.pchip import pchip_build


def cylinder_inertia_spline(r_hist: Num, rho_hist: Num, n_valid: Num) -> Array:
    """
    Build the spline of dI/dr per unit height of cylindrical shells.

    dI/dr is 2 pi * rho * r**3.

    :param r_hist: Radius at every tov row in cm.
    :param rho_hist: Mass density at every tov row in g/cm**3.
    :param n_valid: Number of real rows before the padding.
    :returns: The spline params (x, y, h, m).
    """
    return pchip_build(r_hist, 2.0 * jnp.pi * rho_hist * r_hist**3, n_valid)
