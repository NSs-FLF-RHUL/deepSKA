# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Density against radius."""

from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.pchip import pchip_build


def density_spline(r_hist: Num, rho_hist: Num, n_valid: Num) -> Array:
    """
    Build the spline of density against radius.

    :param r_hist: Radius at every tov row in cm.
    :param rho_hist: Mass density at every tov row in g/cm**3.
    :param n_valid: Number of real rows before the padding.
    :returns: The spline params (x, y, h, m).
    """
    return pchip_build(r_hist, rho_hist, n_valid)
