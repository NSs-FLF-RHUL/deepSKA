# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Mass of the star against density.

- Mass inside a given density.
- Mass fraction outside a density, the x axis of fig 2.
"""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.pchip import pad_increasing, pchip_build, pchip_eval


def m_of_rho(rho: Num, m_hist: Num, rho_hist: Num, n_valid: Num) -> Array:
    """
    Calculate the mass enclosed at a given density.

    The history repeats rows past the surface so the padding gets fixed up first, else
    pchip gives nan.

    :param rho: Mass density in g/cm**3.
    :param m_hist: Enclosed mass at every tov row in g.
    :param rho_hist: Mass density at every tov row in g/cm**3.
    :param n_valid: Number of real rows before the padding.
    :returns: The enclosed mass in g.
    """
    # minus the density as a spline wants its x increasing
    x = pad_increasing(-rho_hist, n_valid)
    m_spline = pchip_build(x, m_hist, n_valid)
    return jax.vmap(pchip_eval, in_axes=(None, 0))(m_spline, jnp.atleast_1d(-rho))


def dm_m(
    rho: Num,
    m_star: Num,
    m_hist: Num,
    rho_hist: Num,
    n_valid: Num,
) -> Array:
    """
    Calculate the fraction of the stars mass lying outside a density.

    :param rho: Mass density in g/cm**3.
    :param m_star: Mass of the star in g.
    :param m_hist: Enclosed mass at every tov row in g.
    :param rho_hist: Mass density at every tov row in g/cm**3.
    :param n_valid: Number of real rows before the padding.
    :returns: The mass fraction dM / M.
    """
    return (m_star - m_of_rho(rho, m_hist, rho_hist, n_valid)) / m_star
