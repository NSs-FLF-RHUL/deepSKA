# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Mass fraction outside a density, the x axis of fig 2."""

from jax import Array

from deepska.glitch.glitch_rise_model.star.m_of_rho import m_of_rho
from deepska.glitch.kinds import Num


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
