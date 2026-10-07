# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Mass inside a given density."""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.pad_increasing import pad_increasing
from deepska.glitch.numerics.pchip_build import pchip_build
from deepska.glitch.numerics.pchip_eval import pchip_eval


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
