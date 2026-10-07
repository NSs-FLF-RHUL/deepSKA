# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Negele & Vautherin 1973 inner crust.

P = n_b e^S dS/dx in MeV with x = ln(n_b / 1e35).
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import m_u, mev
from deepska.glitch.kinds import Num


def p_inner(rho: Num, rho_d: Num, rho_cc: Num) -> Array:
    """
    Calculate the pressure of the inner crust, Negele & Vautherin 1973.

    :param rho: Mass density in g/cm**3, clamped to between rho_d and rho_cc.
    :param rho_d: Neutron drip density in g/cm**3.
    :param rho_cc: Crust-core transition density in g/cm**3.
    :returns: The pressure in erg/cm**3.
    """
    # clamped to the fits range so it never gets asked outside it
    n_b = jnp.clip(rho, rho_d, rho_cc) / m_u
    x = jnp.log(n_b / 1e35)
    # the fit
    s = (
        0.28822899
        + 0.59150523 * x
        + 0.090185940 * x**2
        - 0.11025614 * x**3
        + 0.029377479 * x**4
        - 0.0032618465 * x**5
        + 0.00013543555 * x**6
    )
    ds = (
        0.59150523
        + 0.18037188 * x
        - 0.33076842 * x**2
        + 0.11750992 * x**3
        - 0.016309233 * x**4
        + 0.00081261330 * x**5
    )
    return n_b * ds * jnp.exp(s) * mev  # mev turns it into erg/cm^3
