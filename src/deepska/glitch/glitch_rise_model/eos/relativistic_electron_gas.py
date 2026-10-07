# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Relativistic degenerate electron gas, for the outer crust."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import c, hbar, m_u
from deepska.glitch.kinds import Num


def p_outer(rho: Num) -> Array:
    """
    Calculate the pressure of the outer crust, a relativistic electron gas.

    With 0.4 electrons per nucleon mass.

    :param rho: Mass density in g/cm**3.
    :returns: The pressure in erg/cm**3.
    """
    return (
        (hbar * c)
        * (3 * jnp.pi**2 / m_u) ** (4 / 3)
        * (0.4 * rho) ** (4 / 3)
        / (12 * jnp.pi**2)
    )
