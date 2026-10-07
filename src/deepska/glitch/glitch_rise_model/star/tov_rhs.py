# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Right hand side of the tov equations."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import c, g_newton
from deepska.glitch.kinds import Arr, Fn, Num


def tov_rhs(x: Arr, r: Num, rho_of_p: Fn, p_min: Num) -> Array:
    """
    Calculate the right hand side of the tov equations.

    :param x: Enclosed mass in g and pressure in erg/cm**3, as (m, p).
    :param r: Radius in cm.
    :param rho_of_p: The inverse eos, rho(P).
    :param p_min: Surface pressure in erg/cm**3, the pressure is clipped at it.
    :returns: The gradients dm/dr and dp/dr.
    """
    m, p = x
    # clipped at p_min so the padding past the surface stays safe
    rho = rho_of_p(jnp.maximum(p, p_min))
    dm = 4.0 * jnp.pi * r**2 * rho
    dp = (
        -g_newton
        * (rho + p / c**2)
        * (m + 4.0 * jnp.pi * r**3 * p / c**2)
        / (r * (r - 2.0 * g_newton * m / c**2))
    )
    return jnp.array([dm, dp])
