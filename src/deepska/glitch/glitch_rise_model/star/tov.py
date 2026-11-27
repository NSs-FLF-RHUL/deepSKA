# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Tov from the crust core boundary out to the surface.

Runs all n_tov steps so we can vmap, n_valid says how many rows are real before the
padding. The right hand side lives here too since nothing else uses it.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import c, g_newton
from deepska.glitch.kinds import Arr, Fn, Num, StarInputs, Steps
from deepska.glitch.numerics.ode import solve_ode


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


def tov(
    p0: Num, p_min: Num, inverse: Fn, inputs: StarInputs
) -> tuple[Array | int, ...]:
    """
    Integrate the tov equations from the crust-core boundary to the surface.

    Starts from p0 and m0 at r0 and stops when the pressure hits p_min.

    :param p0: Pressure at the crust-core boundary in erg/cm**3.
    :param p_min: Surface pressure in erg/cm**3.
    :param inverse: The inverse eos, rho(P).
    :param inputs: The star inputs, for m0, r0, dr and n_tov.
    :returns r_star: Radius of the surface in cm.
    :returns m_star: Mass of the star in g.
    :returns n_valid: Number of real rows before the padding.
    :returns r_hist: Radius at every row in cm.
    :returns m_hist: Enclosed mass at every row in g.
    :returns p_hist: Pressure at every row in erg/cm**3.
    """

    # tov right hand side with this stars inverse eos filled in
    def rhs(x: Num, r: Num) -> Array:
        return tov_rhs(x, r, inverse, p_min)

    r_hist, x_hist, r_star, x_end, n_valid, _ = solve_ode(
        rhs,
        jnp.array([inputs.m0, p0]),
        Steps(inputs.r0, inputs.dr, inputs.n_tov),
        stop=(1, p_min),
    )
    return r_star, x_end[0], n_valid, r_hist, x_hist[:, 0], x_hist[:, 1]
