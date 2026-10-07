# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Tov from the crust core boundary out to the surface.

Runs all n_tov steps so we can vmap, n_valid says how many rows are real before the
padding.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.star.tov_rhs import tov_rhs
from deepska.glitch.kinds import Fn, Num, StarInputs, Steps
from deepska.glitch.numerics.ode import solve_ode


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

    r_hist, x_hist, r_star, x_end, n_valid = solve_ode(
        rhs,
        jnp.array([inputs.m0, p0]),
        Steps(inputs.r0, inputs.dr, inputs.n_tov),
        stop=(1, p_min),
    )
    return r_star, x_end[0], n_valid, r_hist, x_hist[:, 0], x_hist[:, 1]
