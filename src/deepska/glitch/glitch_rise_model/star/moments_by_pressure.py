# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Everything the spin equations need from the star, with the crust edges found by P.

The same wiring as moments.py, but the crust-core radius and the drip radius are where
the TOV pressure falls to P_t and P_drip, not where rho(r) = rho_cc and rho_d. At a
Maxwell join rho jumps (eps_2 in the core to eps_1 in the crust) while P is the same on
both sides, so by rho the radius moves in 10 cm TOV steps as the inputs change (a
sawtooth in dnu, wrong derivatives); by P it moves smoothly. Used by the metamodel,
whose TOV starts at the centre.

Contents:

- r_at_p: the radius where the TOV pressure is a given P.
- moments_by_pressure: the star, as moments, with the two edges by P.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.star.density import (
    density_profile,
    density_spline,
)
from deepska.glitch.glitch_rise_model.star.equilibrium_glitch import equilibrium_glitch
from deepska.glitch.glitch_rise_model.star.inertia import (
    cylinder_inertia_spline,
    inertia,
    sphere_inertia_spline,
)
from deepska.glitch.glitch_rise_model.star.shells import shells
from deepska.glitch.glitch_rise_model.star.tov import tov
from deepska.glitch.kinds import Arr, Fn, Num, Splines, Star, StarInputs


def r_at_p(r_hist: Arr, p_hist: Arr, n_valid: Num, p_min: Num, p_q: Num) -> Array:
    """
    Find the radius where the TOV pressure is p_q, straight in log P between rows.

    The padding rows past the surface are replaced by pressures that keep falling, so
    the rows stay in order for the interpolation.

    :param r_hist: Radius at every TOV row in cm.
    :param p_hist: Pressure at every TOV row in erg/cm**3.
    :param n_valid: Number of real rows before the padding.
    :param p_min: Surface pressure in erg/cm**3.
    :param p_q: The pressure to find, in erg/cm**3, between p_min and the start.
    :returns: The radius in cm.
    """
    i = jnp.arange(p_hist.shape[0])
    below = p_min * jnp.exp(-(i - n_valid + 1.0))  # padding rows: keep going down
    p_safe = jnp.where(i < n_valid, p_hist, below)
    return jnp.interp(-jnp.log(p_q), -jnp.log(p_safe), r_hist)


def moments_by_pressure(
    inputs: StarInputs, p0: Num, inverse: Fn, p_t: Num, p_drip: Num
) -> Star:
    """
    Build the star and everything the spin equations need from it, edges by P.

    :param inputs: The star inputs.
    :param p0: Pressure at r0, where the TOV starts, in erg/cm**3.
    :param inverse: The inverse eos, rho(P).
    :param p_t: Pressure at the crust-core boundary in erg/cm**3.
    :param p_drip: Pressure at neutron drip in erg/cm**3.
    :returns: The built star: radii, moments of inertia, eq 23 and the shells.
    """
    p_min = inputs.eos(inputs.rho_min, inputs.rho_d, inputs.rho_cc)  # the surface
    r_star, m_star, n_valid, r_hist, m_hist, p_hist = tov(p0, p_min, inverse, inputs)
    rho_hist = density_profile(p_hist, p_min, inverse)
    splines = Splines(
        density_spline(r_hist, rho_hist, n_valid),
        sphere_inertia_spline(r_hist, rho_hist, n_valid),
        cylinder_inertia_spline(r_hist, rho_hist, n_valid),
    )
    r_cc = r_at_p(r_hist, p_hist, n_valid, p_min, p_t)  # the crust-core boundary
    r_drip = r_at_p(r_hist, p_hist, n_valid, p_min, p_drip)
    i_tot, i_crust_total, i_cyl_unit, h, i_sf, i_core, i_crust = inertia(
        m_star, r_star, r_cc, r_drip, splines
    )
    dnu_equil = equilibrium_glitch(i_sf, i_tot, inputs.dom_crit)  # eq 23
    a_shell, x, dx, rho_shell = shells(r_cc, r_drip, h, splines, inputs.n_shells)
    return Star(
        r_star,
        m_star,
        r_drip,
        i_tot,
        i_crust_total,
        i_cyl_unit,
        h,
        i_sf,
        i_core,
        i_crust,
        dnu_equil,
        a_shell,
        x,
        dx,
        rho_shell,
        m_hist,
        rho_hist,
        n_valid,
    )
