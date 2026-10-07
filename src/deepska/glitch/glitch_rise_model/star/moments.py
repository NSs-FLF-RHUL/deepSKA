# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Everything the spin equations need from the star.

Just wiring, runs the star pieces in the order they depend on each other.
"""

from deepska.glitch.glitch_rise_model.star.cylinder_inertia_spline import (
    cylinder_inertia_spline,
)
from deepska.glitch.glitch_rise_model.star.density_profile import density_profile
from deepska.glitch.glitch_rise_model.star.density_spline import density_spline
from deepska.glitch.glitch_rise_model.star.drip_radius import drip_radius
from deepska.glitch.glitch_rise_model.star.equilibrium_glitch import equilibrium_glitch
from deepska.glitch.glitch_rise_model.star.inertia import inertia
from deepska.glitch.glitch_rise_model.star.shells import shells
from deepska.glitch.glitch_rise_model.star.sphere_inertia_spline import (
    sphere_inertia_spline,
)
from deepska.glitch.glitch_rise_model.star.tov import tov
from deepska.glitch.kinds import Fn, Num, Splines, Star, StarInputs


def moments(inputs: StarInputs, p0: Num, p_min: Num, inverse: Fn) -> Star:
    """
    Build the star and everything the spin equations need from it.

    :param inputs: The star inputs.
    :param p0: Pressure at the crust-core boundary in erg/cm**3.
    :param p_min: Surface pressure in erg/cm**3.
    :param inverse: The inverse eos, rho(P).
    :returns: The built star: radii, moments of inertia, eq 23 and the shells.
    """
    r_star, m_star, n_valid, r_hist, m_hist, p_hist = tov(p0, p_min, inverse, inputs)
    rho_hist = density_profile(p_hist, p_min, inverse)
    splines = Splines(
        density_spline(r_hist, rho_hist, n_valid),
        sphere_inertia_spline(r_hist, rho_hist, n_valid),
        cylinder_inertia_spline(r_hist, rho_hist, n_valid),
    )
    r_drip = drip_radius(splines.rho_r, inputs.rho_d, inputs.r0, r_star)
    i_tot, i_crust_total, i_cyl_unit, h, i_sf, i_core, i_crust = inertia(
        m_star, r_star, inputs.r0, r_drip, splines
    )
    dnu_equil = equilibrium_glitch(i_sf, i_tot, inputs.dom_crit)  # eq 23
    a_shell, x, dx, rho_shell = shells(inputs.r0, r_drip, h, splines, inputs.n_shells)
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
