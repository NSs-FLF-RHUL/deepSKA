# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Moments of inertia of the 3 bits of the star.

Pinned crust superfluid, core superfluid, and the rest. The model choices (0.35, 0.95,
the cylinder) each have their own file.
"""

from jax import Array

from deepska.glitch.glitch_rise_model.star.core_inertia import core_inertia
from deepska.glitch.glitch_rise_model.star.cylinder_half_height import (
    cylinder_half_height,
)
from deepska.glitch.glitch_rise_model.star.cylinder_inertia import cylinder_inertia
from deepska.glitch.glitch_rise_model.star.rest_inertia import rest_inertia
from deepska.glitch.glitch_rise_model.star.star_inertia import star_inertia
from deepska.glitch.kinds import Num, Splines
from deepska.glitch.numerics.pchip_integral import pchip_integral


def inertia(
    m_star: Num, r_star: Num, r0: Num, r_drip: Num, splines: Splines
) -> tuple[Array, ...]:
    """
    Calculate every moment of inertia the spin equations need.

    :param m_star: Mass of the star in g.
    :param r_star: Radius of the surface in cm.
    :param r0: Radius of the crust-core boundary in cm.
    :param r_drip: Radius of neutron drip in cm.
    :param splines: The splines of density and dI/dr against radius.
    :returns i_tot: Whole star, in g cm**2.
    :returns i_crust_total: Crust from spherical shells, in g cm**2.
    :returns i_cyl_unit: Crust per unit height of cylindrical shells, in g cm.
    :returns h: Half-height of the cylinder in cm.
    :returns i_sf: Pinned superfluid of the inner crust, in g cm**2.
    :returns i_core: Core superfluid, in g cm**2.
    :returns i_crust: Everything else, in g cm**2.
    """
    i_tot = star_inertia(m_star, r_star)  # moment of inertia for whole star
    # crust I from spherical shells: crust-core to surface
    i_crust_total = pchip_integral(splines.di_sphere, r0, r_star)
    # crust I per unit height of cylindrical shells
    i_cyl_unit = pchip_integral(splines.di_cylinder, r0, r_star)
    h = cylinder_half_height(i_crust_total, i_cyl_unit)
    # pinned superfluid cylindrical shells of height 2h, inner crust only
    i_sf = cylinder_inertia(splines.di_cylinder, r0, r_drip, h)
    i_core = core_inertia(i_tot, i_crust_total)
    i_crust = rest_inertia(i_tot, i_core, i_sf)
    return i_tot, i_crust_total, i_cyl_unit, h, i_sf, i_core, i_crust
