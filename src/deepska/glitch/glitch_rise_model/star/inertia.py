# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Moments of inertia of the 3 bits of the star.

- Splines of dI/dr for spherical and cylindrical shells.
- Moment of inertia for whole star.
- Moment of inertia of cylindrical shells.
- Height of the cylinder that the crust is modeled as.
- Moment of inertia of the core superfluid.
- Moment of inertia of whatever isnt superfluid.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num, Splines
from deepska.glitch.numerics.pchip import pchip_build, pchip_integral


def sphere_inertia_spline(r_hist: Num, rho_hist: Num, n_valid: Num) -> Array:
    """
    Build the spline of dI/dr of spherical shells.

    dI/dr is 8 pi / 3 * rho * r**4.

    :param r_hist: Radius at every tov row in cm.
    :param rho_hist: Mass density at every tov row in g/cm**3.
    :param n_valid: Number of real rows before the padding.
    :returns: The spline params (x, y, h, m).
    """
    return pchip_build(r_hist, 8.0 * jnp.pi / 3.0 * rho_hist * r_hist**4, n_valid)


def cylinder_inertia_spline(r_hist: Num, rho_hist: Num, n_valid: Num) -> Array:
    """
    Build the spline of dI/dr per unit height of cylindrical shells.

    dI/dr is 2 pi * rho * r**3.

    :param r_hist: Radius at every tov row in cm.
    :param rho_hist: Mass density at every tov row in g/cm**3.
    :param n_valid: Number of real rows before the padding.
    :returns: The spline params (x, y, h, m).
    """
    return pchip_build(r_hist, 2.0 * jnp.pi * rho_hist * r_hist**3, n_valid)


def star_inertia(m_star: Num, r_star: Num) -> Array:
    """
    Calculate the moment of inertia of the whole star, 0.35 M R**2.

    :param m_star: Mass of the star in g.
    :param r_star: Radius of the surface in cm.
    :returns: The moment of inertia in g cm**2.
    """
    return 0.35 * m_star * r_star**2


def cylinder_inertia(di_cylinder: Num, a: Num, b: Num, h: Num) -> Array:
    """
    Calculate the moment of inertia of cylindrical shells between two radii.

    :param di_cylinder: Spline of dI/dr per unit height.
    :param a: Inner radius in cm.
    :param b: Outer radius in cm.
    :param h: Half-height of the cylinder in cm.
    :returns: The moment of inertia in g cm**2.
    """
    return 2.0 * h * pchip_integral(di_cylinder, a, b)


def cylinder_half_height(i_crust_total: Num, i_cyl_unit: Num) -> Array:
    """
    Calculate the half-height of the cylinder the crust is modelled as.

    The cylinder (total height 2h) has the same moment of inertia as the spherical
    crust.

    :param i_crust_total: Moment of inertia of the crust in g cm**2.
    :param i_cyl_unit: Moment of inertia of the crust per unit height in g cm.
    :returns: The half-height h in cm.
    """
    return i_crust_total / (2.0 * i_cyl_unit)


def core_inertia(i_tot: Num, i_crust_total: Num) -> Array:
    """
    Calculate the moment of inertia of the core neutron superfluid.

    Taken as 95% of the cores moment of inertia.

    :param i_tot: Moment of inertia of the whole star in g cm**2.
    :param i_crust_total: Moment of inertia of the crust in g cm**2.
    :returns: The moment of inertia of the core superfluid in g cm**2.
    """
    return 0.95 * (i_tot - i_crust_total)


def rest_inertia(i_tot: Num, i_core: Num, i_sf: Num) -> Array:
    """
    Calculate the moment of inertia of whatever isnt superfluid.

    The crust lattice plus the charged 5% of the core.

    :param i_tot: Moment of inertia of the whole star in g cm**2.
    :param i_core: Moment of inertia of the core superfluid in g cm**2.
    :param i_sf: Moment of inertia of the pinned crust superfluid in g cm**2.
    :returns: The moment of inertia of the rest in g cm**2.
    """
    return i_tot - i_core - i_sf


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
