# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The cylindrical shells the inner crust superfluid is split into."""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.star.inertia import cylinder_inertia
from deepska.glitch.kinds import Num, Splines
from deepska.glitch.numerics.midpoint_edges import midpoint_edges
from deepska.glitch.numerics.pchip import pchip_eval


def shells(
    r0: Num, r_drip: Num, h: Num, splines: Splines, n_shells: int
) -> tuple[Array, ...]:
    """
    Split the inner crust superfluid into cylindrical shells.

    :param r0: Radius of the crust-core boundary in cm.
    :param r_drip: Radius of neutron drip in cm.
    :param h: Half-height of the cylinder in cm.
    :param splines: The splines of density and dI/dr against radius.
    :param n_shells: Number of shells.
    :returns a_shell: Moment of inertia of each shell in g cm**2.
    :returns x: Radius of each shell as a fraction of r_drip.
    :returns dx: Spacing of x.
    :returns rho_shell: Mass density at each shell centre in g/cm**3.
    """
    r_shell = jnp.linspace(r0, r_drip, n_shells)
    edges = midpoint_edges(r_shell, r0, r_drip)
    # one per shell, pairing edge i with edge i + 1
    a_shell = jax.vmap(cylinder_inertia, in_axes=(None, 0, 0, None))(
        splines.di_cylinder, edges[:-1], edges[1:], h
    )
    # density at each shell centre, from the rho(r) spline
    rho_shell = jax.vmap(pchip_eval, in_axes=(None, 0))(splines.rho_r, r_shell)
    # n_shells points, n_shells - 1 gaps
    x, dx = r_shell / r_drip, (1.0 - r0 / r_drip) / (n_shells - 1)
    return a_shell, x, dx, rho_shell
