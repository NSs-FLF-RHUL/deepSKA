# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Where the inner crust ends."""

from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.invert import invert
from deepska.glitch.numerics.pchip import pchip_eval


def drip_radius(rho_r: Num, rho_d: Num, r0: Num, r_star: Num) -> Array:
    """
    Find the radius of neutron drip, where rho(r) = rho_d.

    :param rho_r: Spline of density against radius.
    :param rho_d: Neutron drip density in g/cm**3.
    :param r0: Radius of the crust-core boundary in cm.
    :param r_star: Radius of the surface in cm.
    :returns: The radius of neutron drip in cm.
    """

    # density at radius r, from the rho(r) spline
    def rho_at(r: Num) -> Array:
        return pchip_eval(rho_r, r)

    return invert(rho_at, rho_d, r0, r_star, n=40)
