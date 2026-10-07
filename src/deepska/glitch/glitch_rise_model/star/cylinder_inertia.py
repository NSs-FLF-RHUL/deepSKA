# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Moment of inertia of cylindrical shells."""

from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.pchip_integral import pchip_integral


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
