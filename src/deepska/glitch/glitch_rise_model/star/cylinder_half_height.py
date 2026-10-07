# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Height of the cylinder the crust is modelled as."""

from jax import Array

from deepska.glitch.kinds import Num


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
