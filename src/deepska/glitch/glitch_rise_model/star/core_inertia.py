# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Moment of inertia of the core superfluid."""

from jax import Array

from deepska.glitch.kinds import Num


def core_inertia(i_tot: Num, i_crust_total: Num) -> Array:
    """
    Calculate the moment of inertia of the core neutron superfluid.

    Taken as 95% of the cores moment of inertia.

    :param i_tot: Moment of inertia of the whole star in g cm**2.
    :param i_crust_total: Moment of inertia of the crust in g cm**2.
    :returns: The moment of inertia of the core superfluid in g cm**2.
    """
    return 0.95 * (i_tot - i_crust_total)
