# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Moment of inertia of whatever isnt superfluid."""

from jax import Array

from deepska.glitch.kinds import Num


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
