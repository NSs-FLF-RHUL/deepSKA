# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Eq 23."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import muhz
from deepska.glitch.kinds import Num


def equilibrium_glitch(i_sf: Num, i_tot: Num, dom_crit: Num) -> Array:
    """
    Calculate the equilibrium glitch size, eq 23.

    :param i_sf: Moment of inertia of the pinned crust superfluid in g cm**2.
    :param i_tot: Moment of inertia of the whole star in g cm**2.
    :param dom_crit: Initial lag of the superfluid over the crust in rad/s.
    :returns: The glitch size in muHz.
    """
    return (i_sf / i_tot) * dom_crit / (2.0 * jnp.pi) / muhz
