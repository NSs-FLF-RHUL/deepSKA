# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Eq 22, the crust."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num


def crust_rate(
    d_sf: Num,
    d_core: Num,
    a_shell: Num,
    i_crust: Num,
    i_core: Num,
) -> Array:
    """
    Calculate the rate of change of the crust spin, eq 22.

    :param d_sf: Rate of change of the spin of every shell in rad/s**2.
    :param d_core: Rate of change of the core spin in rad/s**2.
    :param a_shell: Moment of inertia of every shell in g cm**2.
    :param i_crust: Moment of inertia of the crust component in g cm**2.
    :param i_core: Moment of inertia of the core superfluid in g cm**2.
    :returns: The rate in rad/s**2.
    """
    # angular momentum the shells and core lose all lands on the crust
    return -jnp.sum(a_shell * d_sf) / i_crust - i_core * d_core / i_crust
