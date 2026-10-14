# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The state y is the shells then the core then the crust."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Arr, Num


def pack_state(om_sf: Num, om_core: Num, om_crust: Num) -> Array:
    """
    Join the three spins into one state.

    :param om_sf: Spin of every shell in rad/s.
    :param om_core: Spin of the core superfluid in rad/s.
    :param om_crust: Spin of the crust in rad/s.
    :returns: The state y, shells then core then crust.
    """
    return jnp.concatenate([om_sf, jnp.array([om_core, om_crust])])


def unpack_state(y: Arr) -> tuple[Array, Array, Array]:
    """
    Split the state back into its three spins.

    :param y: The state, or a whole history with the state along the last axis.
    :returns om_sf: Spin of every shell in rad/s.
    :returns om_core: Spin of the core superfluid in rad/s.
    :returns om_crust: Spin of the crust in rad/s.
    """
    # last axis so it works on a whole history too
    return y[..., :-2], y[..., -2], y[..., -1]
