# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Where the spin equations start from."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.spin.spin_state import pack_state
from deepska.glitch.kinds import Num


def initial_spin(omega0: Num, dom_crit: Num, n_shells: int) -> Array:
    """
    Make the state the spin equations start from.

    Shells lead by dom_crit, core and crust start at omega0.

    :param omega0: Spin of the pulsar in rad/s.
    :param dom_crit: Initial lag of the superfluid over the crust in rad/s.
    :param n_shells: Number of shells.
    :returns: The spin state, shells then core then crust, in rad/s.
    """
    return pack_state(jnp.full(n_shells, omega0 + dom_crit), omega0, omega0)
