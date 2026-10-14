# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Keep function for solve_ode so the sweeps dont eat memory."""

from jax import Array

from deepska.glitch.glitch_rise_model.spin.spin_state import unpack_state
from deepska.glitch.kinds import Num


def crust_only(y: Num) -> Array:
    """
    Pick the crust spin out of the state.

    :param y: The spin state, shells then core then crust, in rad/s.
    :returns: The spin of the crust in rad/s.
    """
    return unpack_state(y)[2]
