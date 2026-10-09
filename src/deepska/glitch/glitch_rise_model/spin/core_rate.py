# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Eq 21, the core superfluid."""

from jax import Array

from deepska.glitch.kinds import Num


def core_rate(om_core: Num, om_crust: Num, b_core: Num) -> Array:
    """
    Calculate the rate of change of the core superfluid spin, eq 21.

    :param om_core: Spin of the core superfluid in rad/s.
    :param om_crust: Spin of the crust in rad/s.
    :param b_core: Mutual friction coefficient of the core.
    :returns: The rate in rad/s**2.
    """
    return 2 * om_core * b_core * (om_crust - om_core)
