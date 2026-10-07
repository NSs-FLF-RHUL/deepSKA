# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Eq 20, the pinned superfluid in the crust shells."""

from jax import Array

from deepska.glitch.kinds import Num


def superfluid_rate(
    om_sf: Num,
    om_crust: Num,
    der: Num,
    b_shell: Num,
    x: Num,
) -> Array:
    """
    Calculate the rate of change of the spin of every shell, eq 20.

    :param om_sf: Spin of every shell in rad/s.
    :param om_crust: Spin of the crust in rad/s.
    :param der: Gradient of om_sf with respect to x.
    :param b_shell: Mutual friction coefficient at every shell.
    :param x: Radius of every shell as a fraction of r_drip.
    :returns: The rate in rad/s**2.
    """
    # x * der is the gradient term
    return b_shell * (2 * om_sf + x * der) * (om_crust - om_sf)
