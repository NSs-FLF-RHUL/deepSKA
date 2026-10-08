# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Right hand side of the spin equations.

- Eq 20, the pinned superfluid in the crust shells.
- Eq 21, the core superfluid.
- Eq 22, the crust.

rhs2 just unpacks y, calls eq 20, 21, 22, packs the rates back up.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.spin.spin_state import pack_state, unpack_state
from deepska.glitch.kinds import Coupling, Num, Star
from deepska.glitch.numerics.central_difference import central_difference


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


def core_rate(om_core: Num, om_crust: Num, b_core: Num) -> Array:
    """
    Calculate the rate of change of the core superfluid spin, eq 21.

    :param om_core: Spin of the core superfluid in rad/s.
    :param om_crust: Spin of the crust in rad/s.
    :param b_core: Mutual friction coefficient of the core.
    :returns: The rate in rad/s**2.
    """
    return 2 * om_core * b_core * (om_crust - om_core)


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


def rhs2(y: Num, _t: Num, coupling: Coupling, star: Star) -> Array:
    """
    Calculate the right hand side of the spin equations.

    :param y: The spin state, shells then core then crust, in rad/s.
    :param _t: Time, not used as nothing depends on it directly, rk4 passes it anyway.
    :param coupling: B at every shell and B of the core.
    :param star: The built star.
    :returns: The rate of change of the state in rad/s**2.
    """
    om_sf, om_core, om_crust = unpack_state(y)
    der = central_difference(om_sf, star.dx)
    d_sf = superfluid_rate(om_sf, om_crust, der, coupling.b_shell, star.x)  # Eq 20
    d_core = core_rate(om_core, om_crust, coupling.b_core)  # Eq 21
    d_crust = crust_rate(d_sf, d_core, star.a_shell, star.i_crust, star.i_core)  # Eq 22
    return pack_state(d_sf, d_core, d_crust)
