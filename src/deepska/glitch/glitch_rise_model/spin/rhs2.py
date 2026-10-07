# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Right hand side of the spin equations.

Just unpacks y, calls eq 20, 21, 22 each in their own file, packs the rates back up.
"""

from jax import Array

from deepska.glitch.glitch_rise_model.spin.core_rate import core_rate
from deepska.glitch.glitch_rise_model.spin.crust_rate import crust_rate
from deepska.glitch.glitch_rise_model.spin.spin_state import pack_state, unpack_state
from deepska.glitch.glitch_rise_model.spin.superfluid_rate import superfluid_rate
from deepska.glitch.kinds import Coupling, Num, Star
from deepska.glitch.numerics.central_difference import central_difference


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
