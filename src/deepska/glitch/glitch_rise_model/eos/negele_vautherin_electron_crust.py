# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The whole crust: electron gas below neutron drip, Negele & Vautherin above it.

They dont match at rho_d though!!! Just below it P is about 2.2x higher than just above
it, so rho(P) has a jump there.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.eos.negele_vautherin_1973 import p_inner
from deepska.glitch.glitch_rise_model.eos.relativistic_electron_gas import p_outer
from deepska.glitch.kinds import Num


def p_crust(rho: Num, rho_d: Num, rho_cc: Num) -> Array:
    """
    Calculate the pressure of the whole crust.

    :param rho: Mass density in g/cm**3.
    :param rho_d: Neutron drip density in g/cm**3, where the two eos meet.
    :param rho_cc: Crust-core transition density in g/cm**3.
    :returns: The pressure in erg/cm**3.
    """
    # electron gas below drip, Negele & Vautherin from it
    return jnp.where(rho < rho_d, p_outer(rho), p_inner(rho, rho_d, rho_cc))
