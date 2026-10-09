# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The drag profiles B(rho) the model can be run with.

- make_b_profile: the profile of one pinning case.
- make_b_profile_flat: a profile that is the same at every density.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.drag.b_eval import b_eval
from deepska.glitch.glitch_rise_model.drag.p_b import p_b
from deepska.glitch.kinds import Fn, Num


def make_b_profile(k: Num) -> Fn:
    """
    Make the drag profile B(rho) of one pinning case.

    :param k: Which case, 0 = A, 1 = B, 2 = C.
    :returns: Function giving B at a mass density in g/cm**3.
    """

    def b_profile(rho: Num) -> Array:
        return b_eval(p_b, k, rho)

    return b_profile


def make_b_profile_flat(b_value: Num) -> Fn:
    """
    Make a drag profile that is the same at every density.

    :param b_value: The value of B.
    :returns: Function giving b_value at any mass density.
    """

    def b_profile_flat(rho: Num) -> Array:
        return jnp.full_like(rho, b_value)  # same B at every density

    return b_profile_flat
