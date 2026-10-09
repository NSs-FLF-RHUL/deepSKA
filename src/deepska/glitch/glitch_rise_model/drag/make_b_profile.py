# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""A drag profile from one of the pinning cases."""

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
