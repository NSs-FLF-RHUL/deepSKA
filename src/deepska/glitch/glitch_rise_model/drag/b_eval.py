# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""B at any density, from the case splines."""

from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.loglog_pchip_eval import loglog_pchip_eval


def b_eval(p_b: tuple[Array, ...], k: Num, rho: Num) -> Array:
    """
    Evaluate the mutual friction coefficient B of one pinning case.

    :param p_b: The log-log splines of B against density, one per case.
    :param k: Which case, 0 = A, 1 = B, 2 = C.
    :param rho: Mass density in g/cm**3.
    :returns: B at that density.
    """
    params = tuple(p_[k] for p_ in p_b)  # layer k of every entry of (x, y, h, m)
    return loglog_pchip_eval(params, rho)  # back from log B
