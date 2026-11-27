# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The model read off at the times of the data bins."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num


def at_bins(res: Num, t_hist: Num, t_bins: Num) -> Array:
    """
    Interpolate the model residual onto the times of the data bins.

    jnp so it works inside a jitted sweep too.

    :param res: Timing residual at every step in ms.
    :param t_hist: Time of every step in s.
    :param t_bins: Bin centres in s.
    :returns: The residual at each bin centre in ms.
    """
    return jnp.interp(t_bins, t_hist, res)
