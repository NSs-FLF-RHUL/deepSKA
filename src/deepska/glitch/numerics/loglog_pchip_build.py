# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Pchip in log x and log y."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.pchip_build import pchip_build


def loglog_pchip_build(x: Num, y: Num) -> Array:
    """
    Build a pchip spline in log x and log y.

    :param x: Knot positions, positive and increasing.
    :param y: Value at each knot, positive.
    :returns: The spline params (x, y, h, m), in logs.
    """
    return pchip_build(jnp.log(x), jnp.log(y))
