# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Reading a log-log pchip back off."""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.pchip_eval import pchip_eval


def loglog_pchip_eval(params: tuple[Array, ...], xq: Num) -> Array:
    """
    Evaluate a log-log pchip spline.

    :param params: Spline params from loglog_pchip_build.
    :param xq: Where to evaluate, one value or an array.
    :returns: The value of the spline at xq, out of logs.
    """
    # below the first knot, hold the first knots value
    xq = jnp.maximum(xq, jnp.exp(params[0][0]))

    def log_y_at(x: Num) -> Array:
        return pchip_eval(params, jnp.log(x))

    # atleast_1d so a single value works too
    return jnp.exp(jax.vmap(log_y_at)(jnp.atleast_1d(xq)))
