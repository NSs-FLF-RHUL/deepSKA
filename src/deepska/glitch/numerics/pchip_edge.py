# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Slope at the two end knots of a pchip.

End knots only have a neighbour on one side so the interior formula doesnt work.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num


def pchip_edge(h_a: Num, h_b: Num, d_a: Num, d_b: Num) -> Array:
    """
    Calculate the slope at an end knot of a pchip.

    One sided estimate, then clamped.

    :param h_a: Width of the gap next to the end.
    :param h_b: Width of the gap after that.
    :param d_a: Gradient over the gap next to the end.
    :param d_b: Gradient over the gap after that.
    :returns: The slope at the end knot.
    """
    m = ((2.0 * h_a + h_b) * d_a - h_a * d_b) / (h_a + h_b)
    # points the wrong way, flatten it
    m = jnp.where(jnp.sign(m) != jnp.sign(d_a), 0.0, m)
    # the two gaps disagree, cap at 3x the first slope
    return jnp.where(
        (jnp.sign(d_a) != jnp.sign(d_b)) & (jnp.abs(m) > 3.0 * jnp.abs(d_a)),
        3.0 * d_a,
        m,
    )
