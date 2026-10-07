# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""One step of bisection."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Fn, Num


def bis(
    carry: tuple[Array, Array], f: Fn, target: Num, up: Num
) -> tuple[tuple[Array, Array], float]:
    """
    Do one step of bisection.

    :param carry: The bracket (low, high) around the root.
    :param f: The function being inverted.
    :param target: The value f has to reach.
    :param up: True if f goes up with x.
    :returns carry: The half of the bracket that the root is in.
    :returns out: Always 0.0, as scan wants an output per step.
    """
    low, high = carry
    mid = 0.5 * (low + high)
    go_up = (f(mid) < target) == up  # the root lies above mid
    # 0.0 as scan wants an output per step
    return (jnp.where(go_up, mid, low), jnp.where(go_up, high, mid)), 0.0
