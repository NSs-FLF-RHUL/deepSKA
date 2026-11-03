# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Roots from values already on a grid, then polished by guarded false position.

- first_sign_change: the first pair of neighbouring grid points, inside a range, where
  a function changes sign. That pair is the bracket.
- cut_points and keep_piece: one step of guarded false position. f is worked out at
  two points, where the straight line through the two ends crosses 0 (regula falsi)
  and the middle (bisection), and the first piece where the sign still changes is
  kept. The line closes in fast on a smooth f, the middle makes sure the bracket at
  least halves, so a jump in f is pinned down as bisection would. Finish with
  linear_root across the last bracket.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.linear_root import linear_root


def first_sign_change(
    x: Num, f: Num, lo: Num, hi: Num
) -> tuple[tuple[Array, Array, Array, Array], Array]:
    """
    Find the first neighbouring pair of grid points in [lo, hi] where f changes sign.

    A NaN never counts as a sign change.

    :param x: The grid, increasing.
    :param f: f at each grid point.
    :param lo: The low end of the range.
    :param hi: The high end of the range.
    :returns bracket: (x_a, x_b, f_a, f_b), the pair.
    :returns found: False when no pair in the range changes sign.
    """
    inside = (x >= lo) & (x <= hi) & jnp.isfinite(f)
    flips = inside[:-1] & inside[1:] & (jnp.sign(f[:-1]) != jnp.sign(f[1:]))
    i = jnp.argmax(flips)  # the first one, or 0 when there is none
    return (x[i], x[i + 1], f[i], f[i + 1]), flips.any()


def cut_points(bracket: tuple[Num, Num, Num, Num]) -> Array:
    """
    Make the two points one step of guarded false position works f out at, in order.

    :param bracket: (x_a, x_b, f_a, f_b).
    :returns: The line's root and the middle, the lower one first, shape (2,).
    """
    a, b, fa, fb = bracket
    x_line, x_mid = linear_root(a, b, fa, fb, 0.0), 0.5 * (a + b)
    return jnp.stack([jnp.minimum(x_line, x_mid), jnp.maximum(x_line, x_mid)])


def keep_piece(
    bracket: tuple[Num, Num, Num, Num], cuts: Num, f_cuts: Num
) -> tuple[Array, Array, Array, Array]:
    """
    Keep the first of the three pieces the two cuts make where the sign still changes.

    :param bracket: (x_a, x_b, f_a, f_b).
    :param cuts: The two points, from cut_points.
    :param f_cuts: f at the two points.
    :returns: The new bracket.
    """
    a, b, fa, fb = bracket
    xs = jnp.stack([a, cuts[0], cuts[1], b])  # in order
    fs = jnp.stack([fa, f_cuts[0], f_cuts[1], fb])
    i = jnp.argmax(jnp.sign(fs[:-1]) != jnp.sign(fs[1:]))
    return xs[i], xs[i + 1], fs[i], fs[i + 1]
