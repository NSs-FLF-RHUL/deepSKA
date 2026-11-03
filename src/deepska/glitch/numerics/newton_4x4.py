# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Damped newton for four unknowns, with the slopes and curvatures written by hand.

Used by the crust cell of the eos. Nothing here is autodiff: the slopes g_i and the
curvatures H_ij come in as functions the caller wrote out, crust doc eqs (gd) to (hgg).

Each step, crust doc section 7:

- Downhill only: lambda is added to the diagonal of H until its leading minors are
  positive, so the step points down.
- The newton jump u -> u - H^-1 g, component by component with minors, eqs (32) to (37).
- Ranges: an unknown that stepped out of its box is pulled back to the edge.
- No overshoot: the jump is halved until eps drops.

Contents:

- minor: one 3x3 minor of H.
- newton_step_4x4: the jump, with the four cofactor rows written out.
- downhill_only: the four leading minor tests.
- add_lambda: the diagonal shift that passes them.
- ranges: the clip to the box.
- no_overshoot: the backtracking.
- newton_4x4: the loop, with pins for unknowns held fixed.
"""

import jax
import jax.numpy as jnp
from jax import Array, lax

from deepska.glitch.kinds import Arr, CellEnergy, Fn, Num, Ranges


def minor(h: Arr, i: int, j: int) -> Array:
    """
    Calculate the minor M_ij of a 4x4 matrix: strike row i and column j, 1-based.

    :param h: The 4x4 matrix.
    :param i: The row struck out, 1 to 4.
    :param j: The column struck out, 1 to 4.
    :returns: The determinant of the 3x3 left, expanded along its first row.
    """
    rows = jnp.array([r for r in range(4) if r != i - 1])  # the rows kept, in order
    cols = jnp.array([c for c in range(4) if c != j - 1])  # the columns kept, in order
    m = h[rows][:, cols]
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


def newton_step_4x4(u: Arr, g: Arr, h: Arr) -> tuple[Array, Array]:
    """
    Take the newton jump u -> u - H^-1 g with the inverse written out by cofactors.

    Crust doc eqs (32) to (37): the sixteen minors, the determinant expanded along the
    first row, and each unknown's update on its own line. H is symmetric, so the row
    minors serve for the column ones.

    :param u: The four unknowns, (4,).
    :param g: The four slopes d eps/d u_i, (4,).
    :param h: The sixteen curvatures d2 eps/d u_i d u_j, (4, 4).
    :returns new: The four unknowns after the jump, (4,).
    :returns det: The determinant of H.
    """
    m11, m12, m13, m14 = minor(h, 1, 1), minor(h, 1, 2), minor(h, 1, 3), minor(h, 1, 4)
    m21, m22, m23, m24 = minor(h, 2, 1), minor(h, 2, 2), minor(h, 2, 3), minor(h, 2, 4)
    m31, m32, m33, m34 = minor(h, 3, 1), minor(h, 3, 2), minor(h, 3, 3), minor(h, 3, 4)
    m41, m42, m43, m44 = minor(h, 4, 1), minor(h, 4, 2), minor(h, 4, 3), minor(h, 4, 4)
    det = h[0, 0] * m11 - h[0, 1] * m12 + h[0, 2] * m13 - h[0, 3] * m14
    g1, g2, g3, g4 = g[0], g[1], g[2], g[3]
    u1 = u[0] - (m11 * g1 - m12 * g2 + m13 * g3 - m14 * g4) / det
    u2 = u[1] - (-m21 * g1 + m22 * g2 - m23 * g3 + m24 * g4) / det
    u3 = u[2] - (m31 * g1 - m32 * g2 + m33 * g3 - m34 * g4) / det
    u4 = u[3] - (-m41 * g1 + m42 * g2 - m43 * g3 + m44 * g4) / det
    return jnp.stack([u1, u2, u3, u4]), det


def downhill_only(h: Arr) -> Array:
    """
    Test that H points downhill by its four leading minors.

    :param h: The 4x4 curvature matrix.
    :returns: True when H_11, the 2x2 and 3x3 leading minors and det H are all positive.
    """
    first = h[0, 0]
    second = h[0, 0] * h[1, 1] - h[0, 1] ** 2
    third = minor(h, 4, 4)
    det = (
        h[0, 0] * minor(h, 1, 1)
        - h[0, 1] * minor(h, 1, 2)
        + h[0, 2] * minor(h, 1, 3)
        - h[0, 3] * minor(h, 1, 4)
    )
    return (first > 0) & (second > 0) & (third > 0) & (det > 0)


def add_lambda(h: Arr, *, lam0: float = 1e-8, n_double: int = 127) -> Array:
    """
    Shift the diagonal of H by the smallest lambda that makes it point downhill.

    Every doubling lambda, 2 lambda, 4 lambda, ... is tried at once and the first that
    passes downhill_only is kept, 0 when H was already fine. If none passes the step is
    useless and no_overshoot keeps u. The off-diagonals never change.

    :param h: The 4x4 curvature matrix.
    :param lam0: The first shift tried.
    :param n_double: How many doublings of it are tried.
    :returns: H with lambda added to its diagonal.
    """
    lams = jnp.concatenate([jnp.zeros(1), lam0 * 2.0 ** jnp.arange(n_double)])

    def passes(lam: Num) -> Array:
        return downhill_only(h + lam * jnp.eye(4))

    lam = lams[jnp.argmax(jax.vmap(passes)(lams))]  # the first that passes
    return h + lam * jnp.eye(4)


def ranges(u: Arr, box: Ranges) -> Array:
    """
    Keep each unknown inside its range.

    One that stepped out is pulled back to the edge it crossed. An open range like
    0 < x < 1 is given a touch inside, 1e-6 and 1 - 1e-6; no upper bound is jnp.inf.

    :param u: The four unknowns, (4,).
    :param box: The lower and upper edges, (4,) each.
    :returns: The unknowns, clipped.
    """
    return jnp.clip(u, box.lo, box.hi)


def no_overshoot(eps: Fn, u: Arr, new: Arr, box: Ranges, *, n_halve: int = 40) -> Array:
    """
    Shrink the jump u -> new until eps drops.

    Every halving is tried at once: the whole step, 1/2, 1/4, ... and 0, stay put. Each
    trial point is first pulled into the ranges, and the largest step that drops wins.
    A nan never counts as a drop.

    :param eps: The function being minimised, of a (4,) point.
    :param u: Where the step starts, (4,).
    :param new: Where the whole jump lands, (4,).
    :param box: The ranges each unknown is kept inside.
    :param n_halve: How many halvings of the jump are tried.
    :returns: The point kept, (4,).
    """
    before = eps(u)
    fractions = jnp.concatenate([2.0 ** -jnp.arange(n_halve), jnp.zeros(1)])

    def trial(t: Num) -> Array:  # where a step of this fraction lands, inside the box
        return ranges(u + t * (new - u), box)

    trials = jax.vmap(trial)(fractions)
    drops = jax.vmap(eps)(trials) < before
    k = jnp.argmax(drops)  # the first fraction that drops is the largest one
    return jnp.where(jnp.any(drops), trials[k], u)


def newton_4x4(
    fns: CellEnergy, u: Arr, box: Ranges, n: int = 40, free: Arr | None = None
) -> tuple[Array, Array]:
    """
    Minimise eps from a starting point, crust doc section 7.

    Each step: downhill only, the newton jump, no overshoot inside the ranges. It stops
    when the slope is gone, when no step lowers eps any more, or after n steps. A pinned
    unknown stays where it starts: its slope is set to 0 and its row and column of H to
    the unit ones, so free = (1, 1, 0, 1) is the three-unknown solve with n_g held at 0.

    For autodiff one clean newton step is then taken from the answer held fixed. It
    leaves the answer where it is, the slope is already 0, but gives it the right
    derivative with respect to whatever eps depends on, d u/d theta = -H^-1 d g/d theta
    (the implicit function theorem), instead of whatever the damped steps left behind.

    :param fns: eps with its slopes and curvatures by hand, each of a (4,) point.
    :param u: Where to start, (4,).
    :param box: The ranges each unknown is kept inside.
    :param n: The most steps taken.
    :param free: 1 for an unknown that moves, 0 for one pinned, (4,). None frees all.
    :returns u: The answer, (4,).
    :returns g_norm: The size of the slope there, 0 at a true minimum.
    """
    tol = 1e-10  # a slope this small has gone
    close = (
        1e-3  # a slope this small is near enough for the autodiff step to be trusted
    )
    f = jnp.ones(4) if free is None else jnp.asarray(free, float)

    def slopes(u: Arr) -> Array:  # the four slopes, 0 for a pinned unknown
        return f * fns.slopes(u)

    def curvatures(u: Arr) -> Array:  # a pinned unknown gets a unit row and column
        return fns.curvatures(u) * jnp.outer(f, f) + jnp.diag(1 - f)

    def g_norm(u: Arr) -> Array:
        return jnp.sqrt(jnp.sum(slopes(u) ** 2))

    def step(_: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        done, u = carry
        g = slopes(u)
        new, _det = newton_step_4x4(u, g, add_lambda(curvatures(u)))  # the jump
        v = no_overshoot(fns.eps, u, new, box)  # halves the jump, keeps the ranges
        small = jnp.sqrt(jnp.sum(g**2)) < tol  # the slope is gone
        keep = done | small | jnp.all(v == u)  # or no step lowers eps any more
        return keep, jnp.where(keep, u, v)

    start = (jnp.zeros((), bool), jnp.asarray(u, float))
    _, u = lax.fori_loop(0, n, step, start)
    s = lax.stop_gradient(u)
    g = slopes(s)
    clean, _det = newton_step_4x4(s, g, curvatures(s))
    converged = jnp.sqrt(jnp.sum(g**2)) < close  # only where the loop got close
    u = jnp.where(converged, clean, u)
    return u, g_norm(u)
