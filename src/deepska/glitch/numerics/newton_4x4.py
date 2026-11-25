# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Damped Newton for four unknowns, with the slopes and curvatures written by hand.

Used by the crust cell of the metamodel equation of state. Nothing here is autodiff:
the slopes g_i and the curvatures H_ij come in as functions the caller wrote out. Every
step of every start runs three checks, in this order:

1. Is H a bowl? If not, lambda is added to its diagonal until it is.
2. The jump u -> u - H^-1 g, with H^-1 written out by the minors.
3. In range and downhill: the largest fraction of the jump that lands inside the box
   with eps lower than before.

Contents:

- minor: one 3x3 minor of H.
- det_h: the determinant of H.
- is_bowl: the four leading minor tests, whether H is a bowl.
- shift: the diagonal shift that passes them.
- jump: the Newton jump.
- backtrack: cut the jump until it is in range and downhill.
- newton: the loop from one start, with pins for unknowns held fixed.
"""

import jax
import jax.numpy as jnp
from jax import Array, lax

from deepska.glitch.kinds import Arr, Fn, Num, Objective, Ranges


def minor(h: Arr, i: int, j: int) -> Array:
    """
    Calculate the minor M_ij of a 4x4 matrix: strike row i and column j, 1-based.

    :param h: The 4x4 matrix.
    :param i: The row struck out, 1 to 4.
    :param j: The column struck out, 1 to 4.
    :returns: The determinant of the 3x3 left, expanded along its first row.
    """
    rows = [r for r in range(4) if r != i - 1]
    cols = [c for c in range(4) if c != j - 1]
    m = h[jnp.array(rows)][:, jnp.array(cols)]
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


def det_h(h: Arr) -> Array:
    """
    Calculate det H = H11 M11 - H12 M12 + H13 M13 - H14 M14.

    :param h: The 4x4 matrix.
    :returns: Its determinant, expanded along the first row.
    """
    return (
        h[0, 0] * minor(h, 1, 1)
        - h[0, 1] * minor(h, 1, 2)
        + h[0, 2] * minor(h, 1, 3)
        - h[0, 3] * minor(h, 1, 4)
    )


def is_bowl(h: Arr) -> Array:
    """
    Test that H is a bowl.

    H11 > 0, H11 H22 - H12^2 > 0, M44 > 0 and det H > 0: the four leading minors.

    :param h: The 4x4 curvature matrix.
    :returns: True when all four are positive.
    """
    first = h[0, 0] > 0
    second = h[0, 0] * h[1, 1] - h[0, 1] ** 2 > 0
    third = minor(h, 4, 4) > 0
    fourth = det_h(h) > 0
    return first & second & third & fourth


def shift(h: Arr, *, lam0: float = 1e-8, n_double: int = 127) -> Array:
    """
    Add lambda to the diagonal of H until it is a bowl.

    lambda is 0, then lam0, 2 lam0, 4 lam0, ... (n_double doublings), and the first
    that passes is_bowl is kept. If none passes the last (biggest) is used, so the jump
    is tiny and backtrack keeps the point.

    The first that passes is found in one go. The bowl test is the four leading minors,
    which pass exactly when the smallest eigenvalue of H + lambda is above 0. So go
    straight to the first doubling past -(smallest eigenvalue of H), and test it with
    its two neighbours each side instead of every one.

    :param h: The 4x4 curvature matrix.
    :param lam0: The first shift tried.
    :param n_double: How many doublings of it there are.
    :returns: H with lambda added to its four diagonal entries.
    """
    lams = jnp.concatenate([jnp.zeros(1), lam0 * 2.0 ** jnp.arange(n_double)])
    need = -jnp.linalg.eigvalsh(h)[0]  # lambda has to be above this
    steps = jnp.ceil(jnp.log2(jnp.maximum(need, 1e-300) / lam0)).astype(int) + 1
    guess = jnp.where(need <= 0, 0, jnp.clip(steps, 1, n_double))
    tries = jnp.clip(guess + jnp.arange(-2, 3), 0, n_double)

    def passes(k: Num) -> Array:
        return is_bowl(h + lams[k] * jnp.eye(4))

    ok = jax.vmap(passes)(tries)
    k = jnp.where(ok.any(), tries[jnp.argmax(ok)], n_double)
    return h + lams[k] * jnp.eye(4)


def jump(u: Arr, g: Arr, h: Arr) -> Array:
    """
    Take the Newton jump u -> u - H^-1 g, with H^-1 written out by the minors.

    u1 -> u1 - ( M11 g1 - M12 g2 + M13 g3 - M14 g4) / det H
    u2 -> u2 - (-M21 g1 + M22 g2 - M23 g3 + M24 g4) / det H
    u3 -> u3 - ( M31 g1 - M32 g2 + M33 g3 - M34 g4) / det H
    u4 -> u4 - (-M41 g1 + M42 g2 - M43 g3 + M44 g4) / det H

    H is symmetric, so the row minors serve for the column ones.

    :param u: The four unknowns, (4,).
    :param g: The four slopes d eps / d u_i, (4,).
    :param h: The curvatures d2 eps / d u_i d u_j, (4, 4).
    :returns: The four unknowns after the jump, (4,).
    """
    m = [[minor(h, i, j) for j in range(1, 5)] for i in range(1, 5)]
    det = det_h(h)
    g1, g2, g3, g4 = g
    d1 = (m[0][0] * g1 - m[0][1] * g2 + m[0][2] * g3 - m[0][3] * g4) / det
    d2 = (-m[1][0] * g1 + m[1][1] * g2 - m[1][2] * g3 + m[1][3] * g4) / det
    d3 = (m[2][0] * g1 - m[2][1] * g2 + m[2][2] * g3 - m[2][3] * g4) / det
    d4 = (-m[3][0] * g1 + m[3][1] * g2 - m[3][2] * g3 + m[3][3] * g4) / det
    return u - jnp.stack([d1, d2, d3, d4])


def backtrack(eps: Fn, u: Arr, new: Arr, box: Ranges, *, n_halve: int = 40) -> Array:
    """
    Keep the largest fraction of the jump that is in range and downhill.

    The full jump is tried, then 1/2, 1/4, ... of it (n_halve fractions in all), and the
    largest that lands with all four unknowns inside the box and eps lower than before
    is taken. A nan never counts as lower. If none does, the point stays put.

    :param eps: The function being minimised, of a (4,) point.
    :param u: Where the step starts, (4,).
    :param new: Where the whole jump lands, (4,).
    :param box: The ranges each unknown is kept inside.
    :param n_halve: How many fractions of the jump are tried.
    :returns: The point kept, (4,).
    """
    before = eps(u)
    fractions = 2.0 ** -jnp.arange(n_halve)

    def trial(t: Num) -> Array:
        return u + t * (new - u)

    trials = jax.vmap(trial)(fractions)

    def inside(w: Arr) -> Array:
        return jnp.all((w >= box.lo) & (w <= box.hi))

    works = jax.vmap(inside)(trials) & (jax.vmap(eps)(trials) < before)
    k = jnp.argmax(works)  # the first that works is the largest fraction
    return jnp.where(works.any(), trials[k], u)


def newton(
    fns: Objective, u: Arr, box: Ranges, free: Arr, n: int = 30
) -> tuple[Array, Array]:
    """
    Minimise eps from one start: n steps of shift, jump, backtrack.

    A pinned unknown (free = 0) has its slope set to 0 and its row and column of H set
    to the unit ones, so it never moves: free = (1, 1, 0, 1) is the three-unknown solve
    with the third held. A start stops when |g| < 1e-10 or when no fraction lowers eps,
    and then stays put for the rest of the n steps, so every start runs the same loop
    and the starts can be vmapped.

    For derivatives with respect to whatever eps depends on, one clean undamped Newton
    step is then taken from the answer held fixed, when |g| < 1e-3 there. It leaves the
    answer where it is (the slope is already 0) but gives it the right derivative,
    d u / d theta = -H^-1 d g / d theta, the implicit function theorem, instead of
    whatever the damped steps left behind.

    :param fns: eps with its slopes and curvatures by hand, each of a (4,) point.
    :param u: Where to start, (4,). It is clipped into the box first.
    :param box: The ranges each unknown is kept inside.
    :param free: 1 for an unknown that moves, 0 for one pinned, (4,).
    :param n: The number of steps.
    :returns u: The answer, (4,).
    :returns g_norm: The size of the slope there, 0 at a true minimum.
    """
    gone_below = 1e-10  # a slope this small has gone
    close = 1e-3  # a slope this small is near enough for the clean step to be trusted

    def g_of(w: Arr) -> Array:  # the four slopes, 0 for a pinned unknown
        return free * fns.slopes(w)

    def h_of(w: Arr) -> Array:  # a pinned unknown gets a unit row and column
        return fns.curvatures(w) * jnp.outer(free, free) + jnp.diag(1 - free)

    def step(_i: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        done, w = carry
        g = g_of(w)
        new = backtrack(fns.eps, w, jump(w, g, shift(h_of(w))), box)
        gone = jnp.sqrt(jnp.sum(g**2)) < gone_below
        stop = done | gone | jnp.all(new == w)
        return stop, jnp.where(stop, w, new)

    start = (jnp.zeros((), bool), jnp.clip(u, box.lo, box.hi))
    _, u = lax.fori_loop(0, n, step, start)

    held = lax.stop_gradient(u)
    g = g_of(held)
    clean = jump(held, g, h_of(held))
    u = jnp.where(jnp.sqrt(jnp.sum(g**2)) < close, clean, u)
    return u, jnp.sqrt(jnp.sum(g_of(u) ** 2))
