# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The crust: the cell at every density, drip and the pressure.

At each of 600 densities the cell is found by the damped Newton of
numerics/newton_4x4.py from eight fixed starts, and the lowest converged one with a
proton wins. Every density is solved on its own, from the same eight starts, so the
whole crust is one vmap over the densities with a vmap over the starts inside it.

Shared arguments, the same in every function here:

- u: the four unknowns of a cell (d_cl, n_cl, n_g, A), (4,).
- n_b: the baryon density of the cell in fm^-3.
- c_sat, c_sym: the five coefficients each of symmetric matter and of the symmetry
  energy, in MeV.
- ns: the saturation density n_sat in fm^-3.

Contents:

- box_at: the ranges the Newton never leaves.
- starts_at: the eight starts at one density.
- winner: the cell at one density.
- mu_n_cell: the neutron chemical potential of a cell.
- p_cell: the pressure of a cell.
- crust: the winner at every density, with drip, mu_n and P.
"""

import jax
import jax.numpy as jnp
from jax import Array, lax

from deepska.glitch.data.constants import hbarc, me, mn, mp
from deepska.glitch.data.metamodel_inputs import crust_free, crust_starts
from deepska.glitch.kinds import Arr, Num, Objective, Ranges
from deepska.glitch.metamodel.cell import (
    a_c,
    cell_parts,
    curvatures,
    e_fs,
    eps_cell,
    slopes,
)
from deepska.glitch.metamodel.sums import e_bulk, e_q1
from deepska.glitch.numerics.newton_4x4 import jump, newton


def box_at(n_b: Num) -> Ranges:
    """
    Make the box at one density, the ranges the Newton never leaves.

    1e-6 <= d_cl <= 1 - 1e-6, n_b (1 + 1e-6) <= n_cl <= 0.3,
    0 <= n_g <= n_b (1 - 1e-6), 1 <= A <= 1e6.

    :returns: The lower and upper edges, (4,) each.
    """
    low = jnp.array([1e-6, n_b * (1 + 1e-6), 0.0, 1.0])
    high = jnp.array([1 - 1e-6, 0.3, n_b * (1 - 1e-6), 1e6])
    return Ranges(low, high)


def starts_at(n_b: Num) -> Array:
    """
    Make the eight starts at one density.

    The n_g column of crust_starts is a fraction of n_b, so it is multiplied by n_b,
    and each start is clipped into the box.

    :returns: The eight starts, (8, 4).
    """
    rows = crust_starts.at[:, 2].multiply(n_b)
    box = box_at(n_b)
    return jnp.clip(rows, box.lo, box.hi)


def winner(n_b: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> tuple[Array, Array, Array]:
    """
    Find the cell at one density: Newton from all eight starts, then who wins.

    A start may win only if |g| < 1e-4 (it converged) and Z = A (1 - d_cl) / 2 >= 1 (it
    has a proton); of those the lowest eps wins. If none passes, the lowest eps is
    taken anyway and its |g| is the warning: such a point can never hold the join and
    never enters the table.

    For derivatives with respect to the inputs, one clean Newton step is taken from the
    winner only, after it is chosen. Which start wins is held fixed, so a losing start,
    which may not have converged, never reaches the derivative.

    :returns u: The winning cell, (4,).
    :returns eps: Its energy density in MeV fm^-3.
    :returns g_norm: The size of its slope, below 1e-4 when it converged.
    """
    converged = 1e-4  # a start may win only with a slope below this
    close = 1e-3  # a slope this small is near enough for the clean step to be trusted

    def eps_u(u: Arr) -> Array:
        return eps_cell(u, n_b, c_sat, c_sym, ns)

    def slopes_u(u: Arr) -> Array:
        return slopes(u, n_b, c_sat, c_sym, ns)

    def curvatures_u(u: Arr) -> Array:
        return curvatures(u, n_b, c_sat, c_sym, ns)

    fns = Objective(eps_u, slopes_u, curvatures_u)
    box = box_at(n_b)

    def one(start: Arr, free: Arr) -> tuple[Array, Array]:
        return newton(fns, start, box, free, n=30)

    # the choice is made on held values, so a start that ended on nan (it lost) can
    # never put a nan into the winner's derivatives
    ends, g_norm = lax.stop_gradient(jax.vmap(one)(starts_at(n_b), crust_free))

    eps = jax.vmap(eps_u)(ends)
    eps = jnp.where(jnp.isfinite(eps), eps, jnp.inf)

    z = ends[:, 3] * (1 - ends[:, 0]) / 2
    ok = (g_norm < converged) & (z >= 1) & jnp.isfinite(eps)
    k = jnp.argmin(jnp.where(ok | ~ok.any(), eps, jnp.inf))

    # derivatives with respect to the inputs: one clean Newton step from the winner
    # held fixed (same value, right derivative), only where it converged
    held, free = ends[k], crust_free[k]
    g = free * slopes_u(held)
    h = curvatures_u(held) * jnp.outer(free, free)
    h = h + jnp.diag(1 - free)
    near = jnp.sqrt(jnp.sum(lax.stop_gradient(g) ** 2)) < close
    h_safe = jnp.where(near, h, jnp.eye(4))  # a winner that did not converge: no step
    g_safe = jnp.where(near, g, 0.0)
    u_win = jump(held, g_safe, h_safe)
    return u_win, eps_u(u_win), g_norm[k]


def mu_n_cell(u: Arr, n_b: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the neutron chemical potential of a cell.

    n_g > 0: mu_n = m_n + E(n_g, 1) + n_g [E_sat'(n_g) + E_sym'(n_g)]
    n_g = 0: mu_n = (1 - Y_p) m_n + Y_p (m_p + mu_e) + E(n_cl, d_cl) + E_FS / A
                    + v a_c Y_p^2 A^(2/3) 1/2 (1 - v^(-2/3))

    with mu_e = m_e sqrt(1 + x^2), the electrons' at n_e.

    :returns: mu_n in MeV.
    """
    d_cl, n_cl, n_g, a = u
    v, y_p, n_e, _ = cell_parts(n_b, d_cl, n_cl, n_g)

    gas = (
        mn
        + e_bulk(n_g, 1.0, c_sat, c_sym, ns)
        + n_g * (e_q1(n_g, c_sat, ns) + e_q1(n_g, c_sym, ns))
    )

    x = hbarc * (3 * jnp.pi**2 * n_e) ** (1 / 3) / me
    mu_el = me * jnp.sqrt(1 + x**2)
    no_gas = (
        (1 - y_p) * mn
        + y_p * (mp + mu_el)
        + e_bulk(n_cl, d_cl, c_sat, c_sym, ns)
        + e_fs(v, y_p, n_cl, a)
        + v * a_c(n_cl) * y_p**2 * a ** (2 / 3) * 0.5 * (1 - v ** (-2 / 3))
    )
    return jnp.where(n_g > 0, gas, no_gas)


def p_cell(u: Arr, n_b: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the pressure of a cell, P = n_b mu_n - eps.

    :returns: P in MeV fm^-3.
    """
    return n_b * mu_n_cell(u, n_b, c_sat, c_sym, ns) - eps_cell(
        u, n_b, c_sat, c_sym, ns
    )


def crust(c_sat: Arr, c_sym: Arr, ns: Num) -> dict[str, Array]:
    """
    Find the crust: the winner at every one of 600 densities.

    The densities run from 1e-9 to 0.16 fm^-3, evenly spaced in log (73 per decade, a
    3% step), all vmapped.

    Drip is where mu_n of the cell reaches m_n: a straight line, in log n_b, between
    the last grid point without gas and the first whose winner has n_g > 0. So it moves
    smoothly with the inputs, not in 3% grid steps. A walk that never drips is a walk
    the minimiser got wrong: dripped is False and the draw is dropped with "no drip".

    :returns: By name: n_b (600,), cells (600, 4), eps, g_norm, mu_n and p (600,) each,
        dripped (True or False), n_drip in fm^-3 and p_drip, P at drip, on the same
        straight line.
    """
    n_b = jnp.logspace(-9, jnp.log10(0.16), 600)

    def win_at(n: Num) -> tuple[Array, Array, Array]:
        return winner(n, c_sat, c_sym, ns)

    cells, eps, g_norm = jax.vmap(win_at)(n_b)

    def mu_at(u: Arr, n: Num) -> Array:
        return mu_n_cell(u, n, c_sat, c_sym, ns)

    mu_n = jax.vmap(mu_at)(cells, n_b)
    p = n_b * mu_n - eps

    gas = cells[:, 2] > 0
    dripped = gas.any()
    # the first grid point with gas, i, and the one before it; which pair is held fixed
    i = lax.stop_gradient(jnp.maximum(jnp.argmax(gas), 1))
    t = jnp.clip((mn - mu_n[i - 1]) / (mu_n[i] - mu_n[i - 1]), 0.0, 1.0)
    log_n_drip = jnp.log(n_b[i - 1]) + t * (jnp.log(n_b[i]) - jnp.log(n_b[i - 1]))
    n_drip = jnp.where(dripped, jnp.exp(log_n_drip), jnp.nan)
    p_drip = jnp.where(dripped, p[i - 1] + t * (p[i] - p[i - 1]), jnp.nan)

    return {
        "n_b": n_b,
        "cells": cells,
        "eps": eps,
        "g_norm": g_norm,
        "mu_n": mu_n,
        "p": p,
        "dripped": dripped,
        "n_drip": n_drip,
        "p_drip": p_drip,
    }
