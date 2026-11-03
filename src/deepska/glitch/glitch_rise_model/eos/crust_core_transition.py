# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The crust-core transition n_cc, crust doc section 10, step by step.

delta_E(n_b) = (eps_cell - eps_uni) / n_b compares the best cell against uniform
matter with no proton exchange term (C10). The sphere cell is only trusted up to a
fill fraction of 1/8 (C5), so first the pasta density n_pa where v = 1/8 is found,
then:

- Case 1, delta_E(n_pa) >= 0: the crossing came earlier, a search below n_pa.
- Case 2, delta_E(n_pa) < 0: delta_E is carried on from n_pa as a straight line with
  slope (P_cell - P_uni) / n_pa^2, and n_cc is where the line reaches 0, eq (ncc).
- Case 3, the line does not climb: a plain crossing between n_pa and 0.12.
- Case 0, no crossing anywhere, or uniform matter already lower at 0.01: no
  transition, n_cc = nan and the set is dropped.

Contents:

- cell: one full minimisation at an n_b, seeded by the walk's nearest answer.
- delta_e_of: the comparison, from a cell already found.
- n_cc: the pasta condition, Neill 2026.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from deepska.glitch.data.crust_starts import fixed3
from deepska.glitch.glitch_rise_model.eos.crust import (
    Starts,
    WalkCarry,
    as_cell,
    pressure,
    run_starts,
    starts_at,
)
from deepska.glitch.glitch_rise_model.eos.outer_core import uniform
from deepska.glitch.kinds import Bulk, Cell, Crust, Num, Transition
from deepska.glitch.numerics.false_position import (
    cut_points,
    first_sign_change,
    keep_piece,
)
from deepska.glitch.numerics.linear_root import linear_root


class Best(NamedTuple):
    """
    The best cell at one n_b, what cell gives back.

    Attributes:
        eps: Array
            The cell energy density in MeV fm^-3.
        p: Array
            The pressure in MeV fm^-3.
        v: Array
            The volume fraction of the cluster.
        cell: Cell
            The four unknowns.
        g_norm: Array
            The size of the slope at the answer, 0 when the minimiser got there.

    """

    eps: Array
    p: Array
    v: Array
    cell: Cell
    g_norm: Array


def cell(nb: Num, w: Crust, bulk: Bulk) -> Best:
    """
    Run one full minimisation at an n_b, seeded by the walk's nearest answer.

    The six four-unknown starts of doc section 7, as the walk takes after drip, lowest
    eps wins. The search for n_cc runs from 0.01 up, always past drip.

    :param nb: Baryon density in fm^-3.
    :param w: The crust walk.
    :param bulk: The metamodel inputs.
    :returns: eps, P, v, the cell and the slope size at the answer.
    """
    i = jnp.searchsorted(w.nb, nb)
    prev = jnp.array([w.dcl[i], w.ncl[i], w.ng[i], w.a_cl[i]])
    yes = jnp.ones((), bool)
    starts = starts_at(nb, WalkCarry(prev3=prev, prev4=prev, has4=yes, dripped=yes))
    n_three = 1 + len(fixed3)  # starts_at puts the three-unknown starts first
    four = Starts(starts.u[n_three:], starts.pins[n_three:])
    lanes = run_starts(nb, four, bulk)
    k = jnp.argmin(lanes.eps)
    best = as_cell(lanes.ends[k])
    vv = (nb - best.ng) / (best.ncl - best.ng)
    return Best(lanes.eps[k], pressure(best, nb, bulk), vv, best, lanes.g_norm[k])


def delta_e_of(c: Best | Crust, eps_uni: Num, nb: Num, no_cell_slope: float) -> Array:
    """
    Calculate delta_E from a cell already found, one point or the whole walk.

    :param c: The cell, from cell, or the walk itself for every grid point at once.
    :param eps_uni: Uniform matter's energy density, no proton exchange term.
    :param nb: Baryon density in fm^-3.
    :param no_cell_slope: A slope above this means the minimiser found no cluster.
    :returns: delta_E, counted as core where there is no cluster.
    """
    de = (c.eps - eps_uni) / nb
    v = (nb - c.ng) / (c.ncl - c.ng) if isinstance(c, Crust) else c.v
    no_cell = (c.g_norm > no_cell_slope) | (v >= 1)
    return jnp.where(no_cell, jnp.abs(de) + 1.0, de)


def n_cc(
    w: Crust, bulk: Bulk, *, n_polish: int = 16, no_cell_slope: float = 1e-4
) -> Transition:
    """
    Find the crust-core transition by the pasta condition, Neill 2026.

    The walk has already found the cell at every grid point, so v and delta_E are known
    there for free. Each root is bracketed by the first grid pair where the sign
    changes, then polished by guarded false position, two full cell minimisations side
    by side a step. See PhD/data/writing/ncc-speedup.pdf.

    :param w: The crust walk.
    :param bulk: The metamodel inputs.
    :param n_polish: Polishing steps for each root. Each at least halves the bracket,
        about 0.0025 fm^-3 wide from the grid, so 16 gets it below 4e-8 fm^-3 even
        where v or delta_E jumps.
    :param no_cell_slope: A slope above this means the minimiser found no cluster.
    :returns: n_cc, the case it came from, n_pa and delta_E(n_pa).
    """
    # the ranges of doc section 10 in fm^-3, and the fill fraction the sphere cell is
    # trusted up to (C5)
    low, n_pa_high, high = 0.01, 0.07, 0.12
    v_pasta = 1 / 8

    def uni_eps(n: Num) -> Array:  # uniform matter, no proton exchange term (C10)
        return uniform(n, bulk, exchange=False).eps

    # v and delta_E at every grid point, from the cells the walk already found
    nb = w.nb
    v_grid = (nb - w.ng) / (w.ncl - w.ng)
    de_grid = delta_e_of(w, jax.vmap(uni_eps)(nb), nb, no_cell_slope)

    def cc_search(n_pa: Num, de_pa: Num) -> tuple[tuple[Array, ...], Array]:
        # step 3a or 3c, the search for delta_E = 0: below n_pa for case 1, above it
        # for case 3. Only one is needed, so one search with the range picked by the
        # case. n_pa goes into the grid, as one end of either range
        order = jnp.argsort(jnp.append(nb, n_pa))
        x, f = jnp.append(nb, n_pa)[order], jnp.append(de_grid, de_pa)[order]
        early = de_pa >= 0
        lo, hi = jnp.where(early, low, n_pa), jnp.where(early, n_pa, high)
        return first_sign_change(x, f, lo, hi)

    # step 0, a trusted crust at all, at the first grid point from 0.01 up: uniform
    # matter already lower, or the cluster already past 1/8 of the cell, means no
    # transition and the set is dropped
    i_low = jnp.searchsorted(nb, low)
    no_crust = (de_grid[i_low] >= 0) | (v_grid[i_low] > v_pasta)
    bracket_pa, found_pa = first_sign_change(nb, v_grid - v_pasta, low, n_pa_high)

    # steps 1 to 3 in one loop, so the cell minimiser is compiled once. Each pass runs
    # two cells side by side: passes 0 to n_polish - 1 polish n_pa (step 1), pass
    # n_polish runs the cell at n_pa itself (step 2), the passes after that polish the
    # delta_E = 0 search (step 3a or 3c)
    def one_pass(k: Array, s: tuple[Array, ...]) -> tuple[Array, ...]:
        bracket, n_pa, c_pa = s[:4], s[4], s[5:]
        at_pa = k == n_polish
        # step 1 ends: with no crossing in the range v is short of 1/8 all through it,
        # so n_pa is held at 0.07
        n_pa_now = jnp.where(found_pa, linear_root(*bracket, 0.0), n_pa_high)
        x = jnp.where(at_pa, jnp.stack([n_pa_now, n_pa_now]), cut_points(bracket))
        c = jax.vmap(partial(cell, w=w, bulk=bulk))(x)
        de_x = delta_e_of(c, jax.vmap(uni_eps)(x), x, no_cell_slope)
        f_x = jnp.where(k < n_polish, c.v - v_pasta, de_x)  # step 1, then step 3
        polished = keep_piece(bracket, x, f_x)
        # step 2, the cell against uniform matter at n_pa, then the bracket of step 3
        searched, _ = cc_search(n_pa_now, de_x[0])
        pick = partial(jnp.where, at_pa)  # this pass's answer on pass n_polish
        return (
            *(pick(a, b) for a, b in zip(searched, polished, strict=True)),
            pick(n_pa_now, n_pa),
            *(pick(a, b) for a, b in zip((c.p[0], de_x[0]), c_pa, strict=True)),
        )

    zero = jnp.zeros(())
    start = (*bracket_pa, zero, zero, zero)  # bracket, n_pa, P_cell(n_pa), delta_E_pa
    end = lax.fori_loop(0, 2 * n_polish + 1, one_pass, start)
    n_pa, p_cell, de_pa = end[4], end[5], end[6]
    ncc_search = linear_root(*end[:4], 0.0)
    _, crosses = cc_search(n_pa, de_pa)
    early = de_pa >= 0
    # step 3b, case 2: delta_E carried on as a straight line from n_pa, eq (ncc). The
    # line must climb, P_cell above P_uni, and reach 0 inside (n_pa, 0.12]
    p_uni = uniform(n_pa, bulk, exchange=False).p
    ncc_2 = n_pa - n_pa**2 * de_pa / (p_cell - p_uni)
    line_ok = (p_cell > p_uni) & (ncc_2 > n_pa) & (ncc_2 <= high)

    ncc = jnp.where(
        no_crust,
        jnp.nan,
        jnp.where(
            early | ~line_ok,
            jnp.where(crosses, ncc_search, jnp.nan),
            ncc_2,
        ),
    )
    # a search that found no sign change is case 0 too, whichever side it looked
    searched = jnp.where(crosses, jnp.where(early, 1, 3), 0)
    case = jnp.where(no_crust, 0, jnp.where(~early & line_ok, 2, searched))
    return Transition(ncc, case, n_pa, de_pa)
