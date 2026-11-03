# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The crust-core transition n_cc, crust doc section 10, step by step.

delta_E(n_b) = (eps_cell - eps_uni) / n_b compares the best cell against uniform
matter with no proton exchange term (C10). The sphere cell is only trusted up to a
fill fraction of 1/8 (C5), so first the pasta density n_pa where v = 1/8 is found,
then:

- Case 1, delta_E(n_pa) >= 0: the crossing came earlier, a plain bisection below n_pa.
- Case 2, delta_E(n_pa) < 0: delta_E is carried on from n_pa as a straight line with
  slope (P_cell - P_uni) / n_pa^2, and n_cc is where the line reaches 0, eq (ncc).
- Case 3, the line does not climb: a plain crossing between n_pa and 0.12.
- Case 0, no crossing anywhere, or uniform matter already lower at 0.01: no
  transition, n_cc = nan and the set is dropped.

Contents:

- cell: one full minimisation at an n_b, seeded by the walk's nearest answer.
- delta_e: the comparison.
- plain_crossing: the Neill 2023 way, delta_E = 0 by bisection.
- n_cc: the pasta condition, Neill 2026.
"""

from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.eos.crust import (
    WalkCarry,
    as_cell,
    pressure,
    run_starts,
    starts_at,
)
from deepska.glitch.glitch_rise_model.eos.outer_core import uniform
from deepska.glitch.kinds import Bulk, Cell, Crust, Num, Transition
from deepska.glitch.numerics.invert import invert


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

    The eleven starts of doc section 7 with both flags on, lowest eps wins.

    :param nb: Baryon density in fm^-3.
    :param w: The crust walk.
    :param bulk: The metamodel inputs.
    :returns: eps, P, v, the cell and the slope size at the answer.
    """
    i = jnp.searchsorted(w.nb, nb)
    prev = jnp.array([w.dcl[i], w.ncl[i], w.ng[i], w.a_cl[i]])
    yes = jnp.ones((), bool)
    starts = starts_at(nb, WalkCarry(prev3=prev, prev4=prev, has4=yes, dripped=yes))
    lanes = run_starts(nb, starts, bulk)
    k = jnp.argmin(lanes.eps)
    best = as_cell(lanes.ends[k])
    vv = (nb - best.ng) / (best.ncl - best.ng)
    return Best(lanes.eps[k], pressure(best, nb, bulk), vv, best, lanes.g_norm[k])


def delta_e(nb: Num, w: Crust, bulk: Bulk, *, no_cell_slope: float = 1e-4) -> Array:
    """
    Calculate (eps_cell - eps_uni) / n_b, eq (Delta), in MeV per nucleon.

    eps_uni has no proton exchange term (C10). Where the cell minimiser finds no
    cluster, the slope not gone or the cluster filling the cell, there is no crust, so
    the cell cannot win there: it counts as core, a positive delta_E, rather than
    trusting a false minimum.

    :param nb: Baryon density in fm^-3.
    :param w: The crust walk.
    :param bulk: The metamodel inputs.
    :param no_cell_slope: A slope above this means the minimiser found no cluster.
    :returns: delta_E.
    """
    eps_uni = uniform(nb, bulk, exchange=False).eps
    c = cell(nb, w, bulk)
    de = (c.eps - eps_uni) / nb
    no_cell = (c.g_norm > no_cell_slope) | (c.v >= 1)
    return jnp.where(no_cell, jnp.abs(de) + 1.0, de)


def plain_crossing(
    w: Crust, bulk: Bulk, *, lo: float = 0.01, hi: float = 0.10, n_bisect: int = 30
) -> Array:
    """
    Find n_cc where eps_cell = eps_uni by bisection, Neill 2023.

    :param w: The crust walk.
    :param bulk: The metamodel inputs.
    :param lo: The lower bracket in fm^-3.
    :param hi: The upper bracket in fm^-3.
    :param n_bisect: Halvings of the bracket.
    :returns: n_cc in fm^-3, between lo and hi.
    """
    de = partial(delta_e, w=w, bulk=bulk)
    return invert(de, 0.0, lo, hi, n=n_bisect)


def n_cc(
    w: Crust, bulk: Bulk, *, n_bisect: int = 30, no_cell_slope: float = 1e-4
) -> Transition:
    """
    Find the crust-core transition by the pasta condition, Neill 2026.

    :param w: The crust walk.
    :param bulk: The metamodel inputs.
    :param n_bisect: Halvings of each bracket.
    :param no_cell_slope: A slope above this means the minimiser found no cluster.
    :returns: n_cc, the case it came from, n_pa and delta_E(n_pa).
    """
    # the brackets of doc section 10 in fm^-3, and the fill fraction the sphere cell
    # is trusted up to (C5)
    low, n_pa_high, high = 0.01, 0.07, 0.12
    v_pasta = 1 / 8
    de = partial(delta_e, w=w, bulk=bulk, no_cell_slope=no_cell_slope)

    def fill(nb: Num) -> Array:
        return cell(nb, w, bulk).v

    # step 1, the pasta density: where v reaches 1/8 between 0.01 and 0.07, as v only
    # grows with n_b
    n_pa = invert(fill, v_pasta, low, n_pa_high, n=n_bisect)
    # step 2, the cell against uniform matter there
    de_pa = de(n_pa)
    # step 3, case 1: the crossing came earlier, a plain crossing between 0.01 and n_pa
    ncc_1 = invert(de, 0.0, low, n_pa, n=n_bisect)
    # step 4, case 2: delta_E carried on as a straight line from n_pa, eq (ncc). The
    # line must climb, P_cell above P_uni, and reach 0 inside the bracket. If not, fall
    # back to the plain crossing between n_pa and 0.12; and with no sign change there
    # either there is no transition, so n_cc is nan and the set is dropped
    p_cell = cell(n_pa, w, bulk).p
    p_uni = uniform(n_pa, bulk, exchange=False).p
    ncc_2 = n_pa - n_pa**2 * de_pa / (p_cell - p_uni)
    de_hi = de(high)
    ncc_3 = invert(de, 0.0, n_pa, high, n=n_bisect)
    crosses = de_pa * de_hi < 0
    fallback = jnp.where(crosses, ncc_3, jnp.nan)
    line_ok = (p_cell > p_uni) & (ncc_2 > n_pa) & (ncc_2 <= high)
    case_2 = jnp.where(line_ok, ncc_2, fallback)
    # no crust at 0.01, uniform matter already lower there: no transition at all
    no_crust = de(low) >= 0
    early = de_pa >= 0
    ncc = jnp.where(no_crust, jnp.nan, jnp.where(early, ncc_1, case_2))
    case = jnp.where(
        no_crust,
        0,
        jnp.where(early, 1, jnp.where(line_ok, 2, jnp.where(crosses, 3, 0))),
    )
    return Transition(ncc, case, n_pa, de_pa)
