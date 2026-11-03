# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The crust: one compressible liquid drop cell at each n_b, minimised by hand.

Crust doc sections 7 to 9 and 11. A cell is a cluster of A nucleons at density n_cl
and asymmetry dcl in a neutron gas of density n_g, with the electrons spread over the
cell. Its energy density eps(dcl, n_cl, n_g, A) is written out piece by piece, and so
are its four slopes and sixteen curvatures, which the 4x4 newton takes as they are:
nothing is autodiff. Each n_b is solved from eleven starts at once and the lowest eps
wins; the walk up the grid carries the last answer along as the first start.

Contents:

- v, y_p, e_gas, r_cl, a_c, f, sigma_s, sigma_c: the pieces that move with the cell.
- e_c, e_s, e_v, efs_a: coulomb, surface, curvature and their sum per nucleon.
- n_e, x, eps_e: the electrons.
- eps: the cell energy density, doc eq (epsx).
- sigma_s1_over to f2_over, mu_e: helper derivatives.
- slopes: the four d eps/d u, doc eqs (gd), (gc), (gg), (gA).
- curvatures: the sixteen d2 eps/d u d u, doc eqs (hAA) to (hgg).
- mu_n, pressure: doc section 8, P = n_b mu_n - eps.
- box_at, as_cell, run_starts, starts_at: the newton at one n_b from eleven starts.
- walk: the whole grid, three zones, the previous answer carried along.

Units: MeV and fm throughout. The surface and curvature tension numbers are held
fixed, not fitted to the bulk inputs; they live in data/nuclear_constants.py with the
other constants, and the minimiser's fixed starting points in data/crust_starts.py.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from deepska.glitch.data.crust_starts import fe56, fixed3, fixed4
from deepska.glitch.data.nuclear_constants import (
    b_s,
    beta_c,
    e2,
    hbarc,
    me,
    mn,
    mp,
    sig_top,
)
from deepska.glitch.glitch_rise_model.eos.outer_core import (
    e_bulk,
    e_q,
    e_q1,
    e_q2,
    eps_lep,
)
from deepska.glitch.kinds import (
    Arr,
    Bulk,
    Cell,
    CellEnergy,
    Crust,
    Lanes,
    Num,
    Ranges,
    Starts,
)
from deepska.glitch.numerics.newton_4x4 import newton_4x4


# the pieces that move with the cell
def v(nb: Num, ncl: Num, ng: Num) -> Array:
    """Calculate the volume fraction of the cluster, (n_b - n_g) / (n_cl - n_g)."""
    return (nb - ng) / (ncl - ng)


def y_p(dcl: Num) -> Array:
    """Calculate the proton fraction in the cluster, (1 - dcl) / 2."""
    return 0.5 * (1 - dcl)


def e_gas(ng: Num, bulk: Bulk) -> Array:
    """Calculate the energy per neutron of the gas, E(n_g, delta = 1)."""
    return e_bulk(ng, 1.0, bulk)


def r_cl(ncl: Num) -> Array:
    """Calculate the cluster radius per nucleon^(1/3), (3 / (4 pi n_cl))^(1/3), fm."""
    return (3 / (4 * jnp.pi * ncl)) ** (1 / 3)


def a_c(ncl: Num) -> Array:
    """Calculate the coulomb constant 3 e^2 / (5 r_cl) in MeV."""
    return 0.6 * e2 / r_cl(ncl)


def f(vv: Num) -> Array:
    """Calculate the wigner-seitz lattice correction 1 - 3/2 v^(1/3) + v/2."""
    return 1 - 1.5 * vv ** (1 / 3) + 0.5 * vv


def sigma_s(yp: Num) -> Array:
    """
    Calculate the surface tension in MeV fm^-2.

    sigma_0 (2^(p+1) + b_s) / (Y_p^-p + b_s + (1 - Y_p)^-p) with sigma_0 = 1.1,
    b_s = 29.91 and p = 3 held fixed. Not fitted to the bulk inputs.
    """
    return sig_top / (yp ** (-3) + b_s + (1 - yp) ** (-3))


def sigma_c(yp: Num) -> Array:
    """Calculate the curvature tension, sigma_s (beta - Y_p) / 2 in MeV fm^-1."""
    return 0.5 * sigma_s(yp) * (beta_c - yp)


def e_c(cell: Cell, nb: Num) -> Array:
    """Calculate the coulomb energy per nucleon, a_c Y_p^2 f(v) A^(2/3)."""
    vv = v(nb, cell.ncl, cell.ng)
    return a_c(cell.ncl) * y_p(cell.dcl) ** 2 * f(vv) * cell.a_cl ** (2 / 3)


def e_s(cell: Cell) -> Array:
    """Calculate the surface energy per nucleon, 4 pi r_cl^2 sigma_s A^(-1/3)."""
    return (
        4
        * jnp.pi
        * r_cl(cell.ncl) ** 2
        * sigma_s(y_p(cell.dcl))
        * cell.a_cl ** (-1 / 3)
    )


def e_v(cell: Cell) -> Array:
    """Calculate the curvature energy per nucleon, 8 pi r_cl sigma_c A^(-2/3)."""
    return 8 * jnp.pi * r_cl(cell.ncl) * sigma_c(y_p(cell.dcl)) * cell.a_cl ** (-2 / 3)


def efs_a(cell: Cell, nb: Num) -> Array:
    """Calculate the finite size energy per nucleon, coulomb + surface + curvature."""
    return e_c(cell, nb) + e_s(cell) + e_v(cell)


def n_e(cell: Cell, nb: Num) -> Array:
    """Calculate the electron density, equal to the proton density v n_cl Y_p."""
    return v(nb, cell.ncl, cell.ng) * cell.ncl * y_p(cell.dcl)


def x(cell: Cell, nb: Num) -> Array:
    """Calculate the electron relativity parameter, hbar c (3 pi^2 n_e)^(1/3) / m_e."""
    return hbarc * (3 * jnp.pi**2 * n_e(cell, nb)) ** (1 / 3) / me


def eps_e(cell: Cell, nb: Num) -> Array:
    """Calculate the electron energy density in MeV fm^-3."""
    return eps_lep(n_e(cell, nb), me)


# the cell energy density, doc eq (epsx)
def eps(cell: Cell, nb: Num, bulk: Bulk) -> Array:
    """
    Calculate the cell energy density, doc eq (epsx), in MeV fm^-3.

    Rest mass, the cluster with its finite size terms, the gas, the electrons, and the
    proton rest mass difference taken off.

    :param cell: The four unknowns.
    :param nb: Baryon density of the cell in fm^-3.
    :param bulk: The metamodel inputs.
    :returns: The energy density.
    """
    vv = v(nb, cell.ncl, cell.ng)
    cluster = e_bulk(cell.ncl, cell.dcl, bulk) + efs_a(cell, nb)
    return (
        nb * mn
        + vv * cell.ncl * cluster
        + (1 - vv) * cell.ng * e_gas(cell.ng, bulk)
        + eps_e(cell, nb)
        - (mn - mp) * vv * cell.ncl * y_p(cell.dcl)
    )


# helper derivatives
def sigma_s1_over(yp: Num) -> Array:
    """Calculate sigma_s' / sigma_s, 3 [Y_p^-4 - (1 - Y_p)^-4] over the denominator."""
    return 3 * (yp ** (-4) - (1 - yp) ** (-4)) / (yp ** (-3) + b_s + (1 - yp) ** (-3))


def sigma_c1_over(yp: Num) -> Array:
    """Calculate sigma_c' / sigma_c = sigma_s' / sigma_s - 1 / (beta - Y_p)."""
    return sigma_s1_over(yp) - 1 / (beta_c - yp)


def sigma_s2_over(yp: Num) -> Array:
    """Calculate sigma_s'' / sigma_s, 2 (sigma_s' / sigma_s)^2 minus the second term."""
    second = (
        12 * (yp ** (-5) + (1 - yp) ** (-5)) / (yp ** (-3) + b_s + (1 - yp) ** (-3))
    )
    return 2 * sigma_s1_over(yp) ** 2 - second


def sigma_c2_over(yp: Num) -> Array:
    """Calculate sigma_c'' / sigma_c from sigma_s'' / sigma_s and sigma_s' / sigma_s."""
    return sigma_s2_over(yp) - 2 * sigma_s1_over(yp) / (beta_c - yp)


def f1_over(vv: Num) -> Array:
    """Calculate f' / f with f' = 1/2 - v^(-2/3) / 2."""
    return (0.5 - 0.5 * vv ** (-2 / 3)) / f(vv)


def f2_over(vv: Num) -> Array:
    """Calculate f'' / f with f'' = v^(-5/3) / 3."""
    return (vv ** (-5 / 3) / 3) / f(vv)


def mu_e(cell: Cell, nb: Num) -> Array:
    """Calculate the electron chemical potential, m_e sqrt(1 + x^2) in MeV."""
    return me * jnp.sqrt(1 + x(cell, nb) ** 2)


# the four slopes, doc eqs (gd), (gc), (gg), (gA)
def slopes(cell: Cell, nb: Num, bulk: Bulk) -> Array:
    """
    Calculate the four slopes of eps, doc eqs (gd), (gc), (gg) and (gA), by hand.

    :param cell: The four unknowns.
    :param nb: Baryon density of the cell in fm^-3.
    :param bulk: The metamodel inputs.
    :returns: d eps / d dcl, d eps / d n_cl, d eps / d n_g, d eps / d A, (4,).
    """
    dcl, ncl, ng, a_cl = cell
    vv, yp, dd = v(nb, ncl, ng), y_p(dcl), ncl - ng
    ec, es, ev = e_c(cell, nb), e_s(cell), e_v(cell)
    esym = e_q(ncl, bulk.csym, bulk.ns)
    esat1, esym1 = e_q1(ncl, bulk.csat, bulk.ns), e_q1(ncl, bulk.csym, bulk.ns)
    w = e_bulk(ncl, dcl, bulk) + ec + es + ev  # the cluster energy per nucleon
    eg = e_gas(ng, bulk)
    eg1 = e_q1(ng, bulk.csat, bulk.ns) + e_q1(ng, bulk.csym, bulk.ns)  # d E_g / d n_g
    m = mu_e(cell, nb) - (mn - mp)
    ss, sc, f1 = sigma_s1_over(yp), sigma_c1_over(yp), f1_over(vv)

    g_dcl = vv * ncl * (2 * dcl * esym - ec / yp - (es * ss + ev * sc) / 2 - m / 2)
    g_ncl = vv * ncl * (
        esat1 + dcl**2 * esym1 + (ec - 2 * es - ev) / (3 * ncl) - vv * ec * f1 / dd
    ) - vv * ng / dd * (w - eg + yp * m)
    g_ng = (1 - vv) * (
        eg + ng * eg1 - (ncl * (w + yp * m + vv * ec * f1) - ng * eg) / dd
    )
    g_a = vv * ncl * (2 * ec - es - 2 * ev) / (3 * a_cl)
    return jnp.stack([g_dcl, g_ncl, g_ng, g_a])


# the sixteen curvatures, doc eqs (hAA) to (hgg) and the six symmetric ones
def curvatures(cell: Cell, nb: Num, bulk: Bulk) -> Array:
    """
    Calculate the sixteen curvatures of eps, doc eqs (hAA) to (hgg), by hand.

    The ten distinct ones are written out and the six symmetric ones copied across.

    :param cell: The four unknowns.
    :param nb: Baryon density of the cell in fm^-3.
    :param bulk: The metamodel inputs.
    :returns: H in the order (dcl, n_cl, n_g, A) by (dcl, n_cl, n_g, A), (4, 4).
    """
    dcl, ncl, ng, a_cl = cell
    vv, yp, dd = v(nb, ncl, ng), y_p(dcl), ncl - ng
    p = vv * ncl
    ec, es, ev = e_c(cell, nb), e_s(cell), e_v(cell)
    esym = e_q(ncl, bulk.csym, bulk.ns)
    esat1, esym1 = e_q1(ncl, bulk.csat, bulk.ns), e_q1(ncl, bulk.csym, bulk.ns)
    esat2, esym2 = e_q2(ncl, bulk.csat, bulk.ns), e_q2(ncl, bulk.csym, bulk.ns)
    w = e_bulk(ncl, dcl, bulk) + ec + es + ev
    eg = e_gas(ng, bulk)
    eg1 = e_q1(ng, bulk.csat, bulk.ns) + e_q1(ng, bulk.csym, bulk.ns)
    eg2 = e_q2(ng, bulk.csat, bulk.ns) + e_q2(ng, bulk.csym, bulk.ns)
    xx = x(cell, nb)
    r = jnp.sqrt(1 + xx**2)
    m = me * r - (mn - mp)
    ss, sc = sigma_s1_over(yp), sigma_c1_over(yp)
    ss2, sc2 = sigma_s2_over(yp), sigma_c2_over(yp)
    f1, f2 = f1_over(vv), f2_over(vv)
    bracket_d = 2 * dcl * esym - ec / yp - (es * ss + ev * sc) / 2 - m / 2  # of (gd)
    bracket_c1 = (
        esat1 + dcl**2 * esym1 + (ec - 2 * es - ev) / (3 * ncl) - vv * ec * f1 / dd
    )  # the first bracket of (gc)
    bracket_c2 = w - eg + yp * m  # the second bracket of (gc)
    mm = w + yp * m + vv * ec * f1  # the round bracket of (gg)
    bracket_g = eg + ng * eg1 - (ncl * mm - ng * eg) / dd  # the square bracket of (gg)

    h_aa = p * (-2 * ec + 4 * es + 10 * ev) / (9 * a_cl**2)
    h_da = -p / (6 * a_cl) * (4 * ec / yp - es * ss - 2 * ev * sc)
    h_ca = -(vv * ng / dd) * (2 * ec - es - 2 * ev) / (3 * a_cl) + p * (
        2 * (ec + es + ev) / (9 * ncl * a_cl) - 2 * vv * ec * f1 / (3 * a_cl * dd)
    )
    h_ga = -(1 - vv) * ncl / (3 * a_cl * dd) * (2 * ec - es - 2 * ev + 2 * vv * ec * f1)
    h_dd = p * (
        2 * esym
        + ec / (2 * yp**2)
        + (es * ss2 + ev * sc2) / 4
        + me * xx**2 / (12 * yp * r)
    )
    h_dc = -(vv * ng / dd) * bracket_d + p * (
        2 * dcl * esym1
        - ec / (3 * ncl * yp)
        + (2 * es * ss + ev * sc) / (6 * ncl)
        + vv * ec * f1 / (yp * dd)
        + me * xx**2 * ng / (6 * ncl * dd * r)
    )
    h_dg = -((1 - vv) * ncl / dd) * bracket_d + p * (
        (1 - vv) * ec * f1 / (yp * dd) + me * xx**2 * (1 - vv) / (6 * vv * dd * r)
    )
    h_cc = (
        -(2 * vv * ng / dd) * bracket_c1
        + (2 * vv * ng / dd**2) * bracket_c2
        + p
        * (
            esat2
            + dcl**2 * esym2
            + (-2 * ec + 10 * es + 4 * ev) / (9 * ncl**2)
            - 2 * vv * ec * f1 / (3 * ncl * dd)
            + 2 * vv * ec * f1 / dd**2
            + vv**2 * ec * f2 / dd**2
        )
        + me * xx**2 * vv * ng**2 * yp / (3 * ncl * dd**2 * r)
    )
    h_cg = (
        -((1 - vv) * ncl / dd) * bracket_c1
        + (((1 - vv) * ng - vv * ncl) / dd**2) * bracket_c2
        + p
        * (
            -(1 - vv) * ec * f1 / (3 * ncl * dd)
            + vv * (1 - vv) * ec * f2 / dd**2
            + (1 - 2 * vv) * ec * f1 / dd**2
        )
        + (vv * ng / dd)
        * (
            (1 - vv) * ec * f1 / dd
            + eg1
            + me * xx**2 * (1 - vv) * yp / (3 * vv * dd * r)
        )
    )
    h_gg = ((1 - vv) / dd) * bracket_g + (1 - vv) * (
        2 * eg1
        + ng * eg2
        + (eg + ng * eg1) / dd
        - (ncl * mm - ng * eg) / dd**2
        + ((1 - vv) * ncl / dd**2)
        * (2 * ec * f1 + vv * ec * f2 + me * xx**2 * yp / (3 * vv * r))
    )
    return jnp.array(
        [
            [h_dd, h_dc, h_dg, h_da],
            [h_dc, h_cc, h_cg, h_ca],
            [h_dg, h_cg, h_gg, h_ga],
            [h_da, h_ca, h_ga, h_aa],
        ]
    )


# pressure, doc section 8: P = n_b mu_n - eps
def mu_n(cell: Cell, nb: Num, bulk: Bulk) -> Array:
    """
    Calculate the neutron chemical potential, the cost of one more neutron, in MeV.

    Inner crust, n_g > 0: the gas value m_n + E(n_g, 1) + n_g [E_sat' + E_sym'] at n_g.
    Outer crust, n_g = 0: d eps / d n_b with the unknowns held fixed, only
    v = n_b / n_cl moves.

    :param cell: The four unknowns.
    :param nb: Baryon density of the cell in fm^-3.
    :param bulk: The metamodel inputs.
    :returns: mu_n.
    """
    dcl, ncl, ng, a_cl = cell
    eg1 = e_q1(ng, bulk.csat, bulk.ns) + e_q1(ng, bulk.csym, bulk.ns)
    inner = mn + e_gas(ng, bulk) + ng * eg1
    vv, yp = v(nb, ncl, ng), y_p(dcl)
    outer = (
        (1 - yp) * mn
        + yp * (mp + mu_e(cell, nb))
        + e_bulk(ncl, dcl, bulk)
        + efs_a(cell, nb)
        + vv * a_c(ncl) * yp**2 * a_cl ** (2 / 3) * 0.5 * (1 - vv ** (-2 / 3))
    )
    return jnp.where(ng > 0, inner, outer)


def pressure(cell: Cell, nb: Num, bulk: Bulk) -> Array:
    """Calculate the pressure, doc section 8: P = n_b mu_n - eps, in MeV fm^-3."""
    return nb * mu_n(cell, nb, bulk) - eps(cell, nb, bulk)


# one n_b: newton from several starts at once, keep the lowest eps
class WalkCarry(NamedTuple):
    """
    What the walk carries from one n_b to the next.

    Attributes:
        prev3: Array
            The last three-unknown answer, (4,).
        prev4: Array
            The last four-unknown answer, (4,).
        has4: Array
            A four-unknown answer exists.
        dripped: Array
            The neutron gas has appeared.

    """

    prev3: Array
    prev4: Array
    has4: Array
    dripped: Array


def box_at(nb: Num, *, tiny: float = 1e-6, a_max: float = 1e6) -> Ranges:
    """
    Make the ranges of doc section 7 at one n_b.

    0 < dcl < 1, n_b < n_cl (so v < 1), 0 <= n_g < n_b, A > 0, the open ends a touch
    inside.

    :param nb: Baryon density in fm^-3.
    :param tiny: How far inside an open range the edges sit.
    :param a_max: The upper edge for A, which has no real bound.
    :returns: The lower and upper edges, (4,) each.
    """
    lo = jnp.array([tiny, nb * (1 + tiny), 0.0, 1.0])
    hi = jnp.array([1 - tiny, 1.0, nb * (1 - tiny), a_max])
    return Ranges(lo, hi)


def as_cell(u: Arr) -> Cell:
    """Read a (4,) point as a cell."""
    return Cell(u[0], u[1], u[2], u[3])


def run_starts(nb: Num, starts: Starts, bulk: Bulk, *, n: int = 60) -> Lanes:
    """
    Run the newton from every start at once at one n_b.

    :param nb: Baryon density in fm^-3.
    :param starts: The starting points with their pins.
    :param bulk: The metamodel inputs.
    :param n: Newton steps.
    :returns: Where each start finished, eps there (inf for a start that blew up) and
        the size of the slope there.
    """

    def energy(u: Arr) -> Array:
        return eps(as_cell(u), nb, bulk)

    def slope(u: Arr) -> Array:
        return slopes(as_cell(u), nb, bulk)

    def curvature(u: Arr) -> Array:
        return curvatures(as_cell(u), nb, bulk)

    fns = CellEnergy(energy, slope, curvature)
    run = partial(newton_4x4, fns, box=box_at(nb), n=n)
    ends, g_norm = jax.vmap(run)(starts.u, free=starts.pins)  # every start at once
    e = jax.vmap(energy)(ends)
    return Lanes(ends, jnp.where(jnp.isfinite(e), e, jnp.inf), g_norm)


def starts_at(nb: Num, carry: WalkCarry) -> Starts:
    """
    Make the eleven starts of doc section 7 at one n_b, with their pins.

    Three-unknown starts, n_g pinned at 0: the previous three-unknown answer and the
    four fixed rows. Four-unknown starts: the previous four-unknown answer (the first
    time, the three-unknown one with n_g = n_b / 2), the base answer with n_g = 0.3 n_b
    and 0.8 n_b, and the three fixed rows. The base is the three-unknown answer before
    drip and the four-unknown one after.

    :param nb: Baryon density in fm^-3.
    :param carry: The previous answers and the two flags.
    :returns: One row per start, (11, 4), with 1 for a free unknown and 0 for a pinned
        one in the pins.
    """

    def with_ng(u: Arr, r: Num) -> Array:  # u with n_g set to a fraction r of n_b
        return jnp.array([u[0], u[1], r * nb, u[3]])

    three = jnp.vstack([carry.prev3[None], fixed3])
    first = jnp.where(carry.has4, carry.prev4, with_ng(carry.prev3, 0.5))
    base = jnp.where(carry.dripped, carry.prev4, carry.prev3)
    fixed = jnp.stack([with_ng(fixed4[i], fixed4[i, 2]) for i in range(len(fixed4))])
    four = jnp.vstack(
        [first[None], with_ng(base, 0.3)[None], with_ng(base, 0.8)[None], fixed]
    )
    pin3 = jnp.array([1.0, 1.0, 0.0, 1.0])  # three unknowns: n_g pinned at 0
    free4 = jnp.ones(4)  # four unknowns
    pins = jnp.vstack(
        [jnp.tile(pin3, (len(three), 1)), jnp.tile(free4, (len(four), 1))]
    )
    return Starts(jnp.vstack([three, four]), pins)


# the walk over n_b, doc sections 7 and 11
def walk(
    nbs: Arr,
    bulk: Bulk,
    *,
    n_drip_free: float = 5e-5,
    n_newton: int = 60,
    converged: float = 1e-4,
) -> Crust:
    """
    Solve the cell at every n_b of an increasing grid, doc sections 7 and 11.

    Three zones. Below n_drip_free only the three-unknown starts count. Then the
    four-unknown solve is watched for a neutron gas with a lower eps, the moment of
    drip: the lowest four-unknown lane that converged with a gas beats the
    three-unknown answer. Lanes that did not converge are left out of that test, as one
    run into the edge of the box (a one-neutron cluster, say) can sit lower without
    being a cell. After drip the four-unknown answer is taken. The previous answers are
    carried along as the first starts.

    :param nbs: The grid in fm^-3, increasing.
    :param bulk: The metamodel inputs.
    :param n_drip_free: Below this n_b drip is certainly not reached, so n_g = 0 only.
    :param n_newton: Newton steps at each n_b.
    :param converged: A slope size below this is a lane that got to a minimum.
    :returns: One entry per n_b.
    """
    n_three = 1 + len(fixed3)  # starts_at puts the three-unknown starts first

    def best(lanes: Lanes, ok: Arr) -> tuple[Array, Array, Array]:
        # the lowest eps lane of those where ok is true
        e = jnp.where(ok, lanes.eps, jnp.inf)
        k = jnp.argmin(e)
        return lanes.ends[k], e[k], lanes.g_norm[k]

    def at(carry: WalkCarry, nb: Array) -> tuple[WalkCarry, tuple[Array, ...]]:
        starts = starts_at(nb, carry)
        three = Starts(starts.u[:n_three], starts.pins[:n_three])
        four = Starts(starts.u[n_three:], starts.pins[n_three:])
        # each zone runs only the lanes it uses: zone 0 the five three-unknown ones,
        # zone 1 both groups to watch for drip, zone 2 the six four-unknown ones. One
        # cond per group, so each group's newton is compiled once. A group not run has
        # eps inf, so it is never picked
        before = nb < n_drip_free  # zone 0: n_g = 0 only
        zone = jnp.where(before, 0, jnp.where(carry.dripped, 2, 1))
        lanes3 = lax.cond(
            zone != 2,  # noqa: PLR2004, zone 2
            lambda: run_starts(nb, three, bulk, n=n_newton),
            lambda: Lanes(three.u, jnp.full(n_three, jnp.inf), jnp.zeros(n_three)),
        )
        n_four = len(four.u)
        lanes4 = lax.cond(
            zone != 0,
            lambda: run_starts(nb, four, bulk, n=n_newton),
            lambda: Lanes(four.u, jnp.full(n_four, jnp.inf), jnp.zeros(n_four)),
        )
        end3, e3, gn3 = best(lanes3, jnp.ones(n_three, bool))
        low4 = best(lanes4, jnp.ones(n_four, bool))  # the lowest four-unknown lane
        # the moment of drip: a converged lane with a neutron gas, lower than no gas
        real_gas = (lanes4.ends[:, 2] > 0) & (lanes4.g_norm < converged)
        gas_lane = best(lanes4, real_gas)
        gas = gas_lane[1] < e3
        use4 = ~before & (
            carry.dripped | gas
        )  # zone 2 after drip, or the moment of drip
        # the four-unknown answer: that gas lane at the moment of drip, after it the
        # lowest four-unknown lane, as before
        end4, e4, gn4 = (
            jnp.where(carry.dripped, a, b) for a, b in zip(low4, gas_lane, strict=True)
        )
        new = WalkCarry(
            prev3=jnp.where(before | ~carry.dripped, end3, carry.prev3),
            prev4=jnp.where(before, carry.prev4, jnp.where(use4, end4, low4[0])),
            has4=carry.has4 | ~before,
            dripped=carry.dripped | use4,
        )
        picked = (
            jnp.where(use4, end4, end3),
            jnp.where(use4, e4, e3),
            jnp.where(use4, gn4, gn3),
        )
        return new, (*picked, zone)

    no = jnp.zeros((), bool)
    carry0 = WalkCarry(prev3=fe56, prev4=fe56, has4=no, dripped=no)
    _, (ends, e, gn, zone) = lax.scan(at, carry0, nbs)
    cells = Cell(ends[:, 0], ends[:, 1], ends[:, 2], ends[:, 3])
    p = jax.vmap(pressure, in_axes=(0, 0, None))(cells, nbs, bulk)
    mu = jax.vmap(mu_n, in_axes=(0, 0, None))(cells, nbs, bulk)
    return Crust(
        nb=nbs,
        dcl=cells.dcl,
        ncl=cells.ncl,
        ng=cells.ng,
        a_cl=cells.a_cl,
        eps=e,
        p=p,
        mu_n=mu,
        g_norm=gn,
        zone=zone,
        z=cells.a_cl * y_p(cells.dcl),
        v=v(nbs, cells.ncl, cells.ng),
    )
