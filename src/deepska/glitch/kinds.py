# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Type names and records used across the glitch module.

The names keep the hints short. The records group the inputs that always get passed
together (a star, a time grid, the spin settings), as ruff will not take more than 5
arguments on one function, and the outputs that come back together (uniform matter,
the crust walk, the core table, the eos), so a caller reads them by name rather than
by position.
"""

from collections.abc import Callable
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from jax import Array

# an array, jax or numpy
Arr: TypeAlias = Array | np.ndarray
# a number or an array of them
Num: TypeAlias = Arr | float | int
# any function of arrays: the eos, a drag profile, a right hand side
Fn: TypeAlias = Callable[..., Array]
# the inputs of a run by name, numbers, lists and functions mixed
Inputs: TypeAlias = dict[str, Any]
# what per_run keeps from a run
Outputs: TypeAlias = dict[str, Array]
# x, y, line format, style, what figure draws for one curve
Curve: TypeAlias = tuple[Num, Num, str, dict[str, Any]]
# line format and style for one curve
Style: TypeAlias = tuple[str, dict[str, Any]]
# cmap, norm, label, ticks, format, for one colour bar
Bar: TypeAlias = tuple[Any, Any, str, list[float] | None, str | None]
Colour: TypeAlias = tuple[float, float, float]  # red, green, blue
# swept values or results, one array per name
Grid: TypeAlias = dict[str, np.ndarray]
Rows: TypeAlias = list[tuple[Any, ...]]  # rows of a table


class Steps(NamedTuple):
    """Grid for the ode solver: where it starts, the step size, how many steps."""

    t0: Num
    dt: Num
    n: int


class StarInputs(NamedTuple):
    """Everything needed to build the star."""

    eos: Fn  # P(rho, rho_d, rho_cc), the crust eos
    rho_d: Num  # neutron drip density
    rho_cc: Num  # crust core density
    rho_min: Num  # lowest density we go to, the surface
    m0: Num  # core mass
    r0: Num  # core radius
    dom_crit: Num  # initial lag of the superfluid over the crust, for eq 23
    dr: Num  # tov step
    n_tov: int  # tov steps
    n_shells: int  # shells the inner crust superfluid is split into


class Spin(NamedTuple):
    """Inputs of the spin equations, apart from the star and the couplings."""

    omega0: Num  # the pulsars spin, rad/s
    dom_crit: Num  # initial lag of the superfluid over the crust, rad/s
    dt: Num  # time step
    n_t: int  # time steps
    profile_steps: list[int]  # the steps whose superfluid profile is kept


class Star(NamedTuple):
    """The built star, what moments gives back."""

    r_star: Array  # radius of the surface
    m_star: Array  # mass
    r_drip: Array  # radius of neutron drip
    i_tot: Array  # moment of inertia of the whole star
    i_crust_total: Array  # of the crust, from spherical shells
    i_cyl_unit: Array  # of the crust per unit height, from cylindrical shells
    h: Array  # half height of the cylinder
    i_sf: Array  # of the pinned crust superfluid
    i_core: Array  # of the core superfluid
    i_crust: Array  # of everything else
    dnu_equil: Array  # eq 23, muHz
    a_shell: Array  # moment of inertia of each shell
    x: Array  # shell radius / r_drip
    dx: Array  # spacing of x
    rho_shell: Array  # density at each shell
    m_hist: Array  # mass at every tov row
    rho_hist: Array  # density at every tov row
    n_valid: Array | int  # real tov rows before the padding


class Splines(NamedTuple):
    """The 3 splines in radius that the moments of inertia come from."""

    rho_r: tuple[Array, ...]  # rho(r)
    di_sphere: tuple[Array, ...]  # dI/dr of spherical shells
    di_cylinder: tuple[Array, ...]  # dI/dr per unit height of cylindrical shells


class Coupling(NamedTuple):
    """Mutual friction of one run."""

    b_shell: Num  # B at every shell
    b_core: Num  # B of the core


class Knot(NamedTuple):
    """One knot of the pinning tables for one case."""

    e_p: Num  # pinning energy
    n_s: Num  # superfluid neutron density
    a: Num  # lattice spacing
    length: Num  # length scale, r_n for cases A and B, xi for C


class Case(NamedTuple):
    """The constants that differ between the three pinning cases."""

    c: Num  # the constant in front of the ratio
    p: Num  # power of the lattice spacing
    q: Num  # power of the length scale


class PinningTables(NamedTuple):
    """The pinning tables of b_data, one value per inner crust domain."""

    e_pb: Arr
    e_s: Arr
    e_l: Arr
    n_s: Arr
    r_n: Arr
    xi: Arr
    a: Arr
    delta: Num
    rho_b_data: Arr


class Vela(NamedTuple):
    """The Vela 2016 glitch timing residuals, binned and lined up on the glitch."""

    t_bins: np.ndarray  # bin centres from the glitch on, s
    vela_bin: np.ndarray  # mean residual in each, ms
    dt_shift: float  # the residual at t = 0, the model gets lifted by it
    cum_data: np.ndarray  # running sum above dt_shift
    t_all: np.ndarray  # every bin incl before the glitch
    vela_all: np.ndarray
    t_raw: np.ndarray  # every single pulse
    res_raw: np.ndarray


class Drawn(NamedTuple):
    """The runs of a sweep that get drawn."""

    shown: Grid  # their inputs
    res: Arr  # their whole residuals


class PaperArrays(NamedTuple):
    """Every array the papers table and figures are drawn from."""

    dnu_equil: Num
    b_all: Arr
    delta_v_all: Arr
    f_all: Arr
    rho_b: Arr
    b_curves: Arr
    dm_m: Arr
    dm_m_dots: Arr
    r_km: Arr
    profiles: Arr
    dnu_all: Arr
    dnu_sweep: Arr
    rho_b_data: Arr
    bcore_range: Arr


class Axes(NamedTuple):
    """Limits, labels and title of one figure."""

    xlim: tuple[float | None, float | None]
    ylim: tuple[float | None, float | None]
    xlabel: str
    ylabel: str
    title: str
    xlog: bool = False
    ylog: bool = False
    offset: bool = True  # False gives plain numbers on the axes, no shared offset


class Batching(NamedTuple):
    """How a sweep is split up."""

    batch: int  # runs side by side in one batch
    cores: int  # batches going at once


class Store(NamedTuple):
    """
    What the ode solver stores: a piece every step, the whole state at a few.

    Attributes:
        keep: Fn | None, default None
            What to store every step, `None` for the whole state.
        snapshot_steps: list[int] | None, default None
            The steps to keep the whole state at, `None` for no snapshots.

    """

    keep: Fn | None = None
    snapshot_steps: list[int] | None = None


class Bulk(NamedTuple):
    """
    The metamodel inputs: two sets of five coefficients and n_sat.

    Attributes:
        csat: Arr
            E_sat, 0, K_sat, Q_sat, Z_sat in MeV; the second is always 0.
        csym: Arr
            E_sym, L_sym, K_sym, Q_sym, Z_sym in MeV.
        ns: Num
            The saturation density n_sat in fm^-3.

    """

    csat: Arr
    csym: Arr
    ns: Num


class Uniform(NamedTuple):
    """
    Uniform matter at one density, what uniform gives back. MeV and fm.

    Attributes:
        eps: Array
            The energy density.
        p: Array
            The pressure.
        delta: Array
            The isospin asymmetry, 1 - 2 x_p.
        mu_n: Array
            The neutron chemical potential.

    """

    eps: Array
    p: Array
    delta: Array
    mu_n: Array


class Cell(NamedTuple):
    """
    The four unknowns of one crust cell, crust doc section 7.

    Attributes:
        dcl: Num
            The isospin asymmetry of the cluster, 1 - 2 Y_p.
        ncl: Num
            The density inside the cluster in fm^-3.
        ng: Num
            The density of the neutron gas in fm^-3.
        a_cl: Num
            The mass number A of the cluster.

    """

    dcl: Num
    ncl: Num
    ng: Num
    a_cl: Num


class CellEnergy(NamedTuple):
    """
    What the 4x4 newton minimises: eps with its slopes and curvatures by hand.

    Attributes:
        eps: Fn
            eps(u) at a (4,) point.
        slopes: Fn
            The four d eps/d u_i at a (4,) point, (4,).
        curvatures: Fn
            The sixteen d2 eps/d u_i d u_j at a (4,) point, (4, 4).

    """

    eps: Fn
    slopes: Fn
    curvatures: Fn


class Ranges(NamedTuple):
    """
    The box the 4x4 newton keeps each unknown inside.

    Attributes:
        lo: Arr
            The lower edges, (4,).
        hi: Arr
            The upper edges, (4,).

    """

    lo: Arr
    hi: Arr


class Starts(NamedTuple):
    """
    The starting points of the crust cell minimiser at one n_b, what starts_at makes.

    Attributes:
        u: Array
            One row (dcl, n_cl, n_g, A) per start, (11, 4).
        pins: Array
            1 for an unknown that moves and 0 for one held where it starts, (11, 4).

    """

    u: Array
    pins: Array


class Lanes(NamedTuple):
    """
    Every start run at once at one n_b, what run_starts gives back.

    Attributes:
        ends: Array
            Where each start finished, (11, 4).
        eps: Array
            The cell energy density there, inf for a start that blew up.
        g_norm: Array
            The size of the slope there, 0 at a true minimum.

    """

    ends: Array
    eps: Array
    g_norm: Array


class Crust(NamedTuple):
    """
    The crust walk, one entry per n_b of the grid. MeV and fm.

    Attributes:
        nb: Array
            The grid.
        dcl: Array
            The isospin asymmetry of the best cell's cluster.
        ncl: Array
            The density inside its cluster.
        ng: Array
            The density of its neutron gas.
        a_cl: Array
            The mass number of its cluster.
        eps: Array
            The cell energy density.
        p: Array
            The pressure.
        mu_n: Array
            The neutron chemical potential.
        g_norm: Array
            The size of the slope at the answer, 0 when the minimiser got there.
        zone: Array
            0 before drip is possible, 1 while it is looked for, 2 after it.
        z: Array
            The proton number of the cluster, A Y_p.
        v: Array
            The volume fraction of the cluster.

    """

    nb: Array
    dcl: Array
    ncl: Array
    ng: Array
    a_cl: Array
    eps: Array
    p: Array
    mu_n: Array
    g_norm: Array
    zone: Array
    z: Array
    v: Array


class Core(NamedTuple):
    """
    The outer core from n_cc to 2 n_sat. MeV and fm.

    Attributes:
        nb: Array
            The grid.
        eps: Array
            The energy density.
        p: Array
            The pressure.
        delta: Array
            The isospin asymmetry, 1 - 2 x_p.
        mu_n: Array
            The neutron chemical potential.

    """

    nb: Array
    eps: Array
    p: Array
    delta: Array
    mu_n: Array


class Transition(NamedTuple):
    """
    The crust-core transition, what n_cc gives back, crust doc section 10.

    Attributes:
        n_cc: Array
            The transition density in fm^-3, nan when there is no transition.
        case: Array
            0 none, 1 a plain crossing below n_pa, 2 the line from n_pa, 3 a crossing
            above n_pa.
        n_pa: Array
            The pasta density, where the cluster fills 1/8 of the cell.
        de_pa: Array
            delta_E at n_pa in MeV per nucleon.

    """

    n_cc: Array
    case: Array
    n_pa: Array
    de_pa: Array


class Eos(NamedTuple):
    """
    One metamodel eos, crust doc section 11.

    Attributes:
        crust: Crust
            The crust walk.
        core: Core
            The outer core table.
        transition: Transition
            The crust-core transition.
        n_drip: Array
            The first grid point with a neutron gas, in fm^-3, nan when the walk
            never dripped.
        in_crust: Array
            Per crust grid point, True below n_cc.

    """

    crust: Crust
    core: Core
    transition: Transition
    n_drip: Array
    in_crust: Array


class EosTable(NamedTuple):
    """
    One metamodel eos as a plain table, with the core carried on for whole stars.

    Attributes:
        table: dict[str, np.ndarray]
            nb, eps, p and the crust flag at every point, MeV and fm.
        n_drip: float
            The neutron drip density in fm^-3.
        n_cc: float
            The crust-core transition density in fm^-3.

    """

    table: dict[str, np.ndarray]
    n_drip: float
    n_cc: float
