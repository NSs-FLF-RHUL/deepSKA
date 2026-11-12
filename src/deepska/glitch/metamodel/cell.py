# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The crust cell: its energy, four slopes and ten curvatures.

A compressible liquid drop: one spherical cluster of A nucleons in a cell at baryon
density n_b, in a gas of neutrons and a uniform gas of electrons. The four unknowns are

    u = (d_cl, n_cl, n_g, A)

the cluster's asymmetry, the density inside it, the neutron gas density and the number
of nucleons in the cluster. The cell is the u that makes the energy density lowest.
The surface and curvature tensions are those of Grams et al 2022, held fixed.

The slopes g = d eps / du and the curvatures H = d2 eps / du du are written out by
hand, from the method document, in its shorthand. Nothing is autodiff.

Shared arguments, the same in every function here:

- u: the four unknowns (d_cl, n_cl, n_g, A), (4,); n_cl and n_g in fm^-3.
- n_b: the baryon density of the cell in fm^-3.
- c_sat, c_sym: the five coefficients each of symmetric matter and of the symmetry
  energy, in MeV.
- ns: the saturation density n_sat in fm^-3.
- v: the fraction of the cell's volume the cluster fills.
- y_p: the proton fraction of the cluster.

Contents:

- cell_parts: the volume fraction, proton fraction and densities.
- r_cl, a_c, f_v, sigma_s, sigma_c, e_fs: the finite size energy and its pieces.
- eps_cell: the energy density of the cell.
- shorthand, b_parts: the shorthand of the method for the derivatives.
- slopes: the four slopes.
- curvatures: the ten curvatures as the 4x4 matrix.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import e2, hbarc, me, mn, mp
from deepska.glitch.data.metamodel_inputs import (
    alpha_sigma_c_over_sigma,
    b_s,
    beta,
    p_sigma,
    sigma_sat,
)
from deepska.glitch.kinds import Arr, Num
from deepska.glitch.metamodel.sums import e_bulk, e_q, e_q1, e_q2
from deepska.glitch.metamodel.uniform import eps_lep


def cell_parts(
    n_b: Num, d_cl: Num, n_cl: Num, n_g: Num
) -> tuple[Array, Array, Array, Array]:
    """
    Calculate the volume fraction, the proton fraction and the two densities.

    v = (n_b - n_g) / (n_cl - n_g), Y_p = (1 - d_cl) / 2,
    n_p = n_e = v n_cl Y_p, n_n = n_b - n_p.

    :param d_cl: The cluster's asymmetry.
    :param n_cl: The density inside the cluster in fm^-3.
    :param n_g: The neutron gas density in fm^-3.
    :returns v: The fraction of the cell the cluster fills.
    :returns y_p: The proton fraction of the cluster.
    :returns n_p: The proton density of the cell, equal to the electrons', fm^-3.
    :returns n_n: The neutron density of the cell in fm^-3.
    """
    v = (n_b - n_g) / (n_cl - n_g)
    y_p = 0.5 * (1 - d_cl)
    n_p_cell = v * n_cl * y_p
    n_n_cell = n_b - n_p_cell
    return v, y_p, n_p_cell, n_n_cell


def r_cl(n_cl: Num) -> Num:
    """
    Calculate r_cl = (3 / (4 pi n_cl))^(1/3), the radius one nucleon takes up.

    :param n_cl: The density inside the cluster in fm^-3.
    :returns: r_cl in fm. The cluster's radius is r_cl A^(1/3).
    """
    return (3 / (4 * jnp.pi * n_cl)) ** (1 / 3)


def a_c(n_cl: Num) -> Num:
    """
    Calculate the Coulomb coefficient a_c = 3/5 e^2 / r_cl.

    :param n_cl: The density inside the cluster in fm^-3.
    :returns: a_c in MeV.
    """
    return 0.6 * e2 / r_cl(n_cl)


def f_v(v: Num) -> Num:
    """
    Calculate the Coulomb screening f(v) = 1 - 3/2 v^(1/3) + 1/2 v.

    :returns: f, 1 for a lone cluster and 0 when the cluster fills the cell.
    """
    return 1 - 1.5 * v ** (1 / 3) + 0.5 * v


def sigma_s(y_p: Num) -> Num:
    """
    Calculate the surface tension at a proton fraction.

    sigma_s = sigma_sat (2^(p+1) + b_s) / (Y_p^-p + b_s + (1 - Y_p)^-p)

    :returns: sigma_s in MeV fm^-2.
    """
    return (
        sigma_sat
        * (2 ** (p_sigma + 1) + b_s)
        / (y_p ** (-p_sigma) + b_s + (1 - y_p) ** (-p_sigma))
    )


def sigma_c(y_p: Num) -> Num:
    """
    Calculate the curvature tension at a proton fraction.

    sigma_c = alpha sigma_c,sat / sigma_sat sigma_s (beta - Y_p)

    :returns: sigma_c in MeV fm^-1.
    """
    return alpha_sigma_c_over_sigma * sigma_s(y_p) * (beta - y_p)


def e_fs(v: Num, y_p: Num, n_cl: Num, a: Num) -> Num:
    """
    Calculate the finite size energy per nucleon: Coulomb, surface and curvature.

    E_FS / A = a_c Y_p^2 f(v) A^(2/3) + 4 pi r_cl^2 sigma_s A^(-1/3)
               + 8 pi r_cl sigma_c A^(-2/3)

    :param n_cl: The density inside the cluster in fm^-3.
    :param a: The number of nucleons in the cluster, A.
    :returns: E_FS / A in MeV.
    """
    coulomb = a_c(n_cl) * y_p**2 * f_v(v) * a ** (2 / 3)
    surface = 4 * jnp.pi * r_cl(n_cl) ** 2 * sigma_s(y_p) * a ** (-1 / 3)
    curvature = 8 * jnp.pi * r_cl(n_cl) * sigma_c(y_p) * a ** (-2 / 3)
    return coulomb + surface + curvature


def eps_cell(u: Arr, n_b: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the energy density of the cell.

    eps = eps_mass + eps_cluster + eps_gas + eps_e, with

    eps_mass = n_n m_n + n_p m_p
    eps_cluster = v n_cl [E(n_cl, d_cl) + E_FS / A]
    eps_gas = (1 - v) n_g E(n_g, 1)

    and eps_e the electrons at n_e = n_p.

    :returns: eps in MeV fm^-3.
    """
    d_cl, n_cl, n_g, a = u
    v, y_p, n_p_cell, n_n_cell = cell_parts(n_b, d_cl, n_cl, n_g)
    eps_mass = n_n_cell * mn + n_p_cell * mp
    eps_cluster = (
        v * n_cl * (e_bulk(n_cl, d_cl, c_sat, c_sym, ns) + e_fs(v, y_p, n_cl, a))
    )
    eps_gas = (1 - v) * n_g * e_bulk(n_g, 1.0, c_sat, c_sym, ns)
    eps_e = eps_lep(n_p_cell, me)
    return eps_mass + eps_cluster + eps_gas + eps_e


def shorthand(u: Arr, n_b: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> dict[str, Array]:
    """
    Calculate the shorthand of the method for the derivatives.

    p = v n_cl, d = n_cl - n_g, W = E(n_cl, d_cl) + E_C + E_S + E_V with E_C, E_S and
    E_V the Coulomb, surface and curvature energies per nucleon, E_g = E(n_g, 1) with
    E_g' and E_g'' (sat + sym at n_g), r = sqrt(1 + x^2) for the electrons,
    m = m_e r - (m_n - m_p), the sums at n_cl, and s1, s2, c1, c2, f1, f2, the
    logarithmic derivatives of sigma_s, sigma_c and f.

    :returns: Every piece by name.
    """
    d_cl, n_cl, n_g, a = u
    v, y_p, n_e, _ = cell_parts(n_b, d_cl, n_cl, n_g)

    e_c = a_c(n_cl) * y_p**2 * f_v(v) * a ** (2 / 3)
    e_s = 4 * jnp.pi * r_cl(n_cl) ** 2 * sigma_s(y_p) * a ** (-1 / 3)
    e_v = 8 * jnp.pi * r_cl(n_cl) * sigma_c(y_p) * a ** (-2 / 3)

    x = hbarc * (3 * jnp.pi**2 * n_e) ** (1 / 3) / me
    r = jnp.sqrt(1 + x**2)

    bottom = y_p ** (-3) + b_s + (1 - y_p) ** (-3)
    s1 = 3 * (y_p ** (-4) - (1 - y_p) ** (-4)) / bottom
    s2 = 2 * s1**2 - 12 * (y_p ** (-5) + (1 - y_p) ** (-5)) / bottom

    return {
        "d_cl": d_cl,
        "n_cl": n_cl,
        "n_g": n_g,
        "a": a,
        "v": v,
        "y_p": y_p,
        "p": v * n_cl,
        "d": n_cl - n_g,
        "e_c": e_c,
        "e_s": e_s,
        "e_v": e_v,
        "w": e_bulk(n_cl, d_cl, c_sat, c_sym, ns) + e_c + e_s + e_v,
        "e_g": e_bulk(n_g, 1.0, c_sat, c_sym, ns),
        "e_g1": e_q1(n_g, c_sat, ns) + e_q1(n_g, c_sym, ns),
        "e_g2": e_q2(n_g, c_sat, ns) + e_q2(n_g, c_sym, ns),
        "x": x,
        "r": r,
        "m": me * r - (mn - mp),
        "e_sym": e_q(n_cl, c_sym, ns),
        "e_sat1": e_q1(n_cl, c_sat, ns),
        "e_sym1": e_q1(n_cl, c_sym, ns),
        "e_sat2": e_q2(n_cl, c_sat, ns),
        "e_sym2": e_q2(n_cl, c_sym, ns),
        "s1": s1,
        "s2": s2,
        "c1": s1 - 1 / (beta - y_p),
        "c2": s2 - 2 * s1 / (beta - y_p),
        "f1": (0.5 - 0.5 * v ** (-2 / 3)) / f_v(v),
        "f2": (v ** (-5 / 3) / 3) / f_v(v),
    }


def b_parts(h: dict[str, Array]) -> tuple[Array, Array, Array, Array, Array]:
    """
    Calculate B_delta, B_c1, B_c2, M and B_g of the method, from its shorthand.

    :param h: The shorthand, from shorthand.
    :returns: B_delta, B_c1, B_c2, M and B_g, in that order, in MeV.
    """
    d_cl, n_cl, n_g, v, y_p, d = (
        h["d_cl"],
        h["n_cl"],
        h["n_g"],
        h["v"],
        h["y_p"],
        h["d"],
    )
    e_c, e_s, e_v, m = h["e_c"], h["e_s"], h["e_v"], h["m"]
    b_d = (
        2 * d_cl * h["e_sym"] - e_c / y_p - (e_s * h["s1"] + e_v * h["c1"]) / 2 - m / 2
    )
    b_c1 = (
        h["e_sat1"]
        + d_cl**2 * h["e_sym1"]
        + (e_c - 2 * e_s - e_v) / (3 * n_cl)
        - v * e_c * h["f1"] / d
    )
    b_c2 = h["w"] - h["e_g"] + y_p * m
    big_m = h["w"] + y_p * m + v * e_c * h["f1"]
    b_g = h["e_g"] + n_g * h["e_g1"] - (n_cl * big_m - n_g * h["e_g"]) / d
    return b_d, b_c1, b_c2, big_m, b_g


def slopes(u: Arr, n_b: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the four slopes g = d eps / du, by hand.

    g_delta = p B_delta
    g_c = p B_c1 - v n_g / d B_c2
    g_g = (1 - v) B_g
    g_A = p (2 E_C - E_S - 2 E_V) / (3 A)

    :returns: The slopes in the order (d_cl, n_cl, n_g, A), (4,).
    """
    h = shorthand(u, n_b, c_sat, c_sym, ns)
    b_d, b_c1, b_c2, _, b_g = b_parts(h)
    p, v, n_g, d, a = h["p"], h["v"], h["n_g"], h["d"], h["a"]
    g_d = p * b_d
    g_c = p * b_c1 - v * n_g / d * b_c2
    g_g = (1 - v) * b_g
    g_a = p * (2 * h["e_c"] - h["e_s"] - 2 * h["e_v"]) / (3 * a)
    return jnp.array([g_d, g_c, g_g, g_a])


def curvatures(u: Arr, n_b: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the ten curvatures H = d2 eps / du du, by hand.

    The ten of the method document, one line each below, put into the symmetric 4x4
    matrix in the order (d_cl, n_cl, n_g, A).

    :returns: H, (4, 4).
    """
    h = shorthand(u, n_b, c_sat, c_sym, ns)
    b_d, b_c1, b_c2, big_m, b_g = b_parts(h)
    d_cl, n_cl, n_g, a = h["d_cl"], h["n_cl"], h["n_g"], h["a"]
    v, y_p, p, d, x, r = h["v"], h["y_p"], h["p"], h["d"], h["x"], h["r"]
    e_c, e_s, e_v, f1, f2 = h["e_c"], h["e_s"], h["e_v"], h["f1"], h["f2"]
    e_g, e_g1, e_g2 = h["e_g"], h["e_g1"], h["e_g2"]
    mx = me * x**2  # m_e x^2

    h_aa = p * (-2 * e_c + 4 * e_s + 10 * e_v) / (9 * a**2)
    h_da = -p / (6 * a) * (4 * e_c / y_p - e_s * h["s1"] - 2 * e_v * h["c1"])
    h_ca = -v * n_g / d * (2 * e_c - e_s - 2 * e_v) / (3 * a) + p * (
        2 * (e_c + e_s + e_v) / (9 * n_cl * a) - 2 * v * e_c * f1 / (3 * a * d)
    )
    h_ga = -(1 - v) * n_cl / (3 * a * d) * (2 * e_c - e_s - 2 * e_v + 2 * v * e_c * f1)
    h_dd = p * (
        2 * h["e_sym"]
        + e_c / (2 * y_p**2)
        + (e_s * h["s2"] + e_v * h["c2"]) / 4
        + mx / (12 * y_p * r)
    )
    h_dc = -v * n_g / d * b_d + p * (
        2 * d_cl * h["e_sym1"]
        - e_c / (3 * n_cl * y_p)
        + (2 * e_s * h["s1"] + e_v * h["c1"]) / (6 * n_cl)
        + v * e_c * f1 / (y_p * d)
        + mx * n_g / (6 * n_cl * d * r)
    )
    h_dg = -(1 - v) * n_cl / d * b_d + p * (
        (1 - v) * e_c * f1 / (y_p * d) + mx * (1 - v) / (6 * v * d * r)
    )
    h_cc = (
        -2 * v * n_g / d * b_c1
        + 2 * v * n_g / d**2 * b_c2
        + p
        * (
            h["e_sat2"]
            + d_cl**2 * h["e_sym2"]
            + (-2 * e_c + 10 * e_s + 4 * e_v) / (9 * n_cl**2)
            - 2 * v * e_c * f1 / (3 * n_cl * d)
            + 2 * v * e_c * f1 / d**2
            + v**2 * e_c * f2 / d**2
        )
        + mx * v * n_g**2 * y_p / (3 * n_cl * d**2 * r)
    )
    h_cg = (
        -(1 - v) * n_cl / d * b_c1
        + ((1 - v) * n_g - v * n_cl) / d**2 * b_c2
        + p
        * (
            -(1 - v) * e_c * f1 / (3 * n_cl * d)
            + v * (1 - v) * e_c * f2 / d**2
            + (1 - 2 * v) * e_c * f1 / d**2
        )
        + v
        * n_g
        / d
        * ((1 - v) * e_c * f1 / d + e_g1 + mx * (1 - v) * y_p / (3 * v * d * r))
    )
    h_gg = (1 - v) / d * b_g + (1 - v) * (
        2 * e_g1
        + n_g * e_g2
        + (e_g + n_g * e_g1) / d
        - (n_cl * big_m - n_g * e_g) / d**2
        + (1 - v) * n_cl / d**2 * (2 * e_c * f1 + v * e_c * f2 + mx * y_p / (3 * v * r))
    )
    return jnp.array(
        [
            [h_dd, h_dc, h_dg, h_da],
            [h_dc, h_cc, h_cg, h_ca],
            [h_dg, h_cg, h_gg, h_ga],
            [h_da, h_ca, h_ga, h_aa],
        ]
    )
