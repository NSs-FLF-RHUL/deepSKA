# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Uniform matter in beta equilibrium: the core of the eos.

Neutrons, protons, electrons and muons at a baryon density n. The asymmetry delta is
the one where the charges balance, found by bisection. Then the energy density, the
neutron chemical potential and the pressure, and for the stable and causal checks
dP/dn and the sound speed. dP/dn is written out from the second derivatives of the
energy density, not differenced, with delta moving along beta equilibrium.

Shared arguments, the same in every function here:

- n: the baryon density in fm^-3.
- delta: the asymmetry (n_n - n_p) / n.
- c_sat, c_sym: the five coefficients each of symmetric matter and of the symmetry
  energy, in MeV.
- ns: the saturation density n_sat in fm^-3.
- mu: the electron chemical potential mu_e in MeV.

Contents:

- n_p, n_l, mu_e, charge, delta_beta: the composition.
- eps_lep, eps_uni: the energy density.
- mu_n_uni, p_uni: the pressure.
- dmu_e_dn_p, g1, g2, eps_dd, eps_nd, eps_nn, dp_dn, cs2: stability and causality.
- core_at: the energy density, pressure and chemical potential at one density.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import e2, hbarc, me, mmu, mn, mp
from deepska.glitch.data.metamodel_inputs import x_c
from deepska.glitch.kinds import Arr, Num
from deepska.glitch.metamodel.sums import e_bulk, e_q, e_q1, e_q2
from deepska.glitch.numerics.invert import invert


def n_p(n: Num, delta: Num) -> Num:
    """
    Calculate the proton density, n_p = n (1 - delta) / 2.

    :returns: n_p in fm^-3.
    """
    return 0.5 * n * (1 - delta)


def n_l(mu: Num, m_l: float) -> Array:
    """
    Calculate the density of one kind of lepton at a chemical potential.

    n_l(mu) = [mu^2 - m_l^2]^(3/2) / (3 pi^2 (hbar c)^3), and 0 when mu < m_l.

    :param m_l: The lepton's rest energy in MeV, me for electrons or mmu for muons.
    :returns: n_l in fm^-3.
    """
    above = mu > m_l
    x = jnp.where(above, mu**2 - m_l**2, 1.0)  # 1.0 on the 0 branch keeps grad finite
    return jnp.where(above, x**1.5, 0.0) / (3 * jnp.pi**2 * hbarc**3)


def mu_e(n: Num, delta: Num, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the electron chemical potential of beta equilibrium.

    mu_e(delta) = (m_n - m_p) + 4 E_sym(n) delta + (3/pi)^(1/3) e^2 n_p(delta)^(1/3)

    :returns: mu_e in MeV.
    """
    return (
        (mn - mp)
        + 4 * e_q(n, c_sym, ns) * delta
        + (3 / jnp.pi) ** (1 / 3) * e2 * n_p(n, delta) ** (1 / 3)
    )


def charge(n: Num, delta: Num, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the charge balance F(delta) = n_e(mu_e) + n_mu(mu_e) - n_p(delta).

    :returns: F in fm^-3, 0 at the delta of beta equilibrium.
    """
    mu = mu_e(n, delta, c_sym, ns)
    return n_l(mu, me) + n_l(mu, mmu) - n_p(n, delta)


def delta_beta(n: Num, c_sym: Arr, ns: Num) -> Array:
    """
    Find the asymmetry delta of beta equilibrium: the root of F(delta) on [0, 1].

    By the bisection of numerics/invert.py, 30 halvings and then its straight line
    finish, so the result is smooth in the inputs. Not 40 halvings: past 30 the last
    bracket is so narrow that rounding noise spoils the slope of the finish, and with
    it the derivative.

    :returns: delta, no units.
    """

    def f(delta: Num) -> Array:
        """F(delta) with n, c_sym and ns held fixed, so invert searches over delta."""
        return charge(n, delta, c_sym, ns)

    return invert(f, 0.0, 0.0, 1.0, n=30)


def eps_lep(n_lep: Num, m_l: float) -> Array:
    """
    Calculate the energy density of one kind of lepton, a free Fermi gas.

    eps_l = m_l^4 / (8 pi^2 (hbar c)^3) [x_l (2 x_l^2 + 1) sqrt(1 + x_l^2) - asinh x_l]

    with x_l = hbar c (3 pi^2 n_l)^(1/3) / m_l, and eps_l = 0 when n_l = 0.

    :param n_lep: The lepton's density in fm^-3.
    :param m_l: The lepton's rest energy in MeV, me for electrons or mmu for muons.
    :returns: eps_l in MeV fm^-3, rest energy included.
    """
    some = n_lep > 0
    x = hbarc * (3 * jnp.pi**2 * jnp.where(some, n_lep, 1.0)) ** (1 / 3) / m_l
    front = m_l**4 / (8 * jnp.pi**2 * hbarc**3)
    inside = x * (2 * x**2 + 1) * jnp.sqrt(1 + x**2) - jnp.arcsinh(x)
    return jnp.where(some, front * inside, 0.0)


def eps_uni(n: Num, delta: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the energy density of uniform matter.

    eps_uni = eps_mass + eps_nuc + eps_exch + eps_e + eps_mu, with n_n = n - n_p and

    eps_mass = n_n m_n + n_p m_p
    eps_nuc = n E(n, delta)
    eps_exch = -3/4 (3/pi)^(1/3) e^2 n_p^(4/3)

    :returns: eps_uni in MeV fm^-3.
    """
    np_ = n_p(n, delta)
    nn_ = n - np_
    mu = mu_e(n, delta, c_sym, ns)
    eps_mass = nn_ * mn + np_ * mp
    eps_nuc = n * e_bulk(n, delta, c_sat, c_sym, ns)
    eps_exch = -0.75 * (3 / jnp.pi) ** (1 / 3) * e2 * np_ ** (4 / 3)
    eps_e = eps_lep(n_l(mu, me), me)
    eps_mu = eps_lep(n_l(mu, mmu), mmu)
    return eps_mass + eps_nuc + eps_exch + eps_e + eps_mu


def mu_n_uni(n: Num, delta: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the neutron chemical potential of uniform matter.

    mu_n,uni = m_n + E(n, delta) + n [E_sat'(n) + delta^2 E_sym'(n)]
               + 2 E_sym(n) delta (1 - delta)

    :returns: mu_n,uni in MeV.
    """
    return (
        mn
        + e_bulk(n, delta, c_sat, c_sym, ns)
        + n * (e_q1(n, c_sat, ns) + delta**2 * e_q1(n, c_sym, ns))
        + 2 * e_q(n, c_sym, ns) * delta * (1 - delta)
    )


def p_uni(n: Num, delta: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the pressure of uniform matter, P_uni = n mu_n,uni - eps_uni.

    :returns: P_uni in MeV fm^-3.
    """
    return n * mu_n_uni(n, delta, c_sat, c_sym, ns) - eps_uni(
        n, delta, c_sat, c_sym, ns
    )


def dmu_e_dn_p(mu: Num) -> Array:
    """
    Calculate d mu_e / d n_p, how the leptons answer to one more proton.

    d mu_e / d n_p = [(mu_e sqrt(mu_e^2 - m_e^2) + mu_e sqrt(mu_e^2 - m_mu^2))
                     / (pi^2 (hbar c)^3)]^(-1)

    with the muon term only when mu_e > m_mu.

    :returns: d mu_e / d n_p in MeV fm^3.
    """
    muons = mu > mmu
    electron = mu * jnp.sqrt(mu**2 - me**2)
    muon = jnp.where(muons, mu * jnp.sqrt(jnp.where(muons, mu**2 - mmu**2, 1.0)), 0.0)
    return 1 / ((electron + muon) / (jnp.pi**2 * hbarc**3))


def g1(n: Num, delta: Num, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate g' = -4/3 x_c n_p^(1/3) + mu_e, with x_c = 3/4 (3/pi)^(1/3) e^2.

    :returns: g' in MeV.
    """
    return -4 / 3 * x_c * n_p(n, delta) ** (1 / 3) + mu_e(n, delta, c_sym, ns)


def g2(n: Num, delta: Num, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate g'' = -4/9 x_c n_p^(-2/3) + d mu_e / d n_p.

    :returns: g'' in MeV fm^3.
    """
    mu = mu_e(n, delta, c_sym, ns)
    return -4 / 9 * x_c * n_p(n, delta) ** (-2 / 3) + dmu_e_dn_p(mu)


def eps_dd(n: Num, delta: Num, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate d2 eps / d delta2 = 2 n E_sym + n^2 / 4 g''.

    :returns: The second derivative of eps_uni with respect to delta, MeV fm^-3.
    """
    return 2 * n * e_q(n, c_sym, ns) + n**2 / 4 * g2(n, delta, c_sym, ns)


def eps_nd(n: Num, delta: Num, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the mixed derivative d2 eps / dn d delta.

    d2 eps / dn d delta = 1/2 (m_n - m_p) + 2 delta E_sym + 2 n delta E_sym'
                          - 1/2 g' - n (1 - delta) / 4 g''

    :returns: The mixed second derivative of eps_uni, MeV.
    """
    return (
        0.5 * (mn - mp)
        + 2 * delta * e_q(n, c_sym, ns)
        + 2 * n * delta * e_q1(n, c_sym, ns)
        - 0.5 * g1(n, delta, c_sym, ns)
        - n * (1 - delta) / 4 * g2(n, delta, c_sym, ns)
    )


def eps_nn(n: Num, delta: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate d2 eps / dn2 at fixed delta.

    d2 eps / dn2 = 2 (E_sat' + delta^2 E_sym') + n (E_sat'' + delta^2 E_sym'')
                   + g'' ((1 - delta) / 2)^2

    :returns: The second derivative of eps_uni with respect to n, MeV fm^3.
    """
    return (
        2 * (e_q1(n, c_sat, ns) + delta**2 * e_q1(n, c_sym, ns))
        + n * (e_q2(n, c_sat, ns) + delta**2 * e_q2(n, c_sym, ns))
        + g2(n, delta, c_sym, ns) * ((1 - delta) / 2) ** 2
    )


def dp_dn(n: Num, delta: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate dP/dn along beta equilibrium, for the stable check.

    dP/dn = n (d2 eps/dn2 - (d2 eps/dn d delta)^2 / (d2 eps/d delta2))

    :returns: dP/dn in MeV.
    """
    return n * (
        eps_nn(n, delta, c_sat, c_sym, ns)
        - eps_nd(n, delta, c_sym, ns) ** 2 / eps_dd(n, delta, c_sym, ns)
    )


def cs2(n: Num, delta: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the sound speed squared, c_s^2 = n / (eps + P) dP/dn.

    :returns: c_s^2 in units of c^2, below 1 when causal.
    """
    eps = eps_uni(n, delta, c_sat, c_sym, ns)
    p = p_uni(n, delta, c_sat, c_sym, ns)
    return n / (eps + p) * dp_dn(n, delta, c_sat, c_sym, ns)


def core_at(n: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> tuple[Array, Array, Array]:
    """
    Calculate uniform matter in beta equilibrium at one density.

    :returns eps: eps_uni in MeV fm^-3.
    :returns p: P_uni in MeV fm^-3.
    :returns mu_n: mu_n,uni in MeV.
    """
    delta = delta_beta(n, c_sym, ns)
    return (
        eps_uni(n, delta, c_sat, c_sym, ns),
        p_uni(n, delta, c_sat, c_sym, ns),
        mu_n_uni(n, delta, c_sat, c_sym, ns),
    )
