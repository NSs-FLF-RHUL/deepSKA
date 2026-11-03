# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The outer core: uniform n, p, e, mu matter from the ten metamodel inputs.

Margueron, Hoffmann Casali & Gulminelli 2018 (PRC 97, 025805) write the energy per
nucleon as two Taylor sums in chi = (n - n_sat) / (3 n_sat), one for symmetric matter
and one for the symmetry energy, each term with a low density correction u_k so the
sums go to 0 with n. Outer core doc sections 2 to 5. The crust reuses the sums and
the lepton gas.

Contents:

- terms: k and k! for the sums.
- e_q: one of the two sums, E_sat with csat or E_sym with csym.
- e_q1, e_q2: its first and second density derivatives, by hand.
- e_bulk: the energy per nucleon E_sat + E_sym delta^2.
- n_lep, eps_lep: a free relativistic lepton gas.
- composition: delta from charge neutrality, by bisection.
- mu_n_outer: the neutron chemical potential.
- uniform: eps, P, delta and mu_n at one density.

Units: MeV and fm throughout. The star code converts to cgs. The constants live in
data/nuclear_constants.py.
"""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.nuclear_constants import b_low, e2, hbarc, me, mmu, mn, mp
from deepska.glitch.kinds import Arr, Bulk, Num, Uniform
from deepska.glitch.numerics.invert import invert


def terms(c: Arr) -> tuple[Array, Array]:
    """
    Make the term indices k and k! of a sum with these coefficients.

    :param c: The coefficients, one per term.
    :returns k: 0 up to one less than the number of coefficients.
    :returns k_fact: k!.
    """
    k = jnp.arange(c.shape[0])
    return k, jnp.cumprod(jnp.maximum(k, 1))


def e_q(n: Num, c: Arr, ns: Num) -> Array:
    """
    Calculate one of the two metamodel sums at one density.

    E_sat with c = csat, E_sym with c = csym: the sum over k of c_k / k! chi^k u_k, with
    u_k = 1 - (-3 chi)^(N - k) exp(-b n / n_sat) the low density fix and N the number
    of coefficients, five.

    :param n: Baryon density in fm^-3.
    :param c: The five coefficients in MeV.
    :param ns: Saturation density in fm^-3.
    :returns: The sum in MeV.
    """
    k, k_fact = terms(c)
    n_terms = c.shape[0]
    chi = (n - ns) / (3 * ns)
    u = 1 - (-3 * chi) ** (n_terms - k) * jnp.exp(-b_low * n / ns)
    return jnp.sum(c / k_fact * chi**k * u)


def e_q1(n: Num, c: Arr, ns: Num) -> Array:
    """
    Calculate d E_q / dn by hand.

    The sum over k of c_k / k! [k chi^(k-1) / (3 n_sat) u_k + chi^k u_k'].

    :param n: Baryon density in fm^-3.
    :param c: The five coefficients in MeV.
    :param ns: Saturation density in fm^-3.
    :returns: The derivative in MeV fm^3.
    """
    k, k_fact = terms(c)
    n_terms = c.shape[0]
    chi = (n - ns) / (3 * ns)
    ex = jnp.exp(-b_low * n / ns)
    u = 1 - (-3 * chi) ** (n_terms - k) * ex
    u1 = (
        ex
        / ns
        * (
            (n_terms - k) * (-3 * chi) ** (n_terms - 1 - k)
            + b_low * (-3 * chi) ** (n_terms - k)
        )
    )
    # k chi^(k-1) / (3 n_sat), 0 for k = 0
    dchi = jnp.where(k > 0, k * chi ** jnp.maximum(k - 1, 0), 0.0) / (3 * ns)
    return jnp.sum(c / k_fact * (dchi * u + chi**k * u1))


def e_q2(n: Num, c: Arr, ns: Num) -> Array:
    """
    Calculate d2 E_q / dn2 by hand.

    The sum over k of c_k / k! [k (k-1) chi^(k-2) / (9 n_sat^2) u_k
    + 2 k chi^(k-1) / (3 n_sat) u_k' + chi^k u_k''].

    :param n: Baryon density in fm^-3.
    :param c: The five coefficients in MeV.
    :param ns: Saturation density in fm^-3.
    :returns: The second derivative in MeV fm^6.
    """
    k, k_fact = terms(c)
    n_terms = c.shape[0]
    chi = (n - ns) / (3 * ns)
    ex = jnp.exp(-b_low * n / ns)
    u = 1 - (-3 * chi) ** (n_terms - k) * ex
    u1 = (
        ex
        / ns
        * (
            (n_terms - k) * (-3 * chi) ** (n_terms - 1 - k)
            + b_low * (-3 * chi) ** (n_terms - k)
        )
    )
    # the (5 - k)(4 - k) term is 0 for k = 4, where its power would go negative
    last = jnp.where(k < n_terms - 1, 1.0, 0.0)
    pw = jnp.maximum(n_terms - 2 - k, 0)
    u2 = (
        -ex
        / ns**2
        * (
            (n_terms - k) * (n_terms - 1 - k) * last * (-3 * chi) ** pw
            + 2 * b_low * (n_terms - k) * (-3 * chi) ** (n_terms - 1 - k)
            + b_low**2 * (-3 * chi) ** (n_terms - k)
        )
    )
    dchi = jnp.where(k > 0, k * chi ** jnp.maximum(k - 1, 0), 0.0) / (3 * ns)
    ddchi = jnp.where(k > 1, k * (k - 1) * chi ** jnp.maximum(k - 2, 0), 0.0) / (
        9 * ns**2
    )
    return jnp.sum(c / k_fact * (ddchi * u + 2 * dchi * u1 + chi**k * u2))


def e_bulk(n: Num, delta: Num, bulk: Bulk) -> Array:
    """
    Calculate the bulk energy per nucleon, E_sat(n) + E_sym(n) delta^2.

    :param n: Baryon density in fm^-3.
    :param delta: Isospin asymmetry, (n_n - n_p) / n.
    :param bulk: The metamodel inputs.
    :returns: The energy per nucleon in MeV.
    """
    return e_q(n, bulk.csat, bulk.ns) + e_q(n, bulk.csym, bulk.ns) * delta**2


def n_lep(mu: Num, m: Num) -> Array:
    """
    Calculate the number density of a free lepton gas at a chemical potential.

    0 below threshold. The inactive branch sees a constant, so its derivatives stay
    finite.

    :param mu: Chemical potential in MeV.
    :param m: Rest energy of the lepton in MeV.
    :returns: The number density in fm^-3.
    """
    above = mu > m
    x = jnp.where(above, mu**2 - m**2, 1.0)
    return jnp.where(above, x**1.5, 0.0) / (3 * jnp.pi**2 * hbarc**3)


def eps_lep(nl: Num, m: Num) -> Array:
    """
    Calculate the energy density of a free relativistic lepton gas.

    With x the relativity parameter, m^4 / (8 pi^2 (hbar c)^3) [x (2 x^2 + 1)
    sqrt(1 + x^2) - asinh x]. No lepton: a constant inside, so derivatives stay finite.

    :param nl: Number density in fm^-3.
    :param m: Rest energy of the lepton in MeV.
    :returns: The energy density in MeV fm^-3.
    """
    some = nl > 0
    x = hbarc * (3 * jnp.pi**2 * jnp.where(some, nl, 1.0)) ** (1 / 3) / m
    front = m**4 / (8 * jnp.pi**2 * hbarc**3)
    inside = x * (2 * x**2 + 1) * jnp.sqrt(1 + x**2) - jnp.arcsinh(x)
    return jnp.where(some, front * inside, 0.0)


def composition(
    n: Num, bulk: Bulk, *, exchange: bool = True, n_bisect: int = 40
) -> tuple[Array, Array]:
    """
    Find the isospin asymmetry of uniform matter from charge neutrality, by bisection.

    n_e + n_mu = n_p with mu_e = (m_n - m_p) + 4 E_sym delta + the proton exchange
    term. The bisection is held fixed for autodiff and one newton step is taken from
    its answer: same value, and the derivative of delta with respect to the inputs
    comes out right (the implicit function theorem), which P = n mu_n - eps needs.

    :param n: Baryon density in fm^-3.
    :param bulk: The metamodel inputs.
    :param exchange: Keep the proton exchange term in mu_e; it is dropped under C10.
    :param n_bisect: Halvings in the charge neutrality bisection.
    :returns delta: The isospin asymmetry.
    :returns mu_e: The electron chemical potential in MeV.
    """
    esym = e_q(n, bulk.csym, bulk.ns)
    xc = (3 / jnp.pi) ** (1 / 3) * e2 if exchange else 0.0

    def n_p(d: Num) -> Array:
        return 0.5 * n * (1 - d)

    def mu_e(d: Num) -> Array:
        return (mn - mp) + 4 * esym * d + xc * n_p(d) ** (1 / 3)

    def neutral(d: Num) -> Array:  # charge neutrality, 0 at the answer
        return n_lep(mu_e(d), me) + n_lep(mu_e(d), mmu) - n_p(d)

    d = jax.lax.stop_gradient(invert(neutral, 0.0, 0.0, 1.0, n=n_bisect))
    d = d - neutral(d) / jax.grad(neutral)(d)
    return d, mu_e(d)


def mu_n_outer(n: Num, delta: Num, bulk: Bulk) -> Array:
    """
    Calculate the neutron chemical potential of uniform matter.

    Outer core doc: m_n + E + n [E_sat' + delta^2 E_sym'] + 2 E_sym delta (1 - delta).

    :param n: Baryon density in fm^-3.
    :param delta: Isospin asymmetry.
    :param bulk: The metamodel inputs.
    :returns: mu_n in MeV.
    """
    esat1, esym1 = e_q1(n, bulk.csat, bulk.ns), e_q1(n, bulk.csym, bulk.ns)
    esym = e_q(n, bulk.csym, bulk.ns)
    return (
        mn
        + e_bulk(n, delta, bulk)
        + n * (esat1 + delta**2 * esym1)
        + 2 * esym * delta * (1 - delta)
    )


def uniform(
    n: Num, bulk: Bulk, *, exchange: bool = True, n_bisect: int = 40
) -> Uniform:
    """
    Calculate eps, P, delta and mu_n of uniform matter at one density.

    eps is the rest mass, the nucleons, the proton exchange term and the two lepton
    gases; P = n mu_n - eps.

    :param n: Baryon density in fm^-3.
    :param bulk: The metamodel inputs.
    :param exchange: Keep the proton exchange term; it is dropped under C10.
    :param n_bisect: Halvings in the charge neutrality bisection.
    :returns: eps and P in MeV fm^-3, delta, and mu_n in MeV.
    """
    d, mu = composition(n, bulk, exchange=exchange, n_bisect=n_bisect)
    n_p = 0.5 * n * (1 - d)
    xc = 0.75 * (3 / jnp.pi) ** (1 / 3) * e2 if exchange else 0.0
    eps = (
        0.5 * n * (1 + d) * mn
        + n_p * mp  # rest mass
        + n * e_bulk(n, d, bulk)  # nucleons
        - xc * n_p ** (4 / 3)  # proton exchange
        + eps_lep(n_lep(mu, me), me)
        + eps_lep(n_lep(mu, mmu), mmu)  # leptons
    )
    mun = mu_n_outer(n, d, bulk)
    return Uniform(eps, n * mun - eps, d, mun)
