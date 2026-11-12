# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The two metamodel sums and their derivatives.

The metamodel of Margueron et al 2018 writes the energy per nucleon of nuclear matter
as two Taylor sums around saturation. E_q(n) with c = c_sat is E_sat(n), symmetric
matter, and with c = c_sym is E_sym(n), the symmetry energy. The five terms k = 0..4
are written out, with no loop and no factorial, so each derivative with respect to n is
plain algebra and stays finite at n_sat.

Shared arguments, the same in every function here:

- n: the baryon density in fm^-3.
- k: which term of the sum, 0 to 4, a Python int.
- c: the five coefficients c_q,k for k = 0..4 in MeV, c_sat or c_sym.
- ns: the saturation density n_sat in fm^-3.

Contents:

- chi: the expansion variable.
- u, u1, u2: the low density correction u_k and its first two derivatives.
- e_q, e_q1, e_q2: the sum and its first two derivatives.
- e_bulk: bulk matter at an asymmetry delta.
- e_pnm: pure neutron matter.
"""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.metamodel_inputs import b_low
from deepska.glitch.kinds import Arr, Num


def chi(n: Num, ns: Num) -> Num:
    """
    Calculate the expansion variable chi = (n - n_sat) / (3 n_sat).

    :returns: chi, no units.
    """
    return (n - ns) / (3 * ns)


def u(n: Num, k: int, ns: Num) -> Array:
    """
    Calculate the low density correction u_k = 1 - (-3 chi)^(5-k) e^(-b n / n_sat).

    :returns: u_k, which goes to 0 at n = 0 and to 1 at high density.
    """
    x = -3 * chi(n, ns)
    return 1 - x ** (5 - k) * jnp.exp(-b_low * n / ns)


def u1(n: Num, k: int, ns: Num) -> Array:
    """
    Calculate u_k', the derivative of u_k with respect to n.

    u_k' = e^(-b n / n_sat) / n_sat [(5-k)(-3 chi)^(4-k) + b (-3 chi)^(5-k)]

    :returns: u_k' in fm^3.
    """
    x = -3 * chi(n, ns)
    ex = jnp.exp(-b_low * n / ns)
    return ex / ns * ((5 - k) * x ** (4 - k) + b_low * x ** (5 - k))


def u2(n: Num, k: int, ns: Num) -> Array:
    """
    Calculate u_k'', the second derivative of u_k with respect to n.

    u_k'' = -e^(-b n / n_sat) / n_sat^2 [(5-k)(4-k)(-3 chi)^(3-k)
            + 2b (5-k)(-3 chi)^(4-k) + b^2 (-3 chi)^(5-k)]

    The (5-k)(4-k) term is 0 for k = 4, where its power would go negative.

    :returns: u_k'' in fm^6.
    """
    last = 4  # the last term of the sum, the one without the first piece
    x = -3 * chi(n, ns)
    ex = jnp.exp(-b_low * n / ns)
    first = (5 - k) * (4 - k) * x ** (3 - k) if k < last else 0.0
    return (
        -ex
        / ns**2
        * (first + 2 * b_low * (5 - k) * x ** (4 - k) + b_low**2 * x ** (5 - k))
    )


def e_q(n: Num, c: Arr, ns: Num) -> Array:
    """
    Calculate the sum E_q(n) = sum_k c_q,k / k! chi^k u_k.

    The five terms k = 0..4 are written out.

    :returns: E_q in MeV.
    """
    x = chi(n, ns)
    return (
        c[0] * u(n, 0, ns)
        + c[1] * x * u(n, 1, ns)
        + c[2] / 2 * x**2 * u(n, 2, ns)
        + c[3] / 6 * x**3 * u(n, 3, ns)
        + c[4] / 24 * x**4 * u(n, 4, ns)
    )


def e_q1(n: Num, c: Arr, ns: Num) -> Array:
    """
    Calculate E_q', the derivative of the sum with respect to n.

    E_q'(n) = sum_k c_q,k / k! [k chi^(k-1) / (3 n_sat) u_k + chi^k u_k']

    Written out for k = 0..4. The k = 0 term has no k chi^(k-1) part.

    :returns: E_q' in MeV fm^3.
    """
    x = chi(n, ns)
    a = 3 * ns  # d chi / dn = 1 / a
    return (
        c[0] * u1(n, 0, ns)
        + c[1] * (1 / a * u(n, 1, ns) + x * u1(n, 1, ns))
        + c[2] / 2 * (2 * x / a * u(n, 2, ns) + x**2 * u1(n, 2, ns))
        + c[3] / 6 * (3 * x**2 / a * u(n, 3, ns) + x**3 * u1(n, 3, ns))
        + c[4] / 24 * (4 * x**3 / a * u(n, 4, ns) + x**4 * u1(n, 4, ns))
    )


def e_q2(n: Num, c: Arr, ns: Num) -> Array:
    """
    Calculate E_q'', the second derivative of the sum with respect to n.

    E_q''(n) = sum_k c_q,k / k! [k(k-1) chi^(k-2) / (9 n_sat^2) u_k
               + 2k chi^(k-1) / (3 n_sat) u_k' + chi^k u_k'']

    Written out for k = 0..4. The k = 0 term has no k chi^(k-1) part, and the k <= 1
    terms have no k(k-1) chi^(k-2) part.

    :returns: E_q'' in MeV fm^6.
    """
    x = chi(n, ns)
    a = 3 * ns  # d chi / dn = 1 / a
    return (
        c[0] * u2(n, 0, ns)
        + c[1] * (2 / a * u1(n, 1, ns) + x * u2(n, 1, ns))
        + c[2]
        / 2
        * (2 / a**2 * u(n, 2, ns) + 4 * x / a * u1(n, 2, ns) + x**2 * u2(n, 2, ns))
        + c[3]
        / 6
        * (
            6 * x / a**2 * u(n, 3, ns)
            + 6 * x**2 / a * u1(n, 3, ns)
            + x**3 * u2(n, 3, ns)
        )
        + c[4]
        / 24
        * (
            12 * x**2 / a**2 * u(n, 4, ns)
            + 8 * x**3 / a * u1(n, 4, ns)
            + x**4 * u2(n, 4, ns)
        )
    )


def e_bulk(n: Num, delta: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the energy per nucleon of bulk matter at an asymmetry.

    E(n, delta) = E_sat(n) + E_sym(n) delta^2

    :param delta: The asymmetry (n_n - n_p) / n, 0 symmetric and 1 pure neutrons.
    :param c_sat: The five coefficients of symmetric matter in MeV.
    :param c_sym: The five coefficients of the symmetry energy in MeV.
    :returns: E in MeV.
    """
    return e_q(n, c_sat, ns) + e_q(n, c_sym, ns) * delta**2


def e_pnm(n: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> Array:
    """
    Calculate the energy per nucleon of pure neutron matter, E(n, 1).

    E(n, 1) = E_sat(n) + E_sym(n)

    :param c_sat: The five coefficients of symmetric matter in MeV.
    :param c_sym: The five coefficients of the symmetry energy in MeV.
    :returns: E in MeV.
    """
    return e_q(n, c_sat, ns) + e_q(n, c_sym, ns)
