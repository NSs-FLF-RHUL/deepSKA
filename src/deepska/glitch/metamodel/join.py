# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The Maxwell join: where the crust ends and the core starts.

No pasta: the crust is sphere cells up to 0.16 fm^-3 and the join is where they meet
uniform matter at equal pressure and equal neutron chemical potential. The crust ends
at n_1 and the core starts at n_2 > n_1, with the density jumping between them.

At every crust point n_1 the core density n_2 with the same pressure is found by
bisection, and Delta mu = mu_cell(n_1) - mu_n,uni(n_2) says who wins there: below 0
the crust, above 0 the core. The join is where Delta mu changes sign. When it changes
sign more than once, the pair that brackets the energy crossing n_cross is taken.

Contents:

- n2_of: the core density with a given pressure.
- next_partner: the next candidate of each crust point.
- join: the join itself, with its flags.
"""

import jax
import jax.numpy as jnp
from jax import Array, lax

from deepska.glitch.kinds import Arr, Num
from deepska.glitch.metamodel.uniform import core_at, delta_beta, p_uni
from deepska.glitch.numerics.invert import invert


def n2_of(
    p_cell: Num,
    c_sat: Arr,
    c_sym: Arr,
    ns: Num,
    *,
    bracket: tuple[float, float] = (0.005, 0.16),
) -> Array:
    """
    Find the core density n_2 with P_uni(n_2) = P_cell(n_1), for a crust point n_1.

    By bisection on the exact P_uni (invert, 30 halvings on the bracket): n_2 is varied
    until P_uni(n_2) - P_cell(n_1) = 0. Not 40 halvings: the value is the same to 15
    digits, but at 40 the last bracket (1e-13) is so narrow that rounding noise in P
    (1e-14) puts the derivative 0.5% off; at 30 it is 0.0005%.

    :param p_cell: P_cell(n_1) in MeV fm^-3.
    :param c_sat: The five coefficients of symmetric matter in MeV.
    :param c_sym: The five coefficients of the symmetry energy in MeV.
    :param ns: The saturation density n_sat in fm^-3.
    :param bracket: The densities the search runs between, in fm^-3.
    :returns: n_2 in fm^-3, nan when P_cell is outside P_uni of the bracket.
    """
    low, high = bracket

    def p_core(n: Num) -> Array:
        """P_uni at n, with its delta from the bisection on the charge balance."""
        return p_uni(n, delta_beta(n, c_sym, ns), c_sat, c_sym, ns)

    n2 = invert(p_core, p_cell, low, high, n=30)
    inside = (p_cell >= p_core(low)) & (p_cell <= p_core(high))
    return jnp.where(inside, n2, jnp.nan)


def next_partner(ok: Arr, *, max_gap: int = 3) -> tuple[Array, Array]:
    """
    Find the partner of each crust point: the next candidate after it.

    The partner of point i is the next point j > i with ok[j], at most max_gap grid
    steps on, so a hole of unconverged points is stepped over but a wide one is not.

    :param ok: Whether each crust point is a candidate.
    :param max_gap: The most grid steps between a point and its partner.
    :returns j: The partner of each point, the point itself when it has none, so
        indexing with it is always safe.
    :returns live: True where a point is a candidate with a partner.
    """
    n = ok.shape[0]
    idx = jnp.arange(n)
    nxt = lax.cummin(jnp.where(ok, idx, n), axis=0, reverse=True)  # first ok at >= i
    nxt = jnp.append(nxt[1:], n)  # first ok strictly after i
    live = ok & (nxt < n) & (nxt - idx <= max_gap)
    return jnp.where(live, nxt, idx), live


def join(
    crust_out: dict[str, Array], c_sat: Arr, c_sym: Arr, ns: Num
) -> dict[str, Array]:
    """
    Find the Maxwell join from the crust.

    In order: n_2 and Delta mu at every crust point; the candidates (past drip,
    converged, n_2 found and above n_1); the sign changes of Delta mu between a
    candidate and its partner; the energy crossing n_cross, the first place
    eps_cell - eps_uni goes from below 0 to above; the choice, the first sign
    change with n_1 < n_cross < n_2, else the nearest to it, else the first; and the
    join at the chosen pair, a straight line between its two points.

    Only the chosen pair goes into the derivative. Which pair is held fixed and the
    join is worked out again at that pair alone, so a 0/0 at a crust point that was not
    chosen never reaches it. The values are the same.

    :param crust_out: The crust, from crust.
    :param c_sat: The five coefficients of symmetric matter in MeV.
    :param c_sym: The five coefficients of the symmetry energy in MeV.
    :param ns: The saturation density n_sat in fm^-3.
    :returns: By name: n1 and n2 in fm^-3, p_t in MeV fm^-3, mu_t in MeV, eps_1 and
        eps_2 in MeV fm^-3, jump = n2 / n1 - 1, big_jump (the jump is above 5%),
        n_cross in fm^-3, n_flips (how many sign changes), join_flag (more than one)
        and joined. When joined is False there is no sign change, "no join", and the
        join values are nan.
    """
    converged = 1e-4  # a crust point counts only with a slope below this
    n_low = 0.005  # the low end of the bracket n_2 is searched on, fm^-3
    big = 0.05  # a jump above this is flagged
    n_b, p_c, mu_c, eps_c = (
        crust_out["n_b"],
        crust_out["p"],
        crust_out["mu_n"],
        crust_out["eps"],
    )

    def n2_at(p: Num) -> Array:
        return n2_of(p, c_sat, c_sym, ns)

    def core_n(n: Num) -> tuple[Array, Array, Array]:
        return core_at(n, c_sat, c_sym, ns)

    # at every crust point n_1: n_2 by the bisection, then Delta mu
    n2 = jax.vmap(n2_at)(p_c)
    _, _, mu_2 = jax.vmap(core_n)(jnp.where(jnp.isfinite(n2), n2, 0.1))
    d_mu = lax.stop_gradient(mu_c - mu_2)  # < 0: crust wins, > 0: core wins (choice)
    n2 = lax.stop_gradient(n2)

    # candidates
    n_drip = jnp.where(crust_out["dripped"], crust_out["n_drip"], jnp.inf)
    past_drip_converged = (n_b >= n_drip) & (crust_out["g_norm"] < converged)
    ok = past_drip_converged & jnp.isfinite(n2) & (n2 > n_b) & jnp.isfinite(d_mu)

    # sign changes: candidate i and its partner j with Delta mu of opposite sign
    j, live = next_partner(ok)
    flips = live & (jnp.sign(d_mu) != jnp.sign(d_mu[j]))
    t = d_mu / (d_mu - d_mu[j])  # straight line, where Delta mu = 0
    n1_all = n_b + t * (n_b[j] - n_b)
    n2_all = jax.vmap(n2_at)(p_c + t * (p_c[j] - p_c))

    # which one: the energy crossing n_cross
    eps_u, _, _ = jax.vmap(core_n)(jnp.maximum(n_b, n_low))
    d_eps = eps_c - eps_u
    jc, live_c = next_partner(past_drip_converged & (n_b >= n_low))
    cross = live_c & (d_eps < 0) & (d_eps[jc] >= 0)
    kc = jnp.argmax(cross)
    tc = d_eps[kc] / (d_eps[kc] - d_eps[jc[kc]])
    n_cross = jnp.where(cross.any(), n_b[kc] + tc * (n_b[jc[kc]] - n_b[kc]), jnp.nan)

    brackets = flips & (n1_all < n_cross) & (n_cross < n2_all)
    miss = jnp.where(
        flips, jnp.abs(jnp.log(n_cross / jnp.sqrt(n1_all * n2_all))), jnp.inf
    )
    k = jnp.where(
        brackets.any(),
        jnp.argmax(brackets),  # the first that brackets n_cross
        jnp.where(jnp.isfinite(n_cross), jnp.argmin(miss), jnp.argmax(flips)),
    )
    n_flips = flips.sum()

    # the join, at the chosen pair. Which pair is a choice (held, no derivative); the
    # values are then worked out afresh at that pair only, so a 0/0 or nan at any other
    # crust point can never reach the derivatives
    k = lax.stop_gradient(k)
    jk = lax.stop_gradient(j[k])
    joined_held = lax.stop_gradient(flips.any())

    def d_mu_at(i: Num) -> Array:
        return mu_c[i] - core_n(n2_at(p_c[i]))[2]

    dmu_i, dmu_j = d_mu_at(k), d_mu_at(jk)
    gap = jnp.where(joined_held, dmu_i - dmu_j, 1.0)
    tk = jnp.where(joined_held, dmu_i / gap, 0.0)
    n1 = n_b[k] + tk * (n_b[jk] - n_b[k])
    p_1 = p_c[k] + tk * (p_c[jk] - p_c[k])
    mu_t = mu_c[k] + tk * (mu_c[jk] - mu_c[k])
    eps_1 = eps_c[k] + tk * (eps_c[jk] - eps_c[k])
    n_2 = n2_at(p_1)
    eps_2, p_t, _ = core_n(jnp.where(jnp.isfinite(n_2), n_2, 0.1))  # P_t the core's own
    jump = n_2 / n1 - 1

    joined = flips.any()  # no sign change: "no join"

    def keep(x: Num) -> Array:
        return jnp.where(joined, x, jnp.nan)

    return {
        "n1": keep(n1),
        "n2": keep(n_2),
        "p_t": keep(p_t),
        "mu_t": keep(mu_t),
        "eps_1": keep(eps_1),
        "eps_2": keep(eps_2),
        "jump": keep(jump),
        "big_jump": joined & (jump > big),
        "n_cross": n_cross,
        "n_flips": n_flips,
        "join_flag": n_flips > 1,
        "joined": joined,
    }
