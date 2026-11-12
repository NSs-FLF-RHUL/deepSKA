# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The table: the equation of state as one list, and P(eps).

One list from low to high density: the crust cells below n_1, the crust end, the core
start at n_2 and the core up to the central density n_c. The table always has the same
number of rows, 600 + 1 + 400 = 1001, so draws can be vmapped. Like the TOV, which runs
all its steps because how many reach the surface is not known in advance, how many
crust cells are kept is not known in advance either.

Contents:

- core_table: the core from n_2 to n_c.
- crust_join_core: the crust, the join and the core for one set of inputs.
- table_fixed: the table.
- p_of_eps_fixed: from the inputs to P at given energy densities.
"""

import jax
import jax.numpy as jnp
from jax import Array, lax

from deepska.glitch.kinds import Arr, Num
from deepska.glitch.metamodel.crust import crust
from deepska.glitch.metamodel.join import join
from deepska.glitch.metamodel.uniform import core_at


def core_table(n_2: Num, n_c: Num, c_sat: Arr, c_sym: Arr, ns: Num) -> dict[str, Array]:
    """
    Calculate the core on 400 evenly spaced densities from n_2 to n_c.

    The first is the core start, n_2 itself. eps_uni and P_uni are those of uniform
    matter in beta equilibrium.

    :param n_2: The density the core starts at, in fm^-3.
    :param n_c: The central density in fm^-3.
    :param c_sat: The five coefficients of symmetric matter in MeV.
    :param c_sym: The five coefficients of the symmetry energy in MeV.
    :param ns: The saturation density n_sat in fm^-3.
    :returns: By name: n_b in fm^-3, eps and p in MeV fm^-3, (400,) each.
    """
    n = jnp.linspace(n_2, n_c, 400)

    def at(n_i: Num) -> tuple[Array, Array]:
        eps, p, _ = core_at(n_i, c_sat, c_sym, ns)
        return eps, p

    eps, p = jax.vmap(at)(n)
    return {"n_b": n, "eps": eps, "p": p}


def crust_join_core(
    c_sat: Arr, c_sym: Arr, ns: Num, n_c: Num
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array]]:
    """
    Build the crust, the join and the core for one set of inputs.

    :param c_sat: The five coefficients of symmetric matter in MeV.
    :param c_sym: The five coefficients of the symmetry energy in MeV.
    :param ns: The saturation density n_sat in fm^-3.
    :param n_c: The central density in fm^-3.
    :returns crust_out: The crust, from crust.
    :returns join_out: The join, from join.
    :returns core_out: The core, from core_table.
    """
    crust_out = crust(c_sat, c_sym, ns)
    join_out = join(crust_out, c_sat, c_sym, ns)
    core_out = core_table(join_out["n2"], n_c, c_sat, c_sym, ns)
    return crust_out, join_out, core_out


def table_fixed(
    crust_out: dict[str, Array], join_out: dict[str, Array], core_out: dict[str, Array]
) -> dict[str, Array]:
    """
    Build the table, with a fixed number of rows so it can be vmapped.

    The rows, from low to high density: the 600 crust rows, the crust end
    (n_1, eps_1, P_t (1 - 1e-12)), a hair below P_t so the splines see a strictly
    rising P, and the 400 core rows, the first of them the core start (n_2, eps_2, P_t).

    A crust cell is kept when it is below n_1, its winner converged (|g| < 1e-4) and
    its P is below P_t. A cell that is left out is replaced by a point on the straight
    line, in log-log, between the kept cells either side, at a density no other row
    has. So no row repeats, and the star's splines can take the table as it is; and
    every kept cell is in the table exactly. A left-out cell below the last kept one
    sits at its own grid density; the ones past it are spread evenly in log between the
    last kept one and n_1.

    :param crust_out: The crust, from crust.
    :param join_out: The join, from join.
    :param core_out: The core, from core_table.
    :returns: By name: n_b in fm^-3, eps and p in MeV fm^-3, and crust, (1001,) each.
        crust is True up to and including the crust end and False from n_2.
    """
    converged = 1e-4  # a crust cell is kept only with a slope below this
    n_b = crust_out["n_b"]
    n_1 = join_out["n1"]
    end_p = join_out["p_t"] * (1 - 1e-12)
    used = (
        (n_b < n_1)
        & (crust_out["p"] < join_out["p_t"])
        & (crust_out["g_norm"] < converged)
    )

    # the kept points, and the crust end, to draw the lines through: a left-out row
    # repeats the last kept one here, which the line just steps over
    idx = jnp.arange(used.shape[0])
    last = lax.cummax(jnp.where(used, idx, -1), axis=0)  # the last kept at or before
    last = jnp.where(last < 0, jnp.argmax(used), last)  # before the first: the first
    log_n_kept = jnp.log(jnp.concatenate([n_b[last], n_1[None]]))
    log_eps_kept = jnp.log(
        jnp.concatenate([crust_out["eps"][last], join_out["eps_1"][None]])
    )
    log_p_kept = jnp.log(jnp.concatenate([crust_out["p"][last], end_p[None]]))

    # where each crust row goes: its own density up to the last kept point, then the
    # rest spread evenly in log from there to just below n_1
    k = lax.stop_gradient(last[-1])  # the last kept point
    tail = (idx - k) / (idx.shape[0] - k)  # 0 at the last kept, below 1 at the last row
    log_n_tail = jnp.log(n_b[k]) + tail * (jnp.log(n_1) - jnp.log(n_b[k]))
    log_n = jnp.where(idx <= k, jnp.log(n_b), log_n_tail)

    eps = jnp.exp(jnp.interp(log_n, log_n_kept, log_eps_kept))
    p = jnp.exp(jnp.interp(log_n, log_n_kept, log_p_kept))
    n_crust = idx.shape[0]  # the crust rows; the row after them is the crust end
    n_rows = n_crust + 1 + core_out["n_b"].shape[0]
    return {
        "n_b": jnp.concatenate([jnp.exp(log_n), n_1[None], core_out["n_b"]]),
        "eps": jnp.concatenate([eps, join_out["eps_1"][None], core_out["eps"]]),
        "p": jnp.concatenate([p, end_p[None], core_out["p"]]),
        "crust": jnp.arange(n_rows) <= n_crust,
    }


def p_of_eps_fixed(
    values: Arr, n_c: Num, eps: Num
) -> tuple[Array, Array, dict[str, Array]]:
    """
    Calculate P at given energy densities from the inputs, vmappable over draws.

    Under vmap a draw can't be dropped part way, so it runs to the end and is dropped
    afterwards with values[ok].

    :param values: The 12 values of a draw, from draw_values.
    :param n_c: The central density in fm^-3.
    :param eps: The energy densities to give P at, in MeV fm^-3, one or an array.
    :returns p: P at eps in MeV fm^-3, nan when the draw is dropped.
    :returns ok: True when the draw is kept: it dripped, it joined and P rises.
    :returns why: The reason, one True or False each, True where that check failed:
        "no drip", "no join", "P falls".
    """
    e_sat, n_sat, k_sat, q_sat, z_sat, e_sym, l_sym, k_sym, q_sym, z_sym = values[2:]
    c_sat = jnp.array([e_sat, 0.0, k_sat, q_sat, z_sat])
    c_sym = jnp.array([e_sym, l_sym, k_sym, q_sym, z_sym])

    crust_out, join_out, core_out = crust_join_core(c_sat, c_sym, n_sat, n_c)
    t = table_fixed(crust_out, join_out, core_out)

    p_rises = jnp.all(jnp.diff(t["eps"]) > 0) & jnp.all(jnp.diff(t["p"]) > 0)
    ok = crust_out["dripped"] & join_out["joined"] & p_rises
    why = {
        "no drip": ~crust_out["dripped"],
        "no join": ~join_out["joined"],
        "P falls": ~p_rises,
    }

    p = jnp.interp(eps, t["eps"], t["p"])
    return jnp.where(ok, p, jnp.nan), ok, why
