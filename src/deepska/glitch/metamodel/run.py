# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The runs: many draws on every core, and the sensitivity of the rise at one point.

A run of many draws has two stages. Stage 1 is cheap and runs on every draw: the draw
and the five checks on the inputs. Stage 2 is the equation of state, the star and the
glitch rise, and runs only on the draws that passed stage 1, about 3% of them. Both
stages are spread over every device jax has, one share of the draws on each. For a
CPU that is one device per core when XLA_FLAGS is set before jax is imported, as
scripts/metamodel.py does (cores, at the top of the script, every core but one):

    XLA_FLAGS=--xla_force_host_platform_device_count=<number of cores>

One core is left free: with every core taken, jax 0.10 locks up in stage 2, each share
waiting on helper threads that have no core left to run on.

Contents:

- stage_1_on_core, stage_2_one, stage_2_on_core: the two stages on one core's share.
- on_every_core: one stage spread over the cores.
- run_draws: the whole run, and why each rejected draw was rejected.
- dnu_of, rise_sensitivity: the derivative of the rise with respect to all 13 inputs.
"""

import math
import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from deepska.glitch.data.metamodel_inputs import priors
from deepska.glitch.data.paper_inputs import dt, n_t
from deepska.glitch.kinds import Arr, Fn, Num, Tree
from deepska.glitch.metamodel.draw import draw_and_check, draw_b
from deepska.glitch.metamodel.star import rise_of


def stage_1_on_core(keys: Arr) -> tuple[Array, Array, Array, dict[str, Array]]:
    """
    Run stage 1 on one core's share of the keys.

    2000 draws at a time, vmapped inside each batch: all of them at once would not fit
    in memory, as the stable and causal checks are 1600 points each.

    :param keys: The random keys of this core's draws.
    :returns: values, n_c, ok and why of every draw, as draw_and_check gives them.
    """
    return lax.map(draw_and_check, keys, batch_size=2000)


def stage_2_one(
    draw: tuple[Arr, Num, Num, Num], *, rise_every: int = 20
) -> dict[str, Any]:
    """
    Run stage 2 for one accepted draw: the eos, the star and the rise.

    :param draw: The draw, as (values, n_c, b_core, b_crust).
    :param rise_every: Every this many steps of the rise is kept, 20 is every 0.12 s.
    :returns: What rise_of gives, with dnu thinned to every rise_every steps.
    """
    values, n_c, b_core, b_crust = draw
    out = rise_of(values, n_c, b_core, b_crust)
    out["dnu"] = out["dnu"][rise_every - 1 :: rise_every]
    return out


def stage_2_on_core(
    draws: tuple[Arr, Arr, Arr, Arr], *, rise_every: int = 20
) -> dict[str, Any]:
    """
    Run stage 2 on one core's share of the draws, one draw at a time.

    One at a time is the fastest, measured on 18 cores: 135 ms a draw, against 140 at
    4 at a time, 158 at 8 and 213 at 16.

    :param draws: This core's draws, as (values, n_c, b_core, b_crust).
    :param rise_every: Every this many steps of the rise is kept.
    :returns: What stage_2_one gives, for every draw.
    """
    return lax.map(partial(stage_2_one, rise_every=rise_every), draws)


def on_every_core(stage: Fn, arrays: Tree) -> Tree:
    """
    Run one stage on every core at once.

    The arrays, whose leading axis is the draws, are split evenly over the cores, the
    stage is run on each core's share, and the answers are put back in order. The draws
    are padded with copies of the first up to a multiple of the number of cores, and
    the copies are dropped afterwards.

    :param stage: The stage on one core's share, made to run on every core by jax.pmap.
    :param arrays: An array or any nest of arrays, one row per draw.
    :returns: What the stage gives, one row per draw.
    """
    n_cores = jax.local_device_count()
    n = jax.tree_util.tree_leaves(arrays)[0].shape[0]
    per_core = -(-n // n_cores)  # rounded up
    pad = per_core * n_cores - n

    def split(x: Arr) -> np.ndarray:
        x = np.asarray(x)  # gathered back from the cores of an earlier stage
        x = np.concatenate([x, np.repeat(x[:1], pad, axis=0)])
        return x.reshape(n_cores, per_core, *x.shape[1:])

    def join_back(x: Arr) -> Arr:
        return x.reshape(n_cores * per_core, *x.shape[2:])[:n]

    out = stage(jax.tree_util.tree_map(split, arrays))
    return jax.tree_util.tree_map(join_back, out)


def run_draws(n_draws: int, *, seed: int = 0, rise_every: int = 20) -> dict[str, Any]:
    """
    Run n_draws draws: all 13 inputs, the checks, the eos, the star and the rise.

    :param n_draws: How many draws.
    :param seed: The seed of the random draws.
    :param rise_every: Every this many steps of each rise is kept.
    :returns: By name: n_draws; n_cores; n_run, how many draws passed the checks and
        went into stage 2; n_kept; counts, how many draws each reason rejected, each
        rejected draw counted once, under the first reason it fails; seconds, the time
        of the two stages; t, the time of each kept step of the rise in s; and for the
        kept draws values, n_c, b_core, b_crust, dnu in muHz, dnu_equil in muHz,
        r_star in km and m_star in solar masses.
    """
    start = time.time()
    keys = jax.random.split(jax.random.PRNGKey(seed), n_draws)

    # stage 1: the five checks on every draw
    # jax hands the answers back before the cores have finished, so wait for them
    # before the clock is read
    values, n_c, ok_1, why_1 = jax.block_until_ready(
        on_every_core(jax.pmap(stage_1_on_core), keys)
    )
    keys, values, n_c = keys[ok_1], values[ok_1], n_c[ok_1]
    b_core, b_crust = jax.vmap(draw_b)(keys)
    time_1 = time.time() - start

    # stage 2: eos, star, rise on the ones that passed
    stage_2 = jax.pmap(partial(stage_2_on_core, rise_every=rise_every))
    out = jax.block_until_ready(on_every_core(stage_2, (values, n_c, b_core, b_crust)))
    ok_2, why_2 = out["ok"], out["why"]
    time_2 = time.time() - start - time_1

    # each rejected draw counted once, under the first reason it fails
    counts: dict[str, int] = {}
    failed = jnp.zeros(n_draws, bool)
    for name in [
        "neutron gas",
        "symmetric matter",
        "E_sym <= 0",
        "unstable",
        "acausal",
    ]:
        counts[name] = int((~failed & why_1[name]).sum())
        failed = failed | why_1[name]
    failed = jnp.zeros(len(ok_2), bool)
    for name in ["no drip", "no join", "P falls", "no surface"]:
        counts[name] = int((~failed & why_2[name]).sum())
        failed = failed | why_2[name]

    keep = np.asarray(ok_2)
    result: dict[str, Any] = {
        "n_draws": n_draws,
        "n_cores": jax.local_device_count(),
        "n_run": len(ok_2),
        "n_kept": int(ok_2.sum()),
        "counts": counts,
        "seconds": (time_1, time_2),
        "values": values[keep],
        "n_c": n_c[keep],
        "b_core": b_core[keep],
        "b_crust": b_crust[keep],
    }
    for name in ["dnu", "dnu_equil", "r_star", "m_star"]:
        result[name] = np.asarray(out[name])[keep]
    result["t"] = dt * rise_every * np.arange(1, result["dnu"].shape[1] + 1)
    return result


def dnu_of(values: Arr, n_c: Num, log_b_core: Num, log_b_crust: Num) -> Array:
    """
    Calculate the rise alone, with B given as log B.

    :param values: The 12 values of a draw.
    :param n_c: The central density in fm^-3.
    :param log_b_core: The log of the mutual friction in the core.
    :param log_b_crust: The log of the mutual friction in the crust.
    :returns: dnu at every time step in muHz.
    """
    out = rise_of(values, n_c, jnp.exp(log_b_core), jnp.exp(log_b_crust))
    return out["dnu"]


def rise_sensitivity(
    values: Arr, n_c: float, b_core: float, b_crust: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Calculate d dnu(t) / dx at one point, for all 13 inputs.

    By forward mode, jax.jacfwd, which gives all 13 in one go, about 35 s with the
    compile. B is differentiated as log B and the result divided by B.

    :param values: The 12 values the derivative is taken at; r_gas and r_sym are not
        used.
    :param n_c: The central density in fm^-3.
    :param b_core: The mutual friction in the core.
    :param b_crust: The mutual friction in the crust.
    :returns t: The time of every step of the rise in s.
    :returns slopes: d dnu / dx at every time step, in muHz per unit of x, by name:
        the ten inputs in the order of priors, then "n_c", "B_core" and "B_crust".
    """
    args = (values, n_c, math.log(b_core), math.log(b_crust))
    d_values, d_n_c, d_log_b_core, d_log_b_crust = jax.jit(
        jax.jacfwd(dnu_of, argnums=(0, 1, 2, 3))
    )(*args)
    slopes = {name: np.asarray(d_values[:, 2 + j]) for j, name in enumerate(priors)}
    slopes["n_c"] = np.asarray(d_n_c)
    slopes["B_core"] = np.asarray(d_log_b_core) / b_core
    slopes["B_crust"] = np.asarray(d_log_b_crust) / b_crust
    t = dt * np.arange(1, n_t + 1)
    return t, slopes
