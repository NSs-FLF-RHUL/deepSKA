# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Any inputs swept at once, every combination (a matrix) or paired up.

- make_sweep_run: the two halves of a run in the form a sweep wants.
- sweep_paired: a paired sweep, run i takes value i of every list.
- sweep_matrix: a matrix sweep, every combination of the swept values.
- run_sweep: run a sweep over any of the inputs.
- save_sweep: save a sweep as one npz file.

per_run(dnu, profiles, star, p) picks what to keep from each run, p is that runs full
inputs by name. b_profile in inputs can be a list of profile functions, then sweeping
"b_profile" over 0, 1, 2 .. picks which one each run uses.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from deepska.glitch.glitch_rise_model.run.param_to_star import param_to_star
from deepska.glitch.glitch_rise_model.run.records_of import records_of
from deepska.glitch.glitch_rise_model.run.star_to_output import star_to_output
from deepska.glitch.kinds import Batching, Fn, Grid, Inputs, Outputs, Star, StarInputs
from deepska.glitch.numerics.flat_grid import flat_grid
from deepska.glitch.numerics.in_batches import in_batches
from deepska.glitch.numerics.in_threads import in_threads
from deepska.glitch.numerics.pick_function import pick_function


def make_sweep_run(
    inputs: Inputs, per_run: Callable[..., Outputs]
) -> tuple[Callable[..., Star], Callable[..., Outputs]]:
    """
    Make the two halves of a run in the form a sweep needs.

    b_profile in inputs can be one profile function or a list of them, a swept
    "b_profile" then picks one by number.

    :param inputs: The inputs by name, used for anything that isnt swept.
    :param per_run: Function that picks what to keep from each run.
    :returns star_of: Function building the star for the swept inputs of one run.
    :returns run_on: Function doing one run on a built star.
    """
    b_profiles: list[Fn] = (
        inputs["b_profile"]
        if isinstance(inputs["b_profile"], list)
        else [inputs["b_profile"]]
    )

    # the star for one set of swept star inputs
    def star_of(d: Inputs) -> Star:
        return param_to_star(records_of({**inputs, **d})[0])

    # one run on a built star, d is every swept input of this run
    def run_on(star: Star, d: Inputs) -> Outputs:
        p = {**inputs, **d}
        # which profile, a number as vmap cant go over functions
        k = jnp.asarray(d.get("b_profile", 0)).astype(int)

        def b_profile(rho: Array) -> Array:
            return pick_function(b_profiles, k, rho)

        p["b_profile"] = b_profile
        _, spin = records_of(p)
        dnu, profiles = star_to_output(star, b_profile, p["b_core"], spin)
        return per_run(dnu, profiles, star, p)

    return star_of, run_on


def sweep_paired(
    values: dict[str, Array],
    star_of: Callable[..., Star],
    run_on: Callable[..., Outputs],
    batching: Batching,
) -> Grid:
    """
    Run a paired sweep, where run i takes value i of every list.

    Each run needs its own star if a star input is swept, else one star does for all of
    them.

    :param values: The values to try for each swept input, all the same length.
    :param star_of: Function building the star for one set of swept inputs.
    :param run_on: Function doing one run on a built star.
    :param batching: Runs per batch and batches going at once.
    :returns: The kept outputs, one entry per run.
    """
    names = list(values)
    star_swept = any(n in StarInputs._fields for n in names)
    one_star = None if star_swept else star_of({})

    def one(i: Array) -> Outputs:
        d = {n: values[n][i] for n in names}
        return run_on(star_of(d) if one_star is None else one_star, d)

    runs = len(values[names[0]])
    return in_batches(
        jax.jit(jax.vmap(one)), jnp.arange(runs), batching.batch, batching.cores
    )


def sweep_matrix(
    values: dict[str, Array],
    star_of: Callable[..., Star],
    run_on: Callable[..., Outputs],
    batching: Batching,
) -> Grid:
    """

    Run every combination of the swept values.

    he star only gets built once per combination of the star inputs, then every spin
    input is vmapped on top of each star, as a star takes far longer than a spin run.

    :param values: The values to try for each swept input.
    :param star_of: Function building the star for one set of swept inputs.
    :param run_on: Function doing one run on a built star.
    :param batching: Runs per batch and batches going at once.
    :returns: The kept outputs, with one axis per swept input.
    """
    batch, cores = batching
    names = list(values)
    star_names = [n for n in names if n in StarInputs._fields]
    spin_names = [n for n in names if n not in StarInputs._fields]
    star_draws, star_shape = flat_grid({n: values[n] for n in star_names})
    spin_draws, spin_shape = flat_grid({n: values[n] for n in spin_names})
    # how many different stars, how many runs on each
    n_stars, n_spins = int(np.prod(star_shape)), int(np.prod(spin_shape))

    def a_star(i: Array) -> Star:
        return star_of({n: star_draws[n][i] for n in star_names})

    # every star built once, vmapped, 50 at a time as a star is heavier than a run
    stars = in_batches(
        jax.jit(jax.vmap(a_star)), jnp.arange(n_stars), min(batch, 50), cores
    )

    # some stars x some spin combinations, vmapped over both
    def a_job(o_b: Star, s_idx: Array, t_idx: Array) -> Outputs:
        def on_star(o: Star, i: Array) -> Outputs:
            def on_spin(j: Array) -> Outputs:
                d = {
                    **{n: star_draws[n][i] for n in star_names},
                    **{n: spin_draws[n][j] for n in spin_names},
                }
                return run_on(o, d)

            return jax.vmap(on_spin)(t_idx)

        return jax.vmap(on_star)(o_b, s_idx)

    a_job_jit = jax.jit(a_job)  # so each shape of job compiles once
    t_batch = min(n_spins, batch)
    s_batch = max(1, batch // t_batch)
    t_chunks = [
        np.arange(t, min(t + t_batch, n_spins)) for t in range(0, n_spins, t_batch)
    ]
    s_chunks = [
        np.arange(s, min(s + s_batch, n_stars)) for s in range(0, n_stars, s_batch)
    ]
    jobs = [(s_idx, t_idx) for s_idx in s_chunks for t_idx in t_chunks]

    def run_job(job: tuple[np.ndarray, np.ndarray]) -> Outputs:
        s_idx, t_idx = job
        o_b = jax.tree_util.tree_map(lambda a: a[s_idx], stars)  # just these stars
        return jax.tree_util.tree_map(np.asarray, a_job_jit(o_b, s_idx, t_idx))

    parts = in_threads(run_job, jobs, cores)
    rows = [
        jax.tree_util.tree_map(
            lambda *p: np.concatenate(p, axis=1),
            *parts[r * len(t_chunks) : (r + 1) * len(t_chunks)],
        )
        for r in range(len(s_chunks))
    ]
    # (stars, spin combinations, ...)
    flat = jax.tree_util.tree_map(lambda *p: np.concatenate(p, axis=0), *rows)

    # the order it was worked out in, star inputs first
    worked = star_names + spin_names
    order = [worked.index(n) for n in names]

    # back to one axis per swept input, in the order they were given
    def to_grid(a: np.ndarray) -> np.ndarray:
        a = a.reshape(star_shape + spin_shape + a.shape[2:])
        return a.transpose(order + list(range(len(order), a.ndim)))

    return jax.tree_util.tree_map(to_grid, flat)


def run_sweep(
    inputs: Inputs,
    sweep_over: Inputs,
    per_run: Callable[..., Outputs],
    batching: Batching,
    *,
    matrix: bool,
) -> tuple[Grid, Grid, dict[str, Array]]:
    """
    Run a sweep over any of the inputs.

    :param inputs: The inputs by name, used for anything that isnt swept.
    :param sweep_over: The values to try for each swept input, by name.
    :param per_run: Function of (dnu, profiles, star, p) picking what to keep.
    :param batching: Runs per batch and batches going at once.
    :param matrix: True for every combination of the values, False for paired up.
    :returns out: What per_run kept from every run.
    :returns swept: The swept values.
    :returns draws: The inputs of every run as flat lists.
    """
    values = {n: jnp.asarray(v, dtype=float) for n, v in sweep_over.items()}
    swept = {n: np.asarray(v) for n, v in values.items()}
    star_of, run_on = make_sweep_run(inputs, per_run)
    if not matrix:
        return sweep_paired(values, star_of, run_on, batching), swept, values
    draws, _shape = flat_grid(values)
    return sweep_matrix(values, star_of, run_on, batching), swept, draws


def save_sweep(folder: str | Path, out: Grid, values: Grid) -> str:
    """
    Save a sweep as one npz file.

    Every output goes in, plus the swept values as values_<name>.

    :param folder: Folder to save into, made if it isnt there.
    :param out: The outputs of the sweep.
    :param values: The swept values.
    :returns: The full path of the file.
    """
    where = Path(folder)
    path = where / (
        "sweep_{}_{}.npz".format("_".join(values), time.strftime("%Y-%m-%d_%H%M"))
    )
    where.mkdir(parents=True, exist_ok=True)
    # Any as the numpy stubs think one of the keys could be allow_pickle
    arrays: dict[str, Any] = {k: np.asarray(v) for k, v in out.items()}
    arrays |= {"values_" + n: v for n, v in values.items()}
    np.savez(path, **arrays)
    return str(path.resolve())
