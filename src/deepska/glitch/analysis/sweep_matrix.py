# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
A matrix sweep: every combination of the swept values.

The star only gets built once per combination of the star inputs, then every spin input
is vmapped on top of each star, as a star takes far longer than a spin run.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from deepska.glitch.kinds import Batching, Grid, Outputs, Star, StarInputs
from deepska.glitch.numerics.flat_grid import flat_grid
from deepska.glitch.numerics.in_batches import in_batches
from deepska.glitch.numerics.in_threads import in_threads


def sweep_matrix(
    values: dict[str, Array],
    star_of: Callable[..., Star],
    run_on: Callable[..., Outputs],
    batching: Batching,
) -> Grid:
    """
    Run every combination of the swept values.

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
