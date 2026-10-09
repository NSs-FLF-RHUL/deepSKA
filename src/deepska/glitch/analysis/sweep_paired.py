# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""A paired sweep: run i takes value i of every list."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Batching, Grid, Outputs, Star, StarInputs
from deepska.glitch.numerics.in_batches import in_batches


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
