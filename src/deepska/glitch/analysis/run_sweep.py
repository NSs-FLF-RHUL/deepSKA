# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Any inputs swept at once, every combination (a matrix) or paired up.

per_run(dnu, profiles, star, p) picks what to keep from each run, p is that runs full
inputs by name. b_profile in inputs can be a list of profile functions, then sweeping
"b_profile" over 0, 1, 2 .. picks which one each run uses.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax import Array

from deepska.glitch.analysis.make_sweep_run import make_sweep_run
from deepska.glitch.analysis.sweep_matrix import sweep_matrix
from deepska.glitch.analysis.sweep_paired import sweep_paired
from deepska.glitch.kinds import Batching, Grid, Inputs, Outputs
from deepska.glitch.numerics.flat_grid import flat_grid


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
