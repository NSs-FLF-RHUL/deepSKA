# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""A few values at a time so we dont blow the memory."""

import jax
import numpy as np

from deepska.glitch.kinds import Fn, Grid, Num
from deepska.glitch.numerics.in_threads import in_threads


def in_batches(run: Fn, values: Grid, batch: int, cores: int) -> Grid | np.ndarray:
    """
    Run a function over many values, a batch at a time.

    The batches go side by side in threads.

    :param run: Function taking one batch, giving an array or any nest of arrays.
    :param values: The values to split up.
    :param batch: Number of values in one batch.
    :param cores: Number of batches going at once.
    :returns: The outputs of every batch joined end to end.
    """
    batches = [values[i : i + batch] for i in range(0, len(values), batch)]
    parts = in_threads(run, batches, cores)

    def join(*p: Num) -> np.ndarray:
        return np.concatenate([np.asarray(a) for a in p])

    return jax.tree_util.tree_map(join, *parts)
