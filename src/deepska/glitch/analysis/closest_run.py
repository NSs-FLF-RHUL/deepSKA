# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The run of a sweep nearest the Vela data."""

import numpy as np

from deepska.glitch.kinds import Arr, Grid


def closest_run(
    rms: Arr, values: Grid, *, matrix: bool
) -> tuple[tuple[int, ...], dict[str, float]]:
    """
    Find the run of a sweep that is closest to the Vela data.

    :param rms: Rms distance of every run from the Vela bins in ms.
    :param values: The swept values, one array per input.
    :param matrix: True for a matrix sweep, False for a paired one.
    :returns best: Where the run sits in the sweep.
    :returns where: The value of every swept input at that run.
    """
    names = list(values)
    best = tuple(int(i) for i in np.unravel_index(np.nanargmin(rms), rms.shape))
    # paired runs share one index
    at = (
        dict(zip(names, best, strict=False))
        if matrix
        else dict.fromkeys(names, best[0])
    )
    return best, {n: values[n][i] for n, i in at.items()}
