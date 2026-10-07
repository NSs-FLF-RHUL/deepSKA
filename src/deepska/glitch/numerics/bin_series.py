# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Binning for a time series, used on the Vela residuals."""

import numpy as np

from deepska.glitch.kinds import Arr


def bin_series(t: Arr, y: Arr, width: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Average a time series in bins, with one bin centred on t = 0.

    As the Vela data wants a bin on the glitch.

    :param t: Time of every sample, increasing.
    :param y: Value of every sample.
    :param width: Bin width, in the units of t.
    :returns centre: Centre of every bin.
    :returns means: Mean of y in every bin, nan if the bin is empty.
    """
    edges = np.arange(np.floor(t[0] / width) * width - width / 2, t[-1] + width, width)
    idx = np.digitize(t, edges) - 1
    # nan if a bin is empty
    means = np.array(
        [
            y[idx == i].mean() if np.any(idx == i) else np.nan
            for i in range(len(edges) - 1)
        ]
    )
    centre = edges[:-1] + width / 2
    return centre, means
