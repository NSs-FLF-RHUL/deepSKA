# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Cutting the binned Vela data off at a time."""

import numpy as np

from deepska.glitch.kinds import Arr


def vela_up_to(
    t_bins: Arr, vela_bin: Arr, cum_data: Arr, t_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cut the binned Vela data off at a time.

    :param t_bins: Bin centres in s.
    :param vela_bin: Mean residual in each bin in ms.
    :param cum_data: Running sum of the residuals in ms.
    :param t_max: Latest time to keep in s.
    :returns: The three arrays with only the bins up to t_max.
    """
    keep = t_bins <= t_max
    return t_bins[keep], vela_bin[keep], cum_data[keep]
