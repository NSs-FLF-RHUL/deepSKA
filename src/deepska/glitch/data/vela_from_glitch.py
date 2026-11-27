# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The binned Vela data from the glitch onwards."""

import numpy as np

from deepska.glitch.kinds import Arr


def vela_from_glitch(
    centre: Arr, vela_bin: Arr
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Cut the binned Vela data to start at the glitch.

    :param centre: Centre of every bin in s from the glitch.
    :param vela_bin: Mean residual in every bin in ms.
    :returns t_bins: Bin centres from the glitch on in s.
    :returns vela_from0: Mean residual in each of those bins in ms.
    :returns dt_shift: Residual in the glitch bin in ms.
    :returns cum_data: Running sum of the residuals above dt_shift in ms.
    """
    # the bin at the glitch, nearest to 0 rather than an exact float test
    g0 = int(np.argmin(np.abs(centre)))
    # the model residual gets lifted by this so both start in the same place
    dt_shift = vela_bin[g0]
    t_bins, vela_from0 = centre[g0:], vela_bin[g0:]
    cum_data = np.cumsum(vela_from0 - dt_shift)
    return t_bins, vela_from0, dt_shift, cum_data
