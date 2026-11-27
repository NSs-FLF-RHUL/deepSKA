# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Running sum of the model residual at the data bins, for fig 12."""

import numpy as np

from deepska.glitch.analysis.at_bins import at_bins
from deepska.glitch.kinds import Num


def cumulative_residual(
    res: Num, t_hist: Num, t_bins: Num, dt_shift: Num
) -> np.ndarray:
    """
    Calculate the cumulative model residual at the Vela bins.

    :param res: Timing residual at every step in ms.
    :param t_hist: Time of every step in s.
    :param t_bins: Bin centres in s.
    :param dt_shift: Residual at the glitch in ms, taken off so the sum starts at 0.
    :returns: The running sum of the residual over the bins in ms.
    """
    # the model read off at the data's bin times
    model_bins = at_bins(res, t_hist, t_bins)
    # dt_shift off first so it starts at 0 like the data
    return np.cumsum(model_bins - dt_shift)
