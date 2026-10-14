# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Vela 2016 glitch timing residuals, Palfreyman et al 2018."""

import numpy as np

from deepska.glitch.data.constants import day, ms, ns
from deepska.glitch.data.vela_from_glitch import vela_from_glitch
from deepska.glitch.numerics.bin_series import bin_series


def vela_data(
    path: str, t_g: float, t_0: float, width: float
) -> tuple[np.ndarray, ...]:
    """
    Read the Vela timing residuals and bin them.

    The pre glitch mean is taken off first.

    :param path: Path to the csv of mjd and residual in ns.
    :param t_g: Glitch time in MJD.
    :param t_0: Time in MJD that the pre glitch mean is taken up to.
    :param width: Bin width in s.
    :returns t_bins: Bin centres from the glitch on in s.
    :returns vela_from0: Mean residual in each of those bins in ms.
    :returns dt_shift: Residual in the glitch bin in ms.
    :returns cum_data: Running sum of the residuals above dt_shift in ms.
    :returns centre: Centre of every bin in s, including before the glitch.
    :returns vela_bin: Mean residual in every bin in ms.
    :returns sec: Time of every pulse in s from the glitch.
    :returns res: Residual of every pulse in ms.
    """
    mjd, res = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    # take off the pre glitch mean, ns -> ms
    res = (res - res[mjd <= t_0].mean()) * ns / ms
    sec = (mjd - t_g) * day
    centre, vela_bin = bin_series(sec, res, width)  # one bin centred on t = 0
    t_bins, vela_from0, dt_shift, cum_data = vela_from_glitch(centre, vela_bin)
    return t_bins, vela_from0, dt_shift, cum_data, centre, vela_bin, sec, res
