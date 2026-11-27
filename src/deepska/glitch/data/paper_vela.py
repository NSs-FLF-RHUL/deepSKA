# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The Vela data as the paper figures use it."""

from deepska.glitch.data.paper_inputs import t_0, t_g, vela_bin_width

from deepska.glitch.kinds import Vela, Arr
import numpy as np
from deepska.glitch.numerics.bin_series import bin_series
from deepska.glitch.data.constants import day, ms, ns
import requests
from pathlib import Path

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


def paper_vela(path: str, t_max: float) -> Vela:
    """
    Load the Vela data with the glitch time and bin width of the paper.

    :param path: Path to the csv of residuals.
    :param t_max: Time after the glitch in s that the binned arrays are cut at.
    :returns: The Vela arrays the figures need.
    """
    t_bins, vela_bin, dt_shift, cum_data, t_all, vela_all, t_raw, res_raw = vela_data(
        path, t_g, t_0, vela_bin_width
    )
    t_bins, vela_bin, cum_data = vela_up_to(t_bins, vela_bin, cum_data, t_max)
    return Vela(t_bins, vela_bin, dt_shift, cum_data, t_all, vela_all, t_raw, res_raw)
