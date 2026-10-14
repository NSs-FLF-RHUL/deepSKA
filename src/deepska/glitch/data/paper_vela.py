# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The Vela data as the paper figures use it."""

from deepska.glitch.data.paper_inputs import t_0, t_g, vela_bin_width
from deepska.glitch.data.vela_data import vela_data
from deepska.glitch.data.vela_up_to import vela_up_to
from deepska.glitch.kinds import Vela


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
