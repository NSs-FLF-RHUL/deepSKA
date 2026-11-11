# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
What gets kept from each run of a sweep.

A few numbers and the residual at the Vela bins, not the whole rise.
"""

from collections.abc import Callable

import jax.numpy as jnp

from deepska.glitch.analysis.at_bins import at_bins
from deepska.glitch.analysis.phase_of_rise import phase_of_rise
from deepska.glitch.analysis.timing_residual import timing_residual
from deepska.glitch.kinds import Arr, Inputs, Num, Outputs, Star, Vela


def make_run_summary(
    t_hist: Num, vela: Vela, dt: Num, *, keep_dnu: bool
) -> Callable[..., Outputs]:
    """
    Make the per_run function that keeps a summary of each run.

    The summary is the peak, when it happens, the end value, eq 23, the star and the rms
    distance from the Vela bins in ms.

    :param t_hist: Time of every step in s.
    :param vela: The Vela data.
    :param dt: Time step in s.
    :param keep_dnu: Also keep the whole rise, 160 KB per run so only for small sweeps.
    :returns: Function of (dnu, profiles, star, p) giving the summary.
    """

    def run_summary(dnu: Arr, _profiles: Num, star: Star, p: Inputs) -> Outputs:
        # this runs own omega0, as it may be swept
        res = timing_residual(phase_of_rise(dnu, dt), p["omega0"], vela.dt_shift)
        bins = at_bins(res, t_hist, vela.t_bins)
        out = {
            "peak": dnu.max(),
            "t_peak": (jnp.argmax(dnu) + 1) * dt,
            "end": dnu[-1],
            "dnu_equil": star.dnu_equil,
            "r_star": star.r_star,
            "m_star": star.m_star,
            "r_drip": star.r_drip,
            "rms": jnp.sqrt(jnp.nanmean((bins - vela.vela_bin) ** 2)),
            "bins": bins,
        }
        if keep_dnu:
            out["dnu"] = dnu
        return out

    return run_summary
