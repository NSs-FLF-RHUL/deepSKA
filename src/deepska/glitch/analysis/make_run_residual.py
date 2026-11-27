# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The whole timing residual of a run, for the few runs that get drawn as curves."""

from collections.abc import Callable

from deepska.glitch.analysis.phase_of_rise import phase_of_rise
from deepska.glitch.analysis.timing_residual import timing_residual
from deepska.glitch.kinds import Arr, Inputs, Num, Outputs, Star


def make_run_residual(dt: Num, dt_shift: Num) -> Callable[..., Outputs]:
    """
    Make the per_run function that keeps the whole timing residual.

    :param dt: Time step in s.
    :param dt_shift: Residual at the glitch in ms, the model is lifted by it.
    :returns: Function of (dnu, profiles, star, p) giving the residual in ms.
    """

    def run_residual(dnu: Arr, _profiles: Num, _star: Star, p: Inputs) -> Outputs:
        return {"res": timing_residual(phase_of_rise(dnu, dt), p["omega0"], dt_shift)}

    return run_residual
