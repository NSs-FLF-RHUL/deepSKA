# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Curves for a superfluid profile figure."""

from deepska.glitch.kinds import Arr, Curve, Num
from deepska.glitch.plots.curves import curves


def curves_profile(
    r_km: Num,
    profiles_run: Arr,
    profile_steps: list[int],
    dt: Num,
    steps: list[int],
) -> list[Curve]:
    """
    Make the curves of a superfluid profile figure.

    :param r_km: Radius of every shell in km.
    :param profiles_run: Superfluid spin of every shell at the kept steps, in rad/s.
    :param profile_steps: The steps that were kept.
    :param dt: Time step in s.
    :param steps: The steps to draw, a subset of profile_steps.
    :returns: One curve per step, labelled with its time.
    """
    ys = [profiles_run[profile_steps.index(s)] for s in steps]
    styles = [("-", {"label": "t = %g s" % (s * dt)}) for s in steps]
    return curves(r_km, ys, styles)
