# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Curves for the seven profiles of the paper."""

from deepska.glitch.data.paper_labels import profile_styles
from deepska.glitch.kinds import Curve, Num
from deepska.glitch.plots.curves import curves


def curves_seven(x: Num, ys: Num) -> list[Curve]:
    """
    Make the curves of the seven drag profiles of the paper.

    :param x: The x values, shared by every curve.
    :param ys: The y values, one row per profile, flat ones then A, B, C.
    :returns: The curves, in the line styles of the paper.
    """
    return curves(x, ys, profile_styles)
