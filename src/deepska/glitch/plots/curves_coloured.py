# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Curves for the runs of a sweep."""

from deepska.glitch.kinds import Colour, Curve, Num
from deepska.glitch.plots.curves import curves


def curves_coloured(x: Num, ys: Num, colours: list[Colour]) -> list[Curve]:
    """
    Make one line per run of a sweep, each in its own colour.

    Thin and no labels as theres too many for a legend.

    :param x: The x values, shared by every curve.
    :param ys: The y values, one row per run.
    :param colours: The colour of each run.
    :returns: The curves.
    """
    styles = [("-", {"color": c, "linewidth": 0.8}) for c in colours]
    return curves(x, ys, styles)
