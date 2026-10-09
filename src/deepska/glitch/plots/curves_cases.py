# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Curves for the three pinning cases, with dots at their knots."""

from deepska.glitch.data.paper_labels import case_styles, dot_styles
from deepska.glitch.kinds import Curve, Num
from deepska.glitch.plots.curves import curves


def curves_cases(x: Num, ys: Num, x_dots: Num, y_dots: Num) -> list[Curve]:
    """
    Make a line for each pinning case with dots at its knots.

    :param x: The x values of the lines.
    :param ys: The y values of the lines, one row per case.
    :param x_dots: The x values of the knots.
    :param y_dots: The y values of the knots, one row per case.
    :returns: The curves, each line followed by its dots.
    """
    lines = curves(x, ys, case_styles)
    dots = curves(x_dots, y_dots, dot_styles)
    # line then its dots, case by case
    return [c for pair in zip(lines, dots, strict=False) for c in pair]
