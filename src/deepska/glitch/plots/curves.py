# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Curves in the form figure takes."""

from deepska.glitch.kinds import Arr, Curve, Num, Style


def curves(x: Num, ys: Arr, styles: list[Style]) -> list[Curve]:
    """
    Pair each row of ys with its style, in the form figure takes.

    :param x: The x values, shared by every curve.
    :param ys: The y values, one row per curve.
    :param styles: A (fmt, style dict) per curve, same length as ys.
    :returns: A list of (x, y, fmt, style), one per curve.
    """
    return [(x, y, fmt, style) for y, (fmt, style) in zip(ys, styles, strict=False)]
