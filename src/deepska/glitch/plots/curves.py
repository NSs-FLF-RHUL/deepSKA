# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Curves in the form figure takes.

- curves: pair each row of y values with its style.
- curves_coloured: one thin line per run of a sweep.
- curves_profile: the superfluid profile at chosen times.
- curves_seven: the seven drag profiles of the paper.
- curves_cases: the three pinning cases, with dots at their knots.
"""

from deepska.glitch.data.paper_labels import case_styles, dot_styles, profile_styles
from deepska.glitch.kinds import Arr, Colour, Curve, Num, Style


def curves(x: Num, ys: Arr, styles: list[Style]) -> list[Curve]:
    """
    Pair each row of ys with its style, in the form figure takes.

    :param x: The x values, shared by every curve.
    :param ys: The y values, one row per curve.
    :param styles: A (fmt, style dict) per curve, same length as ys.
    :returns: A list of (x, y, fmt, style), one per curve.
    """
    return [(x, y, fmt, style) for y, (fmt, style) in zip(ys, styles, strict=False)]


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


def curves_seven(x: Num, ys: Num) -> list[Curve]:
    """
    Make the curves of the seven drag profiles of the paper.

    :param x: The x values, shared by every curve.
    :param ys: The y values, one row per profile, flat ones then A, B, C.
    :returns: The curves, in the line styles of the paper.
    """
    return curves(x, ys, profile_styles)


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
