# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Two tables and one figure for the sensitivity script.

How each output answers to each input, how far each output is from converged, and the
rise against time.
"""

import logging

import numpy as np

from deepska.glitch.data.constants import km
from deepska.glitch.kinds import Axes, Num, Outputs
from deepska.glitch.plots.figure import figure
from deepska.glitch.plots.table import table

log = logging.getLogger(__name__)

shown = ["peak", "end", "dnu_equil", "r_star", "rms"]  # the outputs in the tables
header = f"{'':14} {'peak':>9} {'at 120 s':>9} {'eq 23':>9} {'R':>9} {'rms':>9}"


def sensitivity_output(
    names: list[str],
    base: Outputs,
    slopes: Outputs,
    grid_rows: list[tuple[str, Outputs]],
    t_hist: Num,
) -> None:
    """
    Print the two sensitivity tables and draw the figure.

    :param names: The inputs that were varied.
    :param base: The outputs as set.
    :param slopes: The slope of each output against each input.
    :param grid_rows: A (label, outputs) row per grid variant, from grid_check.
    :param t_hist: Time of every step in s.
    """
    log.info(
        "as set: peak %.2f muHz, at 120 s %.3f muHz, eq 23 %.3f muHz, "
        "R %.3f km, rms %.4f ms",
        base["peak"],
        base["end"],
        base["dnu_equil"],
        base["r_star"] / km,
        base["rms"],
    )
    log.info("")
    rows = [
        (n, *tuple(float(slopes[k][i] / base[k]) for k in shown))
        for i, n in enumerate(names)
    ]
    table(
        "percent change in each output for a 1 percent change in each input",
        header,
        rows,
        "%-14s %9.3f %9.3f %9.3f %9.3f %9.3f",
    )
    log.info("")
    as_set = grid_rows[0][1]
    rows = [
        (label, *tuple(100 * float(out[k] / as_set[k] - 1) for k in shown))
        for label, out in grid_rows[1:]
    ]
    table(
        "percent change in each output when a grid is made finer (x2) "
        "or coarser (x0.5)",
        header,
        rows,
        "%-14s %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f",
    )

    curves = [
        (t_hist, np.abs(slopes["dnu"][:, i]), "-", {"label": n})
        for i, n in enumerate(names)
    ]
    axes = Axes(
        (-1, 60),
        (1e-3, None),
        "t (s)",
        "| d dnu / d ln(input) |  (muHz per e-fold)",
        "sensitivity of the glitch rise to each input",
        ylog=True,
    )
    figure(curves, axes)
