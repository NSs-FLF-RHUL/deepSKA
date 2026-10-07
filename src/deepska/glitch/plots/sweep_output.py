# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The one figure of a sweep.

The timing residual of the drawn runs over the Vela data, with a colour bar for each
swept input (up to three).
"""

from deepska.glitch.kinds import Axes, Drawn, Grid, Num, Vela
from deepska.glitch.plots.curves_coloured import curves_coloured
from deepska.glitch.plots.figure import figure
from deepska.glitch.plots.sweep_colours import sweep_colours
from deepska.glitch.plots.vela_curves import vela_curves


def sweep_output(
    values: Grid, drawn: Drawn, runs: int, t_hist: Num, vela: Vela
) -> None:
    """
    Draw the one figure of a sweep.

    :param values: The swept values of every input.
    :param drawn: The inputs and the whole residuals of the runs to draw.
    :param runs: Number of runs the sweep did in all.
    :param t_hist: Time of every step in s.
    :param vela: The Vela data.
    """
    colours, bars = sweep_colours(drawn.shown, values)
    _pulses, _bins, _cum, from0 = vela_curves(vela, top=True)
    axes = Axes(
        (-1, 120),
        (-0.35, 0.25),
        "t (s)",
        "timing residuals (ms)",
        f"timing residuals against the Vela data, {runs} runs, {len(colours)} drawn",
    )
    figure([*curves_coloured(t_hist, drawn.res, colours), from0], axes, bars=bars)
