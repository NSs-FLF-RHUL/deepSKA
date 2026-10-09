# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Colours and colour bars for a sweep."""

from deepska.glitch.kinds import Bar, Colour, Grid
from deepska.glitch.plots.colour_scale import channel_scale, colour, unit

LOG_ABOVE = 30  # an input spanning more than this factor gets a log colour scale
MAX_TICKS = 6  # up to this many values, a tick at each one


def sweep_colours(shown: Grid, values: Grid) -> tuple[list[Colour], list[Bar]]:
    """
    Make a colour for each drawn run and a colour bar for each swept input.

    Only the first three inputs get shown.

    :param shown: The inputs of the drawn runs.
    :param values: The swept values of every input.
    :returns colours: The colour of each drawn run.
    :returns bars: One colour bar per swept input.
    """
    # a colour only has three things to change, any more inputs dont show
    names = list(values)[:3]
    # log for inputs spanning decades
    logs = {
        n: bool(values[n].min() > 0 and values[n].max() / values[n].min() > LOG_ABOVE)
        for n in names
    }
    us = [unit(shown[n], values[n].min(), values[n].max(), log=logs[n]) for n in names]
    colours = [colour(*[u[i] for u in us]) for i in range(len(shown[names[0]]))]
    bars = []
    for c, n in enumerate(names):
        cmap, norm = channel_scale(
            c, float(values[n].min()), float(values[n].max()), log=logs[n]
        )
        # few values: tick each one
        ticks = list(values[n]) if len(values[n]) <= MAX_TICKS else None
        bars.append((cmap, norm, n, ticks, None if logs[n] else "%.3g"))
    return colours, bars
