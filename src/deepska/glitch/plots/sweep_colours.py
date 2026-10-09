# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Colours and colour bars for a sweep.

- colour: a colour from three numbers between 0 and 1.
- unit: scale a value to between 0 and 1.
- channel_scale: the colour map and scale for one colour bar.
- sweep_colours: a colour per drawn run and a bar per swept input.

The first swept input sets the hue, the second how bright, the third how strong the
colour is. The curves and the colour bars both come from here so they always match.
"""

import colorsys
from typing import Any

import numpy as np
from matplotlib.colors import ListedColormap, LogNorm, Normalize

from deepska.glitch.kinds import Bar, Colour, Grid, Num

LOG_ABOVE = 30  # an input spanning more than this factor gets a log colour scale
MAX_TICKS = 6  # up to this many values, a tick at each one


def colour(u1: float, u2: float = 0.5, u3: float = 1.0) -> Colour:
    """
    Make a colour from three numbers between 0 and 1.

    :param u1: Hue, purple round to yellow.
    :param u2: Brightness, dark to light.
    :param u3: Strength, grey to full colour.
    :returns: The colour as red, green, blue.
    """
    return colorsys.hls_to_rgb(0.75 - 0.62 * u1, 0.25 + 0.5 * u2, 0.3 + 0.7 * u3)


def unit(v: Num, low: Num, high: Num, *, log: bool) -> np.ndarray:
    """
    Scale a value to between 0 and 1.

    :param v: The value.
    :param low: The value that maps to 0.
    :param high: The value that maps to 1.
    :param log: True to scale in log.
    :returns: The scaled value, clipped to 0..1.
    """
    v, low, high = (np.log(v), np.log(low), np.log(high)) if log else (v, low, high)
    return np.clip((v - low) / (high - low), 0.0, 1.0)


def channel_scale(channel: int, low: Num, high: Num, *, log: bool) -> tuple[Any, Any]:
    """
    Make the colour map and scale for one colour bar.

    :param channel: Which part of the colour, 0 hue, 1 brightness, 2 strength.
    :param low: Value at the bottom of the bar.
    :param high: Value at the top of the bar.
    :param log: True for a log scale.
    :returns cmap: The colour map.
    :returns norm: The scale from values to colours.
    """
    u = np.linspace(0, 1, 256)
    # hue, at middle brightness and full strength
    ramp = [
        [colour(x) for x in u],
        # brightness, shown in grey
        [colorsys.hls_to_rgb(0.0, 0.25 + 0.5 * x, 0.0) for x in u],
        # strength, shown in blue
        [colour(0.25, 0.5, x) for x in u],
    ][channel]
    return ListedColormap(ramp), (LogNorm(low, high) if log else Normalize(low, high))


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
