# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
One figure, the only file that draws.

curves is a list of (x, y, fmt, style), fmt like "-" ":" "o" "o-" and style a dict of
colour, label etc.
"""

import matplotlib.pyplot as plt

from deepska.glitch.kinds import Axes, Bar, Curve


def figure(curves: list[Curve], axes: Axes, bars: list[Bar] | None = None) -> None:
    """
    Draw one figure.

    :param curves: A list of (x, y, fmt, style), one per curve.
    :param axes: Limits, labels and title.
    :param bars: A (cmap, norm, label, ticks, format) per colour bar, None for none.
    """
    if bars:
        # wider so the extra colour bars dont squash the plot
        plt.figure(figsize=(6.4 + 1.3 * (len(bars) - 1), 4.8))
    else:
        plt.figure()
    if axes.xlog:
        plt.xscale("log")
    if axes.ylog:
        plt.yscale("log")
    for x, y, fmt, style in curves:
        plt.plot(x, y, fmt, **style)
    plt.xlim(*axes.xlim)
    plt.ylim(*axes.ylim)  # (None, None) lets it pick its own
    if not axes.offset:
        plt.ticklabel_format(useOffset=False)
    plt.xlabel(axes.xlabel)
    plt.ylabel(axes.ylabel)
    plt.title(axes.title)
    plt.legend()
    ax = plt.gca()
    # reversed as each new bar goes in next to the plot, so the first ends up nearest
    for cmap, norm, label, ticks, fmt in reversed(bars or []):
        plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=ax,
            label=label,
            ticks=ticks,
            format=fmt,
        )
    plt.show()
