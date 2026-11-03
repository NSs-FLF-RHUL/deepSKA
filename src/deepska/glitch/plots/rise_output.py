# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The glitch rise of one run: the frequency change of the crust against time."""

import numpy as np

from deepska.glitch.kinds import Axes, Num
from deepska.glitch.plots.figure import figure


def rise_output(t: Num, dnu: Num, title: str) -> None:
    """
    Draw the glitch rise, the frequency change of the crust.

    :param t: Time of every step in s.
    :param dnu: Frequency change of the crust at every step in muHz.
    :param title: What the run was, for the title.
    """
    figure(
        [(np.array(t), np.array(dnu), "-", {"color": "tab:blue", "label": "crust"})],
        Axes((0, None), (None, None), "t (s)", r"$\Delta\nu$ ($\mu$Hz)", title),
    )
