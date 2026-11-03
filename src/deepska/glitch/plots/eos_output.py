# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The three figures of one metamodel eos: P(n_b), eps(n_b) and P(eps).

All three are log-log, MeV and fm, with the crust points and the core points drawn
apart and a dashed line at n_cc on the two against n_b.
"""

import numpy as np

from deepska.glitch.kinds import Axes, Curve
from deepska.glitch.plots.figure import figure


def split(x: np.ndarray, y: np.ndarray, crust: np.ndarray) -> list[Curve]:
    """
    Split one curve into its crust part, in blue, and its core part, in red.

    :param x: The x values.
    :param y: The y values.
    :param crust: True where the point lies in the crust.
    :returns: The crust curve then the core curve.
    """
    style = {"markersize": 3}
    return [
        (x[crust], y[crust], "o-", {"color": "tab:blue", "label": "crust", **style}),
        (x[~crust], y[~crust], "o-", {"color": "tab:red", "label": "core", **style}),
    ]


def at_ncc(n_cc: float, y: np.ndarray) -> Curve:
    """
    Make a dashed upright line at n_cc spanning the y values.

    :param n_cc: The crust-core transition density in fm^-3.
    :param y: The y values the line has to span, on a log axis.
    :returns: The line.
    """
    pos = y[y > 0]
    label = f"n_cc = {n_cc:.4f} fm$^{{-3}}$"
    return (
        [n_cc, n_cc],
        [pos.min(), pos.max()],
        "--",
        {"color": "grey", "label": label},
    )


def eos_output(table: dict[str, np.ndarray], n_cc: float) -> None:
    """
    Draw P(n_b), eps(n_b) and P(eps) for one eos.

    :param table: The joined eos, nb, eps, p and the crust flag, from joined.
    :param n_cc: The crust-core transition density in fm^-3.
    """
    nb, eps, p, crust = table["nb"], table["eps"], table["p"], table["crust"]
    n_label = r"$n_b$ (fm$^{-3}$)"
    p_label = r"$P$ (MeV fm$^{-3}$)"
    eps_label = r"$\varepsilon$ (MeV fm$^{-3}$)"
    free = (None, None)  # let matplotlib pick the limits
    figure(
        [*split(nb, p, crust), at_ncc(n_cc, p)],
        Axes(free, free, n_label, p_label, "pressure", xlog=True, ylog=True),
    )
    figure(
        [*split(nb, eps, crust), at_ncc(n_cc, eps)],
        Axes(free, free, n_label, eps_label, "energy density", xlog=True, ylog=True),
    )
    figure(
        split(eps, p, crust),
        Axes(free, free, eps_label, p_label, "the eos", xlog=True, ylog=True),
    )
