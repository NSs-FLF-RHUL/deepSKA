# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Two tables and three figures for the metamodel script.

Why draws were rejected and the range of each input among the kept ones, as tables.
Then every kept rise, the rejected draws by reason, and the sensitivity of the rise to
each input. These three are drawn here and not by figure.py, which has no bar chart,
no thousand lines at once and no symlog axis. Each gives its figure back, so the
script decides what to do with it.

Contents:

- metamodel_output: the two tables.
- rises_figure: every kept draw's glitch rise.
- reasons_figure: the rejected draws, by reason.
- sensitivity_figure: d dnu(t) / dx for every input.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from deepska.glitch.data.metamodel_inputs import (
    b_core_prior,
    b_crust_prior,
    n_c_prior,
    priors,
)
from deepska.glitch.kinds import Arr
from deepska.glitch.plots.table import table

log = logging.getLogger(__name__)


def metamodel_output(result: dict[str, Any]) -> None:
    """
    Print the two tables of a run.

    How many draws each reason rejected, then the range of each of the 13 inputs among
    the kept draws against its prior.

    :param result: The run, from run_draws.
    """
    n_draws = result["n_draws"]
    log.info(
        "%s draws on %d cores: checks %.0f s, eos + star + rise on %s %.0f s",
        f"{n_draws:,}",
        result["n_cores"],
        result["seconds"][0],
        f"{result['n_run']:,}",
        result["seconds"][1],
    )
    log.info("")
    reasons = [
        (name, f"{count:,}", 100 * count / n_draws)
        for name, count in [*result["counts"].items(), ("KEPT", result["n_kept"])]
    ]
    table(
        "why each draw was rejected, counted under the first reason it fails",
        f"{'why rejected':<18}{'count':>10}{'% of all':>10}",
        reasons,
        "%-18s%10s%9.2f%%",
    )
    log.info("")
    columns = [
        (name, result["values"][:, 2 + j], box)
        for j, (name, box) in enumerate(priors.items())
    ]
    columns += [
        ("n_c", result["n_c"], n_c_prior),
        ("B_core", result["b_core"], b_core_prior),
        ("B_crust", result["b_crust"], b_crust_prior),
    ]
    ranges = [
        (name, float(col.min()), float(col.max()), f"[{low:g}, {high:g}]")
        for name, col, (low, high) in columns
    ]
    table(
        "the range of each input among the kept draws",
        f"{'input':<8}{'kept min':>12}{'kept max':>12}{'prior':>22}",
        ranges,
        "%-8s%12.4g%12.4g%22s",
    )


def rises_figure(t: Arr, dnu: Arr, **style: str | float) -> Figure:
    """
    Draw every kept draw's glitch rise on one figure, with their median.

    :param t: The time of each step in s, (steps,).
    :param dnu: The rise of every kept draw in muHz, (draws, steps).
    :param style: Passed on to the lines of the draws, over the defaults here.
    :returns: The figure.
    """
    top = 99.5  # the y axis stops at this percentile of the peaks, not at the highest
    options: dict[str, Any] = {
        "colors": "C0",
        "linewidths": 0.4,
        "alpha": 0.08,
        **style,
    }
    lines = np.stack([np.broadcast_to(t, dnu.shape), dnu], axis=-1)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.add_collection(LineCollection(list(lines), **options))
    ax.plot(t, np.median(dnu, axis=0), color="k", lw=2, label="median")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(min(0.0, float(dnu.min())), float(np.percentile(dnu.max(axis=1), top)))
    ax.set_xlabel("time since glitch (s)")
    ax.set_ylabel(r"$\Delta\nu \ (\mu\mathrm{Hz})$")
    ax.set_title(f"Glitch rises of all {dnu.shape[0]:,} kept draws (13 inputs drawn)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def reasons_figure(
    counts: dict[str, int], n_draws: int, n_kept: int, **style: str | float
) -> Figure:
    """
    Draw the rejected draws as a bar chart: the reason along x, how many up y.

    The y axis is log, so a reason that rejected none shows no bar, only its 0.

    :param counts: How many draws each reason rejected, from run_draws.
    :param n_draws: How many draws there were.
    :param n_kept: How many were kept.
    :param style: Passed on to the bars, over the defaults here.
    :returns: The figure.
    """
    meaning = {
        "neutron gas": "neutron gas\nenergy far from\nNeill et al",
        "symmetric matter": "symmetric matter\nenergy far from\nDrischler et al",
        "E_sym <= 0": r"$E_{sym} \leq 0$" + "\nsomewhere in\n[0.02, $n_c$]",
        "unstable": "unstable\n" + r"($dP/dn \leq 0$)",
        "acausal": "faster than\nlight\n" + r"($c_s^2 \geq 1$)",
        "no drip": "no drip\n(no cell with\n$n_g > 0$)",
        "no join": "no join\n" + r"(no $\Delta\mu$" + "\nsign change)",
        "P falls": "P falls\nalong the\ntable",
        "no surface": "no surface\n(TOV out of\nsteps)",
    }
    options: dict[str, Any] = {"color": "C3", **style}
    names = [meaning.get(name, name) for name in counts]
    heights = list(counts.values())
    fig, ax = plt.subplots(figsize=(13, 6.5))
    bars = ax.bar(names, heights, **options)
    for bar, height in zip(bars, heights, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(height, 1) * 1.15,
            f"{height:,}",
            ha="center",
            fontsize=9,
        )
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3 * max(*heights, 1))
    ax.set_ylabel("rejected draws")
    ax.set_title(f"Why draws are rejected ({n_draws:,} draws, {n_kept:,} kept)")
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    return fig


def sensitivity_figure(
    t: Arr, slopes: dict[str, np.ndarray], title: str, **style: str | float
) -> Figure:
    """
    Draw d dnu(t) / dx for every input on one figure.

    The y axis is symlog, log either side of 0 and straight between -1e-3 and 1e-3, as
    the slopes are in different units and span nine decades. So the heights say what
    the units are, not which input matters more; the shapes and the timing can be
    compared. Solid lines are the symmetric matter inputs, dashed the symmetry energy
    ones, dotted the star and the mutual friction.

    :param t: The time of each step in s.
    :param slopes: d dnu / dx at every step in muHz per unit of x, by name, from
        rise_sensitivity: the ten inputs first, then n_c, B_core and B_crust.
    :param title: The title of the figure.
    :param style: Passed on to every line, over the defaults here.
    :returns: The figure.
    """
    n_sat_like, n_eos = 5, len(priors)  # the symmetric inputs, then all ten
    density = r"$\mathrm{fm}^{-3}$"
    labels = {
        "E_sat": ("$E_{sat}$", "MeV"),
        "n_sat": ("$n_{sat}$", density),
        "K_sat": ("$K_{sat}$", "MeV"),
        "Q_sat": ("$Q_{sat}$", "MeV"),
        "Z_sat": ("$Z_{sat}$", "MeV"),
        "E_sym": ("$E_{sym}$", "MeV"),
        "L_sym": ("$L_{sym}$", "MeV"),
        "K_sym": ("$K_{sym}$", "MeV"),
        "Q_sym": ("$Q_{sym}$", "MeV"),
        "Z_sym": ("$Z_{sym}$", "MeV"),
        "n_c": ("$n_c$", density),
        "B_core": ("$B_{core}$", "unit $B$"),
        "B_crust": ("$B_{crust}$", "unit $B$"),
    }
    fig, ax = plt.subplots(figsize=(11, 7.5))
    colours = plt.get_cmap("tab20")  # 20 colours, in light and dark pairs
    for k, (name, slope) in enumerate(slopes.items()):
        label, unit = labels[name]
        star = k >= n_eos  # n_c and the two B, drawn thicker
        options: dict[str, Any] = {
            "ls": "-" if k < n_sat_like else ("--" if k < n_eos else ":"),
            "lw": 2.4 if star else 1.6,
            "color": colours((2 * k) % 20 + star),
            "label": label + r" ($\mu\mathrm{Hz}$ per " + unit + ")",
            **style,
        }
        ax.plot(t, slope, **options)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time since glitch (s)")
    ax.set_ylabel(
        r"$\partial\Delta\nu(t) / \partial x$ ($\mu\mathrm{Hz}$ per unit of $x$)"
    )
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
