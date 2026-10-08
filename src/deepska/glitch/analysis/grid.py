# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Checking that the outputs dont depend on the numerical grids.

- The numerical grids made finer or coarser by a factor.
- How far the outputs are from converged on the numerical grids.
"""

import numpy as np

from deepska.glitch.analysis.make_run_summary import make_run_summary
from deepska.glitch.glitch_rise_model.run.param_to_output import param_to_output
from deepska.glitch.glitch_rise_model.run.records_of import records_of
from deepska.glitch.kinds import Inputs, Outputs, Vela



def grid_variants(inputs: Inputs, factors: list[float]) -> list[tuple[str, Inputs]]:
    """
    Make the numerical grids finer or coarser, one at a time.

    The extent stays the same, so a finer grid is more points over the same radius or
    the same 120 s.

    :param inputs: The inputs of the run by name.
    :param factors: Factors to change each grid by, 2 is twice the points.
    :returns: A (label, changed inputs) pair per grid and factor.
    """
    variants = []
    for f in factors:
        variants.append(
            (
                f"tov grid x{f:g}",
                {"dr": inputs["dr"] / f, "n_tov": round(inputs["n_tov"] * f)},
            )
        )
        variants.append(
            (
                f"time grid x{f:g}",
                {"dt": inputs["dt"] / f, "n_t": round(inputs["n_t"] * f)},
            )
        )
        variants.append(
            (
                f"shells x{f:g}",
                {"n_shells": round((inputs["n_shells"] - 1) * f) + 1},
            )
        )
    return variants


def grid_check(
    inputs: Inputs, variants: list[tuple[str, Inputs]], vela: Vela
) -> list[tuple[str, Outputs]]:
    """
    Repeat the same run on each grid variant.

    A plain loop not a vmap as every variant has different array sizes.

    :param inputs: The inputs of the run by name.
    :param variants: Label and changed inputs of each variant, from grid_variants.
    :param vela: The Vela data.
    :returns: A (label, outputs) row per variant, the first is the inputs as set.
    """
    rows = []
    for label, change in [("as set", {}), *variants]:
        p = {**inputs, **change, "profile_steps": [0]}  # no profiles needed here
        t_hist = p["dt"] * np.arange(1, p["n_t"] + 1)
        star_inputs, spin = records_of(p)
        dnu, profiles, star = param_to_output(
            star_inputs, p["b_profile"], p["b_core"], spin
        )
        summary = make_run_summary(t_hist, vela, p["dt"], keep_dnu=False)
        rows.append((label, summary(dnu, profiles, star, p)))
    return rows