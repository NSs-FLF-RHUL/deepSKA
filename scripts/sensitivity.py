# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
How sensitive the glitch rise is to each input, and to the numerical grids.

Spin inputs by autodiff, star inputs by a small nudge each way, grids by running again
finer and coarser.
"""

import sys

from deepska.glitch.kinds import Outputs

if __name__ == "__main__":
    for name in [
        name
        for name in sys.modules
        if name == "deepska.glitch" or name.startswith("deepska.glitch.")
    ]:
        # forget the glitch files loaded by an earlier run, so this run reads every one
        # fresh from disk
        del sys.modules[name]

import logging
import time

import numpy as np

from deepska.glitch.analysis.grid_check import grid_check
from deepska.glitch.analysis.grid_variants import grid_variants
from deepska.glitch.analysis.make_run_summary import make_run_summary
from deepska.glitch.analysis.sensitivity import sensitivity
from deepska.glitch.data.paper_inputs import (
    PROFILE_STEPS,
    dom_crit,
    dr,
    dt,
    m0,
    n_shells,
    n_t,
    n_tov,
    omega0,
    r0,
    rho_cc,
    rho_d,
    rho_min,
    t_end,
    vela_path,
)
from deepska.glitch.data.paper_vela import paper_vela
from deepska.glitch.glitch_rise_model.drag.make_b_profile import make_b_profile
from deepska.glitch.glitch_rise_model.eos.negele_vautherin_electron_crust import p_crust
from deepska.glitch.plots.sensitivity_output import sensitivity_output

# plain lines on screen, same as print would give
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# settings: change these
# any of the numbers in INPUTS below
inputs_to_vary = [
    "b_core",
    "dom_crit",
    "omega0",
    "rho_d",
    "rho_cc",
    "rho_min",
    "m0",
    "r0",
]
# each grid tried this many times finer, 2 = twice the points, 0.5 = half
grid_factors = [0.5, 2.0]
# the nudge for the star inputs, 0.01 = 1 percent each way
step = 0.01
# make_b_profile(0 1 2) = case A B C, or make_b_profile_flat(1e-2)
b_profile = make_b_profile(0)

# the point the sensitivities are taken at, the papers values
INPUTS = {
    "eos": p_crust,
    "b_profile": b_profile,
    "rho_d": rho_d,
    "rho_cc": rho_cc,
    "rho_min": rho_min,
    "m0": m0,
    "r0": r0,
    "b_core": 1e-2,
    "omega0": omega0,
    "dom_crit": dom_crit,
    "dr": dr,
    "n_tov": n_tov,
    "n_shells": n_shells,
    "dt": dt,
    "n_t": n_t,
    "profile_steps": PROFILE_STEPS,
}


def sensitivity_run() -> tuple[Outputs, Outputs, list[tuple[str, Outputs]]]:
    """
    Run the input sensitivities and the grid check.

    Then prints the tables and draws the figure.

    :returns base: The outputs as set.
    :returns slopes: The slope of each output against each input.
    :returns grid_rows: A (label, outputs) row per grid variant.
    """
    t_start = time.time()
    vela = paper_vela(vela_path, t_end)
    t_hist = dt * np.arange(1, n_t + 1)  # time of every step

    summary = make_run_summary(t_hist, vela, dt, keep_dnu=True)
    base, slopes = sensitivity(INPUTS, inputs_to_vary, summary, step)
    t_inputs = time.time() - t_start
    grid_rows = grid_check(INPUTS, grid_variants(INPUTS, grid_factors), vela)
    sensitivity_output(inputs_to_vary, base, slopes, grid_rows, t_hist)
    log.info("")
    log.info(
        "%d inputs took %.1f s, %d grid runs took %.1f s",
        len(inputs_to_vary),
        t_inputs,
        len(grid_rows),
        time.time() - t_start - t_inputs,
    )
    return base, slopes, grid_rows


if __name__ == "__main__":
    sensitivity_run()
