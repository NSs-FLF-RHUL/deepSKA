# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Sweep whatever inputs you like, every combination or paired.

On the papers star unless a star input is swept. Change the settings and run, results
get saved to sweeps/ in whatever folder you run from.
"""

import sys

from deepska.glitch.kinds import Batching, Drawn, Grid

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
import os
import time

import jax.numpy as jnp
import numpy as np

from deepska.glitch.analysis.closest_run import closest_run
from deepska.glitch.analysis.drawn_runs import drawn_runs
from deepska.glitch.analysis.make_run_residual import make_run_residual
from deepska.glitch.analysis.make_run_summary import make_run_summary
from deepska.glitch.analysis.run_sweep import run_sweep
from deepska.glitch.analysis.save_sweep import save_sweep
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
from deepska.glitch.plots.sweep_output import sweep_output

# plain lines on screen, same as print would give
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# settings: change these
# input name -> the values to try, any of the INPUTS below; first sets the hue, second
# the brightness, third how strong the colour is
sweep_over = {
    "b_core": jnp.logspace(-5, -2, 500),  # core coupling
    # which of b_profiles below, by number
    "b_profile": [0, 1, 2],
    # neutron drip density, g cm^-3
    "rho_d": jnp.linspace(3.0e11, 5.0e11, 20),
}
# True = every combination of the lists above, False = paired up (same length lists)
matrix = True
# the drag profiles to pick from, here cases A B C, could add make_b_profile_flat(1e-2);
# number 0 is used if "b_profile" isnt swept
b_profiles = [make_b_profile(0), make_b_profile(1), make_b_profile(2)]
# also save the whole rise per run, 160 KB each so only for small sweeps
keep_dnu = False
# how many runs get drawn as curves
drawn = 100
# batches run at once, or put a number
cores = os.cpu_count()
# runs side by side in one batch
batch = 125

# the papers values, anything in sweep_over overrides these
INPUTS = {
    "eos": p_crust,
    "b_profile": b_profiles,
    "rho_d": rho_d,
    "rho_cc": rho_cc,
    "rho_min": rho_min,
    "m0": m0,
    "r0": r0,
    "b_core": 5e-5,
    "omega0": omega0,
    "dom_crit": dom_crit,
    "dr": dr,
    "n_tov": n_tov,
    "n_shells": n_shells,
    "dt": dt,
    "n_t": n_t,
    "profile_steps": PROFILE_STEPS,
}
folder = "sweeps"  # made in the folder you run from


def sweep() -> tuple[Grid, Grid]:
    """
    Run the sweep set up at the top of the file.

    Says which run is closest to Vela, saves it and draws it.

    :returns out: What was kept from every run.
    :returns values: The swept values.
    """
    t_start = time.time()
    vela = paper_vela(vela_path, t_end)
    t_hist = dt * np.arange(1, n_t + 1)  # time of every step
    batching = Batching(batch, cores)

    summary = make_run_summary(t_hist, vela, dt, keep_dnu=keep_dnu)
    out, values, draws = run_sweep(INPUTS, sweep_over, summary, batching, matrix=matrix)
    best, where = closest_run(out["rms"], values, matrix=matrix)
    log.info(
        "%d runs over %s, batches of %d, %d at a time, took %.1f s",
        out["rms"].size,
        ", ".join(f"{n} ({len(v)})" for n, v in values.items()),
        batch,
        cores,
        time.time() - t_start,
    )
    log.info(
        "closest to vela: %s, rms %.4f ms, peak dnu %.1f muHz, eq 23 %.2f muHz",
        ", ".join(f"{n} = {v:.4g}" for n, v in where.items()),
        out["rms"][best],
        out["peak"][best],
        out["dnu_equil"][best],
    )
    log.info("saved %s", save_sweep(folder, out, values))

    shown = drawn_runs(draws, out["rms"].size, drawn)
    # the drawn runs again, keeping the whole residual
    residual = make_run_residual(dt, vela.dt_shift)
    curves_out, _, _ = run_sweep(INPUTS, shown, residual, batching, matrix=False)
    sweep_output(values, Drawn(shown, curves_out["res"]), out["rms"].size, t_hist, vela)
    log.info("whole script, with the figures: %.1f s", time.time() - t_start)
    return out, values


if __name__ == "__main__":
    sweep()
