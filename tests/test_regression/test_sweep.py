import sys

from deepska.glitch.kinds import Batching

if __name__ == "__main__":
    for name in [
        name
        for name in sys.modules
        if name == "deepska.glitch" or name.startswith("deepska.glitch.")
    ]:
        # forget the glitch files loaded by an earlier run, so this run reads every one
        # fresh from disk
        del sys.modules[name]

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from deepska.glitch.analysis.closest_run import closest_run
from deepska.glitch.analysis.make_run_summary import make_run_summary
from deepska.glitch.analysis.sweep import run_sweep
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


@pytest.mark.regression
def test_regression_sweep(
    draws_reference_data: Path = Path(__file__).parent / "sweep-draws.npz",
    out_reference_data: Path = Path(__file__).parent / "sweep-out.npz",
    values_reference_data: Path = Path(__file__).parent / "sweep-values.npz",
    best_and_where_reference_data: Path = Path(__file__).parent
    / "sweep-best-and-where.npz",
) -> None:
    """
    Regression test for `scripts/sweep.py`.

    TODO: Update docstring to use descriptive language, rather than reference by file.

    Also, the function body here is a direct copy-paste of the script. Should consider
    moving this code somewhere common inside the package (maybe a workflows / analysis
    folder?) so that it's not repeated.
    """
    vela = paper_vela(vela_path, t_end)
    t_hist = dt * np.arange(1, n_t + 1)  # time of every step
    batching = Batching(batch, cores)

    summary = make_run_summary(t_hist, vela, dt, keep_dnu=keep_dnu)
    out, values, draws = run_sweep(INPUTS, sweep_over, summary, batching, matrix=matrix)
    best, where = closest_run(out["rms"], values, matrix=matrix)

    out_reference = np.load(out_reference_data)
    values_reference = np.load(values_reference_data)
    draws_reference = np.load(draws_reference_data)
    best_and_where_reference = np.load(best_and_where_reference_data)

    for computed, reference in [
        (out, out_reference),
        (values, values_reference),
        (draws, draws_reference),
    ]:
        for name, reference_array in reference.items():
            assert np.allclose(computed[name], reference_array)
    for name, array in best_and_where_reference.items():
        if name == "best":
            assert np.allclose(best, array)
        else:
            assert np.allclose(where[name], array)
