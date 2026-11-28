import sys
from pathlib import Path

if __name__ == "__main__":
    for name in [
        name
        for name in sys.modules
        if name == "deepska.glitch" or name.startswith("deepska.glitch.")
    ]:
        # forget the glitch files loaded by an earlier run, so this run reads every one
        # fresh from disk
        del sys.modules[name]

import numpy as np
import pytest

from deepska.glitch.analysis.make_run_summary import make_run_summary
from deepska.glitch.analysis.sensitivity import sensitivity
from deepska.glitch.data.paper_inputs import PaperInputs, vela_path
from deepska.glitch.data.paper_vela import paper_vela
from deepska.glitch.glitch_rise_model.drag.make_b_profile import make_b_profile
from deepska.glitch.glitch_rise_model.eos.negele_vautherin_electron_crust import p_crust

vela_2016_default = PaperInputs.vela_2016_default()

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
    "rho_d": vela_2016_default.rho_d,
    "rho_cc": vela_2016_default.rho_cc,
    "rho_min": vela_2016_default.rho_min,
    "m0": vela_2016_default.m0,
    "r0": vela_2016_default.r0,
    "b_core": 1e-2,
    "omega0": vela_2016_default.omega0,
    "dom_crit": vela_2016_default.dom_crit,
    "dr": vela_2016_default.dr,
    "n_tov": vela_2016_default.n_tov,
    "n_shells": vela_2016_default.n_shells,
    "dt": vela_2016_default.dt,
    "n_t": vela_2016_default.n_t,
    "profile_steps": vela_2016_default.PROFILE_STEPS,
}


@pytest.mark.regression
def test_regression_sensitivity_run(
    regression_data: Path,
    base_ref: str = "sensitivity-base-outputs.npz",
    slopes_ref: str = "sensitivity-slopes-outputs.npz",
) -> None:
    """
    Regression test for `scripts/sensitivity.py`.

    TODO: Update docstring to use descriptive language, rather than reference by file.

    Also, the function body here is a direct copy-paste of the script. Should consider
    moving this code somewhere common inside the package (maybe a workflows / analysis
    folder?) so that it's not repeated.
    """
    vela = paper_vela(vela_path, vela_2016_default.t_end)
    t_hist = vela_2016_default.dt * np.arange(1, vela_2016_default.n_t + 1)

    summary = make_run_summary(t_hist, vela, vela_2016_default.dt, keep_dnu=True)
    base, slopes = sensitivity(INPUTS, inputs_to_vary, summary, step)

    base_reference = np.load(regression_data / base_ref)
    slopes_reference = np.load(regression_data / slopes_ref)

    for name, array in base_reference.items():
        assert np.allclose(base[name], array)
    for name, array in slopes_reference.items():
        assert np.allclose(slopes[name], array)
