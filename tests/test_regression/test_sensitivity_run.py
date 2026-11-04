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


@pytest.mark.regression
def test_regression_sensitivity_run(
    base_reference_data: Path = Path(__file__).parent / "sensitivity-base-outputs.npz",
    slopes_refrence_data: Path = Path(__file__).parent
    / "sensitivity-slopes-outputs.npz",
) -> None:
    """
    Regression test for `scripts/sensitivity.py`.

    TODO: Update docstring to use descriptive language, rather than reference by file.

    Also, the function body here is a direct copy-paste of the script. Should consider
    moving this code somewhere common inside the package (maybe a workflows / analysis
    folder?) so that it's not repeated.
    """
    vela = paper_vela(vela_path, t_end)
    t_hist = dt * np.arange(1, n_t + 1)

    summary = make_run_summary(t_hist, vela, dt, keep_dnu=True)
    base, slopes = sensitivity(INPUTS, inputs_to_vary, summary, step)

    base_reference = np.load(base_reference_data)
    slopes_reference = np.load(slopes_refrence_data)

    for name, array in base_reference.items():
        assert np.allclose(base[name], array)
    for name, array in slopes_reference.items():
        assert np.allclose(slopes[name], array)
