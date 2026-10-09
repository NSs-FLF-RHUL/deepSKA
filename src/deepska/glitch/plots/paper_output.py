# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Table 2 and the 13 figures of Graber et al 2018, from the arrays of a run.

Works out phase and residuals first, then one call per figure so each ones easy to find
and change. The papers own choices (which steps get drawn, where cases A and C sit) come
from data so nothing is hard coded here.
"""

import logging

import numpy as np

from deepska.glitch.analysis.cumulative_residual import cumulative_residual
from deepska.glitch.analysis.phase_of_rise import phase_of_rise
from deepska.glitch.analysis.timing_residual import timing_residual
from deepska.glitch.data.paper_inputs import k_a, k_c, steps_a, steps_c
from deepska.glitch.data.paper_labels import domains
from deepska.glitch.kinds import Axes, PaperArrays, Spin, Vela
from deepska.glitch.plots.curves import curves_cases, curves_profile, curves_seven
from deepska.glitch.plots.figure import figure
from deepska.glitch.plots.table import table
from deepska.glitch.plots.vela_curves import vela_curves

log = logging.getLogger(__name__)


def paper_output(arrays: PaperArrays, spin: Spin, vela: Vela) -> None:
    """
    Print eq 23 and table 2, then draw the 13 figures of the paper.

    :param arrays: Every array the table and figures are drawn from.
    :param spin: The spin settings the runs used.
    :param vela: The Vela data.
    """
    a = arrays
    b_label = "mutual friction coefficient B"
    profile_labels = ("r (km)", "Omega_sf (rad/s)")
    dv_unit, f_unit = 1e4, 1e15  # table 2 prints dv in 1e4 cm/s and f in 1e15 dyn/cm
    weak, strong = 0, 1  # the 2 core couplings, first axis of the grid
    t_hist = spin.dt * np.arange(1, spin.n_t + 1)  # time of every step
    # phase = running sum of the frequency change
    phi_all = phase_of_rise(a.dnu_all, spin.dt)
    res_all = timing_residual(phi_all, spin.omega0, vela.dt_shift)
    phi_sweep = phase_of_rise(a.dnu_sweep, spin.dt)
    res_sweep = timing_residual(phi_sweep, spin.omega0, vela.dt_shift)

    r_lim = (a.r_km[-1], a.r_km[0])  # radius axis running inwards
    free = (None, None)  # let the axis pick its own range

    vela_pulses, vela_bins, vela_cum, vela_from0 = vela_curves(vela)
    cum_models = [
        cumulative_residual(res_all[weak, k], t_hist, vela.t_bins, vela.dt_shift)
        for k in range(res_all.shape[1])
    ]
    sweep = [
        (t_hist, res_sweep[i], "-", {"label": f"B_core = {a.bcore_range[i]:g}"})
        for i in range(len(a.bcore_range))
    ]

    log.info("Eq 23: dnu_equil = %.2f muHz", float(a.dnu_equil))  # paper gets 16.0
    log.info("")
    rows = [
        (
            domains[j],
            *(float(a.delta_v_all[k, j] / dv_unit) for k in range(3)),
            *(float(a.f_all[k, j] / f_unit) for k in range(3)),
        )
        for j in range(len(domains))
    ]
    table(
        "Table 2: dv in 1e4 cm/s, f in 1e15 dyn/cm",
        "domain   dv(A)    dv(B)    dv(C)     f(A)    f(B)    f(C)",
        rows,
        "%-6s %7.3f  %7.3f  %7.3f   %6.2f  %6.2f  %6.2f",
    )

    # one superfluid profile figure: a run of the grid at the chosen steps
    def profile(core: int, k: int, steps: list[int], title: str) -> None:
        curves = curves_profile(
            a.r_km, a.profiles[core, k], spin.profile_steps, spin.dt, steps
        )
        figure(curves, Axes(r_lim, free, *profile_labels, title, offset=False))

    figure(
        curves_cases(a.rho_b, a.b_curves, a.rho_b_data, a.b_all),
        Axes(
            (4e11, 1.35e14),
            (1e-5, 1e-1),
            "rho (g/cm^3)",
            b_label,
            "1. B against density",
            xlog=True,
            ylog=True,
        ),
    )
    figure(
        curves_cases(a.dm_m, a.b_curves, a.dm_m_dots, a.b_all),
        Axes(
            (0.0, 0.0095),
            (1e-5, 1e-1),
            "dM / M",
            b_label,
            "2. B against mass fraction",
            ylog=True,
        ),
    )
    profile(weak, k_a, steps_a, "3. superfluid profile, case A, weak core")
    profile(weak, k_c, steps_c, "4. superfluid profile, case C, weak core")
    figure(
        curves_seven(t_hist, a.dnu_all[weak]),
        Axes(
            (-2, 60),
            (1, 5e2),
            "t (s)",
            "dnu (muHz)",
            "5. glitch rise, weak core",
            ylog=True,
        ),
    )
    figure(
        curves_seven(t_hist, phi_all[weak]),
        Axes((-1, 60), (-0.0001, 0.0036), "t (s)", "phase", "6. phase, weak core"),
    )
    profile(strong, k_a, steps_a, "7. superfluid profile, case A, strong core")
    profile(strong, k_c, steps_c, "8. superfluid profile, case C, strong core")
    figure(
        curves_seven(t_hist, a.dnu_all[strong]),
        Axes(
            (-0.2, 6),
            (1e-1, 1.2e2),
            "t (s)",
            "dnu (muHz)",
            "9. glitch rise, strong core",
            ylog=True,
        ),
    )
    figure(
        curves_seven(t_hist, phi_all[strong]),
        Axes((-1, 60), (-0.00005, 0.00105), "t (s)", "phase", "10. phase, strong core"),
    )
    figure(
        [vela_pulses, vela_bins, *curves_seven(t_hist, res_all[weak])],
        Axes(
            (-60, 120),
            (-0.4, 0.4),
            "t (s)",
            "timing residuals (ms)",
            "11. timing residuals, weak core",
        ),
    )
    figure(
        [vela_cum, *curves_seven(vela.t_bins, cum_models)],
        Axes(
            (-1, 120),
            (-19, 1),
            "t (s)",
            "cumulative residuals (ms)",
            "12. cumulative residuals, weak core",
        ),
    )
    figure(
        [vela_from0, *sweep],
        Axes(
            (-1, 120),
            (-0.35, 0.25),
            "t (s)",
            "timing residuals (ms)",
            "13. case A, core coupling from 1e-5 to 1e-2",
        ),
    )
