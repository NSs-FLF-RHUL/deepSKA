# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
From the metamodel inputs to a glitch rise. Change the inputs in the box below and run.

The eos is built once (crust walk, n_drip, n_cc, core), then P(n_b), eps(n_b) and
P(eps) are drawn. The star is the inner core (m0 inside r0, where n_b = 2 n_sat) with
the outer core and the crust marched out from it by the TOV, on the same eos. Then one
glitch rise with a fixed B in the core and in every crust shell.
"""

import sys

if __name__ == "__main__":
    for name in [
        name
        for name in sys.modules
        if name == "deepska.glitch" or name.startswith("deepska.glitch.")
    ]:
        # forget the glitch files an earlier run loaded, so edits get picked up
        del sys.modules[name]

import logging

import jax.numpy as jnp
import numpy as np

from deepska.glitch.data.constants import km, msun
from deepska.glitch.data.paper_inputs import PROFILE_STEPS, dom_crit, dt, n_t, omega0
from deepska.glitch.glitch_rise_model.drag.make_b_profile import make_b_profile_flat
from deepska.glitch.glitch_rise_model.eos.metamodel_cgs import rho_of_p
from deepska.glitch.glitch_rise_model.run.metamodel_to_star import (
    metamodel_table,
    metamodel_to_star_inputs,
    pressure_at,
)
from deepska.glitch.glitch_rise_model.run.param_to_star import param_to_star
from deepska.glitch.glitch_rise_model.run.star_to_output import star_to_output
from deepska.glitch.kinds import Bulk, Spin
from deepska.glitch.plots.eos_output import eos_output
from deepska.glitch.plots.rise_output import rise_output

# the inputs, change these and run
# the ten metamodel inputs, Margueron et al 2018, MeV (n_sat in fm^-3)
E_SAT, K_SAT, Q_SAT, Z_SAT = -15.8, 230.0, 300.0, 0.0  # symmetric matter
E_SYM, L_SYM, K_SYM, Q_SYM, Z_SYM = 32.0, 60.0, -100.0, 0.0, 0.0  # symmetry energy
N_SAT = 0.16  # saturation density
# the inner core, where n_b = 2 n_sat: its mass and radius. These suit the inputs
# above, a 1.4 solar mass star; change the inputs a lot and these need changing too
M0 = 0.4745  # solar masses
R0 = 7.109  # km
# the mutual friction, the same B in every crust shell
B_CORE = 5e-5
B_CRUST = 1e-3

# plain lines on screen, same as print would give
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def metamodel_rise() -> None:
    """Build the eos from the inputs above, draw its three figures, then the rise."""
    bulk = Bulk(
        csat=jnp.array([E_SAT, 0.0, K_SAT, Q_SAT, Z_SAT]),
        csym=jnp.array([E_SYM, L_SYM, K_SYM, Q_SYM, Z_SYM]),
        ns=N_SAT,
    )
    # the eos, the slow part, about 6 s: every step is in metamodel_eos.py
    eos = metamodel_table(bulk)
    log.info("n_drip = %.3e, n_cc = %.4f fm^-3", eos.n_drip, eos.n_cc)
    eos_output(eos.table, eos.n_cc)

    # the star: the TOV from the inner core's edge out, finding n_cc and drip
    inputs = metamodel_to_star_inputs(eos, (M0 * msun, R0 * km))
    star = param_to_star(
        inputs,
        rho_of_p=rho_of_p(eos.table),
        p_start=pressure_at(eos.table, 2 * N_SAT),
    )
    log.info(
        "star: R = %.3f km, M = %.3f solar masses, n_cc at %.3f km, drip at %.3f km",
        float(star.r_star) / km,
        float(star.m_star) / msun,
        float(star.x[0] * star.r_drip) / km,  # the first shell sits on n_cc
        float(star.r_drip) / km,
    )

    # the glitch rise
    spin = Spin(omega0, dom_crit, dt, n_t, PROFILE_STEPS)
    dnu, _ = star_to_output(star, make_b_profile_flat(B_CRUST), B_CORE, spin)
    log.info("eq 23 dnu = %.2f muHz", float(star.dnu_equil))
    t = dt * np.arange(1, n_t + 1)  # time of every step
    rise_output(t, dnu, f"glitch rise, B_core = {B_CORE:g}, B_crust = {B_CRUST:g}")


if __name__ == "__main__":
    metamodel_rise()
