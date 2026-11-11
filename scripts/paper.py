# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The whole paper, run this and it prints table 2 and draws the 13 figures."""

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
from jax import Array

from deepska.glitch.data.b_data import rho_b_data
from deepska.glitch.data.constants import km
from deepska.glitch.data.paper_inputs import (
    PROFILE_STEPS,
    b_cores,
    bcore_range,
    dom_crit,
    dr,
    dt,
    k_a,
    m0,
    n_shells,
    n_t,
    n_tov,
    omega0,
    r0,
    rho_b,
    rho_cc,
    rho_d,
    rho_min,
    t_end,
    vela_path,
)
from deepska.glitch.data.paper_vela import paper_vela
from deepska.glitch.glitch_rise_model.drag.b_cases import b_cases
from deepska.glitch.glitch_rise_model.drag.p_b import b_all, delta_v_all, f_all, p_b
from deepska.glitch.glitch_rise_model.eos.negele_vautherin_electron_crust import p_crust
from deepska.glitch.glitch_rise_model.run.paper_b_profiles import paper_b_profiles
from deepska.glitch.glitch_rise_model.run.param_to_star import param_to_star
from deepska.glitch.glitch_rise_model.run.profile_grid import profile_grid
from deepska.glitch.glitch_rise_model.star.dm_m import dm_m
from deepska.glitch.kinds import Arr, PaperArrays, Spin, StarInputs
from deepska.glitch.plots.paper_output import paper_output

# plain lines on screen, same as print would give
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")


def paper() -> None:
    """
    Run the whole paper: table 2 and the 13 figures.

    One star, the grid and the fig 13 sweep on it, then the table and figures.
    """
    star_inputs = StarInputs(
        p_crust, rho_d, rho_cc, rho_min, m0, r0, dom_crit, dr, n_tov, n_shells
    )
    spin = Spin(omega0, dom_crit, dt, n_t, PROFILE_STEPS)
    star = param_to_star(star_inputs)  # one star for every run
    b_profiles = paper_b_profiles()

    # every profile at weak and strong core
    dnu_all, profiles = profile_grid(
        jnp.arange(len(b_profiles)), b_cores, b_profiles, star, spin
    )
    # case A over the 7 core couplings, for fig 13
    dnu_sweep, _ = profile_grid(jnp.array([k_a]), bcore_range, b_profiles, star, spin)

    # fraction of the stars mass outside each density
    def mass_fraction(rho: Arr) -> Array:
        return dm_m(rho, star.m_star, star.m_hist, star.rho_hist, star.n_valid)

    arrays = PaperArrays(
        dnu_equil=star.dnu_equil,
        b_all=b_all,
        delta_v_all=delta_v_all,
        f_all=f_all,
        rho_b=rho_b,
        b_curves=b_cases(p_b, rho_b),
        dm_m=mass_fraction(rho_b),
        dm_m_dots=mass_fraction(rho_b_data),
        r_km=star.x * star.r_drip / km,
        profiles=profiles,
        dnu_all=dnu_all,
        dnu_sweep=dnu_sweep[:, 0],
        rho_b_data=rho_b_data,
        bcore_range=bcore_range,
    )
    paper_output(arrays, spin, paper_vela(vela_path, t_end))


if __name__ == "__main__":
    paper()
