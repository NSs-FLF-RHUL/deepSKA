# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The equation of state into the star and the glitch rise.

The table goes into the star as two log-log splines, P(rho) and rho(P), with
rho = eps / c^2. The TOV starts at the centre, at the central density n_c, and runs
out to the surface. The crust-core boundary and neutron drip are found inside the star
by the pressure, in star/moments_by_pressure.py. The glitch rise is then that of
Graber, Cumming and Andersson 2018, with a flat mutual friction in the crust.

Every number stays jax, with no float and no numpy on the way, so the derivative of the
rise with respect to the inputs goes through.

Contents:

- star_inputs: the star inputs from the table.
- rise_of: from the inputs to the star and the glitch rise.
"""

from typing import Any

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import c, km, mev_fm3, msun
from deepska.glitch.data.metamodel_inputs import n_tov_centre, r_start
from deepska.glitch.data.paper_inputs import (
    PROFILE_STEPS,
    dom_crit,
    dr,
    dt,
    n_shells,
    n_t,
    omega0,
)
from deepska.glitch.glitch_rise_model.drag.make_b_profile import make_b_profile_flat
from deepska.glitch.glitch_rise_model.run.star_to_output import star_to_output
from deepska.glitch.glitch_rise_model.star.moments_by_pressure import (
    moments_by_pressure,
)
from deepska.glitch.kinds import Arr, Fn, Num, Spin, StarInputs
from deepska.glitch.metamodel.table import crust_join_core, table_fixed
from deepska.glitch.metamodel.uniform import core_at
from deepska.glitch.numerics.pchip import loglog_pchip_build, loglog_pchip_eval


def star_inputs(
    table: dict[str, Array], n_drip: Num, eps_1: Num, eps_c: Num, p_c: Num
) -> tuple[StarInputs, Fn, Array]:
    """
    Make the star inputs from the table.

    The TOV starts at the centre. It cannot start at r = 0, where it divides by r, so
    it starts at r_start: a ball of that radius at the central density holds
    m0 = 4/3 pi r_start^3 rho_c, and P there is P_c.

    :param table: The table, from table_fixed, in MeV and fm.
    :param n_drip: The drip density in fm^-3.
    :param eps_1: eps at n_1, the crust-core boundary, in MeV fm^-3.
    :param eps_c: eps at the centre, n_c, in MeV fm^-3.
    :param p_c: P at the centre in MeV fm^-3.
    :returns inputs: The star inputs, in cgs.
    :returns rho_of_p: The inverse eos rho(P), a log-log pchip, in cgs.
    :returns p_0: P at the centre in erg/cm^3.
    """
    rho = table["eps"] * mev_fm3 / c**2  # mass-energy density, g/cm^3
    p = table["p"] * mev_fm3  # erg/cm^3
    p_params = loglog_pchip_build(rho, p)  # P(rho)
    rho_params = loglog_pchip_build(p, rho)  # rho(P)

    def eos(rho_q: Num, _rho_d: Num, _rho_cc: Num) -> Array:
        """P(rho); rho_d and rho_cc are only there for the shape StarInputs wants."""
        shape = jnp.shape(rho_q)
        return jnp.reshape(loglog_pchip_eval(p_params, rho_q), shape)

    def rho_of_p(p_q: Num) -> Array:
        return loglog_pchip_eval(rho_params, p_q)[0]

    rho_d = jnp.interp(n_drip, table["n_b"], rho)
    rho_cc = eps_1 * mev_fm3 / c**2
    rho_c = eps_c * mev_fm3 / c**2
    m0 = 4.0 / 3.0 * jnp.pi * r_start**3 * rho_c
    inputs = StarInputs(
        eos, rho_d, rho_cc, rho[0], m0, r_start, dom_crit, dr, n_tov_centre, n_shells
    )
    return inputs, rho_of_p, p_c * mev_fm3


def rise_of(values: Arr, n_c: Num, b_core: Num, b_crust: Num) -> dict[str, Any]:
    """
    Calculate the star and the glitch rise from the inputs, for one draw.

    The TOV runs from the centre (n_b = n_c) to the surface. For derivatives use
    forward mode, jax.jacfwd: there are 13 inputs and it carries the derivative along
    with the run. jax.grad works but has to go back through every TOV and rise step,
    which takes about 30 minutes.

    :param values: The 12 values of a draw, from draw_values.
    :param n_c: The central density in fm^-3.
    :param b_core: The mutual friction in the core.
    :param b_crust: The mutual friction in the crust, the same in every shell.
    :returns: By name: dnu, the rise at every time step in muHz; dnu_equil in muHz;
        r_star in km; m_star in solar masses; ok, True when the draw is kept (it
        dripped, it joined, P rises along the table and the TOV reached the surface);
        and why, the reason, one True or False each, True where that check failed:
        "no drip", "no join", "P falls", "no surface".
    """
    e_sat, n_sat, k_sat, q_sat, z_sat, e_sym, l_sym, k_sym, q_sym, z_sym = values[2:]
    c_sat = jnp.array([e_sat, 0.0, k_sat, q_sat, z_sat])
    c_sym = jnp.array([e_sym, l_sym, k_sym, q_sym, z_sym])

    crust_out, join_out, core_out = crust_join_core(c_sat, c_sym, n_sat, n_c)
    table = table_fixed(crust_out, join_out, core_out)
    eps_c, p_c, _mu = core_at(n_c, c_sat, c_sym, n_sat)

    inputs, rho_of_p, p_0 = star_inputs(
        table, crust_out["n_drip"], join_out["eps_1"], eps_c, p_c
    )
    p_t, p_drip = join_out["p_t"] * mev_fm3, crust_out["p_drip"] * mev_fm3
    star = moments_by_pressure(inputs, p_0, rho_of_p, p_t, p_drip)
    spin = Spin(omega0, dom_crit, dt, n_t, PROFILE_STEPS)
    dnu, _ = star_to_output(star, make_b_profile_flat(b_crust), b_core, spin)

    rises = jnp.all(jnp.diff(table["eps"]) > 0) & jnp.all(jnp.diff(table["p"]) > 0)
    # the TOV reached the surface pressure before its last step
    surface = star.n_valid < n_tov_centre + 1
    ok = crust_out["dripped"] & join_out["joined"] & rises & surface
    why = {
        "no drip": ~crust_out["dripped"],
        "no join": ~join_out["joined"],
        "P falls": ~rises,
        "no surface": ~surface,
    }
    return {
        "why": why,
        "dnu": dnu,
        "dnu_equil": star.dnu_equil,
        "r_star": star.r_star / km,
        "m_star": star.m_star / msun,
        "ok": ok,
    }
