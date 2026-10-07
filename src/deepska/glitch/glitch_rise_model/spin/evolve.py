# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The spin equations run forward in time from the glitch.

Crust every step but the whole state only at the kept steps, as 20000 steps x 103 adds
up in a sweep.
"""

from jax import Array

from deepska.glitch.glitch_rise_model.spin.crust_only import crust_only
from deepska.glitch.glitch_rise_model.spin.initial_spin import initial_spin
from deepska.glitch.glitch_rise_model.spin.rhs2 import rhs2
from deepska.glitch.glitch_rise_model.spin.spin_state import unpack_state
from deepska.glitch.kinds import Coupling, Num, Spin, Star, Steps
from deepska.glitch.numerics.ode_snapshots import solve_ode_snapshots


def evolve(coupling: Coupling, star: Star, spin: Spin) -> tuple[Array, Array]:
    """
    Solve the spin equations forward in time from the glitch.

    :param coupling: B at every shell and B of the core.
    :param star: The built star.
    :param spin: The spin settings.
    :returns crust: Spin of the crust at every step in rad/s, shape (n_t,).
    :returns om_sf: Spin of every shell at each kept step in rad/s.
    """

    # the spin equations with this star and these couplings filled in
    def rhs(y: Num, t: Num) -> Array:
        return rhs2(y, t, coupling, star)

    # start: shells lead by dom_crit, core and crust at omega0, as many shells as x has
    y0 = initial_spin(spin.omega0, spin.dom_crit, star.x.shape[0])
    crust, states = solve_ode_snapshots(
        rhs, y0, Steps(0.0, spin.dt, spin.n_t), spin.profile_steps, keep=crust_only
    )
    om_sf, _om_core, _om_crust = unpack_state(states)
    return crust, om_sf
