# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Second half of a run, a built star in and the glitch rise out.

The bit that depends on b_core and the drag profile.
"""

from jax import Array

from deepska.glitch.glitch_rise_model.spin.evolve import evolve
from deepska.glitch.glitch_rise_model.spin.frequency_change import frequency_change
from deepska.glitch.kinds import Coupling, Fn, Num, Spin, Star


def star_to_output(
    star: Star, b_profile: Fn, b_core: Num, spin: Spin
) -> tuple[Array, Array]:
    """
    Calculate the glitch rise on a built star.

    :param star: The built star.
    :param b_profile: Function giving B at a mass density in g/cm**3.
    :param b_core: Mutual friction coefficient of the core.
    :param spin: The spin settings.
    :returns dnu: Frequency change of the crust in muHz, shape (n_t,).
    :returns profiles: Superfluid spin in rad/s, shape (kept steps, shells).
    """
    b_shell = b_profile(star.rho_shell)  # the drag profile at the shells
    om_crust, profiles = evolve(Coupling(b_shell, b_core), star, spin)
    return frequency_change(om_crust, spin.omega0), profiles
