# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
One whole run, parameters in and glitch rise out.

Split in 2 so a sweep over b_core can build the star once.
"""

from jax import Array

from deepska.glitch.glitch_rise_model.run.param_to_star import param_to_star
from deepska.glitch.glitch_rise_model.run.star_to_output import star_to_output
from deepska.glitch.kinds import Fn, Num, Spin, Star, StarInputs


def param_to_output(
    inputs: StarInputs, b_profile: Fn, b_core: Num, spin: Spin
) -> tuple[Array, Array, Star]:
    """
    Do one whole run, from the inputs to the glitch rise.

    :param inputs: The star inputs.
    :param b_profile: Function giving B at a mass density in g/cm**3.
    :param b_core: Mutual friction coefficient of the core.
    :param spin: The spin settings.
    :returns dnu: Frequency change of the crust at every step in muHz.
    :returns profiles: Superfluid spin of every shell at the kept steps in rad/s.
    :returns star: The built star, so a sweep over star inputs gets R, M, eq 23 etc.
    """
    star = param_to_star(inputs)
    dnu, profiles = star_to_output(star, b_profile, b_core, spin)
    return dnu, profiles, star
