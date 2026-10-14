# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""One run on a built star, with the drag profile picked by number."""

from jax import Array

from deepska.glitch.glitch_rise_model.run.star_to_output import star_to_output
from deepska.glitch.kinds import Fn, Num, Spin, Star
from deepska.glitch.numerics.pick_function import pick_function


def run_profile(
    k: Num, b_core: Num, b_profiles: list[Fn], star: Star, spin: Spin
) -> tuple[Array, Array]:
    """
    Do one run on a built star, with the drag profile picked by number.

    k is a number not a function so every_pair can vmap over it.

    :param k: Which profile of b_profiles to use.
    :param b_core: Mutual friction coefficient of the core.
    :param b_profiles: The list of drag profile functions.
    :param star: The built star.
    :param spin: The spin settings.
    :returns dnu: Frequency change of the crust at every step in muHz.
    :returns profiles: Superfluid spin of every shell at the kept steps in rad/s.
    """

    # star_to_output wants a function of rho alone
    def b_profile(rho: Num) -> Array:
        return pick_function(b_profiles, k, rho)

    return star_to_output(star, b_profile, b_core, spin)
