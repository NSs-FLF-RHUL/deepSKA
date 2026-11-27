# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Runs over a grid of drag profiles and core couplings, all on the one star."""

from jax import Array

from deepska.glitch.glitch_rise_model.run.run_profile import run_profile
from deepska.glitch.kinds import Fn, Num, Spin, Star
from deepska.glitch.numerics.every_pair import every_pair


def profile_grid(
    ks: Num, b_cores: Num, b_profiles: list[Fn], star: Star, spin: Spin
) -> tuple[Array, Array]:
    """
    Run several drag profiles at several core couplings on one star.

    The papers 2 x 7 grid is ks = 0..6 at the weak and strong core, the fig 13 sweep is
    ks = [4] at bcore_range, same function.

    :param ks: Which profiles to run, as numbers into b_profiles.
    :param b_cores: The core couplings to run them at.
    :param b_profiles: The list of drag profile functions.
    :param star: The built star.
    :param spin: The spin settings.
    :returns dnu: Frequency change in muHz, as (core couplings, profiles, steps).
    :returns profiles: Superfluid profiles in rad/s, stacked the same way.
    """
    dnu, profiles = every_pair(run_profile, ks, b_cores, (b_profiles, star, spin))
    return dnu, profiles
