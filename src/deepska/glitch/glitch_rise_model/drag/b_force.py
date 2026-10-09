# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Pinning force per unit length of vortex, f."""

from jax import Array

from deepska.glitch.data.constants import kappa
from deepska.glitch.kinds import Num


def b_force(delta_v: Num, rho_s: Num) -> Array:
    """
    Calculate the pinning force per unit length of vortex.

    :param delta_v: Critical velocity difference in cm/s.
    :param rho_s: Superfluid mass density in g/cm**3.
    :returns: The pinning force f in dyn/cm.
    """
    return kappa * delta_v * rho_s  # Magnus force at the unpinning velocity
