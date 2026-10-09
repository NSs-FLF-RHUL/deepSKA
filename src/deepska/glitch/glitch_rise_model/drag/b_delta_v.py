# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The velocity difference a vortex needs before it unpins from a nucleus."""

from jax import Array

from deepska.glitch.data.constants import kappa
from deepska.glitch.kinds import Num


def b_delta_v(e_p: Num, rho_s: Num, a: Num, length: Num, delta: Num) -> Array:
    """
    Calculate the velocity difference a vortex needs to unpin from a nucleus.

    :param e_p: Pinning energy in erg.
    :param rho_s: Superfluid mass density in g/cm**3.
    :param a: Lattice spacing in cm.
    :param length: Length scale in cm, r_n for cases A and B and xi for C.
    :param delta: Vortex-nucleus overlap fraction.
    :returns: The critical velocity difference delta_v in cm/s.
    """
    return e_p * delta / (length * a * rho_s * kappa)
