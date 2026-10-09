# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Pinning energy for case A."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Num


def e_pa_of_eb(e_s: Num, e_l: Num) -> Array:
    """
    Calculate the case A pinning energy from the two Epstein-Baym energies.

    :param e_s: The energy E_s of the pinning tables in erg.
    :param e_l: The energy E_l of the pinning tables in erg.
    :returns: The case A pinning energy in erg.
    """
    return jnp.sqrt(e_s**2 + e_s * e_l + 0.5 * e_l**2)
