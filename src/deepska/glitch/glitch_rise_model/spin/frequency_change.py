# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""From the crust spin to the frequency change of the glitch."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import muhz
from deepska.glitch.kinds import Num


def frequency_change(om_crust: Num, omega0: Num) -> Array:
    """
    Convert the crust spin into the frequency change of the glitch.

    :param om_crust: Spin of the crust in rad/s.
    :param omega0: Spin of the pulsar before the glitch in rad/s.
    :returns: The frequency change in muHz, 0 at t = 0.
    """
    return (om_crust - omega0) / (2 * jnp.pi) / muhz
