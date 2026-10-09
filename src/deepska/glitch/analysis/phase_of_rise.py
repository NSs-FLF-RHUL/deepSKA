# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Phase gained through the rise."""

from jax import Array

from deepska.glitch.data.constants import muhz
from deepska.glitch.kinds import Num
from deepska.glitch.numerics.cumulative_integral import cumulative_integral


def phase_of_rise(dnu: Num, dt: Num) -> Array:
    """
    Calculate the phase gained through the rise.

    :param dnu: Frequency change of the crust in muHz.
    :param dt: Time step in s.
    :returns: The phase at every step in cycles.
    """
    # phase = running sum of the frequency change, dnu in muHz so back to Hz first
    return cumulative_integral(dnu * muhz, dt)
