# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""From phase to timing residual."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import ms
from deepska.glitch.kinds import Num


def timing_residual(phi: Num, omega0: Num, dt_shift: Num) -> Array:
    """
    Convert the phase of the rise into a timing residual.

    Minus as the star spinning up means pulses arrive early.

    :param phi: Phase gained through the rise in cycles.
    :param omega0: Spin of the pulsar in rad/s.
    :param dt_shift: Residual at the glitch in ms, the model is lifted by it.
    :returns: The timing residual in ms.
    """
    nu0 = omega0 / (2 * jnp.pi)  # spin frequency
    return -phi / nu0 / ms + dt_shift  # timing residual lifted by dt_shift
