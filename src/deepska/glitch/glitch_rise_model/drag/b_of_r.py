# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""B from the drag to lift ratio R."""

from jax import Array

from deepska.glitch.kinds import Num


def b_of_r(ratio: Num) -> Array:
    """
    Calculate B from the drag to lift ratio R.

    Peaks at 0.5 when R = 1 and drops off either side, so big or small R both give weak
    coupling.

    :param ratio: The drag to lift ratio R.
    :returns: The mutual friction coefficient B.
    """
    return ratio / (1.0 + ratio**2)
