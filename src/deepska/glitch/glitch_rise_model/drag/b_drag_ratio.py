# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""R, the drag to lift ratio that B is built from."""

from jax import Array

from deepska.glitch.data.constants import hbar, kappa, m_u
from deepska.glitch.kinds import Case, Knot, Num


def b_drag_ratio(knot: Knot, case: Case, delta: Num) -> Array:
    """
    Calculate the drag to lift ratio R at one knot for one pinning case.

    Each case has its own constant and its own powers of the lattice spacing and the
    length scale.

    :param knot: Pinning energy, superfluid density, lattice spacing and length scale.
    :param case: The constant and the two powers of this pinning case.
    :param delta: Vortex-nucleus overlap fraction.
    :returns: The drag to lift ratio R.
    """
    rho_s = knot.n_s * m_u
    # same for every case
    common = (m_u / (2.0 * hbar)) ** 0.5 * (knot.e_p * delta / (rho_s * kappa)) ** 0.5
    return case.c * common * knot.a**case.p * knot.length**case.q
