# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The drag coefficient B at one knot for one case."""

import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import m_u
from deepska.glitch.glitch_rise_model.drag.b_delta_v import b_delta_v
from deepska.glitch.glitch_rise_model.drag.b_drag_ratio import b_drag_ratio
from deepska.glitch.glitch_rise_model.drag.b_force import b_force
from deepska.glitch.glitch_rise_model.drag.b_of_r import b_of_r
from deepska.glitch.kinds import Case, Knot, Num


def b(knot: Knot, case: Case, delta: Num) -> tuple[Array, Array, Array]:
    """
    Calculate B, delta_v and f at one knot for one pinning case.

    :param knot: Pinning energy, superfluid density, lattice spacing and length scale.
    :param case: The constant and the two powers of this pinning case.
    :param delta: Vortex-nucleus overlap fraction.
    :returns b: The mutual friction coefficient B.
    :returns delta_v: The critical velocity difference in cm/s.
    :returns f: The pinning force per unit length in dyn/cm.
    """
    # energies come signed in the tables but only the size matters here
    knot = knot._replace(e_p=jnp.abs(knot.e_p))
    rho_s = knot.n_s * m_u
    ratio = b_drag_ratio(knot, case, delta)
    delta_v = b_delta_v(knot.e_p, rho_s, knot.a, knot.length, delta)
    f = b_force(delta_v, rho_s)
    return b_of_r(ratio), delta_v, f
