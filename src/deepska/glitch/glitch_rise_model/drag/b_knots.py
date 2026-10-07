# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""B, delta_v and f at every knot, for the three cases A, B and C."""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.drag.b import b
from deepska.glitch.glitch_rise_model.drag.e_pa_of_eb import e_pa_of_eb
from deepska.glitch.kinds import Case, Knot, PinningTables

# runs b down the knots, same case and delta for all of them
over_knots = jax.vmap(b, in_axes=(Knot(0, 0, 0, 0), None, None))
# then over the cases, each with its own energy, length scale and constants
over_cases = jax.vmap(over_knots, in_axes=(Knot(0, None, None, 0), Case(0, 0, 0), None))


def b_knots(tables: PinningTables) -> tuple[Array, Array, Array]:
    """
    Calculate B, delta_v and f at every knot for the three pinning cases.

    :param tables: The pinning tables, one value per inner crust domain.
    :returns b: B, shape (3, 5) as 3 cases by 5 knots.
    :returns delta_v: The critical velocity difference in cm/s, same shape.
    :returns f: The pinning force per unit length in dyn/cm, same shape.
    """
    # pinning energy: case A, B, C
    e_cases = jnp.stack([e_pa_of_eb(tables.e_s, tables.e_l), tables.e_pb, tables.e_pb])
    # length scale: case A, B, C
    l_cases = jnp.stack([tables.r_n, tables.r_n, tables.xi])
    cases = Case(
        c=jnp.array([2.8, 2.8, 1.0 / (2.0 * jnp.sqrt(jnp.pi))]),
        p=jnp.array([-1.5, -1.5, 0.5]),  # power of a
        q=jnp.array([1.0, 1.0, -1.0]),  # power of the length scale
    )
    knots = Knot(e_cases, tables.n_s, tables.a, l_cases)
    return over_cases(knots, cases, tables.delta)
