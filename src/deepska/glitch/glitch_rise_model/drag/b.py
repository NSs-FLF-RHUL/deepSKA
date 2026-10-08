# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The drag coefficient B at the knots of the pinning tables.

- e_pa_of_eb: the case A pinning energy.
- b_drag_ratio: R, the drag to lift ratio.
- b_delta_v: the velocity difference a vortex needs to unpin.
- b_force: the pinning force per unit length.
- b_of_r: B from R.
- b: B, delta_v and f at one knot for one case.
- b_knots: B, delta_v and f at every knot for the three cases.
"""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.data.constants import hbar, kappa, m_u
from deepska.glitch.kinds import Case, Knot, Num, PinningTables


def e_pa_of_eb(e_s: Num, e_l: Num) -> Array:
    """
    Calculate the case A pinning energy from the two Epstein-Baym energies.

    :param e_s: The energy E_s of the pinning tables in erg.
    :param e_l: The energy E_l of the pinning tables in erg.
    :returns: The case A pinning energy in erg.
    """
    return jnp.sqrt(e_s**2 + e_s * e_l + 0.5 * e_l**2)


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


def b_force(delta_v: Num, rho_s: Num) -> Array:
    """
    Calculate the pinning force per unit length of vortex.

    :param delta_v: Critical velocity difference in cm/s.
    :param rho_s: Superfluid mass density in g/cm**3.
    :returns: The pinning force f in dyn/cm.
    """
    return kappa * delta_v * rho_s  # Magnus force at the unpinning velocity


def b_of_r(ratio: Num) -> Array:
    """
    Calculate B from the drag to lift ratio R.

    Peaks at 0.5 when R = 1 and drops off either side, so big or small R both give weak
    coupling.

    :param ratio: The drag to lift ratio R.
    :returns: The mutual friction coefficient B.
    """
    return ratio / (1.0 + ratio**2)


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
