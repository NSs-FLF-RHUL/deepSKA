# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The metamodel eos as the star code wants it, in cgs.

The density the star code uses is the mass-energy density eps / c^2, which the TOV
needs, not the rest mass density.

Contents:

- to_cgs: the joined table as rho in g/cm^3 and P in erg/cm^3.
- rho_of_p: the inverse eos, rho(P), a log-log pchip.
- p_of_rho: the eos, P(rho, rho_d, rho_cc), the shape StarInputs wants.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from deepska.glitch.data.constants import c, mev_fm3
from deepska.glitch.kinds import Fn, Num
from deepska.glitch.numerics.pchip import loglog_pchip_build, loglog_pchip_eval


def to_cgs(table: dict[str, np.ndarray]) -> tuple[Array, Array]:
    """
    Turn the table into cgs.

    :param table: The table from joined, MeV and fm.
    :returns rho: The mass-energy density eps / c^2 in g/cm^3.
    :returns p: The pressure in erg/cm^3.
    """
    return jnp.array(table["eps"] * mev_fm3 / c**2), jnp.array(table["p"] * mev_fm3)


def rho_of_p(table: dict[str, np.ndarray]) -> Fn:
    """
    Make the inverse eos rho(P), one value in and one out, as the TOV wants.

    :param table: The table from joined.
    :returns: rho in g/cm^3 at a P in erg/cm^3.
    """
    rho, p = to_cgs(table)
    params = loglog_pchip_build(p, rho)

    def inverse(p_q: Num) -> Array:
        return loglog_pchip_eval(params, p_q)[0]

    return inverse


def p_of_rho(table: dict[str, np.ndarray]) -> Fn:
    """
    Make the eos P(rho, rho_d, rho_cc), the shape StarInputs wants.

    rho_d and rho_cc are only there for the shape, the table already knows both.

    :param table: The table from joined.
    :returns: P in erg/cm^3 at a rho in g/cm^3, a scalar in and out or an array.
    """
    rho, p = to_cgs(table)
    params = loglog_pchip_build(rho, p)

    def eos(rho_q: Num, _rho_d: Num, _rho_cc: Num) -> Array:
        shape = jnp.shape(rho_q)
        return jnp.reshape(loglog_pchip_eval(params, rho_q), shape)

    return eos
