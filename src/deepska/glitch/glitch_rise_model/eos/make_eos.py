# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The eos and its inverse for one star."""

from jax import Array

from deepska.glitch.kinds import Fn, Num
from deepska.glitch.numerics.invert_log import invert_log


def make_eos(eos: Fn, rho_d: Num, rho_cc: Num, rho_min: Num) -> tuple[Fn, Fn]:
    """
    Make the eos and its inverse for one star.

    :param eos: The crust eos, P(rho, rho_d, rho_cc) in erg/cm**3.
    :param rho_d: Neutron drip density in g/cm**3.
    :param rho_cc: Crust-core transition density in g/cm**3.
    :param rho_min: Lowest density we go to in g/cm**3.
    :returns p_of: P(rho) with the two densities filled in.
    :returns inverse: Its inverse, rho(P).
    """

    def p_of(rho: Num) -> Array:
        return eos(rho, rho_d, rho_cc)

    # rho from P, done in logs as P spans decades
    def inverse(p_target: Num) -> Array:
        return invert_log(p_of, p_target, rho_min, rho_cc)

    return p_of, inverse
