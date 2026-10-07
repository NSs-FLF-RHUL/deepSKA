# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Moment of inertia for whole star."""

from jax import Array

from deepska.glitch.kinds import Num


def star_inertia(m_star: Num, r_star: Num) -> Array:
    """
    Calculate the moment of inertia of the whole star, 0.35 M R**2.

    :param m_star: Mass of the star in g.
    :param r_star: Radius of the surface in cm.
    :returns: The moment of inertia in g cm**2.
    """
    return 0.35 * m_star * r_star**2
