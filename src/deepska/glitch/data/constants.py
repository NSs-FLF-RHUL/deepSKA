# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""All cgs."""

import jax.numpy as jnp

g_newton, c, msun = 6.67430e-8, 2.99792458e10, 1.98847e33
m_u = 1.66053906660e-24  # g
hbar = 1.054571817e-27  # erg s
kappa = jnp.pi * hbar / m_u  # quantum of circulation, cm^2/s
mev = 1.602176634e-6  # MeV -> erg

# units in cgs, so x / km is x in km, same as MeV above
km = 1e5  # cm
ms = 1e-3  # s
ns = 1e-9  # s
muhz = 1e-6  # Hz
day = 86400.0  # s
