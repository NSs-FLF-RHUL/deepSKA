# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Physical constants: cgs for the star, then MeV and fm for the metamodel eos.

The star code works in cgs. The metamodel equation of state works in MeV and fm with
masses as rest energies, and mev_fm3 turns its energy densities and pressures into the
star's units at the end.
"""

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

# nuclear physics in MeV and fm, for the metamodel equation of state
mev_fm3 = mev * 1e39  # MeV fm^-3 -> erg/cm^3, the eos's units to the star's
hbarc = 197.327  # hbar c, MeV fm
e2 = 1.44  # the electron charge squared, MeV fm
mn, mp = 939.565, 938.272  # neutron and proton rest energies, MeV
me, mmu = 0.511, 105.66  # electron and muon rest energies, MeV
