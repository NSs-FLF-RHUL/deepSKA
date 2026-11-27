# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Pinning tables, one value per inner crust domain, cgs."""

import jax.numpy as jnp

from deepska.glitch.data.constants import mev
from deepska.glitch.kinds import PinningTables

# pinning energy per site, cases B and C
e_pb = jnp.array([0.21, 0.29, -2.74, -0.72, -0.02]) * mev
# the two energies that combine into case A
e_s = jnp.array([0.42, -0.13, -1.64, -1.00, -0.78]) * mev
e_l = jnp.array([0.16, 0.94, 1.40, 1.00, 0.49]) * mev
# superfluid neutron density, 1e-4 fm^-3 -> cm^-3
n_s = jnp.array([4.8, 47.0, 184.0, 436.0, 737.0]) * 1e35
# nuclear radius, fm -> cm
r_n = jnp.array([5.9, 6.7, 7.2, 7.3, 7.2]) * 1e-13
# coherence length, fm -> cm
xi = jnp.array([15.6, 10.1, 12.0, 26.1, 90.8]) * 1e-13
# lattice spacing, fm -> cm
a = jnp.array([90.0, 72.5, 56.1, 39.8, 29.2]) * 1e-13
# vortex-nucleus overlap fraction
delta = 0.01
# density of each domain, g cm^-3
rho_b_data = jnp.array([1.5, 9.6, 33.9, 78.9, 131.0]) * 1e12

# all of the above as one thing to hand to b_knots
tables = PinningTables(e_pb, e_s, e_l, n_s, r_n, xi, a, delta, rho_b_data)
