# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The fixed starting points of the crust cell minimiser, crust doc section 7.

Starting guesses, not physics. Rows are (dcl, n_cl, n_g, A): the cluster asymmetry,
the density inside the cluster in fm^-3, the neutron gas density in fm^-3, and the
mass number. In fixed4 the n_g column is a fraction of n_b, filled in at each density.
"""

import jax.numpy as jnp

# the first guess at the lowest density, iron-56 like
fe56 = jnp.array([0.07, 0.16, 0.0, 56.0])
# the four fixed starts with three unknowns, n_g held at 0
fixed3 = jnp.array(
    [
        [0.15, 0.14, 0.0, 60.0],
        [0.15, 0.14, 0.0, 300.0],
        [0.45, 0.14, 0.0, 60.0],
        [0.45, 0.14, 0.0, 300.0],
    ]
)
# the three fixed starts with four unknowns, n_g as a fraction of n_b
fixed4 = jnp.array(
    [[0.2, 0.14, 0.85, 150.0], [0.4, 0.12, 0.6, 200.0], [0.6, 0.10, 0.9, 400.0]]
)
