# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Glitch rise model of Graber, Cumming & Andersson 2018, written in jax."""

import jax

# float64 has to be switched on before any jnp array is made, importing anything from
# glitch runs this first
jax.config.update("jax_enable_x64", val=True)
