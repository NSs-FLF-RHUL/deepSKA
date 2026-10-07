# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The drag splines, built from the tables in b_data.

Runs once when this file is first imported so the splines only get built once. Gives
b_all, delta_v_all, f_all at the knots (3 cases by 5 knots) and p_b, the 3 log-log
splines stacked up.
"""

import jax

from deepska.glitch.data.b_data import tables
from deepska.glitch.glitch_rise_model.drag.b_knots import b_knots
from deepska.glitch.numerics.loglog_pchip_build import loglog_pchip_build

b_all, delta_v_all, f_all = b_knots(tables)  # knot values for all 3 cases
# one spline per case through them, same densities for all 3
p_b = jax.vmap(loglog_pchip_build, in_axes=(None, 0))(tables.rho_b_data, b_all)
