# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Inputs for the metamodel equation of state, in MeV and fm.

Every fixed number the metamodel needs that is not a physical constant (those are in
constants.py), in the order they are used:

- the low density correction of Margueron et al 2018;
- the surface and curvature tensions of Grams et al 2022, held fixed;
- the ten metamodel inputs as set, and the box each one is drawn from;
- the two low density checks a draw is weighted by, the neutron gas and symmetric
  matter;
- the eight starts of the crust cell minimiser;
- the star: where the TOV starts, and the mutual friction.
"""

import math

import jax.numpy as jnp

from deepska.glitch.data.constants import e2

# strength b of the low density correction, Margueron et al 2018 eq (13)
b_low = 10 * math.log(2.0)

# surface and curvature tensions, Grams et al 2022, held fixed, not fitted to the inputs
sigma_sat = 1.1  # surface tension of symmetric matter, MeV fm^-2
sigma_sym = 2.3  # its isospin dependence, MeV fm^-2
p_sigma = 3  # p, the shape of the surface tension against Y_p
sigma_c_sat = 0.1  # curvature tension of symmetric matter, MeV fm^-1
beta = 0.7
alpha = 5.5
# b_s = 2^p p (p + 1) sigma_sat / sigma_sym - 2^(p+1) = 29.91
b_s = 2**p_sigma * p_sigma * (p_sigma + 1) * sigma_sat / sigma_sym - 2 ** (p_sigma + 1)
alpha_sigma_c_over_sigma = alpha * sigma_c_sat / sigma_sat  # = 0.5 fm
x_c = 0.75 * (3 / math.pi) ** (1 / 3) * e2  # Coulomb exchange, 3/4 (3/pi)^(1/3) e^2

# the ten metamodel inputs as set, Margueron et al 2018, in MeV (n_sat in fm^-3). The
# order here is the order of the ten in every array of values
inputs_as_set = {
    "E_sat": -15.8,
    "n_sat": 0.16,
    "K_sat": 230.0,
    "Q_sat": 300.0,
    "Z_sat": 0.0,
    "E_sym": 32.0,
    "L_sym": 60.0,
    "K_sym": -100.0,
    "Q_sym": 0.0,
    "Z_sym": 0.0,
}
# the priors: each of the ten uniform on its box (low, high). Margueron et al 2018's
# mean plus or minus one spread; Q_sat and Z_sat on the wide ranges of Dinh Thi et al
# 2021
priors = {
    "E_sat": (-16.1, -15.5),  # MeV
    "n_sat": (0.15, 0.16),  # fm^-3
    "K_sat": (210.0, 250.0),  # MeV
    "Q_sat": (-1000.0, 1000.0),  # MeV
    "Z_sat": (-3000.0, 3000.0),  # MeV
    "E_sym": (30.0, 34.0),  # MeV
    "L_sym": (45.0, 75.0),  # MeV
    "K_sym": (-200.0, 0.0),  # MeV
    "Q_sym": (-400.0, 400.0),  # MeV
    "Z_sym": (-1500.0, 500.0),  # MeV
}
n_c_prior = (0.32, 0.6)  # the eleventh number, the central density n_c, fm^-3

# the two weights, each (density in fm^-3, centre in MeV, width in MeV)
# the energy of pure neutron matter: Neill et al 2026, from Schwenk and Pethick's
# unitary gas
neutron_gas_band = (0.0085, 2.912, 0.522)
# the energy of symmetric matter: Drischler, Hebeler and Schwenk 2016
symmetric_matter_band = (0.05, -8.57, 0.75)

# the 8 starts of the crust cell minimiser, rows (d_cl, n_cl, n_g / n_b, A). Starting
# guesses, not physics. n_g is a fraction of n_b, filled in at each density
crust_starts = jnp.array(
    [
        [0.07, 0.16, 0.0, 56.0],  # 1, iron-56, the low density guess
        [0.15, 0.14, 0.0, 60.0],  # 2
        [0.15, 0.14, 0.0, 300.0],  # 3
        [0.45, 0.14, 0.0, 60.0],  # 4
        [0.45, 0.14, 0.0, 300.0],  # 5
        [0.2, 0.14, 0.85, 150.0],  # 6, n_g = 0.85 n_b
        [0.4, 0.12, 0.6, 200.0],  # 7, n_g = 0.6 n_b
        [0.6, 0.10, 0.9, 400.0],  # 8, n_g = 0.9 n_b
    ]
)
# which unknowns move: rows 1 to 5 hold n_g = 0 (three unknowns), rows 6 to 8 all four
crust_free = jnp.array([[1.0, 1.0, 0.0, 1.0]] * 5 + [[1.0, 1.0, 1.0, 1.0]] * 3)

# the star: the TOV from the centre (n_b = n_c) out to the surface
r_start = 10.0  # the TOV starts this far from the centre, cm (r = 0 is 0/0)
n_tov_centre = 200_000  # TOV steps of 10 cm, room for a star up to 20 km
n_c_as_set = 0.4  # central density, fm^-3: a 1.38 M_sun star with the inputs as set

# the mutual friction B, the same in every crust shell
b_core_as_set, b_crust_as_set = 5e-5, 1e-3
# its priors, each uniform in log B on (low, high)
b_core_prior = (3e-5, 1e-4)  # Graber, Cumming and Andersson 2018, the Vela 2016 rise
b_crust_prior = (1e-5, 1e-1)  # below B_core (no overshoot) to strong Kelvin wave drag
