# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Nuclear constants in MeV and fm, for the metamodel equation of state.

The star code works in cgs, see constants.py. The equation of state works in MeV and
fm, and the star code converts at the end.
"""

import math

hbarc = 197.327  # hbar c, MeV fm
e2 = 1.44  # the electron charge squared, MeV fm
mn, mp = 939.565, 938.272  # neutron and proton rest energies, MeV
me, mmu = 0.511, 105.66  # electron and muon rest energies, MeV
# strength b of the metamodel low density correction, Margueron et al 2018 eq (13)
b_low = 10 * math.log(2.0)
# the crust surface tension of crust doc section 7, held fixed and not fitted to the
# bulk inputs: sigma_s = sig_top / (Y_p^-3 + b_s + (1 - Y_p)^-3), where sig_top is
# sigma_0 (2^(p+1) + b_s) with sigma_0 = 1.1 MeV fm^-2 and p = 3, and the curvature
# tension is sigma_c = sigma_s (beta_c - Y_p) / 2
sig_top, b_s, beta_c = 50.50, 29.91, 0.7
