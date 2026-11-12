# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The metamodel equation of state: ten inputs to one table, then the star and the rise.

The nuclear metamodel of Margueron et al 2018 with a compressible liquid drop crust,
joined to the core by a Maxwell construction. One module per part, in the order they
are used:

- sums: the two metamodel sums and their derivatives.
- draw: the priors, the draw and the five checks on the inputs.
- uniform: uniform matter in beta equilibrium, its pressure, dP/dn and
  the sound speed; the core of the eos.
- cell: the crust cell, its energy, four slopes and ten curvatures.
- crust: the eight starts, who wins, every density, drip and P.
- join: the Maxwell join.
- table: the one table, and P against the energy density.
- star: the table into the star and the glitch rise.
- run: many draws on every core, and the sensitivity of the rise to each input.

The damped Newton the crust uses is numerics/newton_4x4.py, the fixed numbers are
data/metamodel_inputs.py and the figures are plots/metamodel_output.py.

Units are MeV and fm throughout: densities in fm^-3, energy densities and pressures in
MeV fm^-3, masses as rest energies. The star works in cgs and star.py converts.

Everything is jax and of fixed shape, so one draw is one jit, many draws are one vmap,
and the derivative of the rise with respect to the inputs runs by forward mode autodiff.
Under vmap a draw cannot be dropped part way: every draw runs to the end and returns
ok, True or False, with the reason, one True or False per check, and the dropped ones
are removed afterwards.

The Negele and Vautherin crust of the paper run, glitch_rise_model/eos, is separate and
unchanged.
"""
