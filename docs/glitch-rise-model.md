# Glitch Rise Model (`deepska.glitch`)

The pulsar glitch rise model of [Graber, Cumming and Andersson 2018, ApJ 865, 23](https://doi.org/10.3847/1538-4357/aad776) written in JAX.
It:

- Builds the crust from the Negele and Vautherin 1973 equation of state with the TOV equations.
- Splits the pinned superfluid into cylindrical shells
- Gets the mutual friction from the Epstein and Baym pinning cases A, B and C.
- Then then runs the spin equations (Eqs 20 to 22 of the paper) forward in time.

Everything is `float64` and can be `vmap`-ed, so sweeps over the inputs are quick.

Four scripts are provided in the `scripts/` folder at the top level of the repository.
These scripts can be executed in an environment with the `deepska` package installed.
Each script has settings that can be configured at the top of the file:

```sh
python -m scripts/paper.py        # Table 2 and the 13 figures of the paper
python -m scripts/sweep.py        # sweep any inputs and compare with the Vela data
python -m scripts/sensitivity.py  # how much each output moves with each input
python -m scripts/metamodel.py    # the metamodel equation of state, see below
```

There is one function per file, so a change to one piece stays in that file.
Names are the paper's symbols in lowercase (`b_core`, `omega0`, `i_sf`, `r_drip`), and inputs that always
go together are grouped in `deepska/glitch/kinds.py` (`StarInputs`, `Spin`, `Star`).
A single run is `param_to_output(star_inputs, b_profile, b_core, spin)`.

Autodiff through the star build (`m0`, `r0`, `rho_d`, `rho_cc`, `rho_min`) should not be trusted.
The surface is found by a straight line across the last TOV row, so the radius is a small sawtooth in those inputs and the gradient comes out as the slope of one tooth.
The sensitivity script uses a 1% nudge each way for those inputs and autodiff for the rest.

The Vela 2016 glitch timing residuals in `deepska/glitch/data/palfreyman2018_vela_glitch_residuals.csv` is the supplementary data file of [Palfreyman et al. 2018, Nature 556, 219](https://doi.org/10.1038/s41586-018-0001-x) (`41586_2018_1_MOESM1_ESM.csv`, the same file that is in [vanessagraber/glitchrises](https://github.com/vanessagraber/glitchrises)).
It is only used to compare the model with the data.

## Metamodel equation of state (`deepska.glitch.metamodel`)

A second equation of state, built from ten nuclear inputs instead of read from a table: the metamodel of [Margueron, Hoffmann Casali and Gulminelli 2018, PRC 97, 025805](https://doi.org/10.1103/PhysRevC.97.025805) with a compressible liquid drop crust, joined to the core by a Maxwell construction.
The Negele and Vautherin crust above is separate and unchanged, and the paper, sweep and sensitivity scripts still use it.

`scripts/metamodel.py` has two functions:

- `run(n_draws)` draws all 13 inputs `n_draws` times (the ten metamodel inputs, the central density `n_c` and the mutual friction `B_core` and `B_crust`), drops the draws the checks reject, and builds the equation of state, the star and the glitch rise of the rest.
  About 3% of draws are kept. It prints why the others were rejected and saves every kept rise and a bar chart of the reasons.
- `sensitivity()` saves the derivative of the rise with respect to each of the 13 inputs, at the inputs as set in `deepska/glitch/data/metamodel_inputs.py`.

It runs on every core but one.
To choose how many, for example the 4 cores a cluster job was given, set `cores` at the top of the script.

There is one file per part, in the order they are used:

- `sums.py`: the two metamodel sums `E_sat(n)` and `E_sym(n)` and their derivatives, the five terms written out.
- `draw.py`: the priors, the draw, and the five checks on the inputs.
  Two weight a draw by how close its energy is to calculations at one low density: of pure neutron matter (Neill et al 2026) and of symmetric matter (Drischler, Hebeler and Schwenk 2016).
  The other three need, everywhere from 0.02 fm^-3 to `n_c`, a positive symmetry energy, a pressure that rises with density, and sound slower than light.
- `uniform.py`: uniform neutron, proton, electron and muon matter in beta equilibrium, the core of the equation of state.
- `cell.py`: the crust cell, with its energy, four slopes and ten curvatures written out by hand. The surface and curvature tensions are those of Grams et al 2022.
- `crust.py`: the cell at each of 600 densities, by `numerics/newton_4x4.py` from eight fixed starts, then neutron drip and the pressure.
- `join.py`: the Maxwell join, where the crust and the core have the same pressure and neutron chemical potential.
- `table.py`: the equation of state as one table of 1001 rows, and the pressure against the energy density.
- `star.py`: the table into the star, with the TOV run from the centre, and the glitch rise.
- `run.py`: many draws on every core, and the sensitivity of the rise.

The fixed numbers live in `deepska/glitch/data/metamodel_inputs.py`, and the constants in MeV and fm in `data/constants.py`.
Units are MeV and fm until `star.py` converts to cgs.

Everything is of fixed shape, so one draw is one `jit` and many draws are one `vmap`.
Under `vmap` a draw cannot be dropped part way: every draw runs to the end and returns `ok`, with the reason as one flag per check, and the dropped ones are removed afterwards.

The derivative of the rise with respect to the inputs should be taken by forward mode (`jax.jacfwd`), which gives all 13 in about a second after compiling.
Reverse mode (`jax.grad`) works but goes back through every TOV and rise step, about 30 minutes.
The crust-core boundary and neutron drip are found inside the star by the pressure (`star/moments_by_pressure.py`), not by the density, as the density jumps at a Maxwell join and the radius would move in steps of one TOV row.
The rise is then continuous in the inputs, with a kink where the join passes a point of the crust grid.

### Variable names

The code uses the short symbols of the equations.

A draw and its inputs:

- `values`: the 12 numbers of a draw, `r_gas` and `r_sym` (the random numbers the two weights are compared with) then the ten inputs.
- `c_sat`: the five coefficients of symmetric matter, E_sat, 0, K_sat, Q_sat and Z_sat, in MeV.
- `c_sym`: the five coefficients of the symmetry energy, E_sym, L_sym, K_sym, Q_sym and Z_sym, in MeV.
- `ns`: the saturation density n_sat, in fm^-3.
- `n_c`: the central density of the star, in fm^-3.
- `b_core`, `b_crust`: the mutual friction in the core and in the crust.
- `ok`: whether a draw is kept. `why`: which check it failed.

Uniform matter (`sums.py`, `uniform.py`):

- `chi`: (n - n_sat) / 3 n_sat, how far the density is from saturation.
- `u`, `u1`, `u2`: the low density correction u_k and its two derivatives.
- `e_q`, `e_q1`, `e_q2`: E_sat or E_sym and its first and second derivatives in density.
- `delta`: the asymmetry (n_n - n_p) / n.
- `eps`, `p`: the energy density and the pressure, in MeV fm^-3.
- `mu_n`: the neutron chemical potential, in MeV.

The crust (`cell.py`, `crust.py`):

- `u`: the four unknowns of a cell, `d_cl`, `n_cl`, `n_g` and `a`: the cluster's asymmetry, the density inside the cluster, the density of the neutron gas around it, and its number of nucleons A.
- `n_b`: the baryon density of the cell.
- `v`: the fraction of the cell the cluster fills.
- `y_p`: the cluster's proton fraction.
- `e_c`, `e_s`, `e_v`: the Coulomb, surface and curvature energies per nucleon.
- `w`: the cluster's energy per nucleon, with those three.
- `s1`, `s2`, `c1`, `c2`, `f1`, `f2`: the first and second log derivatives of the surface tension, the curvature tension and the Coulomb screening f(v).
- `b_d`, `b_c1`, `b_c2`, `big_m`, `b_g`: groups of terms the slopes and the curvatures share.
- `g`, `h`: the four slopes d eps / d u and the 4 x 4 curvatures.
- `n_drip`, `p_drip`: neutron drip, where the cell's mu_n reaches the neutron mass.

The join (`join.py`):

- `n1`, `n2`: the crust ends at n1 and the core starts at n2.
- `p_t`, `mu_t`: the pressure and the neutron chemical potential at the join.
- `n_cross`: where the cell and uniform matter have the same energy density.
- `n_flips`: how many times mu_cell - mu_uni changes sign.

## Credit & Copyright

The `jax` implementation of the Glitch Rise Model was implemented in 2026 Isaac Dodds, Royal Holloway University of London.
