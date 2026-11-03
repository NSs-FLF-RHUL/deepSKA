# Glitch Rise Model (`deepska.glitch`)

The pulsar glitch rise model of [Graber, Cumming and Andersson 2018, ApJ 865, 23](https://doi.org/10.3847/1538-4357/aad776) written in JAX.
It:

- Builds the crust from the Negele and Vautherin 1973 equation of state with the TOV equations.
- Splits the pinned superfluid into cylindrical shells
- Gets the mutual friction from the Epstein and Baym pinning cases A, B and C.
- Then then runs the spin equations (Eqs 20 to 22 of the paper) forward in time.

Everything is `float64` and can be `vmap`-ed, so sweeps over the inputs are quick.

Three scripts are provided in the `scripts/` folder at the top level of the repository.
These scripts can be executed in an environment with the `deepska` package installed.
Each script has settings that can be configured at the top of the file:

```sh
python -m scripts/paper.py        # Table 2 and the 13 figures of the paper
python -m scripts/sweep.py        # sweep any inputs and compare with the Vela data
python -m scripts/sensitivity.py  # how much each output moves with each input
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

## Metamodel equation of state

`deepska/glitch/glitch_rise_model/eos/` also holds a second equation of state, built from the ten metamodel inputs of [Margueron, Hoffmann Casali and Gulminelli 2018, PRC 97, 025805](https://doi.org/10.1103/PhysRevC.97.025805) (`Bulk` in `kinds.py`) instead of read from a table.
It is not wired into the star build above yet; that still uses the Negele and Vautherin table.

- `outer_core.py`: uniform neutron, proton, electron and muon matter, with the composition from charge neutrality.
- `crust.py`: one compressible liquid drop cell at each density, minimised with `numerics/newton_4x4.py` from slopes and curvatures written out by hand.
- `crust_core_transition.py`: the transition density `n_cc` by the pasta condition of Neill 2026.
- `metamodel_eos.py`: the three together, `metamodel_eos(bulk)`.

The fixed numbers live in `deepska/glitch/data/nuclear_constants.py` (MeV and fm) and `data/crust_starts.py` (the minimiser's starting guesses).
Numerical settings such as grid sizes, bisection counts and Newton steps are keyword arguments with defaults.

The crust code follows the equations of the crust document line by line, so it uses the document's short symbols:

| Name                                                       | Quantity                                                                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `dcl`, `ncl`, `ng`, `a_cl`                                 | the four unknowns of a cell: cluster asymmetry 1 - 2 Y_p, density inside the cluster, neutron gas density, mass number A |
| `vv`, `yp`, `dd`                                           | the volume fraction v of the cluster, the proton fraction Y_p, and n_cl - n_g                                            |
| `ec`, `es`, `ev`, `w`                                      | the Coulomb, surface and curvature energies per nucleon, and the cluster energy per nucleon including them               |
| `eg`, `eg1`, `eg2`                                         | the energy per neutron of the gas and its first and second density derivatives                                           |
| `esym`, `esat1`, `esym1`, `esat2`, `esym2`                 | E_sym, and the first and second density derivatives of E_sat and E_sym at the cluster density                            |
| `m`                                                        | the electron chemical potential less the neutron-proton mass difference, mu_e - (m_n - m_p)                              |
| `ss`, `sc`, `ss2`, `sc2`                                   | sigma_s'/sigma_s, sigma_c'/sigma_c and the same ratios for the second derivatives                                        |
| `f1`, `f2`                                                 | f'/f and f''/f of the lattice correction f(v)                                                                            |
| `xx`, `r`                                                  | the electron relativity parameter x and sqrt(1 + x^2)                                                                    |
| `mm`, `bracket_d`, `bracket_c1`, `bracket_c2`, `bracket_g` | the bracketed groups of the slope equations, reused by the curvatures                                                    |
| `g_dcl`, `g_ncl`, `g_ng`, `g_a`                            | the four slopes d eps / d u                                                                                              |
| `h_dd`, `h_dc`, ... `h_aa`                                 | the ten distinct curvatures d2 eps / d u d u                                                                             |
| `u`, `g`, `h`                                              | in `newton_4x4.py`: the point (4,), its slopes (4,) and its curvatures (4, 4)                                            |
| `chi`, `u`, `u1`, `u2`                                     | in `outer_core.py`: (n - n_sat) / 3 n_sat, the low density factor u_k and its two derivatives                            |

## Credit & Copyright

The `jax` implementation of the Glitch Rise Model was implemented in 2026 Isaac Dodds, Royal Holloway University of London.
