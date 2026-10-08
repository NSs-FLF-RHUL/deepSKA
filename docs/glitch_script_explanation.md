# Glitch rise model (`deepska.glitch`)

The pulsar glitch rise model of [Graber, Cumming and Andersson 2018, ApJ 865, 23](https://doi.org/10.3847/1538-4357/aad776) written in JAX.
It:

- Builds the crust from the Negele and Vautherin 1973 equation of state with the TOV equations.
- Splits the pinned superfluid into cylindrical shells
- Gets the mutual friction from the Epstein and Baym pinning cases A, B and C.
- Then then runs the spin equations (Eqs 20 to 22 of the paper) forward in time.

Everything is `float64` and can be `vmap`-ed, so sweeps over the inputs are quick.

Three scripts, each with its settings at the top of the file:

```sh
python -m deepska.glitch.scripts.paper        # Table 2 and the 13 figures of the paper
python -m deepska.glitch.scripts.sweep        # sweep any inputs and compare with the Vela data
python -m deepska.glitch.scripts.sensitivity  # how much each output moves with each input
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
