# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The draw and the checks on the inputs.

One draw is thirteen numbers: the ten metamodel inputs, each uniform on its prior box,
the central density n_c, and the two mutual friction coefficients. Five checks then
say whether the ten make an acceptable equation of state:

- neutron gas: the energy of pure neutron matter at one low density must be close to
  the value of Neill et al 2026 (a weight, compared with a random number);
- symmetric matter: the energy of symmetric matter at one low density must be close
  to the value of Drischler, Hebeler and Schwenk 2016 (also a weight);
- the symmetry energy stays positive up to n_c;
- stable: the pressure rises with density up to n_c;
- causal: sound is slower than light up to n_c.

Under vmap a draw cannot be dropped part way, so every check gives True or False and
the draw is dropped afterwards.

The 12 values of a draw, in order: r_gas and r_sym, the random numbers the two weights
are compared with, then E_sat, n_sat, K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym,
Z_sym in MeV (n_sat in fm^-3). Every function that takes values takes these 12.

Contents:

- draw_in_range: one random value in a range.
- draw_values: the 12 values of a draw.
- low_density_checks: the neutron gas and symmetric matter checks.
- symmetry_energy_positive: the symmetry energy stays positive.
- stable_and_causal: the pressure rises and sound is slower than light.
- draw_n_c: the central density.
- draw_b: the two mutual friction coefficients.
- draw_and_check: one draw with every check as a flag.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from deepska.glitch.data.metamodel_inputs import (
    b_core_prior,
    b_crust_prior,
    n_c_prior,
    neutron_gas_band,
    priors,
    symmetric_matter_band,
)
from deepska.glitch.kinds import Arr, Num
from deepska.glitch.metamodel.sums import e_pnm, e_q
from deepska.glitch.metamodel.uniform import cs2, delta_beta, dp_dn


def draw_in_range(key: Arr, low: Num, high: Num) -> Array:
    """
    Draw a random value between low and high.

    :param key: The random key.
    :param low: The lower end of the range.
    :param high: The upper end of the range.
    :returns: A value uniform on the range.
    """
    return jax.random.uniform(key, minval=low, maxval=high)


def draw_values(key: Arr) -> Array:
    """
    Draw the 12 values, each in its range.

    r_gas and r_sym on [0, 1], then the ten inputs on their prior boxes, in the order
    of priors: E_sat, n_sat, K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym.

    :param key: The random key of this draw.
    :returns: The 12 values, (12,).
    """
    lows = jnp.array([0.0, 0.0, *(low for low, _high in priors.values())])
    highs = jnp.array([1.0, 1.0, *(high for _low, high in priors.values())])
    keys = jax.random.split(key, 12)
    return jax.vmap(draw_in_range)(keys, lows, highs)


def low_density_checks(values: Arr) -> tuple[Array, Array]:
    """
    Test the neutron gas and symmetric matter checks, each on its own.

    Each energy gives a weight, and the check passes when the weight is above the
    draw's random number for it: w_gas > r_gas and w_sym > r_sym.

    w_gas = exp[-1/2 ((E(0.0085, 1) - 2.912) / 0.522)^2], E(0.0085, 1) = E_sat + E_sym
    w_sym = exp[-1/2 ((E(0.05, 0) + 8.57) / 0.75)^2],    E(0.05, 0) = E_sat

    The densities, centres and widths are neutron_gas_band and symmetric_matter_band
    in data/metamodel_inputs.py.

    :param values: The 12 values from draw_values.
    :returns gas: True when the neutron gas check passes.
    :returns symmetric: True when the symmetric matter check passes.
    """
    r_gas, r_sym = values[0], values[1]
    e_sat, n_sat, k_sat, q_sat, z_sat, e_sym, l_sym, k_sym, q_sym, z_sym = values[2:]
    c_sat = jnp.array([e_sat, 0.0, k_sat, q_sat, z_sat])
    c_sym = jnp.array([e_sym, l_sym, k_sym, q_sym, z_sym])

    n_gas, centre_gas, width_gas = neutron_gas_band
    e_gas = e_pnm(n_gas, c_sat, c_sym, n_sat)
    w_gas = jnp.exp(-0.5 * ((e_gas - centre_gas) / width_gas) ** 2)
    n_sm, centre_sym, width_sym = symmetric_matter_band
    e_sm = e_q(n_sm, c_sat, n_sat)
    w_sym = jnp.exp(-0.5 * ((e_sm - centre_sym) / width_sym) ** 2)

    return w_gas > r_gas, w_sym > r_sym


def symmetry_energy_positive(
    values: Arr, n_c: Num, *, n_low: float = 0.02, n_points: int = 300
) -> Array:
    """
    Test that E_sym(n) > 0 at every n in [n_low, n_c].

    E_sym is worked out at all the points at once, vmapped over n.

    :param values: The 12 values from draw_values.
    :param n_c: The central density in fm^-3.
    :param n_low: The lowest density tested, in fm^-3.
    :param n_points: How many evenly spaced densities are tested.
    :returns: True when E_sym stays positive at every point.
    """
    # only n_sat and the five of the symmetry energy are needed here
    n_sat = values[3]
    e_sym, l_sym, k_sym, q_sym, z_sym = values[7:]
    c_sym = jnp.array([e_sym, l_sym, k_sym, q_sym, z_sym])

    n = jnp.linspace(n_low, n_c, n_points)
    e_sym_n = jax.vmap(e_q, in_axes=(0, None, None))(n, c_sym, n_sat)

    return jnp.all(e_sym_n > 0)


def stable_and_causal(
    values: Arr, n_c: Num, *, n_low: float = 0.02, n_points: int = 1600
) -> tuple[Array, Array]:
    """
    Test that dP/dn > 0 (stable) and c_s^2 < 1 (causal) at every n in [n_low, n_c].

    At each point delta comes from the bisection on the charge balance, then dP/dn and
    c_s^2; all the points at once, vmapped over n. 1600 points, not 400: at 400 one
    draw in about 600 had a dip in P narrower than the step, which slipped through.

    :param values: The 12 values from draw_values.
    :param n_c: The central density in fm^-3.
    :param n_low: The lowest density tested, in fm^-3.
    :param n_points: How many evenly spaced densities are tested.
    :returns stable: True when the pressure rises at every point.
    :returns causal: True when sound is slower than light at every point.
    """
    e_sat, n_sat, k_sat, q_sat, z_sat, e_sym, l_sym, k_sym, q_sym, z_sym = values[2:]
    c_sat = jnp.array([e_sat, 0.0, k_sat, q_sat, z_sat])
    c_sym = jnp.array([e_sym, l_sym, k_sym, q_sym, z_sym])

    n = jnp.linspace(n_low, n_c, n_points)

    def at(n_i: Num) -> tuple[Array, Array]:
        delta = delta_beta(n_i, c_sym, n_sat)
        return dp_dn(n_i, delta, c_sat, c_sym, n_sat), cs2(
            n_i, delta, c_sat, c_sym, n_sat
        )

    slope, sound = jax.vmap(at)(n)
    return jnp.all(slope > 0), jnp.all(sound < 1)


def draw_n_c(key: Arr) -> Array:
    """
    Draw n_c, the central density, uniform on its prior.

    It has its own key, folded from the draw's, so the 12 values don't move.

    :param key: The random key of this draw.
    :returns: n_c in fm^-3.
    """
    return draw_in_range(jax.random.fold_in(key, 1), *n_c_prior)


def draw_b(key: Arr) -> tuple[Array, Array]:
    """
    Draw B_core and B_crust, each uniform in log B on its prior.

    Each has its own key, folded from the draw's.

    :param key: The random key of this draw.
    :returns b_core: The mutual friction in the core.
    :returns b_crust: The mutual friction in the crust, the same in every shell.
    """
    log_core = draw_in_range(jax.random.fold_in(key, 2), *np.log(b_core_prior))
    log_crust = draw_in_range(jax.random.fold_in(key, 3), *np.log(b_crust_prior))
    return jnp.exp(log_core), jnp.exp(log_crust)


def draw_and_check(key: Arr) -> tuple[Array, Array, Array, dict[str, Array]]:
    """
    Make one draw and run every check on it, as flags.

    The draw is never dropped here, it is dropped afterwards with values[ok]. Every
    check runs on every draw, as under vmap a draw can't skip the later checks.

    :param key: The random key of this draw.
    :returns values: The 12 values from draw_values.
    :returns n_c: The central density in fm^-3.
    :returns ok: True when every check passes.
    :returns why: One True or False per check, True where that check failed:
        "neutron gas", "symmetric matter", "E_sym <= 0", "unstable", "acausal".
    """
    values = draw_values(key)
    n_c = draw_n_c(key)
    gas, symmetric = low_density_checks(values)
    positive = symmetry_energy_positive(values, n_c)
    stable, causal = stable_and_causal(values, n_c)
    why = {
        "neutron gas": ~gas,
        "symmetric matter": ~symmetric,
        "E_sym <= 0": ~positive,
        "unstable": ~stable,
        "acausal": ~causal,
    }
    ok = gas & symmetric & positive & stable & causal
    return values, n_c, ok, why
