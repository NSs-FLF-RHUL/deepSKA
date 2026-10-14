# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
How much every output moves when each input moves, d out / d ln(input).

Per e-fold, so it compares fairly across inputs of any size. Spin inputs by forward mode
autodiff on the one star, all in one go. Star inputs by a nudge each way instead: the
surface is found by a straight line across the last tov row, so R against any star input
is a tiny sawtooth, and autodiff gives the slope of one tooth not the trend (R against
m0 comes out half what it should).
"""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.run.param_to_output import param_to_output
from deepska.glitch.glitch_rise_model.run.param_to_star import param_to_star
from deepska.glitch.glitch_rise_model.run.records_of import records_of
from deepska.glitch.glitch_rise_model.run.star_to_output import star_to_output
from deepska.glitch.kinds import Arr, Fn, Inputs, Num, Outputs, StarInputs


def sensitivity(
    inputs: Inputs, names: list[str], per_run: Fn, step: Num
) -> tuple[Outputs, Outputs]:
    """
    Calculate how much every output moves with each input.

    The slopes are d out / d ln(input), so per e-fold.

    :param inputs: The inputs by name, the point the slopes are taken at.
    :param names: The inputs to vary.
    :param per_run: Function that picks the outputs from a run.
    :param step: Nudge for the star inputs, 0.01 is 1 percent each way.
    :returns base: The outputs as set.
    :returns slopes: The slope of each output, one entry per input on the last axis.
    """
    spin_names = [n for n in names if n not in StarInputs._fields]
    star_names = [n for n in names if n in StarInputs._fields]

    # the inputs with each of some multiplied by exp(s)
    def scaled(some: list[str], s: Arr) -> Inputs:
        return {**inputs, **{n: inputs[n] * jnp.exp(s[i]) for i, n in enumerate(some)}}

    # star inputs nudged, so the star is built again
    def whole_run(s: Num) -> Outputs:
        p = scaled(star_names, s)
        star_inputs, spin = records_of(p)
        dnu, profiles, star = param_to_output(
            star_inputs, p["b_profile"], p["b_core"], spin
        )
        return per_run(dnu, profiles, star, p)

    star = param_to_star(records_of(inputs)[0])

    # spin inputs nudged, same star
    def on_star(s: Num) -> Outputs:
        p = scaled(spin_names, s)
        _, spin = records_of(p)
        dnu, profiles = star_to_output(star, p["b_profile"], p["b_core"], spin)
        return per_run(dnu, profiles, star, p)

    base = on_star(jnp.zeros(len(spin_names)))
    spin_slopes: Outputs = (
        jax.jacfwd(on_star)(jnp.zeros(len(spin_names))) if spin_names else {}
    )
    star_slopes: Outputs = {}
    if star_names:
        nudges = step * jnp.eye(len(star_names))  # one input at a time
        up, down = jax.vmap(whole_run)(nudges), jax.vmap(whole_run)(-nudges)
        star_slopes = jax.tree_util.tree_map(
            lambda a, b: jnp.moveaxis((a - b) / (2 * step), 0, -1), up, down
        )

    # one column per input, in the order they were asked for
    def in_order(key: str) -> Array:
        return jnp.stack(
            [
                spin_slopes[key][..., spin_names.index(n)]
                if n in spin_names
                else star_slopes[key][..., star_names.index(n)]
                for n in names
            ],
            axis=-1,
        )

    return base, {key: in_order(key) for key in base}
