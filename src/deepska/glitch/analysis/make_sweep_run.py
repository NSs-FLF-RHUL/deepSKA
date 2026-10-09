# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The two halves of a run in the form a sweep wants, swept inputs by name."""

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from deepska.glitch.glitch_rise_model.run.param_to_star import param_to_star
from deepska.glitch.glitch_rise_model.run.records_of import records_of
from deepska.glitch.glitch_rise_model.run.star_to_output import star_to_output
from deepska.glitch.kinds import Fn, Inputs, Outputs, Star
from deepska.glitch.numerics.pick_function import pick_function


def make_sweep_run(
    inputs: Inputs, per_run: Callable[..., Outputs]
) -> tuple[Callable[..., Star], Callable[..., Outputs]]:
    """
    Make the two halves of a run in the form a sweep needs.

    b_profile in inputs can be one profile function or a list of them, a swept
    "b_profile" then picks one by number.

    :param inputs: The inputs by name, used for anything that isnt swept.
    :param per_run: Function that picks what to keep from each run.
    :returns star_of: Function building the star for the swept inputs of one run.
    :returns run_on: Function doing one run on a built star.
    """
    b_profiles: list[Fn] = (
        inputs["b_profile"]
        if isinstance(inputs["b_profile"], list)
        else [inputs["b_profile"]]
    )

    # the star for one set of swept star inputs
    def star_of(d: Inputs) -> Star:
        return param_to_star(records_of({**inputs, **d})[0])

    # one run on a built star, d is every swept input of this run
    def run_on(star: Star, d: Inputs) -> Outputs:
        p = {**inputs, **d}
        # which profile, a number as vmap cant go over functions
        k = jnp.asarray(d.get("b_profile", 0)).astype(int)

        def b_profile(rho: Array) -> Array:
            return pick_function(b_profiles, k, rho)

        p["b_profile"] = b_profile
        _, spin = records_of(p)
        dnu, profiles = star_to_output(star, b_profile, p["b_core"], spin)
        return per_run(dnu, profiles, star, p)

    return star_of, run_on
