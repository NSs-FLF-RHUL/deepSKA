# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
First half of a run, star inputs in and the built star out.

None of this depends on b_core or the drag profile so a sweep over those only needs it
once.
"""

from deepska.glitch.glitch_rise_model.eos.make_eos import make_eos
from deepska.glitch.glitch_rise_model.star.moments import moments
from deepska.glitch.kinds import Star, StarInputs


def param_to_star(inputs: StarInputs) -> Star:
    """
    Build the star from its inputs.

    Eos first, then tov, splines, inertia and shells.

    :param inputs: The star inputs.
    :returns: The built star.
    """
    p_of, inverse = make_eos(inputs.eos, inputs.rho_d, inputs.rho_cc, inputs.rho_min)
    p_min = p_of(inputs.rho_min)  # pressure we integrate down to, the surface
    p0 = p_of(inputs.rho_cc)  # pressure at the crust core boundary, where tov starts
    return moments(inputs, p0, p_min, inverse)
