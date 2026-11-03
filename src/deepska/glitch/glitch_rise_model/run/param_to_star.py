# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
First half of a run, star inputs in and the built star out.

None of this depends on b_core or the drag profile so a sweep over those only needs it
once.
"""

from deepska.glitch.glitch_rise_model.eos.make_eos import make_eos
from deepska.glitch.glitch_rise_model.star.moments import moments
from deepska.glitch.kinds import Fn, Num, Star, StarInputs


def param_to_star(
    inputs: StarInputs, *, rho_of_p: Fn | None = None, p_start: Num | None = None
) -> Star:
    """
    Build the star from its inputs.

    Eos first, then TOV, splines, inertia and shells.

    :param inputs: The star inputs.
    :param rho_of_p: The inverse eos rho(P), when there is one already, as for a table
        eos. None inverts P(rho) by bisection at every step instead.
    :param p_start: The pressure at r0 when r0 is where the inner core ends, so the TOV
        marches the outer core and the crust. None starts at the crust-core boundary,
        at P(rho_cc), as in the paper.
    :returns: The built star.
    """
    p_of, inverse = make_eos(inputs.eos, inputs.rho_d, inputs.rho_cc, inputs.rho_min)
    p_min = p_of(inputs.rho_min)  # pressure we integrate down to, the surface
    inverse = inverse if rho_of_p is None else rho_of_p
    if p_start is None:
        p0 = p_of(inputs.rho_cc)  # pressure at the crust core boundary, TOV starts
        return moments(inputs, p0, p_min, inverse)
    return moments(inputs, p_start, p_min, inverse, inner_core=True)
