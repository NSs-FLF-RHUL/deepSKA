# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The seven drag profiles of the paper."""

from deepska.glitch.data.paper_inputs import b_flat
from deepska.glitch.data.paper_labels import cases
from deepska.glitch.glitch_rise_model.drag.make_b_profile import (
    make_b_profile,
    make_b_profile_flat,
)
from deepska.glitch.kinds import Fn


def paper_b_profiles() -> list[Fn]:
    """
    List the seven drag profiles of the paper.

    :returns: The four flat profiles, then cases A, B and C.
    """
    return [make_b_profile_flat(v) for v in b_flat] + [
        make_b_profile(k) for k in range(len(cases))
    ]
