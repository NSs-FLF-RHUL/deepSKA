# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""From inputs by name to the two records a run takes."""

from deepska.glitch.kinds import Inputs, Spin, StarInputs


def records_of(p: Inputs) -> tuple[StarInputs, Spin]:
    """
    Sort the inputs of a run into the two records the model takes.

    :param p: The inputs of the run by name.
    :returns star_inputs: The star inputs.
    :returns spin: The spin settings.
    """
    return (
        StarInputs(*[p[k] for k in StarInputs._fields]),
        Spin(*[p[k] for k in Spin._fields]),
    )
