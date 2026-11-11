# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Which runs of a sweep get drawn as curves."""

import numpy as np

from deepska.glitch.kinds import Grid


def drawn_runs(draws: Grid, runs: int, drawn: int) -> Grid:
    """
    Pick which runs of a sweep get drawn as curves.

    :param draws: The inputs of every run as flat lists, one per swept input.
    :param runs: Number of runs in the sweep.
    :param drawn: The most runs to draw.
    :returns: The inputs of the picked runs.
    """
    if len(draws) == 1:
        idx = np.arange(0, runs, max(1, runs // drawn))  # one input: evenly spread
    else:
        # several: a random pick so every input gets covered, same pick every time
        idx = np.sort(
            np.random.default_rng(0).choice(runs, min(drawn, runs), replace=False)
        )
    return {n: np.asarray(v)[idx] for n, v in draws.items()}
