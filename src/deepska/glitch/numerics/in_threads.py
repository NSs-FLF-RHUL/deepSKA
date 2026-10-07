# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Threads for running batches side by side."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from deepska.glitch.kinds import Fn


def in_threads(
    run: Fn, jobs: list[tuple[np.ndarray, np.ndarray]], cores: int
) -> list[Any]:
    """
    Run a function on every job, several at once.

    :param run: Function taking one job.
    :param jobs: The jobs.
    :param cores: Number of jobs going at once.
    :returns: The result of every job, in the same order.
    """
    # first job alone so a jitted run compiles once not once per core
    first = run(jobs[0])
    with ThreadPoolExecutor(cores) as pool:
        rest = list(pool.map(run, jobs[1:]))  # map keeps them in order
    return [first, *rest]
