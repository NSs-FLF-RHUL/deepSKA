# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Saving a sweep to disk."""

import time
from pathlib import Path
from typing import Any

import numpy as np

from deepska.glitch.kinds import Grid


def save_sweep(folder: str | Path, out: Grid, values: Grid) -> str:
    """
    Save a sweep as one npz file.

    Every output goes in, plus the swept values as values_<name>.

    :param folder: Folder to save into, made if it isnt there.
    :param out: The outputs of the sweep.
    :param values: The swept values.
    :returns: The full path of the file.
    """
    where = Path(folder)
    path = where / (
        "sweep_{}_{}.npz".format("_".join(values), time.strftime("%Y-%m-%d_%H%M"))
    )
    where.mkdir(parents=True, exist_ok=True)
    # Any as the numpy stubs think one of the keys could be allow_pickle
    arrays: dict[str, Any] = {k: np.asarray(v) for k, v in out.items()}
    arrays |= {"values_" + n: v for n, v in values.items()}
    np.savez(path, **arrays)
    return str(path.resolve())
