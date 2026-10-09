# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The numerical grids made finer or coarser by a factor."""

from deepska.glitch.kinds import Inputs


def grid_variants(inputs: Inputs, factors: list[float]) -> list[tuple[str, Inputs]]:
    """
    Make the numerical grids finer or coarser, one at a time.

    The extent stays the same, so a finer grid is more points over the same radius or
    the same 120 s.

    :param inputs: The inputs of the run by name.
    :param factors: Factors to change each grid by, 2 is twice the points.
    :returns: A (label, changed inputs) pair per grid and factor.
    """
    variants = []
    for f in factors:
        variants.append(
            (
                f"tov grid x{f:g}",
                {"dr": inputs["dr"] / f, "n_tov": round(inputs["n_tov"] * f)},
            )
        )
        variants.append(
            (
                f"time grid x{f:g}",
                {"dt": inputs["dt"] / f, "n_t": round(inputs["n_t"] * f)},
            )
        )
        variants.append(
            (
                f"shells x{f:g}",
                {"n_shells": round((inputs["n_shells"] - 1) * f) + 1},
            )
        )
    return variants
