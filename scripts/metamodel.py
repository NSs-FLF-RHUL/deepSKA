# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The metamodel equation of state, from the draw to the glitch rise.

Two things, both at the bottom of this file:

- run(n_draws): draws all 13 inputs n_draws times, drops the draws the checks reject,
  and builds the eos, the star and the glitch rise of the rest. The only thing to
  change is the number of draws; about 3% of them are kept.
- sensitivity(): how the rise answers to each of the 13 inputs, at the inputs as set
  in deepska/glitch/data/metamodel_inputs.py.

The figures are saved to metamodel_out/ in whatever folder you run from.

The number of draws is set at the bottom, the number of cores at the top.
"""

import os
import sys

if __name__ == "__main__":
    # change this: how many cores to run on, here every core but one, or put a number.
    # XLA treats the CPU as this many devices, one share of the draws on each. Not
    # every core, as with every core taken jax 0.10 locks up (each share waits on
    # helper threads that have no core left to run on). It is up here as it has to be
    # set before jax is imported
    cores = max(1, (os.cpu_count() or 1) - 1)
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cores}"

    for name in [
        name
        for name in sys.modules
        if name == "deepska.glitch" or name.startswith("deepska.glitch.")
    ]:
        # forget the glitch files an earlier run loaded, so edits get picked up
        del sys.modules[name]

import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from matplotlib.figure import Figure

from deepska.glitch.data.metamodel_inputs import (
    b_core_as_set,
    b_crust_as_set,
    inputs_as_set,
    n_c_as_set,
)
from deepska.glitch.metamodel.run import rise_sensitivity, run_draws
from deepska.glitch.plots.metamodel_output import (
    metamodel_output,
    reasons_figure,
    rises_figure,
    sensitivity_figure,
)

# plain lines on screen, same as print would give
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# keep the compiled code on disk, so a rerun skips the compiling (same answers)
jax.config.update(
    "jax_compilation_cache_dir", str(Path("~/.cache/deepska_jax").expanduser())
)

# settings: change these
out_dir = Path("metamodel_out")  # where the figures are saved


def save(fig: Figure, name: str) -> None:
    """
    Save a figure as a png in out_dir.

    :param fig: The figure.
    :param name: Its file name, without the .png.
    """
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=130)
    log.info("saved %s", path)


def run(n_draws: int) -> dict[str, Any]:
    """
    Draw, run the checks, and build the eos, star and rise of every kept draw.

    Prints why draws were rejected and the range of each input among the kept ones,
    and saves two figures: every kept rise, and the rejected draws by reason.

    :param n_draws: How many draws. About 3% are kept.
    :returns: The run, from run_draws: the kept draws' inputs, rises, radii and masses.
    """
    result = run_draws(n_draws)
    metamodel_output(result)
    save(rises_figure(result["t"], result["dnu"]), "metamodel_rises")
    counts = result["counts"]
    save(reasons_figure(counts, n_draws, result["n_kept"]), "metamodel_reasons")
    return result


def sensitivity() -> None:
    """
    Draw the sensitivity of the rise to all 13 inputs, at the inputs as set.

    The ten inputs, n_c, B_core and B_crust are the ones in
    deepska/glitch/data/metamodel_inputs.py. Saves one figure.
    """
    values = jnp.array([0.0, 0.0, *inputs_as_set.values()])  # r_gas, r_sym not used
    t, slopes = rise_sensitivity(values, n_c_as_set, b_core_as_set, b_crust_as_set)
    title = (
        rf"Sensitivity of the glitch rise, inputs as set ($n_c$ = {n_c_as_set} "
        r"$\mathrm{fm}^{-3}$, "
        rf"$B_{{core}}$ = {b_core_as_set:g}, $B_{{crust}}$ = {b_crust_as_set:g})"
    )
    save(sensitivity_figure(t, slopes, title), "metamodel_sensitivity")


if __name__ == "__main__":
    run(5_000)  # change this: the number of draws
    sensitivity()
