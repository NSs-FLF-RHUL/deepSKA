# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
From the ten metamodel inputs to the star inputs.

Like the paper run, but the eos, rho_d and rho_cc all come from the ten metamodel
inputs instead of the Negele and Vautherin table. And the TOV starts where the inner
core ends, at 2 n_sat, and marches the outer core and the crust with the same eos,
instead of starting at the crust-core boundary.

Contents:

- metamodel_table: the eos as one table, with n_drip and n_cc.
- pressure_at: P at one n_b, in cgs.
- metamodel_to_star_inputs: the StarInputs from the table and the inner core.
"""

import jax
import numpy as np

from deepska.glitch.data.paper_inputs import dom_crit, dr, n_shells
from deepska.glitch.glitch_rise_model.eos.metamodel_cgs import p_of_rho, to_cgs
from deepska.glitch.glitch_rise_model.eos.metamodel_eos import joined, metamodel_eos
from deepska.glitch.kinds import Bulk, EosTable, StarInputs


def metamodel_table(bulk: Bulk) -> EosTable:
    """
    Build the eos from the ten inputs as one table.

    :param bulk: The metamodel inputs.
    :returns: The table from n_b = 1e-9 to 2 n_sat, MeV and fm, with n_drip and n_cc.
    """
    out = jax.jit(metamodel_eos)(bulk)  # one compile, not one per small step
    return EosTable(joined(out), float(out.n_drip), float(out.transition.n_cc))


def pressure_at(table: dict[str, np.ndarray], n: float) -> float:
    """
    Read P off the table at one n_b.

    :param table: The table from metamodel_table.
    :param n: The baryon density in fm^-3.
    :returns: P in erg/cm^3.
    """
    _rho, p = to_cgs(table)
    return float(np.exp(np.interp(n, table["nb"], np.log(np.array(p)))))


def metamodel_to_star_inputs(
    eos: EosTable, inner_core: tuple[float, float], *, n_tov: int = 70_000
) -> StarInputs:
    """
    Make the star inputs from the table and the inner core.

    :param eos: The eos, from metamodel_table.
    :param inner_core: The inner core's mass m0 in g and radius r0 in cm, where the
        TOV starts.
    :param n_tov: TOV steps of 10 cm as in the paper. 70,000 is room for 7 km of outer
        core and crust; the 1.4 solar mass star of the default inputs needs 5.5 km.
    :returns: The star inputs, the rest from paper_inputs.
    """
    rho, _p = to_cgs(eos.table)
    nb = eos.table["nb"]
    rho_d, rho_cc = (float(np.interp(n, nb, rho)) for n in (eos.n_drip, eos.n_cc))
    m0, r0 = inner_core
    rho_min = float(rho[0])  # the surface: the table's lowest point
    eos_of = p_of_rho(eos.table)
    return StarInputs(
        eos_of, rho_d, rho_cc, rho_min, m0, r0, dom_crit, dr, n_tov, n_shells
    )
