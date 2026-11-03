# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
The whole picture, crust doc section 11: one eos from the ten metamodel inputs.

The crust walk on a log grid of n_b, the n_cc search between 0.01 and 0.12 fm^-3, then
the outer core with its proton exchange term on a linear grid from n_cc to 2 n_sat.
Everything is fixed shape and jax, so d(anything downstream) / d(inputs) runs by
autodiff.

Contents:

- metamodel_eos: the eos, crust, core and transition.
- joined: the one table, crust points below n_cc then the core, as plain arrays.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from deepska.glitch.glitch_rise_model.eos.crust import walk
from deepska.glitch.glitch_rise_model.eos.crust_core_transition import n_cc
from deepska.glitch.glitch_rise_model.eos.outer_core import uniform
from deepska.glitch.kinds import Bulk, Core, Eos


def metamodel_eos(
    bulk: Bulk,
    *,
    nb_lo: float = 1e-9,
    nb_hi: float = 0.12,
    n_crust: int = 300,
    n_core: int = 400,
) -> Eos:
    """
    Build one eos from the metamodel inputs.

    :param bulk: The metamodel inputs.
    :param nb_lo: The lowest crust density in fm^-3.
    :param nb_hi: The highest crust density in fm^-3, certainly core by then.
    :param n_crust: Log points between them, doc section 11: 300 is 37 per decade.
    :param n_core: Linear points from n_cc to 2 n_sat.
    :returns: The crust walk, the core table, the transition, n_drip and which crust
        points lie below n_cc.
    """
    nbs = jnp.geomspace(nb_lo, nb_hi, n_crust)
    w = walk(nbs, bulk)  # certainly crust and the n_cc search, section 9
    n_drip = w.nb[jnp.argmax(w.ng > 0)]
    tr = n_cc(w, bulk)  # between 0.01 and 0.12, section 10
    nb_core = jnp.linspace(tr.n_cc, 2 * bulk.ns, n_core)  # certainly core
    u = jax.vmap(partial(uniform, bulk=bulk))(nb_core)
    core = Core(nb=nb_core, eps=u.eps, p=u.p, delta=u.delta, mu_n=u.mu_n)
    return Eos(
        crust=w, core=core, transition=tr, n_drip=n_drip, in_crust=w.nb < tr.n_cc
    )


def joined(out: Eos) -> dict[str, np.ndarray]:
    """
    Join the crust points below n_cc and the core points into one table, outside jit.

    :param out: The eos.
    :returns: nb, eps, p and a crust flag, plain numpy arrays.
    """
    c, k = out.crust, out.core
    m = np.array(out.in_crust)
    return {
        "nb": np.concatenate([np.array(c.nb)[m], np.array(k.nb)]),
        "eps": np.concatenate([np.array(c.eps)[m], np.array(k.eps)]),
        "p": np.concatenate([np.array(c.p)[m], np.array(k.p)]),
        "crust": np.concatenate([np.ones(m.sum(), bool), np.zeros(len(k.nb), bool)]),
    }
