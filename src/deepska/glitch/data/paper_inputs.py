# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Inputs for the paper run, Graber et al 2018 on the Vela 2016 glitch, all cgs."""

from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp

from deepska.glitch.data.constants import m_u, msun


@dataclass
class PaperInputs:
    """Container class for Graber et al. 2018 values."""

    dr: float = 10.0  # tov step size
    n_tov: int = 20_000  # number of tov steps
    t_end: float = 120.0  # evolve end time
    n_t: int = 20_000  # number of t_end steps
    PROFILE_STEPS: list = field(
        default_factory=lambda: [
            0,
            20,
            50,
            100,
            300,
            500,
            1000,
            3000,
            5000,
            8000,
            20000,
        ]
    )  # steps whose superfluid profile is kept
    n_shells: float = (
        101  # shells the inner crust superfluid is split into, numerical not physics
    )
    rho_d: float = 4.0e11  # drip we go to
    rho_cc: float = 0.08e39 * m_u  # crust-core we go to
    rho_min: float = 1.0  # lowest density  we go to
    m0: float = 1.4 * msun  # core mass
    r0: float = 1e6  # core radius
    omega0: float = 2 * jnp.pi * 11.195  # Vela spin, rad/s
    dom_crit: float = 6.3e-3  # initial lag of the superfluid over the crust, rad/s
    bcore_range: jnp.ndarray = field(
        default_factory=lambda: jnp.array([1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 5e-4, 1e-2])
    )  # the 7 core couplings, 5e-5 weak core and 1e-2 strong core
    steps_a: list = field(
        default_factory=lambda: [0, 50, 300, 1000, 3000, 8000]
    )  # steps drawn for the case A profiles
    steps_c: list = field(
        default_factory=lambda: [0, 20, 100, 500, 5000, 20000]
    )  # steps drawn for the case C profiles
    b_flat: list = field(
        default_factory=lambda: [1e-1, 1e-2, 1e-3, 1e-4]
    )  # the flat drag profiles
    t_g: float = 57734.4849906  # glitch time, MJD
    t_0: float = 57734.4849521  # where the pre glitch mean stops, MJD
    vela_bin_width: float = 2.0  # s

    dt: float = field(init=False)  # evolve step size
    b_cores: jnp.ndarray = field(init=False)  # weak then strong, for the grid figures
    rho_b: jnp.ndarray = field(init=False)  # densities the B curves get drawn on
    k_a: int = field(
        init=False
    )  # where A sits in the profile list, flat ones come first
    k_c: int = field(
        init=False
    )  # where C sits in the profile list, flat ones come first

    @classmethod
    def vela_2016_default(cls) -> "PaperInputs":
        """Create an instance of the class with attributes set to the paper values."""
        return PaperInputs()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.dt = self.t_end / self.n_t
        self.b_cores = self.bcore_range[jnp.array([3, -1])]
        self.rho_b = jnp.logspace(jnp.log10(self.rho_d), jnp.log10(self.rho_cc), 400)
        self.k_a = len(self.b_flat)
        self.k_c = self.k_a + 2


# sits next to this file
vela_path = str(
    Path(__file__).resolve().parent / "palfreyman2018_vela_glitch_residuals.csv"
)
