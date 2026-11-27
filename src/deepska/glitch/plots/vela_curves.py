# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""The Vela data as curves for figure, the 4 ways the figures show it."""

from deepska.glitch.kinds import Curve, Vela


def vela_curves(vela: Vela, *, top: bool = False) -> tuple[Curve, Curve, Curve, Curve]:
    """
    Make the curves of the Vela data, the 4 ways the figures show it.

    :param vela: The Vela data.
    :param top: True puts the data on top of a fan of model curves.
    :returns pulses: Every single pulse.
    :returns bins: Every bin, including before the glitch.
    :returns cum: The running sum from the glitch on.
    :returns from0: The bins from the glitch on.
    """
    extra = {"zorder": 3} if top else {}
    pulses: Curve = (
        vela.t_raw,
        vela.res_raw,
        "-",
        {"color": "lightgrey", "linewidth": 0.5, "label": "Vela pulses"},
    )
    style = {"color": "red", "label": "Vela, 2 s bins", **extra}
    bins: Curve = (vela.t_all, vela.vela_all, "o-", style)
    cum: Curve = (vela.t_bins, vela.cum_data, "o-", style)
    from0: Curve = (vela.t_bins, vela.vela_bin, "o-", style)
    return pulses, bins, cum, from0
