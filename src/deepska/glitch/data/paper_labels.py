# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Legend labels, colours and line styles of the paper figures."""

import math

from deepska.glitch.data.paper_inputs import b_flat

domains = ["I", "II", "III", "IV", "V"]  # inner crust domains, table 2
# pinning cases as the legend shows them
cases = ["(A)", "(B)", "(C)"]
colours = ["purple", "tab:blue", "orange"]  # one per case
# legend labels for the flat B values
flat = [f"1e{round(math.log10(v))}" for v in b_flat]

# dotted black
flat_styles = [(":", {"color": "black", "label": "B = " + f}) for f in flat]
case_styles = [
    ("-", {"color": colours[k], "label": cases[k]}) for k in range(len(cases))
]
# knot dots, no label
dot_styles = [("o", {"color": colours[k]}) for k in range(len(cases))]
profile_styles = flat_styles + case_styles  # same order as paper_b_profiles
