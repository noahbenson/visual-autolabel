# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/plot/_core.py
# Plotting utilities for the visual_autolabel library.

"""
The `visual_autolabel.plot` package contains utilities for use in and with the
`visual_autolabel` library.
"""

from ._core import (
    add_inferred,
    add_prior,
    add_raterlabels,
    summarize_dist,
    plot_distbars,
    plot_prediction
)

__all__ = [
    "add_inferred",
    "add_prior",
    "add_raterlabels",
    "summarize_dist",
    "plot_distbars",
    "plot_prediction"
]
