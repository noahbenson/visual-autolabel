# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2025/__init__.py

"""Tooling and functions specific to the paper by Benson, Song, et al. (2025).

The `visual_autolabel.benson2025` subpackage contains code that implements the
CNN training described in the associated paper. It should serve both as the
repository of the article's code and as an example of how to use the
`visual_autolabel` library for other parts of cortex and other kinds of data.
"""

from . import config
from . import hcp
from . import nyu

from .analysis import (
    scores,
    all_scores,
    unet,
    all_unets)

#from ._core import osf_repository

# We import the UNet used as a convenience.
from ..image import UNet


# These are descriptions of the features used in the training/analysis.
