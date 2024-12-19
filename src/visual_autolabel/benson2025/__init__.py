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

from ._core import osf_repository

# We import the UNet used as a convenience.
from ..image import UNet

# These are descriptions of the features used in the training/analysis.
input_descriptions = {
    'null': 'Nothing',
    'anat': 'T1 Only',
    't1t2': 'T1 & T2',
    'trac': 'T1 & DWI',
    'nofn': 'T1, T2, & DWI',
    'func': 'T1 & Retinotopy',
    'nodw': 'T1, T2, & Retinotopy',
    'not2': 'T1, DWI, & Retinotopy',
    'full': 'T1, T2, DWI, & Retinotopy',
    'tmpl': 'Benson et al. (2014), using T1',
    'warp': 'Benson & Winawer (2018), using T1 & Retinotopy',
    'rely': 'Inter-rater Reliability'}
output_descriptions = {
    'area': 'Visual Area Labels',
    'ring': 'Visual Ring Labels',
    'sect': 'Visual Sector Labels'}
