# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2024/__init__.py

"""Tooling and functions specific to the paper Benson et al, 2024.
"""

from . import config
from . import hcp
from . import nyu

from .analysis import (
    scores,
    all_scores,
    unet,
    all_unets)

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
