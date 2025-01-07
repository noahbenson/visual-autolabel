import os, sys, json

import pimms, torch
import numpy as np
import scipy as sp
import pandas as pd
import neuropythy as ny
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipyvolume as ipv

import visual_autolabel as va
import visual_autolabel.benson2024 as proj

dataset_cache_path  = '/data/visual-autolabel/datasets'
analysis_path = '/data/visual-autolabel/analysis'
model_cache_path = '/data/visual-autolabel/models'
dwi_filename_pattern = (
    '/data', 'hcp', 'tracts', '{subject}',
    '{hemisphere}.{tract_name}_normalized.mgz')

proj.config.model_cache_path     = model_cache_path
proj.config.dataset_cache_path   = dataset_cache_path
proj.config.analysis_pat         = analysis_path
proj.config.dwi_filename_pattern = dwi_filename_pattern

if len(sys.argv) != 3:
    print("SYNTAX: gendata.py <ventral/dorsal> <subno>")
    sys.exit(1)

if sys.argv[1] == 'ventral':
    dset = proj.hcp.make_datasets(
        'anat', ('hV4', 'VO1', 'VO2'),
        raters=proj.hcp.HCPImageCache._ventral_raters,
        multiproc=False)
else:
    dset = proj.hcp.make_datasets(
        'anat', ('V3a', 'V3b', 'IPS0', 'LO1'),
        raters=proj.hcp.HCPImageCache._dorsal_raters,
        multiproc=False)
val = dset['val']
trn = dset['trn']

ii = int(sys.argv[2])

if ii > len(trn):
    ii -= len(trn)
    trn = val
(inp, outp) = trn[ii]

sys.exit(0)
