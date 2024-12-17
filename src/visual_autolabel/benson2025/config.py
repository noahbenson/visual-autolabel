# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2025/__init__.py

"""Configuration of the visual_autolabel.benson2025 module.

This module contains simple configuration items used in writing the paper Benson
et al, 2024. All of these configuration items are used by the functions of the
`visual_autolabel.benson2025` module when the optional parameter `Ellipsis` (or
`...`) is provided; for example, `hcp_dataset(111312)` is equivalent to calling
`hcp_dataset(111312, dataset_cache_path=benson2025.config.dataset_cache_path`.

Attributes
----------
hcp_sids : numpy array of ints
    The subject-IDs that we use in training and evaluating the CNNs. The HCP
    subjects were generated using the following code-block, which excludes
    subjects who were marked as excluded by Benson et al. (2022).
       >>> np.array(
       ...    [sid for sid in ny.data['hcp_lines'].subject_list
       ...     if ('mean',sid,'lh') not in ny.data['hcp_lines'].exclusions
       ...     if ('mean',sid,'rh') not in ny.data['hcp_lines'].exclusions])
nyu_sids : numpy array of strs
    The subject-IDs that we use as a secondary dataset to assess
    generalizability.  These were all publicly available subjects from the 
    NYU Retinotopy Dataset (Himmelberg et al., 2021) at the time that the paper
    associated with this library was being prepared.
dataset_cache_path : path-like
    The directory in which cache data can be stored. This option is used by all
    module functions when the parameter Ellipsis is provided. The actual cache
    data are stored in the subdirectories NYU and HCP.
model_cache_path : path-like
    The directory where models trained for the project are stored. Each model
    is stored in a subdirectory of this directory whose name is the model tag.
analysis_path : path-like
    The directory in which analysis data (primarily the scores dataframes) are
    stored.
dwi_filename_pattern
    The pattern to be used for DWI filenames. This should be a string or a tuple
    of strings, each of which is formatted (via the `format` method) using the
    named argument `'tract_name'` along with all of the key-value pairs in the
    target that is being loaded.  If the DWI_FILENAME_PATTERN environment
    variable is set, we use it.
"""

import os

import numpy as np


#-------------------------------------------------------------------------------
# hcp_sids
hcp_sids = np.array(
    [100610, 118225, 140117, 158136, 172130, 197348, 214524, 346137,
     412528, 573249, 724446, 825048, 905147, 102311, 159239, 173334,
     182739, 198653, 221319, 352738, 429040, 581450, 725751, 826353,
     910241, 102816, 126426, 145834, 162935, 175237, 185442, 199655,
     233326, 360030, 436845, 585256, 732243, 926862, 104416, 128935,
     146129, 164131, 176542, 186949, 200210, 239136, 365343, 463040,
     601127, 751550, 859671, 927359, 105923, 130114, 146432, 164636,
     177140, 187345, 200311, 246133, 380036, 467351, 617748, 757764,
     942658, 108323, 130518, 146735, 165436, 191033, 200614, 249947,
     381038, 525541, 627549, 765864, 871762, 109123, 131217, 146937,
     167036, 177746, 191336, 201515, 385046, 536647, 638049, 770352,
     872764, 111312, 167440, 178142, 191841, 203418, 257845, 389357,
     541943, 771354, 878776, 958976, 111514, 132118, 150423, 169040,
     178243, 204521, 263436, 393247, 547046, 654552, 782561, 878877,
     966975, 114823, 155938, 192641, 205220, 283543, 395756, 550439,
     671855, 783462, 898176, 971160, 115017, 134829, 156334, 169444,
     180533, 193845, 209228, 318637, 397760, 552241, 680957, 789373,
     899885, 973770, 115825, 135124, 157336, 169747, 181232, 195041,
     212419, 320826, 401422, 562345, 690152, 814649, 901139, 995174,
     116726, 137128, 158035, 171633, 181636, 196144, 330324, 406836,
     572045, 818859, 901442])
hcp_sids.flags['WRITEABLE'] = False

#-------------------------------------------------------------------------------
# nyu_sids
nyu_sids = np.array(
    ['sub-wlsubj001', 'sub-wlsubj004', 'sub-wlsubj006', 'sub-wlsubj007',
     'sub-wlsubj014', 'sub-wlsubj019', 'sub-wlsubj023', 'sub-wlsubj042',
     'sub-wlsubj043', 'sub-wlsubj045', 'sub-wlsubj046', 'sub-wlsubj055',
     'sub-wlsubj056', 'sub-wlsubj057', 'sub-wlsubj062', 'sub-wlsubj064',
     'sub-wlsubj067', 'sub-wlsubj071', 'sub-wlsubj076', 'sub-wlsubj079',
     'sub-wlsubj081', 'sub-wlsubj083', 'sub-wlsubj084', 'sub-wlsubj085',
     'sub-wlsubj086', 'sub-wlsubj087', 'sub-wlsubj088', 'sub-wlsubj090',
     'sub-wlsubj091', 'sub-wlsubj092', 'sub-wlsubj094', 'sub-wlsubj095',
     'sub-wlsubj104', 'sub-wlsubj105', 'sub-wlsubj109', 'sub-wlsubj114',
     'sub-wlsubj115', 'sub-wlsubj116', 'sub-wlsubj117', 'sub-wlsubj118',
     'sub-wlsubj120', 'sub-wlsubj122', 'sub-wlsubj126'])
nyu_sids.flags['WRITEABLE'] = False

#-------------------------------------------------------------------------------
# dataset_cache_path
dataset_cache_path = '/data/visual-autolabel/datasets'

#-------------------------------------------------------------------------------
# model_cache_path
model_cache_path = '/data/visual-autolabel/models'

#-------------------------------------------------------------------------------
# analysis_path
analysis_path = '/data/visual-autolabel/analysis'

#-------------------------------------------------------------------------------
# dwi_filename_pattern
dwi_filename_pattern_init = os.environ.get('DWI_FILENAME_PATTERN')


# Delete extraneous imports.
del np
del os
