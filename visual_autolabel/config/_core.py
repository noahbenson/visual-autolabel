# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/config.py
# Configuration of the visual_autolabel package.
# This file just contains global definitions that are used throughout.

import warnings

import numpy as np
import neuropythy as ny


#-------------------------------------------------------------------------------
# saved_image_size
# The size of images (number of image rows) that get saved to cache. It's
# relatively efficient to downsample images, so keeping this somewhat larger
# than needed is a good idea.
saved_image_size = 512

#-------------------------------------------------------------------------------
# default_image_size
# The default size for images used in model training. This is the number of rows
# in the images.
default_image_size = 128

#-------------------------------------------------------------------------------
# default_partition
# The default training-validation partition. This should be a tuple of either
# the subject count for each category or the subject fraction for each category.
# The training fraction is listed first, and the validation fraction is listed
# second.
default_partition = (0.8, 0.2)

#-------------------------------------------------------------------------------
# sids
# The subject-IDs that we use in training.
if len(ny.data['hcp_lines'].exclusions) == 0:
    warnings.warn('neuropythy "hcp_lines" dataset is missing exclusions')
sids = np.array([sid for sid in ny.data['hcp_lines'].subject_list
                 if ('mean',sid,'lh') not in ny.data['hcp_lines'].exclusions
                 if ('mean',sid,'rh') not in ny.data['hcp_lines'].exclusions])
sids.flags['WRITEABLE'] = False
nyusids = np.array(
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
nyusids.flags['WRITEABLE'] = False
