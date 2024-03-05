# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2024/__init__.py


#===============================================================================
# Initialization

#-------------------------------------------------------------------------------
# Dependencies

import os, sys, json

import numpy as np
import neuropythy as ny

from .. import (
    FlatmapFeature,
    NullFeature,
    train_until,
    load_training,
    autolog
)
from ..plot import (
    add_inferred,
    add_prior,
    add_raterlabels
)


#-------------------------------------------------------------------------------
# Initialization

# DWI Features .................................................................
# If the DWI_FILENAME_PATTERN environment variable is set, extract it. We
# consider this an initialization variable because the "official"
# dwi_filename_pattern variable is in the benson2024 namespace (not 
# benson2024._core). That is where users will change the value; here is just
# where we ge the initial value.
dwi_filename_pattern_init = os.environ.get('DWI_FILENAME_PATTERN')
# Diffusion-weighted feature code; though in fairness, this is largely just code
# for loading the feature from a file.
class DWIFeature(FlatmapFeature):
    """Flatmap features loaded from DWI-based surface data files.
    
    This class provides instructions to the `visual_autolabel` library on how to
    load diffusion-weighted imaging data for use in training PyTorch models. The
    class can be configured by setting the static field `filename_pattern`. This
    field may be a string or a tuple of strings; in either case, all strings are
    first formatted with a dictionary containing all target data as well as the
    `hemisphere` name. If the field is a tuple, then the elements of the tuple
    are joined using `os.path.join`. (The `get_filename` classmethod may
    alternately be rewritten or overloaded.)
    """
    # DWI Filenames ------------------------------------------------------------
    # The filename pattern that we use:
    @classmethod
    def filename_pattern(self):
        """Returns the format pattern for the DWI filenames.
        
        By default, this returns the value found in the environment variable
        `DWI_FILENAME_PATTERN`. It can be changed by changing the value of
        `visual_autolabel.benson2024.dwi_filename_pattern`.
        """
        # We want to extract this from the benson2024 namespace, in case
        # someone changes it manually.
        from visual_autolabel.benson2024 import dwi_filename_pattern as patt
        return patt
    # The method to interpret that filename pattern.
    def get_filename(self, target, view, pattern=None):
        data = dict(target, **view)
        data['tract_name'] = self.tract_name
        flpatt = self.filename_pattern()
        if isinstance(flpatt, str):
            return flpatt.format(**data)
        else:
            flparts = [s.format(**data) for s in flpatt]
            return os.path.join(*flparts)
    # FlatmapFeature overloads -------------------------------------------------
    __slots__ = ('tract_name')
    def __init__(self, tract_name, interp_method=None):
        super().__init__(f'dwi_{tract_name}', interp_method=interp_method)
        self.tract_name = tract_name
    def get_property(self, fmap, target, view={}):
        # Here's the filename:
        filename = self.get_filename(target, view)
        try:
            prop = ny.load(filename)
        except ValueError:
            from warnings import warn
            # If we fail to load the file, we instead give it blank data.
            warn(f"failed to load '{self.property}' file: {filename}")
            prop = np.zeros(np.max(fmap.labels) + 1, dtype=float)
        return prop[fmap.labels]
# Given this type for loading features, we can now make a dictionary of features
# that we will enable the visual_autolabel library to use.
dwi_features = {
    'dwi_OR': DWIFeature('OR'),
    'dwi_VOF': DWIFeature('VOF')
}
hcp_features = dict(
    dwi_features,
    # Add in the 'zeros' feature, which represents all zeros for a null input.
    zeros=NullFeature('zeros')
)

# Training Feature Sets.........................................................
# The base feature-sets we are predicting:
vaonly_properties = ('V1', 'V2', 'V3')
econly_properties = ('E0', 'E1', 'E2', 'E3', 'E4')
# The base feature-sets we use to predict the above labels:
t1only_properties = ('x', 'y', 'z',
                     'curvature', 'convexity',
                     'thickness', 'surface_area')
t2only_properties = ('myelin',)
fnonly_properties = ('prf_x', 'prf_y', 'prf_sigma', 'prf_cod')
dwonly_properties = ('dwi_OR', 'dwi_VOF')
full_properties = (t1only_properties + t2only_properties +
                   dwonly_properties + fnonly_properties)
# The feature-sets by name.
hcp_input_properties = {
    'null': ('zeros',),
    'anat': t1only_properties,
    't1t2': t1only_properties + t2only_properties,
    'func': t1only_properties + fnonly_properties,
    'trac': t1only_properties + dwonly_properties,
    'not2': t1only_properties + fnonly_properties + dwonly_properties,
    'nofn': t1only_properties + t2only_properties + dwonly_properties,
    'nodw': t1only_properties + t2only_properties + fnonly_properties,
    'full': full_properties
}
hcp_output_properties = {
    'area': vaonly_properties,
    'ring': econly_properties,
    'sect': vaonly_properties + econly_properties,
}
# All the feature properties.
hcp_properties = dict(hcp_input_properties, **hcp_output_properties)
# Now the NYU properties.
nyu_input_properties = {
    'null': ('zeros',),
    'anat': t1only_properties,
    'func': t1only_properties + fnonly_properties,
}
nyu_output_properties = {
    'area': vaonly_properties,
}
# All the feature properties.
nyu_properties = dict(nyu_input_properties, **nyu_output_properties)
