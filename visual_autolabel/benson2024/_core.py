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
        if patt is None:
            raise RuntimeError("DWI_FILENAME_PATTERN not set")
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
nyu_features = {}

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


#-------------------------------------------------------------------------------
# Loading Model Data

def _to_model_cache_pseudo_path(model_cache_path):
    from pathlib import Path
    # Parse the model_cache_path.
    if model_cache_path is None:
        from visual_autolabel.config import model_cache_path
    elif isinstance(model_cache_path, Path):
        model_cache_path = os.fspath(model_cache_path)
    if not ny.util.is_pseudo_path(model_cache_path):
        model_cache_path = ny.util.pseudo_path(model_cache_path)
    return model_cache_path
def benson2024_unet(inputs, outputs, part='model', model_cache_path=None):
    """Loads a UNet from the Benson, Song, and Winawer (2024) dataset.
    
    Parameters
    ----------
    inputs : str
        The inputs of the model. This must be one of the following:
        * `'null'`. Blank images.
        * `'anat'`. Anatomical properties that can be derived from a T1-weighted
          image alone; these include the raw midgray coordinates of the
          individual vertices for the hemisphere (`'x'`, `'y'`, `'z'`), values
          derived from them (`'curvature'`, `'convexity'`), the gray-matter
          thickness (`'thickness'`) and the `'surface_area'` property, which
          indicates the unwarped vertex size. Because it is generally assumed
          that a T1-weighted image is always collected in an fMRI or sMRI
          experiment, these features are included in all other feature lists.
        * `'t1t2'`. Anatomical data that can be derived from a T1-weighted and a
          T2*-weighted image. These include all the `anat` properties and the
          `'myelin'` property.
        * `'func'`. Functional PRF measurements; these include `'prf_x'`,
          `'prf_y'`, `'prf_sigma'`, and `'prf_cod'` along with the anatomical
          properties.
        * `'trac'`. Properties that are derived from DWI measurements alone;
          these include `dwi_OR` and `dwi_VOF` along with the anatomical
          properties.
        * `'not2'`. All previous properties except thos derived from a
          T2-weighted image: i.e., all but `'myelin'`.
        * `'nodw'`. All previous properties except for the
          tractography/DWI-based properties; i.e., all but those in `trac`.
        * `'nofn'`. All previous properties except for those derived from
          functional data; i.e., all but those in `func`.
        * `'full'`. All previous properties.
    outputs : str
        The organizational property being predicted; this must be either
        `'area'` to predict visual area labels (V1, V2, V3) or `'ring'` to
        predict iso-eccentric regions (E0, E1, E2, E3, E4).
    part : str or tuple of str, optional
        The part or parts of the model data to load. The default is `'model'`,
        in which case the `UNet` itself is returned. The values `'history'`,
        `'options'`, or `'plan'` can also be used to extract the model's
        components. The value `'all'` can also be provided, in which case a
        dictionary is returned. If a tuple or list of strings is given, then an
        equivalent tuple is returned.
    model_cache_path : pathlike, optional
        The path from which to load the models. The default is `None`, which
        indicates that the `visual_autolabel.config.model_cache_path` should be
        used.

    Returns
    -------
    model : visual_autolabel.UNet
        The trained `UNet` model. This is returned when `part` is `'model'`.
    options : dict
        The global options used in all rounds of the training plan. This is
        returned when `part` is `'options'`. Note that the partition of
        subjects is part of the options.
    plan : list of dict
        The training plan for the model. This is a list of rounds of training,
        each of which is represented by a dictionary of training options. This
        is returned when `part` is `'plan'`.
    history : pandas.DataFrame
        The training history of the model. This is returned when `part` is
        `'history'`.
    """
    parts = ('model','options','plan','history')
    # Parse the model_cache_path first.
    model_cache_path = _to_model_cache_pseudo_path(model_cache_path)
    # Next parse the part argument.
    if not isinstance(part, str):
        from collections.abc import (Sequence, Set)
        if part is Ellipsis:
            part = set(parts)
        if isinstance(part, Sequence):
            return tuple(
                benson2024_unet(
                    inputs, outputs, part=p,
                    model_cache_path=model_cache_path)
                for p in part)
        elif isinstance(part, Set):
            return {
                k: benson2024_unet(
                    inputs, outputs, part=k,
                    model_cache_path=model_cache_path)
                for k in part}
        else:
            raise TypeError(f'unrecognized type for part option: {type(part)}')
    # Now we can descend into the models' directory and load the part.
    pp = model_cache_path.subpath(f'benson2024_{inputs}_{outputs}')
    part = part.lower()
    if part == 'all':
        return benson2024_unet(
            inputs, outputs, part=Ellipsis,
            model_cache_path=model_cache_path)
    elif part == 'model':
        import torch
        from visual_autolabel.benson2024 import UNet
        # We need to load the options to get the base model.
        opts = benson2024_unet(
            inputs, outputs, 'options',
            model_cache_path=model_cache_path)
        base_model = opts['base_model']
        state = torch.load(pp.local_path('model.pt'))
        nfeat = state['layer0.0.weight'].shape[1]
        nsegm = state['conv_last.weight'].shape[0]
        mdl = UNet(nfeat, nsegm, base_model=base_model)
        mdl.load_state_dict(state)
        return mdl
    elif part == 'options':
        with open(pp.local_path('options.json'), 'rt') as f:
             return json.load(f)
    elif part == 'plan':
        with open(pp.local_path('plan.json'), 'rt') as f:
             return json.load(f)
    elif part == 'history':
        import pandas as pd
        return pd.read_csv(
            pp.local_path('training.tsv'),
            sep='\t',
            keep_default_na=False)
    else:
        raise ValueError(f"unrecognized model part: '{part}'")
def benson2024_data(model_cache_path=None):
    """Returns a lazy map of all models in the Benson, Song, and Winawer (2024)
    dataset.

    The return value of this function is a lazily evaluated equivalent of the
    following dictionary:

    ```
    {(inputs, outputs): benson2024_unet(inputs, outputs, 'all')
     for inputs in hcp_input_properties.keys()
     for outputs in ['area', 'ring']}
    ```
    """
    import pimms, neuropythy as ny
    mcp = _to_model_cache_pseudo_path(model_cache_path)
    d = {
        (inputs, outputs): ny.util.curry(
            benson2024_unet,
            inputs, outputs, 'all',
            model_cache_path=mcp)
        for inputs in hcp_input_properties.keys()
        for outputs in ['area', 'ring']}
    return pimms.lazy_map(d)
    
