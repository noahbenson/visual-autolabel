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

# Feature descriptions.
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


#-------------------------------------------------------------------------------
# Loading or Generating Model Partitions for Training

def hcp_partition(cluster_relatives=True, hcp_restricted_path=None):
    """Generates and returns a subject partition for the HCP dataset.

    If no arguments are provided, then this will reproduce the same partition
    as was used for final model training in the paper Benson et al. (2024). In
    order to generate this partition, the neuropythy library must be configured
    to have access to the HCP restricted (behavioral) dataset, which is required
    to identify subjects that are twins or siblings and ensure that they are in
    the same part of the partition.

    This function isn't meant to produce arbitrary partitions; it is meant to
    reproduce the partition used in the paper. In fact, it always produces
    the same partition, and it raises an error if a partition in which relatives
    are clustered is requested but the HCP restricted data is not found.

    See also `visual_autolabel.partition`.

    Parameters
    ----------
    cluster_relatives : boolean, optional
        Whether to ensure that relatives are placed in the same part of the
        partition. In order for this argument to work, the neuropythy library
        must be configured to have access to the HCP restricted (behavioral)
        data (see https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage
        for more information). Alternatively, this data file may be provided
        via the `hcp_restricted_path` option. If `cluster_relatives` is `True`
        then the returned partition is the same partition used in training the
        final HCP dataset models. Otherwise, the returned partition is the same
        partition used in the hyperparameter grid-search.
    hcp_restricted_path : path-like or None, optional
        Where the function should find the HCP restricted datafile. If `None`
        (the default), then neuropythy must be configured to have access to the
        restricted dat ain order for the `cluster_relatives` option to work.
        Otherwise, this must be a filename or path-like object referencing the
        restricted data file.
    """
    # Grab our subject list:
    sids = ny.data['hcp_lines'].subjects.keys()
    # These subjects were excluded due to data issues, so we remove them here.
    sids = np.setdiff1d(list(sids), hcp_partition.excluded_sids)
    # Now, if we aren't clustering relatives, we can just make a choice (we
    # don't need to hide things behind randomness).
    if not cluster_relatives:
        trn = sids[hcp_partition.nocluster_partition_ii]
    else:
        # We start by separating subjects into relatives and nonrelatives; we
        # need to get these lists from neuropythy if they aren't provided.
        if hcp_restricted_path is None:
            rsp = ny.data['hcp_retinotopy'].retinotopy_sibling_pairs
        else:
            # We load the data and use it.
            from neuropythy.datasets import HCPMetaDataset
            ds = HCPMetaDataset(genetic_path=hcp_restricted_path)
            rsp = ds.retinotopy_sibling_pairs
        if rsp is None:
            raise ValueError("no valid HCP genetic data found")
        # We put the MZs, DZs and SBs together:
        mzs = rsp['monozygotic_twins']
        dzs = rsp['dizygotic_twins']
        urs = rsp['unrelated_pairs']
        sbs = rsp['nontwin_siblings']
        relatives = np.vstack([mzs, dzs, sbs])
        ii = np.isin(relatives[:,0], sids) & np.isin(relatives[:,1], sids)
        relatives = relatives[ii, :] 
        others = np.setdiff1d(sids, np.unique(relatives))
        # Now select from these lists.
        (rii, oii) = hcp_partition.cluster_trn_ii
        trn = np.unique(
            np.concatenate(
                [relatives[rii,:].flatten(), others[oii]]))
    val = np.setdiff1d(sids, trn)
    return (trn, val)
hcp_partition.excluded_sids = (
    125525, 131722, 134627, 144226, 148133, 169343, 177645, 178647,
    182436, 192439, 214019, 251833, 644246, 706040, 833249, 861456,
    943862, 951457)
hcp_partition.nocluster_trn_ii = np.array(
    [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  13,  14,
     15,  16,  17,  19,  23,  24,  26,  27,  28,  29,  31,  32,  33,
     34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
     48,  49,  50,  51,  52,  53,  55,  56,  57,  58,  59,  60,  62,
     63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74,  75,  76,
     77,  80,  81,  82,  83,  84,  85,  87,  88,  90,  91,  93,  94,
     95,  96,  98,  100, 102, 103, 104, 105, 106, 107, 108, 109, 110,
     114, 115, 116, 117, 118, 119, 121, 122, 124, 126, 127, 128, 129,
     130, 131, 132, 133, 134, 136, 137, 138, 139, 141, 143, 144, 145,
     146, 147, 148, 150, 151, 152, 153, 154, 157, 158, 159, 161, 162])
hcp_partition.cluster_trn_ii = (
    np.array(
        [0,  1,  2,  4,  5,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
         20, 22, 23, 25, 26, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43,
         44, 45, 46, 47, 48, 49, 51, 53, 55, 57, 58, 59, 60, 61, 62, 63, 64,
         65, 66, 67, 68, 70, 71]),
    np.array(
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 14, 15, 16, 17, 18]))
def nyu_partition():
    """Returns the partition used in training the NYU dataset."""
    # Since there is nothing to obscure in this function, we just return the
    # literal subject IDs for the dataset.
    trn = (
        "sub-wlsubj001", "sub-wlsubj004", "sub-wlsubj006", "sub-wlsubj014",
        "sub-wlsubj023", "sub-wlsubj042", "sub-wlsubj043", "sub-wlsubj045",
        "sub-wlsubj055", "sub-wlsubj056", "sub-wlsubj057", "sub-wlsubj062",
        "sub-wlsubj064", "sub-wlsubj067", "sub-wlsubj071", "sub-wlsubj076",
        "sub-wlsubj079", "sub-wlsubj081", "sub-wlsubj083", "sub-wlsubj084",
        "sub-wlsubj085", "sub-wlsubj086", "sub-wlsubj087", "sub-wlsubj088",
        "sub-wlsubj090", "sub-wlsubj092", "sub-wlsubj095", "sub-wlsubj105",
        "sub-wlsubj114", "sub-wlsubj115", "sub-wlsubj117", "sub-wlsubj118",
        "sub-wlsubj122", "sub-wlsubj126")
    val = (
        "sub-wlsubj116", "sub-wlsubj007", "sub-wlsubj104", "sub-wlsubj091",
        "sub-wlsubj094", "sub-wlsubj120", "sub-wlsubj046", "sub-wlsubj109",
        "sub-wlsubj019")
    return (trn, val)

#-------------------------------------------------------------------------------
# Loading HCP and NYU Datasets

def hcp_dataset(inputs, outputs, sids=None, cache_path=None, features=None):
    """Returns one of the HCP datasets used by Benson et al. (2024).
    
    The dataset returned is specified by the first two parameters, `inputs` and
    `outputs`, which should be names of input and output variables. For example,
    `hcp_dataset('anat', 'area')` will return a dataset that accepts as input
    anatomical data and that predicts visual area boundaries.

    The datasets returned by this function contain all subject IDs used in the
    paper; it does **not** separate the subjects out into training and
    validation subsets. To create datasets for only a subset of the subjects,
    you can pass the `sids` option (either as a named argument or as the third
    positional argument). To obtain the training/validation partition used in
    the paper, see the `hcp_partition` function.
    """
    from ..image import make_datasets
    if sids is None:
        from visual_autolabel import sids as _sids
        sids = _sids
    if cache_path is None:
        from visual_autolabel.config import data_cache_path
        cache_path = data_cache_path
    if features is None:
        features = hcp_features
    if isinstance(inputs, str):
        inputs = hcp_input_properties.get(inputs, (inputs,))
    if isinstance(outputs, str):
        outputs = hcp_output_properties.get(outputs, (outputs,))
    dsets = make_datasets(
        inputs,
        outputs,
        features=features,
        partition=(sids, []),
        cache_path=cache_path)
    return dsets['trn']
def nyu_dataset(inputs, outputs='area',
                sids=None, cache_path=None, features=None):
    """Returns one of the NYU datasets used by Benson et al. (2024).
    
    The dataset returned is specified by the first two parameters, `inputs`
    which should be the names of input variables (`'anat'` or `'func'`). For
    example, `nyu_dataset('anat')` will return a dataset that accepts as
    input anatomical data.

    The datasets returned by this function contain all subject IDs used in the
    paper; it does **not** separate the subjects out into training and
    validation subsets. To create datasets for only a subset of the subjects,
    you can pass the `sids` option (either as a named argument or as the third
    positional argument). To obtain the training/validation partition used in
    the paper, see the `nyu_partition` function.

    """
    from ..image import nyu_make_datasets
    if sids is None:
        from visual_autolabel import nyusids
        sids = nyusids
    if cache_path is None:
        from visual_autolabel.config import nyudata_cache_path
        cache_path = nyudata_cache_path
    if features is None:
        features = nyu_features
    if isinstance(inputs, str):
        inputs = nyu_input_properties.get(inputs, (inputs,))
    if isinstance(outputs, str):
        outputs = nyu_output_properties.get(outputs, (outputs,))
    dsets = nyu_make_datasets(
        inputs,
        outputs,
        features=features,
        partition=(sids, []),
        cache_path=cache_path)
    return dsets['trn']
def hcp_all_datasets(sids=None, cache_path=None, features=None,
                     include_null=False, include_sect=False):
    """Returns a dictionary of all HCP datasets used by Benson et al. (2024).
    
    The dictionary returned uses tuples of `(inputs, outputs)` as keys, for
    example the key `('anat', 'area')` is used for a dataset built for a CNN
    that accepts anatomical data as input and predicts visual area boundaries.

    The datasets returned by this function contain all subject IDs used in the
    paper; it does **not** separate the subjects out into training and
    validation subsets. To create datasets for only a subset of the subjects,
    you can pass the `sids` option (either as a named argument or as the first
    positional argument). To obtain the training/validation partition used in
    the paper, see the `hcp_partition` function.
    """
    return {
        (inp, outp): hcp_dataset(
            inp, outp,
            sids=sids,
            features=features,
            cache_path=cache_path)
        for (inp,inputs) in hcp_input_properties.items()
        for (outp,outputs) in hcp_output_properties.items()
        # We skip null and sect because they were never used.
        if (include_null or inp != 'null')
        if (include_sect or outp != 'sect')}
def nyu_all_datasets(sids=None, cache_path=None, features=None,
                     include_null=False, include_sect=False):
    """Returns a dictionary of all NYU datasets used by Benson et al. (2024).
    
    The dictionary returned uses tuples of `(inputs, outputs)` as keys, for
    example the key `('anat', 'area')` is used for a dataset built for a CNN
    that accepts anatomical data as input and predicts visual area boundaries.

    The datasets returned by this function contain all subject IDs used in the
    paper; it does **not** separate the subjects out into training and
    validation subsets. To create datasets for only a subset of the subjects,
    you can pass the `sids` option (either as a named argument or as the first
    positional argument). To obtain the training/validation partition used in
    the paper, see the `nyu_partition` function.
    """
    return {
        (inp, outp): nyu_dataset(
            inp, outp,
            sids=sids,
            features=features,
            cache_path=cache_path)
        for (inp,inputs) in nyu_input_properties.items()
        for (outp,outputs) in nyu_output_properties.items()
        # We skip null and sect because they were never used.
        if (include_null or inp != 'null')
        if (include_sect or outp != 'sect')}
def hcp_flatmaps(sid, datasets,
                 add_inferred=True, add_prior=True, add_raters=True,
                 data_cache_path=None,
                 model_cache_path=None):
    """Returns a nested lazy-map of all the requested evaluation flatmaps.
    
    This function returns flatmaps that are ready to be used for evaluation of
    the CNN results in this project. The return value is a map whose keys are
    subject IDs and whose values are themselves lazy-maps. The nested lazy-maps
    each have the keys `'lh'` and `'rh'` and values that are the associated 
    flatmaps.
    
    The flatmaps have properties representing the gold-standard and predicted
    boundaries/labels. These properties are added for all datasets given in the
    `datasets` option. If this option is `None` (the default), then all datasets
    from the paper are included.
    """
    from collections.abc import Mapping
    if data_cache_path is None:
        from visual_autolabel.config import data_cache_path as _dcp
        data_cache_path = _dcp
    if model_cache_path is None:
        from visual_autolabel.config import model_cache_path as _mcp
        model_cache_path = _mcp
    # Get the subject:
    sub = ny.data['hcp_lines'].subjects[sid]
    # Add extra properties as requested:
    if add_inferred:
        from visual_autolabel.plot import add_inferred
        sub = add_inferred(sub)
    if add_prior:
        from visual_autolabel.plot import add_prior
        sub = add_prior(sub)
    if add_raters:
        from visual_autolabel.plot import add_raterlabels
        sub = add_raterlabels(sub)
    # Find a target that is appropriate for this subject:
    ds0 = next(iter(datasets.values()))
    targ = next(
        target 
        for target in ds0.targets
        if target['subject'] == sid)
    # We now make an LH and an RH dataset.
    fmaps = []
    for h in ['lh','rh']:
        hem = sub.hemis[h]
        view = {'hemisphere':h}
        fmap = ds0.image_cache.get_flatmap(targ, view=view)
        ps = {
            p: hem.prop(p)[fmap.labels]
            for p in hem.properties.keys()
            if p not in fmap.properties}
        for ((inp,outp),ds) in datasets.items():
            if outp == 'area':
                labelsets = {'visual_area': slice(0,3)}
            elif outp == 'ring':
                labelsets = {'visual_ring': slice(0,5)}
            elif outp == 'sect':
                labelsets = {'visual_area': slice(0,3), 'visual_ring': slice(3,8)}
            else:
                raise ValueError(f"invalid output: {output}")
            mdl = benson2024_unet(
                inp, outp, 'model',
                model_cache_path=model_cache_path)
            labels = ds.predlabels(targ, mdl, view=view, labelsets=labelsets)
            for (k,lbl) in labels.items():
                ps[f"{inp}_{k}"] = lbl
        fmaps.append(fmap.with_prop(ps))
    return tuple(fmaps)
def hcp_all_flatmaps(datasets, sids=None,
                     add_inferred=True, add_prior=True, add_raters=True,
                     data_cache_path=None,
                     model_cache_path=None):
    """Generates a lazy-map of all evaluation flatmaps for all HCP subjects.
    
    See also `hcp_flatmaps`."""
    import pimms
    if sids is None:
        from visual_autolabel.config import sids
    return pimms.lmap(
        {sid: ny.util.curry(
             hcp_flatmaps, sid, datasets,
             add_inferred=add_inferred, 
             add_prior=add_prior, 
             add_raters=add_raters,
             data_cache_path=data_cache_path,
             model_cache_path=model_cache_path)
         for sid in sids})
def nyu_flatmaps(sid, datasets,
                 add_prior=True,
                 data_cache_path=None,
                 model_cache_path=None):
    """Returns a nested lazy-map of all the requested evaluation flatmaps.
    
    This function returns flatmaps that are ready to be used for evaluation of
    the CNN results in this project. The return value is a map whose keys are
    subject IDs and whose values are themselves lazy-maps. The nested lazy-maps
    each have the keys `'lh'` and `'rh'` and values that are the associated 
    flatmaps.
    
    The flatmaps have properties representing the gold-standard and predicted
    boundaries/labels. These properties are added for all datasets given in the
    `datasets` option. If this option is `None` (the default), then all datasets
    from the paper are included.
    """
    from collections.abc import Mapping
    # Find a target that is appropriate for this subject:
    ds0 = next(iter(datasets.values()))
    # Get the subject.
    sub = ds0.image_cache.subjects[sid]
    if data_cache_path is None:
        from visual_autolabel.config import nyudata_cache_path as _dcp
        data_cache_path = _dcp
    if model_cache_path is None:
        from visual_autolabel.config import model_cache_path as _mcp
        model_cache_path = _mcp
    # Add extra properties as requested:
    if add_prior:
        from visual_autolabel.plot import add_prior
        sub = add_prior(sub)
    targ = next(
        target 
        for target in ds0.targets
        if target['subject'] == sid)
    # We now make an LH and an RH dataset.
    fmaps = []
    for h in ['lh','rh']:
        hem = sub.hemis[h]
        view = {'hemisphere':h}
        fmap = ds0.image_cache.get_flatmap(targ, view=view)
        ps = {
            p: hem.prop(p)[fmap.labels]
            for p in hem.properties.keys()
            if p not in fmap.properties}
        for ((inp,outp),ds) in datasets.items():
            labelsets = {'visual_area': slice(0,3)}
            mdl = benson2024_unet(
                inp, outp, 'model',
                model_cache_path=model_cache_path)
            labels = ds.predlabels(targ, mdl, view=view, labelsets=labelsets)
            for (k,lbl) in labels.items():
                ps[f"{inp}_{k}"] = lbl
        fmaps.append(fmap.with_prop(ps))
    return tuple(fmaps)
def nyu_all_flatmaps(datasets, sids=None,
                     add_prior=True,
                     data_cache_path=None,
                     model_cache_path=None):
    """Generates a lazy-map of all evaluation flatmaps for all HCP subjects.
    
    See also `hcp_flatmaps`."""
    import pimms
    if sids is None:
        from visual_autolabel.config import nyusids as sids
    return pimms.lmap(
        {sid: ny.util.curry(
             nyu_flatmaps, sid, datasets,
             add_prior=add_prior, 
             data_cache_path=data_cache_path,
             model_cache_path=model_cache_path)
         for sid in sids})


#-------------------------------------------------------------------------------
# Score Functions
def score_dataframe(hem, suffix, rowinit=None, smooth=0, pair_tags=Ellipsis):
    """Score the HCP hemisphere or subject and return a DataFrame summary.
    
    The Dice-Sørensen coefficient is used to score similarity between visual
    areas predicted by one method and those predicted by another.
    """
    import pandas
    from visual_autolabel import dice_scores
    if pair_tags is Ellipsis:
        pair_tags = score_dataframe.default_pair_tags
    if suffix.startswith('_'):
        suffix = suffix[1:]
    if suffix in ('visual_area', 'area'):
        key = 'area'
    elif suffix in ('visual_ring', 'ring'):
        key = 'ring'
    elif suffix == ('visual_sector', 'sect'):
        key = 'sector'
    suffix = '_visual_' + key
    if rowinit is None:
        rowinit = {}
    # If given (lh,rh) we do each and combine them.
    if isinstance(hem, (list, tuple)) and len(hem) == 2:
        hems = hem
        return pandas.concat(
            [score_dataframe(
                hem, suffix,
                rowinit=dict(rowinit, hemisphere=h),
                smooth=smooth,
                pair_tags=pair_tags)
             for (h,hem) in zip(('lh','rh'), hems)])
    rowinit = dict(rowinit, parcellation=key)
    props = [p for p in hem.properties.keys() if p.endswith(suffix)]
    rows = []
    for p1 in props:
        k1 = p1.split("_")[0]
        lbl1 = hem.prop(p1)
        for p2 in props:
            # We make the comparisons in only one way, and we decide that
            # way via alphabetical sort.
            if p1 >= p2:
                continue
            k2 = p2.split("_")[0]
            lbl2 = hem.prop(p2)
            labels = np.union1d(np.unique(lbl1), np.unique(lbl2))
            # The first value of labels will be zero, which is ignored
            # by the dice_scores function.
            labels = labels[labels > 0]
            # Calculate the dice scores.
            scores = dice_scores(lbl1, lbl2, smooth=smooth)
            assert len(scores) == len(labels), \
                "wrong number of labels/scores"
            # Figure out if this pair of properties has a tag.
            tag = pair_tags.get((k1,k2), '')
            rows.extend(
                [dict(
                     rowinit,
                     method1=k1, method2=k2, tag=tag,
                     label=lbl,
                     score=score)
                 for (lbl,score) in zip(labels, scores)])
            # Add a mean row also
            rows.append(
                dict(
                    rowinit,
                    method1=k1, method2=k2, tag=tag,
                    label='mean',
                    score=np.mean(scores)))
    return pandas.DataFrame(rows)
score_dataframe.default_pair_tags = {
    (kk1, kk2): tag
    for pairings in [
         # Gold-standard Inter-rater Reliability.
        {'A1': (('A2','A3','A4'), 'rely'),
         'A2': (('A3','A4'), 'rely'),
         'A3': (('A4',), 'rely'),
         # Model Prediction Accuracies.
         'anat': (('A1','A2','A3','A4'), 'anat'),
         'full': (('A1','A2','A3','A4'), 'full'),
         'func': (('A1','A2','A3','A4'), 'func'),
         'inf' : (('A1','A2','A3','A4'), 'inf'),
         'nodw': (('A1','A2','A3','A4'), 'nodw'),
         'nofn': (('A1','A2','A3','A4'), 'nofn'),
         'not2': (('A1','A2','A3','A4'), 'not2'),
         'prior':(('A1','A2','A3','A4'), 'prior'),
         't1t2': (('A1','A2','A3','A4'), 't1t2'),
         'trac': (('A1','A2','A3','A4'), 'trac')}]
    for (k1,(k2s,tag)) in pairings.items()
    for k2 in k2s
    for (kk1,kk2) in ((k1,k2), (k2,k1))}


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
    
