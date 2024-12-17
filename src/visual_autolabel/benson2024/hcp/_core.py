# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2024/hcp/_core.py


#===============================================================================
# Initialization

#-------------------------------------------------------------------------------
# Dependencies

import os, json

import numpy as np
import neuropythy as ny

from ...image import (
    FlatmapFeature,
    NullFeature)
from ...plot import (
    add_inferred,
    add_prior,
    add_raterlabels)


#-------------------------------------------------------------------------------
# Initialization

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
        `visual_autolabel.benson2024.config.dwi_filename_pattern`.
        """
        # We want to extract this from the benson2024 namespace, in case
        # someone changes it manually.
        from visual_autolabel.benson2024.config \
            import dwi_filename_pattern as patt
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
    'dwi_VOF': DWIFeature('VOF')}
features = dict(
    dwi_features,
    # Add in the 'zeros' feature, which represents all zeros for a null input.
    zeros=NullFeature('zeros'))

# Training Feature Sets.........................................................
# The base feature-sets we are predicting:
from .._core import (
    vaonly_properties,
    econly_properties,
    t1only_properties,
    fnonly_properties)
t2only_properties = ('myelin',)
dwonly_properties = ('dwi_OR', 'dwi_VOF')
full_properties = (t1only_properties + t2only_properties +
                   dwonly_properties + fnonly_properties)
# The feature-sets by name.
input_properties = {
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
output_properties = {
    'area': vaonly_properties,
    'ring': econly_properties,
    'sect': vaonly_properties + econly_properties,
}
# All the feature properties.
properties = dict(input_properties, **output_properties)


#-------------------------------------------------------------------------------
# Loading or Generating Model Partitions for Training

def partition(cluster_relatives=True, hcp_restricted_path=None):
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
    sids = np.setdiff1d(list(sids), partition.excluded_sids)
    # Now, if we aren't clustering relatives, we can just make a choice (we
    # don't need to hide things behind randomness).
    if not cluster_relatives:
        trn = sids[partition.nocluster_partition_ii]
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
        (rii, oii) = partition.cluster_trn_ii
        trn = np.unique(
            np.concatenate(
                [relatives[rii,:].flatten(), others[oii]]))
    val = np.setdiff1d(sids, trn)
    return (trn, val)
partition.excluded_sids = (
    125525, 131722, 134627, 144226, 148133, 169343, 177645, 178647,
    182436, 192439, 214019, 251833, 644246, 706040, 833249, 861456,
    943862, 951457)
partition.nocluster_trn_ii = np.array(
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
partition.cluster_trn_ii = (
    np.array(
        [0,  1,  2,  4,  5,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
         20, 22, 23, 25, 26, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43,
         44, 45, 46, 47, 48, 49, 51, 53, 55, 57, 58, 59, 60, 61, 62, 63, 64,
         65, 66, 67, 68, 70, 71]),
    np.array(
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 14, 15, 16, 17, 18]))

#-------------------------------------------------------------------------------
# Loading HCP Datasets

def dataset(inputs, outputs,
            partition=None, sids=Ellipsis, cache_path=Ellipsis):
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
    from ._datasets import make_datasets
    if sids is Ellipsis:
        from ..config import hcp_sids
        sids = hcp_sids
    if cache_path is Ellipsis:
        from ..config import dataset_cache_path
        if dataset_cache_path is not None:
            dataset_cache_path = os.path.join(dataset_cache_path, 'HCP')
        cache_path = dataset_cache_path
    if isinstance(inputs, str):
        inputs = input_properties.get(inputs, (inputs,))
    if isinstance(outputs, str):
        outputs = output_properties.get(outputs, (outputs,))
    if partition is None:
        part = (sids, ())
    elif partition is Ellipsis:
        makepart = globals()['partition']
        part = makepart()
    else:
        part = partition
    dsets = make_datasets(
        inputs,
        outputs,
        features=features,
        partition=(sids, []),
        cache_path=cache_path)
    return dsets['trn'] if partition is None else dsets
def all_datasets(sids=Ellipsis, cache_path=Ellipsis,
                 partition=None, include_null=False, include_sect=False):
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
        (inp, outp): dataset(
            inp, outp,
            sids=sids,
            partition=partition,
            cache_path=cache_path)
        for (inp,inputs) in input_properties.items()
        for (outp,outputs) in output_properties.items()
        # We typically skip null and sect because they were never used.
        if (include_null or inp != 'null')
        if (include_sect or outp != 'sect')}
def flatmaps(sid, datasets,
             add_inferred=True, add_prior=True, add_raters=True,
             dataset_cache_path=Ellipsis,
             model_cache_path=Ellipsis):
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
    # Get the subject:
    sub = ny.data['hcp_lines'].subjects[sid]
    # Parse some arguments.
    if dataset_cache_path is Ellipsis:
        from ..config import dataset_cache_path as dcp
        if dcp is not None:
            dcp = os.path.join(dcp, 'HCP')
        dataset_cache_path = dcp
    if model_cache_path is Ellipsis:
        from ..config import model_cache_path as _mcp
        model_cache_path = _mcp
    # Add extra properties as requested:
    if add_inferred:
        from ...plot import add_inferred
        sub = add_inferred(sub)
    if add_prior:
        from ...plot import add_prior
        sub = add_prior(sub)
    if add_raters:
        from ...plot import add_raterlabels
        sub = add_raterlabels(sub)
    targ = next(
        target 
        for target in ds0.targets
        if target['subject'] == sid)
    # We now make an LH and an RH dataset.
    from ..analysis import unet
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
                labelsets = {
                    'visual_area': slice(0,3),
                    'visual_ring': slice(3,8)}
            else:
                raise ValueError(f"invalid output: {output}")
            mdl = unet(
                inp, outp, 'model',
                model_cache_path=model_cache_path)
            labels = ds.predlabels(targ, mdl, view=view, labelsets=labelsets)
            for (k,lbl) in labels.items():
                ps[f"{inp}_{k}"] = lbl
        fmaps.append(fmap.with_prop(ps))
    return tuple(fmaps)
def all_flatmaps(datasets, sids=Ellipsis,
                 add_inferred=True, add_prior=True, add_raters=True,
                 dataset_cache_path=Ellipsis,
                 model_cache_path=Ellipsis):
    """Generates a lazy-map of all evaluation flatmaps for all HCP subjects.
    
    See also `hcp_flatmaps`."""
    import pimms
    if sids is Ellipsis:
        from ..config import hcp_sids as sids
    return pimms.lmap(
        {sid: ny.util.curry(
             flatmaps, sid, datasets,
             add_inferred=add_inferred, 
             add_prior=add_prior, 
             add_raters=add_raters,
             dataset_cache_path=dataset_cache_path,
             model_cache_path=model_cache_path)
         for sid in sids})
