# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2024/nyu/_core.py


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

from .._core import (
    vaonly_properties,
    econly_properties,
    t1only_properties,
    fnonly_properties)
input_properties = {
    'null': ('zeros',),
    'anat': t1only_properties,
    'func': t1only_properties + fnonly_properties,
    # 'fnyu' is used to specify the functional CNN retrained on the NYU dataset.
    'fnyu': t1only_properties + fnonly_properties}
output_properties = {
    'area': vaonly_properties}
# All the properties and additional features (none for this dataset).
properties = dict(input_properties, **output_properties)
features = {}


#-------------------------------------------------------------------------------
# Loading or Generating Model Partitions for Training

def partition():
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
# Loading NYU Datasets

def dataset(inputs, outputs='area',
            partition=None, sids=Ellipsis, cache_path=Ellipsis):
    """Returns one of the NYU datasets used by Benson et al. (2024).
    
    The dataset returned is specified by the first two parameters, `inputs`
    which should be the names of input variables (`'anat'` or `'func'`). For
    example, `dataset('anat')` will return a dataset that accepts as
    input anatomical data.

    The datasets returned by this function contain all subject IDs used in the
    paper; it does **not** separate the subjects out into training and
    validation subsets. To create datasets for only a subset of the subjects,
    you can pass the `sids` option (either as a named argument or as the third
    positional argument). To obtain the training/validation partition used in
    the paper, see the `partition` function.

    """
    from ._datasets import make_datasets
    if sids is Ellipsis:
        from ..config import nyu_sids
        sids = nyu_sids
    if cache_path is Ellipsis:
        from ..config import dataset_cache_path
        if dataset_cache_path is not None:
            dataset_cache_path = os.path.join(dataset_cache_path, 'NYU')
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
    """Returns a dictionary of all NYU datasets used by Benson et al. (2024).
    
    The dictionary returned uses tuples of `(inputs, outputs)` as keys, for
    example the key `('anat', 'area')` is used for a dataset built for a CNN
    that accepts anatomical data as input and predicts visual area boundaries.

    The datasets returned by this function contain all subject IDs used in the
    paper; it does **not** separate the subjects out into training and
    validation subsets. To create datasets for only a subset of the subjects,
    you can pass the `sids` option (either as a named argument or as the first
    positional argument). To obtain the training/validation partition used in
    the paper, see the `nyu.partition` function.
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
             add_prior=True,
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
    # Get the subject.
    sub = ds0.image_cache.subjects[sid]
    # Parse some arguments.
    if dataset_cache_path is Ellipsis:
        from ..config import dataset_cache_path as dcp
        if dcp is not None:
            dcp = os.path.join(dcp, 'NYU')
        dataset_cache_path = dcp
    if model_cache_path is None:
        from ..config import model_cache_path as mcp
        model_cache_path = mcp
    # Add extra properties as requested:
    if add_prior:
        from ...plot import add_prior
        sub = add_prior(sub)
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
            labelsets = {'visual_area': slice(0,3)}
            mdl = unet(
                inp, outp, 'model',
                model_cache_path=model_cache_path)
            labels = ds.predlabels(targ, mdl, view=view, labelsets=labelsets)
            for (k,lbl) in labels.items():
                ps[f"{inp}_{k}"] = lbl
        # We mark the visual_area property of the NYU dataset as gold-standard.
        gold = hem.prop('visual_area')[fmap.labels]
        # (But we want hV4 masked out.)
        gold[gold == 4] = 0
        ps['gold_visual_area'] = gold
        fmaps.append(fmap.with_prop(ps))
    return tuple(fmaps)
def all_flatmaps(datasets, sids=Ellipsis,
                 add_prior=True,
                 dataset_cache_path=Ellipsis,
                 model_cache_path=Ellipsis):
    """Generates a lazy-map of all evaluation flatmaps for all HCP subjects.
    
    See also `flatmaps`."""
    import pimms
    if sids is Ellipsis:
        from ..config import nyu_sids as sids
    return pimms.lmap(
        {sid: ny.util.curry(
             flatmaps, sid, datasets,
             add_prior=add_prior, 
             dataset_cache_path=dataset_cache_path,
             model_cache_path=model_cache_path)
         for sid in sids})
