# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2025/analysis/_core.py


#===============================================================================
# Initialization

#-------------------------------------------------------------------------------
# Dependencies

import os, json
from collections.abc import (Sequence, Set)
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pimms
import neuropythy as ny

from ...util import dice_scores
from ...image import UNet


#-------------------------------------------------------------------------------
# Utility Functions

def analdir(analysis_path, *args, mkdirs=False, mkdir_mode=0o775):
    if analysis_path is None:
        return None
    elif analysis_path is Ellipsis:
        from ..config import analysis_path as ap
        analysis_path = ap
    analysis_path = Path(analysis_path)
    if len(args) == 0:
        return analysis_path
    for arg in args:
        analysis_path = analysis_path / arg
    if mkdirs and not analysis_path.is_dir():
        analysis_path.mkdir(mode=mkdir_mode, parents=True, exist_ok=True)
    return analysis_path
def analfile(analysis_path, *args, mkdirs=False, mkdir_mode=0o775):
    if analysis_path is None:
        return None
    elif analysis_path is Ellipsis:
        from ..config import analysis_path as ap
        analysis_path = ap
    analysis_path = Path(analysis_path)
    if len(args) == 0:
        return analysis_path
    for arg in args:
        analysis_path = analysis_path / arg
    parent = analysis_path.parent
    if mkdirs and not parent.is_dir():
        parent.mkdir(mode=mkdir_mode, parents=True, exist_ok=True)
    return analysis_path


#-------------------------------------------------------------------------------
# Score Functions

def calc_scores(hem, suffix, rowinit=None, smooth=0, pair_tags=None):
    """Score the given hemisphere or subject and return a DataFrame summary.
    
    The Dice-Sørensen coefficient is used to score similarity between visual
    areas predicted by one method and those predicted by another. Methods
    are found by looking at the properties of the (subject or hemisphere) 
    argument; all properties that end with the given suffix are considered to be
    prediced by separate methods, so `'A1_visual_area'` and `'func_visual_area'`
    are considered the methods `'A1'` and `'func'` for `suffix='_visual_area'`.

    Parameters
    ----------
    obj : subject or hemisphere or tuple of hemispheres
        The subject, hemisphere, or hemispheres that are to be scored. If a
        tuple of hemispheres are given, they must be given as `(lh, rh)`.
    suffix : str
        The suffix of properties that are to be scored; typically this is either
        `'_visual_area'` or `'_visual_ring'`.
    rowinit : dict or None, optional
        Data that should be used to initialize each row of the returned
        dataframe object. This argument effectively adds columns whose rows all
        have the same value to the returned dataframe object.
    smooth : float, optional
        The smoothing to apply to the Dice coefficient calculation. See the
        `dice_loss` function for more information. The default is 0.
    pair_tags : 'hcp', 'nyu', or None, optional
        Instructions on how to fill in the `'tag'` column of the returned
        dataframe. If `None` is given, then no tag column is generated. If 
        either `'hcp'` or `'nyu'` is given, then tags appropriate to those
        datasets are generated. Tags typically are applied to comparisons
        against the gold-standard data, signaling which comparisons are likely
        to be analyzed or plotted together. The default is `None`.
    """
    if pair_tags is None:
        pair_tags = {}
    elif isinstance(pair_tags, str):
        pair_tags = pair_tags.lower()
        if pair_tags == 'hcp':
            pair_tags = calc_scores.hcp_default_pair_tags
        if pair_tags == 'nyu':
            pair_tags = calc_scores.nyu_default_pair_tags
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
        return pd.concat(
            [calc_scores(
                hem, suffix,
                rowinit=dict(rowinit, hemisphere=h),
                smooth=smooth,
                pair_tags=pair_tags)
             for (h,hem) in zip(('lh','rh'), hems)],
            ignore_index=True)
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
            if len(scores) != len(labels):
                from warnings import warn
                warn(
                    f"skipping entry {rowinit} with wrong number of"
                    f" labels/scores:\n"
                    f" label {k1} contains {set(np.unique(lbl1))}\n"
                    f" label {k2} contains {set(np.unique(lbl2))}")
            else:
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
    return pd.DataFrame(rows)
calc_scores.hcp_default_pair_tags = {
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
calc_scores.nyu_default_pair_tags = {
    ('anat',  'gold'):  'anat',
    ('func',  'gold'):  'func',
    ('fnyu',  'gold'):  'fnyu',
    ('prior', 'gold'):  'prior',
    ('gold',  'anat'):  'anat',
    ('gold',  'func'):  'func',
    ('gold',  'fnyu'):  'fnyu',
    ('gold',  'prior'): 'prior'}
def scores(dataset, sid,
           overwrite=False,
           analysis_path=Ellipsis,
           dataset_cache_path=Ellipsis,
           model_cache_path=Ellipsis,
           mkdirs=True,
           mkdir_mode=0o775,
           fork=True):
    """Returns the scores dataframe for a single subject from a dataset.
    
    The Dice-Sørensen coefficient is used to score similarity between visual
    areas predicted by one method and those predicted by another. Methods
    are found by looking at the properties of the (subject or hemisphere) 
    argument; all properties that end with the given suffix are considered to be
    prediced by separate methods, so `'A1_visual_area'` and `'func_visual_area'`
    are considered the methods `'A1'` and `'func'` for `suffix='_visual_area'`.

    Parameters
    ----------
    dataset : 'hcp', 'nyu', or 'all', optional
        The dataset whose data is to be returned. The default is `'all'`.
    sid : str or int, optional
        The id of the subject whose scores are to be returned.
    overwrite : boolean, optional
        Whether to recalculate and overwrite existing data. If `False` (the
        default), then data are loaded from the existing data files instead of
        being calculated.
    analysis_path : path-like or Ellipsis or None, optional
        The path from which analysis data should be loaded or to which it should
        be saved. If `Ellipsis` is given (the default), then the value in the
        `benson2025.config.analysis_path` module is used.
    dataset_cache_path : path-like or Ellipsis or None, optional
        The path from which datasets should be cached. If `Ellipsis` is given
        (the default), then the subdirectory named `dataset` of the value
        `benson2025.config.dataset_cache_path` is used.
    model_cache_path : path-like or Ellipsis or None, optional
        The path from which datasets should be cached. If `Ellipsis` is given
        (the default), then the subdirectory named `dataset` of the value
        `benson2025.config.dataset_cache_path` is used.
    mkdirs : boolean, optional
        Whether to make directories that do not exist for saving data files. The
        default is `True`.
    mkdir_mode : int, optional
        If `mkdirs` is `True` and any directories are made, then they are made
        with this mode. The default is `0o775`.
    fork : boolean, optional
        When dataframes must be calculated instead of just loaded, this option
        determines whether they run in separate forked processes (`True`) or in
        the calling process (`False`). The main effect of this is that setting
        `fork` to `True` can slow the calculation very slightly, but it prevents
        cached data from the calculations from accumulating in the calling
        thread. The default is `True`.
    """
    dataset = dataset.upper()
    if dataset == 'HCP':
        from .. import hcp
        module = hcp
    elif dataset == 'NYU':
        from .. import nyu
        module = nyu
    else:
        raise ValueError("dataset must be 'hcp' or 'nyu'")
    # If the subject has already been cached, we just return it.
    sid_filename = analfile(
        analysis_path, 'dice', dataset, f"{sid}.csv",
        mkdirs=mkdirs,
        mkdir_mode=mkdir_mode)
    if not overwrite and sid_filename.is_file():
        return pd.read_csv(sid_filename, keep_default_na=False)
    if fork:
        # This is a simple workaround for memory leaks: at the cost of a bit of
        # time to fork the process, we avoid loading and caching subject data in
        # this (original) process.
        from ...util import forkrun
        return forkrun(
            scores,
            dataset, sid,
            overwrite=overwrite,
            analysis_path=analysis_path,
            dataset_cache_path=dataset_cache_path,
            model_cache_path=model_cache_path,
            mkdirs=mkdirs,
            mkdir_mode=mkdir_mode,
            fork=False)
    # Now make the flatmaps and create the dataframes.
    datasets = module.all_datasets(
        cache_path=dataset_cache_path)
    fmaps = module.flatmaps(
        sid, datasets,
        dataset_cache_path=dataset_cache_path,
        model_cache_path=model_cache_path)
    rowinit = dict(dataset=dataset, sid=sid)
    dfs = []
    # We calculate visual areas for both datasets.
    df = calc_scores(
        fmaps, 'area',
        rowinit=rowinit,
        pair_tags=dataset.lower())
    dfs.append(df)
    # We calculate the visual ring for just the HCP.
    if dataset == 'HCP':
        df = calc_scores(
            fmaps, 'ring',
            rowinit=rowinit,
            pair_tags=dataset.lower())
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # If we have a cache path, save it.
    if sid_filename and (overwrite or not sid_filename.is_file()):
        df.to_csv(sid_filename, index=False)
    return df
def all_scores(dataset='all',
               overwrite=False,
               analysis_path=Ellipsis,
               dataset_cache_path=Ellipsis,
               model_cache_path=Ellipsis,
               mkdirs=True,
               mkdir_mode=0o775,
               fork=True):
    """Returns a dataframe of all model comparisons across all subjects.
    
    This function loads (or calculates) then returns the requested dataframe of
    dice scores for all models compared across all subjects from the dataset of
    Benson et al. (2024).

    Parameters
    ----------
    dataset : 'hcp', 'nyu', or 'all', optional
        The dataset whose data is to be returned. The default is `'all'`.
    overwrite : boolean, optional
        Whether to recalculate and overwrite existing data. If `False` (the
        default), then data are loaded from the existing data files instead of
        being calculated.
    analysis_path : path-like or Ellipsis or None, optional
        The path from which analysis data should be loaded or to which it should
        be saved. If `Ellipsis` is given (the default), then the value in the
        `benson2025.config.analysis_path` module is used.
    dataset_cache_path : path-like or Ellipsis or None, optional
        The path from which datasets should be cached. If `Ellipsis` is given
        (the default), then the subdirectory named `dataset` of the value
        `benson2025.config.dataset_cache_path` is used.
    model_cache_path : path-like or Ellipsis or None, optional
        The path from which datasets should be cached. If `Ellipsis` is given
        (the default), then the subdirectory named `dataset` of the value
        `benson2025.config.dataset_cache_path` is used.
    mkdirs : boolean, optional
        Whether to make directories that do not exist for saving data files. The
        default is `True`.
    mkdir_mode : int, optional
        If `mkdirs` is `True` and any directories are made, then they are made
        with this mode. The default is `0o775`.
    fork : boolean, optional
        When dataframes must be calculated instead of just loaded, this option
        determines whether they run in separate forked processes (`True`) or in
        the calling process (`False`). The main effect of this is that setting
        `forkrun` to `True` can slow the calculation very slightly, but it
        prevents cached data from the calculations from accumulating in the
        calling thread. The default is `True`.
    """
    opts = dict(
        overwrite=overwrite,
        analysis_path=analysis_path,
        dataset_cache_path=dataset_cache_path,
        model_cache_path=model_cache_path,
        mkdirs=mkdirs,
        mkdir_mode=mkdir_mode,
        fork=fork)
    # First thing is to figure out if we can just load the datafile.
    dataset = dataset.lower()
    if dataset == 'all':
        return pd.concat(
            [all_scores(ds, **opts) for ds in ('hcp','nyu')],
            ignore_index=True)
    elif dataset == 'hcp':
        from ..config import hcp_sids
        sids = hcp_sids
    elif dataset == 'nyu':
        from ..config import nyu_sids
        sids = nyu_sids
    else:
        raise ValueError("dataset must be 'hcp', 'nyu', or 'all'")
    # If the dataset has already been cached, we just return it.
    filename = analfile(
        analysis_path, f"{dataset}dice.csv",
        mkdirs=mkdirs,
        mkdir_mode=mkdir_mode)
    if not overwrite and filename and filename.is_file():
        return pd.read_csv(filename, keep_default_na=False)
    # Otherwise, we're going to calculate everything...
    df = pd.concat(
        [scores(dataset, sid, **opts) for sid in sids],
        ignore_index=True)
    if filename and (overwrite or not filename.is_file()):
        df.to_csv(filename, index=False)
    return df
    

#-------------------------------------------------------------------------------
# Loading Model Data

def _to_model_cache_pseudo_path(model_cache_path):
    from .._core import osf_repository
    if ny.util.is_pseudo_path(model_cache_path):
        if model_cache_path.source_path.startswith(f"{osf_repository}/models"):
            return model_cache_path
        raise ValueError("model_cache_path must be a local directory name")
    # Parse the model_cache_path.
    if model_cache_path is Ellipsis:
        from visual_autolabel.benson2025.config import model_cache_path
    elif isinstance(model_cache_path, Path):
        model_cache_path = os.fspath(model_cache_path)
    # Okay, make it a pseudo path for the OSF repo's model's directory!
    return ny.util.pseudo_path(
        f"{osf_repository}/models",
        cache_path=model_cache_path)
def unet(inputs, outputs, part='model', model_cache_path=None):
    """Loads a UNet from the Benson, Song, et al. (2025) dataset.
    
    Parameters
    ----------
    inputs : str
        The inputs of the model. This must be one of the following:
        * `'anat'`. Anatomical properties that can be derived from a T1-weighted
          image alone; these include the raw midgray coordinates of the
          individual vertices for the hemisphere (`'x'`, `'y'`, `'z'`), values
          derived from them (`'curvature'`, `'convexity'`), the gray-matter
          thickness (`'thickness'`) and the `'surface_area'` property, which
          indicates the unwarped vertex size. Because it is generally assumed
          that a T1-weighted image is always collected in an fMRI or sMRI
          experiment, these features are included in all other feature lists.
        * `'t1t2'`. Anatomical data that can be derived from a T1-weighted and a
          T2-weighted image. These include all the `anat` properties and the
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
        * `'fnyu'`. Identical to `'func'` but represents the model that was
          retrained on the NYU dataset.
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
        indicates that the `visual_autolabel.benson2025.config.model_cache_path`
        should be used.

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
    # First parse the part argument.
    if not isinstance(part, str):
        if part is Ellipsis:
            part = set(parts)
        if isinstance(part, Sequence):
            return tuple(
                unet(
                    inputs, outputs, part=p,
                    model_cache_path=model_cache_path)
                for p in part)
        elif isinstance(part, Set):
            return {
                k: unet(
                    inputs, outputs, part=k,
                    model_cache_path=model_cache_path)
                for k in part}
        else:
            raise TypeError(f'unrecognized type for part option: {type(part)}')
    # Next, parse the model_cache_path.
    model_cache_path = _to_model_cache_pseudo_path(model_cache_path)
    # Now we can descend into the models' directory and load the part.
    pp = model_cache_path.subpath(f'{inputs}_{outputs}')
    part = part.lower()
    if part == 'all':
        return unet(
            inputs, outputs, part=Ellipsis,
            model_cache_path=model_cache_path)
    elif part == 'model':
        # We need to load the options to get the base model.
        opts = unet(
            inputs, outputs, 'options',
            model_cache_path=model_cache_path)
        base_model = opts['base_model']
        state = torch.load(pp.local_path('model.pt'), weights_only=True)
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
        return pd.read_csv(
            pp.local_path('training.tsv'),
            sep='\t',
            keep_default_na=False)
    else:
        raise ValueError(f"unrecognized model part: '{part}'")
def all_unets(model_cache_path=None):
    """Returns a lazy map of all models in the Benson, Song, and Winawer (2024)
    dataset.

    The return value of this function is a lazily evaluated equivalent of the
    following dictionary:

    ```
    {(inputs, outputs): unet(inputs, outputs, 'all')
     for inputs in visual_autolabel.benson2025.hcp.input_properties.keys()
     for outputs in ['area', 'ring']}
    ```
    """
    from ..hcp import input_properties
    d = {
        (inputs, outputs): ny.util.curry(
            unet,
            inputs, outputs, 'all',
            model_cache_path=model_cache_path)
        for inputs in input_properties.keys()
        for outputs in ['area', 'ring']}
    return pimms.lazy_map(d)
