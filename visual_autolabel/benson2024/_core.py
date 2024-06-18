# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2024/__init__.py


#===============================================================================
# Initialization

#-------------------------------------------------------------------------------
# Dependencies

import os, json

import numpy as np
import neuropythy as ny

from ..image import (
    FlatmapFeature,
    NullFeature,
    train_until,
    load_training,
    autolog)
from ..plot import (
    add_inferred,
    add_prior,
    add_raterlabels)


#-------------------------------------------------------------------------------
# Initialization


# Training Feature Sets.........................................................
# The base feature-sets we are predicting:
vaonly_properties = ('V1', 'V2', 'V3')
econly_properties = ('E0', 'E1', 'E2', 'E3', 'E4')
# The base feature-sets we use to predict the above labels:
t1only_properties = ('x', 'y', 'z',
                     'curvature', 'convexity',
                     'thickness', 'surface_area')
fnonly_properties = ('prf_x', 'prf_y', 'prf_sigma', 'prf_cod')


#-------------------------------------------------------------------------------
# Score Functions

def score_dataframe(hem, suffix, rowinit=None, smooth=0, pair_tags=None):
    """Score the HCP hemisphere or subject and return a DataFrame summary.
    
    The Dice-Sørensen coefficient is used to score similarity between visual
    areas predicted by one method and those predicted by another.
    """
    import pandas
    from visual_autolabel import dice_scores
    if pair_tags is None:
        pair_tags = {}
    elif isinstance(pair_tags, str):
        pair_tags = pair_tags.lower()
        if pair_tags == 'hcp':
            pair_tags = score_dataframe.hcp_default_pair_tags
        if pair_tags == 'nyu':
            pair_tags = score_dataframe.nyu_default_pair_tags
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
            if len(scores) != len(labels):
                from warnings import warn
                warn(
                    f"skipping entry {rowinit} with wrong number of labels/scores:\n"
                    f" label {k1} contains {set(np.unique(labels))}\n"
                    f" label {k2} contains {set(np.unique(labels))}")
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
        return pandas.DataFrame(rows)
score_dataframe.hcp_default_pair_tags = {
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
score_dataframe.nyu_default_pair_tags = {
    ('anat', 'gold'): 'anat',
    ('func', 'gold'): 'func',
    ('fnyu', 'gold'): 'fnyu',
    ('gold', 'anat'): 'anat',
    ('gold', 'func'): 'func',
    ('gold', 'fnyu'): 'fnyu'}


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
    
