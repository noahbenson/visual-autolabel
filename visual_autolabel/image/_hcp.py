# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/image/_hcp.py
# Code to manage the HCP-specific aspects of the AutoLabeler dataset.


#===============================================================================
# Dependencies

# External Libries -------------------------------------------------------------

import os, sys, warnings
from collections.abc import (Sequence, Mapping)

import numpy as np
import scipy as sp
import nibabel as nib
import neuropythy as ny
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pimms, torch
from torch.utils.data import (Dataset, DataLoader)

# Internal Tools ---------------------------------------------------------------

from ..config import (
    sids,
    default_partition,
    default_image_size,
)
from ..util import (
    partition_id,
    partition as make_partition,
    is_partition,
    trndata,
    valdata,
    convrelu
)
from ._data import (
    ImageCacheDataset,
    BilateralFlatmapImageCache,
    FlatmapFeature,
    LabelFeature,
    LabelDiffFeature,
    LabelUnionFeature,
    LabelIntersectFeature
)


#===============================================================================
# The HCP Feature Cache

class HCPLinesImageCache(BilateralFlatmapImageCache):
    """An ImageCache subclass that handles features of the HCP Occipital Pole.

    The `HCPLinesImageCache` type is a simple overload of the
    `ImageCache` type that includes instructions for plotting most occipital
    pole features. The `target_id` parameters for the class's methods are always
    a tuple of `(subject_ID, hemi)` where `hemi` is either `'lh'` or `'rh'` or
    one of the other HCP hemispheres such as `'lh_LR32k'`.
    """
    # The featuers we know how to make.
    _builtin_features = {
        # Functional Features first.
        'prf_polar_angle':  FlatmapFeature('prf_polar_angle', 'nearest'),
        'prf_eccentricity': FlatmapFeature('prf_eccentricity', 'linear'),
        'prf_cod':          FlatmapFeature('prf_variance_explained', 'linear'),
        'prf_sigma':        FlatmapFeature('prf_radius', 'linear'),
        'prf_x':            FlatmapFeature('prf_x', 'linear'),
        'prf_y':            FlatmapFeature('prf_y', 'linear'),
        # The visual area and visual sector-based features.
        'V1':    LabelFeature('visual_area:1', 'nearest'),
        'V2':    LabelFeature('visual_area:2', 'nearest'),
        'V3':    LabelFeature('visual_area:3', 'nearest'),
        'SV1d0': LabelFeature('visual_sector:1', 'nearest'),
        'SV1d1': LabelFeature('visual_sector:2', 'nearest'),
        'SV1d2': LabelFeature('visual_sector:3', 'nearest'),
        'SV1d3': LabelFeature('visual_sector:4', 'nearest'),
        'SV1d4': LabelFeature('visual_sector:5', 'nearest'),
        'SV1v0': LabelFeature('visual_sector:6', 'nearest'),
        'SV1v1': LabelFeature('visual_sector:7', 'nearest'),
        'SV1v2': LabelFeature('visual_sector:8', 'nearest'),
        'SV1v3': LabelFeature('visual_sector:9', 'nearest'),
        'SV1v4': LabelFeature('visual_sector:10', 'nearest'),
        'SV2d1': LabelFeature('visual_sector:11', 'nearest'),
        'SV2d2': LabelFeature('visual_sector:12', 'nearest'),
        'SV2d3': LabelFeature('visual_sector:13', 'nearest'),
        'SV2d4': LabelFeature('visual_sector:14', 'nearest'),
        'SV2v1': LabelFeature('visual_sector:15', 'nearest'),
        'SV2v2': LabelFeature('visual_sector:16', 'nearest'),
        'SV2v3': LabelFeature('visual_sector:17', 'nearest'),
        'SV2v4': LabelFeature('visual_sector:18', 'nearest'),
        'SV3d1': LabelFeature('visual_sector:19', 'nearest'),
        'SV3d2': LabelFeature('visual_sector:20', 'nearest'),
        'SV3d3': LabelFeature('visual_sector:21', 'nearest'),
        'SV3d4': LabelFeature('visual_sector:22', 'nearest'),
        'SV3v1': LabelFeature('visual_sector:23', 'nearest'),
        'SV3v2': LabelFeature('visual_sector:24', 'nearest'),
        'SV3v3': LabelFeature('visual_sector:25', 'nearest'),
        'SV3v4': LabelFeature('visual_sector:26', 'nearest'),
        # These get a bit complex as they subtract or union pieces together.
        'SV1fov': LabelFeature('visual_sector:1 6'),
        'SV2fov': LabelDiffFeature(
            'visual_area:2--visual_sector:11 12 13 14 15 16 17 18',
            'nearest'),
        'SV3fov': LabelDiffFeature(
            'visual_area:3--visual_sector:19 20 21 22 23 24 25 26',
            'nearest'),
        # Eccentricity regions.
        'E0': LabelDiffFeature(
            'visual_area:1 2 3--visual_sector:2 3 4 5 7 8 9 10 11 12 13 14 15'
            ' 16 17 18 19 20 21 22 23 24 25 26',
            'nearest'),
        'E1': LabelFeature('visual_sector:2 7 11 15 19 23', 'nearest'),
        'E2': LabelFeature('visual_sector:3 8 12 16 20 24', 'nearest'),
        'E3': LabelFeature('visual_sector:4 9 13 17 21 25', 'nearest'),
        'E4': LabelFeature('visual_sector:5 10 14 18 22 26', 'nearest'),
        # Anatomical Features.
        'myelin': FlatmapFeature('myelin', 'linear'),
        # The vertex coordinates themselves; we add these in.
        'x': FlatmapFeature('midgray_x', 'linear'),
        'y': FlatmapFeature('midgray_y', 'linear'),
        'z': FlatmapFeature('midgray_z', 'linear')
    }
    @classmethod
    def builtin_features(cls):
        fs = HCPLinesImageCache._builtin_features
        return dict(BilateralFlatmapImageCache.builtin_features(), **fs)
    @classmethod
    def unpack_target(cls, target):
        if len(target) == 2:
            if isinstance(target, Mapping):
                rater = target['rater']
                sid = target['subject']
            else:
                (rater, sid) = target
        else:
            raise ValueError(
                f"target for {type(self)}.make_flatmap must be one of: "
                "(rater,sid), {'rater':rater, 'subject':sid}")
        return (rater, sid)
    def cache_filename(self, target, feature):
        rater = target['rater']
        subject = target['subject']
        return os.path.join(feature, f"{rater}_{subject}.pt")
    def make_flatmap(self, target, view=None):
        # We may have been given (rater, sid, h) or ((rater, sid), h):
        (rater, sid) = self.unpack_target(target)
        if view is None:
            raise ValueError("HCPLinesImageCache requires a view")
        h = view['hemisphere']
        # Get the subject and hemi.
        sub = ny.data['hcp_lines'].subjects[sid]
        hem = sub.hemis[h]
        # Fix the properties now, if needed:
        (x,y,z) = hem.surface('midgray').coordinates
        hem = hem.with_prop(midgray_x=x, midgray_y=y, midgray_z=z)
        if rater is not None and rater != 'mean':
            # Get the appropriate data from the dataset.
            dat = ny.data['hcp_lines'].subject_labels[rater][sid][h]
            hem = hem.with_prop(
                visual_area=dat['visual_area'],
                visual_sector=dat['visual_sector'])
        # Make the flatmap:
        fmap = ny.to_flatmap('occipital_pole', hem, radius=np.pi/2.25)
        fmap = fmap.with_meta(subject_id=sid, rater=rater, hemisphere=h)
        # And return!
        return fmap
    # We overload fill_image so that we can call down then turn NaNs into 0s.
    def fill_image(self, target, feature, im):
        super().fill_image(target, feature, im)
        im[torch.isnan(im)] = 0
        return im

class HCPLinesDataset(ImageCacheDataset):
    """A PyTorch Dataset object that encapsulates the HCP lines dataset.

    The `HCPLinesDataset` is a PyTorch dataset for use with image-based models
    such as CNNs. The dataset may be configured to use any of a number of known
    features, including features based on the hand-drawn annotations. For a full
    list of possible features, check the `HCPLinesImageCache` type and the
    results of the `HCPLinesImageCache.builtin_features()` method.
    """
    __slots__ = ()
    def __init__(self, inputs, outputs,
                 raters=('A1', 'A2', 'A3', 'A4'),
                 subjects=Ellipsis,
                 exclusions=Ellipsis,
                 image_size=Ellipsis,
                 transform=None,
                 input_transform=None,
                 output_transform=None,
                 hemis='lr',
                 cache_image_size=Ellipsis,
                 cache_path=None,
                 overwrite=False,
                 mkdirs=True,
                 mkdir_mode=0o775,
                 multiproc=True,
                 timeout=None,
                 dtype='float32',
                 memcache=True,
                 normalization=None,
                 features=None,
                 flatmap_cache=True):
        # Make an HCPLines Occipital Image Cache object first.
        imcache = HCPLinesImageCache(
            hemis=hemis,
            image_size=cache_image_size,
            cache_path=cache_path,
            overwrite=overwrite,
            mkdirs=mkdirs,
            mkdir_mode=mkdir_mode,
            multiproc=multiproc,
            timeout=timeout,
            dtype=dtype,
            memcache=memcache,
            normalization=normalization,
            features=features,
            flatmap_cache=flatmap_cache)
        # Figure out the targets dicts of rater and subject.
        dset = ny.data['hcp_lines']
        sids = dset.subject_list
        if subjects is Ellipsis:
            subjects = sids
        if isinstance(raters, str):
            raters = (raters,)
        # Figure out the exclusions next.
        if exclusions is Ellipsis:
            # We get the exclusions from the dataset.
            exclusions = dset.exclusions
        elif exclusions is not None:
            exclusions = set()
        else:
            exclusions = set(exclusions)
        # Step through these and process from (rater, sid, h) into (rater, sid)
        # when necessary.
        tmp = exclusions
        exclusions = set([])
        for excl in tmp:
            if isinstance(excl, tuple) and len(excl) == 1:
                excl = excl[0]
            if isinstance(excl, str):
                if excl in raters:
                    for s in subjects:
                        exclusions.add((excl, s))
            elif isinstance(excl, int):
                if excl in subjects:
                    for r in raters:
                        exclusions.add((r, excl))
            elif len(excl) == 3:
                (r,s,h) = excl
                exclusions.add((r,s))
            elif len(excl) == 2:
                exclusions.add(excl)
            else:
                raise ValueError(f"invalid exclusion: {excl}")
        # Filter out the excluded targets.
        targets = [{'rater':r, 'subject':s}
                   for r in raters for s in subjects
                   if (r,s) not in exclusions]
        targets = tuple(targets)
        # Now go ahead and initialize our superclass using it.
        super().__init__(
            imcache, inputs, outputs, targets,
            image_size=image_size,
            transform=transform,
            input_transform=input_transform,
            output_transform=output_transform)

def make_datasets(in_features, out_features,
                  features=None,
                  partition=default_partition,
                  raters=('A1', 'A2', 'A3', 'A4'),
                  subjects=Ellipsis,
                  exclusions=Ellipsis,
                  image_size=Ellipsis,
                  transform=None,
                  input_transform=None,
                  output_transform=None,
                  hemis='lr',
                  cache_image_size=Ellipsis,
                  cache_path=None,
                  overwrite=False,
                  mkdirs=True,
                  mkdir_mode=0o775,
                  multiproc=True,
                  timeout=None,
                  dtype='float32',
                  memcache=True,
                  normalization=None,
                  flatmap_cache=True):
    """Returns a mapping of training and validation datasets.

    The mapping returned by `make_datasets()` contains, at the top level, the
    keys `'trn'` and `'val'` whose keys are the training and validation
    datasets, respectively. At the next level, the keys are `'anat'`, `'func'`,
    and `'both'` for the dataset input image type. The second level of maps are
    lazy.

    Parameters
    ----------
    features : 'func' or 'anat' or 'both' or None
        The type of input images that the dataset uses: functional data
        (`'func'`), anatomical data (`'anat'`), or both (`'both'`). If `None`
        (the default), then a mapping is returned with each input dataset type
        as values and with `'func'`, `'anat'`, and `'both'` as keys.
    sids : list-like, optional
        An iterable of subject-IDs to be included in the datasets. By default,
        the subject list `visual_autolabel.util.sids` is used.
    partition : partition-like
        How to make the partition of sujbect-IDs; the partition is made using
        `visual_autolabel.utils.partitoin(sids, how=partition)`.
    image_size : int, optional
        The width of the training images, in pixels (default: 512).
    cache_path : str or None, optional
        The path in which the dataset will be cached, or None if no cache is to
        be used (the default).

    Returns
    -------
    nested mapping of HCPVisualDataset objects
        A nested dictionary structure whose values at the bottom are datasets
        for training and validation partitions and for anatomy, function, and
        both. If `features` is `None`, then the return value is equivalent to
        `{f: make_datasets(f) for f in ['anat', 'func', 'both']}`.
    """
    (trn_sids, val_sids) = make_partition(sids, how=partition)
    def curry_fn(sids):
        return (lambda:HCPLinesDataset(in_features, out_features,
                                       subjects=sids,
                                       raters=raters,
                                       features=features,
                                       cache_path=cache_path,
                                       image_size=image_size,
                                       exclusions=exclusions,
                                       transform=transform,
                                       input_transform=input_transform,
                                       output_transform=output_transform,
                                       hemis=hemis,
                                       cache_image_size=cache_image_size,
                                       overwrite=overwrite,
                                       mkdirs=mkdirs,
                                       mkdir_mode=mkdir_mode,
                                       multiproc=multiproc,
                                       timeout=timeout,
                                       dtype=dtype,
                                       memcache=memcache,
                                       normalization=normalization,
                                       flatmap_cache=flatmap_cache))
    return pimms.lmap({'trn': curry_fn(trn_sids),
                       'val': curry_fn(val_sids)})
def make_dataloaders(in_features, out_features,
                     features=None,
                     partition=default_partition,
                     raters=('A1', 'A2', 'A3', 'A4'),
                     subjects=Ellipsis,
                     exclusions=Ellipsis,
                     image_size=Ellipsis,
                     transform=None,
                     input_transform=None,
                     output_transform=None,
                     hemis='lr',
                     cache_image_size=Ellipsis,
                     cache_path=None,
                     overwrite=False,
                     mkdirs=True,
                     mkdir_mode=0o775,
                     multiproc=True,
                     timeout=None,
                     dtype='float32',
                     memcache=True,
                     normalization=None,
                     flatmap_cache=True,
                     datasets=None, 
                     shuffle=True,
                     batch_size=5):
    """Returns a pair of PyTorch dataloaders as a dictionary.

    `make_dataloaders('func')` returns training and validation dataloaders (in a
    dictionary whose keys are `'trn'` and `'val'`) for the functional data of
    HCP. The dataloaders and datasets can be modified with the optional
    arguments.

    Parameters
    ----------
    in_features : list-like of feature names
        A list or tuple of the feature names that are to be used as input.
    out_features : list-like of feature names
        A list or tuple of the feature names that are to be used as outputs.
    features : dict of features, optional
        A dictionary of features to be used when creating the datasets.
    sids : list-like, optional
        An iterable of subject-IDs to be included in the datasets. By default,
        the subject list `visual_autolabel.util.sids` is used.
    partition : partition-like, optional
        How to make the partition of sujbect-IDs; the partition is made using
        `visual_autolabel.utils.partitoin(sids, how=partition)`.
    image_size : int or None, optional
        The width of the training images, in pixels; if `None`, then 512 is
        used (default: `None`).
    cache_path : str or None, optional
        The path in which the dataset will be cached, or None if no cache is to
        be used (the default).
    datasets : None or mapping of datasets, optional
        A mapping of datasets that should be used. If the keys of this mapping
        are `'trn'` and `'val'` then all of the above arguments are ignored and
        these datasets are used for the dataloaders. Otherwise, if `features` is
        a key in `datasets`, then `datasets[features]` is used and the other
        options above are ignored. Otherwise, if `datasets` is `None` (the
        default), then the datasets are created using the above options.
    shuffle : boolean, optional
        Whether to shuffle the IDs when loading samples (default: `True`).
    batch_size : int, optional
        The batch size for samples from the dataloader (default: 5).

    Returns
    -------
    nested mapping of PyTorch DataLoader objects
        A nested dictionary structure whose values at the bottom are PyTorch
        data-loader objects for training and validation partitions and for
        anatomy, function, and both. If `features` is `None`, then the return
        value is equivalent to
        `{f: make_dataloader(f, **kw) for f in ['anat', 'func', 'both']}`.
    """
    # What were we given for datasets?
    if datasets is None:
        # We need to make the datasets using the other options.
        datasets = make_datasets(in_features, out_features,
                                 subjects=subjects,
                                 raters=raters,
                                 features=features,
                                 partition=partition,
                                 cache_path=cache_path,
                                 image_size=image_size,
                                 exclusions=exclusions,
                                 transform=transform,
                                 input_transform=input_transform,
                                 output_transform=output_transform,
                                 hemis=hemis,
                                 cache_image_size=cache_image_size,
                                 overwrite=overwrite,
                                 mkdirs=mkdirs,
                                 mkdir_mode=mkdir_mode,
                                 multiproc=multiproc,
                                 timeout=timeout,
                                 dtype=dtype,
                                 memcache=memcache,
                                 normalization=normalization,
                                 flatmap_cache=flatmap_cache)
    # At this point, datasets must have 'trn' and 'val' entries in order to be
    # valid, or it must be a 2-tuple.
    if not is_partition(datasets):
        raise ValueError("make_dataloaders(): provided datasets are not valid")
    # Okay, now we can make the data-loaders using these datasets.
    trn = trndata(datasets)
    val = valdata(datasets)
    return dict(
        trn=DataLoader(trn, batch_size=batch_size, shuffle=shuffle),
        val=DataLoader(val, batch_size=batch_size, shuffle=shuffle))

