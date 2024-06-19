# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2024/nyu/_datasets.py
# Code to manage the NYU retinotopy-specific aspects of the AutoLabeler dataset.


#===============================================================================
# Dependencies

# External Libries -------------------------------------------------------------

import os
from collections.abc import (Sequence, Mapping)

import numpy as np
import neuropythy as ny
import pimms, torch
from torch.utils.data import (Dataset, DataLoader)

# Internal Tools ---------------------------------------------------------------

from ...util import (
    partition_id,
    partition as make_partition,
    is_partition,
    trndata,
    valdata,
    convrelu,
    rigid_align_cortices
)
from ...image import (
    ImageCacheDataset,
    BilateralFlatmapImageCache,
    FlatmapFeature,
    LabelFeature,
    LabelDiffFeature,
    LabelUnionFeature,
    LabelIntersectFeature
)


#===============================================================================
# The NYU Image Cache

class NYUImageCache(BilateralFlatmapImageCache):
    """An ImageCache subclass for features from the NYU Retinotopy Dataset.
    """
    # The pseudo-path for the NYU retinotopy dataset:
    # We don't declare a cache path because that can be configured using
    # Neuropythy (openneuro datasets go in an openneuro cache directory).
    nyu_retinotopy_pp = ny.util.pseudo_path('s3://openneuro.org/ds003787/')
    # The subject list:
    subject_list = [
        'sub-wlsubj001',
        'sub-wlsubj004',
        'sub-wlsubj006',
        'sub-wlsubj007',
        'sub-wlsubj014',
        'sub-wlsubj019',
        'sub-wlsubj023',
        'sub-wlsubj042',
        'sub-wlsubj043',
        'sub-wlsubj045',
        'sub-wlsubj046',
        'sub-wlsubj055',
        'sub-wlsubj056',
        'sub-wlsubj057',
        'sub-wlsubj062',
        'sub-wlsubj064',
        'sub-wlsubj067',
        'sub-wlsubj071',
        'sub-wlsubj076',
        'sub-wlsubj079',
        'sub-wlsubj081',
        'sub-wlsubj083',
        'sub-wlsubj084',
        'sub-wlsubj085',
        'sub-wlsubj086',
        'sub-wlsubj087',
        'sub-wlsubj088',
        'sub-wlsubj090',
        'sub-wlsubj091',
        'sub-wlsubj092',
        'sub-wlsubj094',
        'sub-wlsubj095',
        'sub-wlsubj104',
        'sub-wlsubj105',
        'sub-wlsubj109',
        'sub-wlsubj114',
        'sub-wlsubj115',
        'sub-wlsubj116',
        'sub-wlsubj117',
        'sub-wlsubj118',
        'sub-wlsubj120',
        #'sub-wlsubj121',
        'sub-wlsubj122',
        'sub-wlsubj126']
    @staticmethod
    def load_subject(subj_id, path=nyu_retinotopy_pp, max_eccen=12.4):
        # Get the FreeSurfer subject directory:
        subpp = path.subpath(f'derivatives/freesurfer/{subj_id}')
        sub = ny.freesurfer_subject(subpp)
        # Get the PRF subpath:
        prfpp = path.subpath(
            f'derivatives/prfanalyze-vista/{subj_id}/ses-nyu3t01/')
        labpp = path.subpath(
            f'derivatives/ROIs/{subj_id}/')
        for h in ['lh', 'rh']:
            hem = sub.hemis[h]
            # Load the retinotopy data for this hemisphere:
            prfs = dict(
                prf_polar_angle=ny.load(
                    prfpp.local_path(f'{h}.angle_adj.mgz')),
                prf_eccentricity=ny.load(
                    prfpp.local_path(f'{h}.eccen.mgz')),
                prf_variance_explained=ny.load(
                    prfpp.local_path(f'{h}.vexpl.mgz')),
                prf_radius=ny.load(
                    prfpp.local_path(f'{h}.sigma.mgz')),
                prf_x=ny.load(
                    prfpp.local_path(f'{h}.x.mgz')),
                prf_y=ny.load(
                    prfpp.local_path(f'{h}.y.mgz')),
                visual_area=ny.load(
                    labpp.local_path(f'{h}.ROIs_V1-4.mgz')))
            # Add scaled eccentricity to the subject:
            prfs['prf_scaled_eccentricity'] = (
                prfs['prf_eccentricity'] / max_eccen * 8)
            prfs['prf_scaled_x'] = prfs['prf_eccentricity'] / max_eccen * 8
            prfs['prf_scaled_y'] = prfs['prf_eccentricity'] / max_eccen * 8
            hem = hem.with_prop(prfs)
            sub = sub.with_hemi({h: hem})
        return sub
    # The NYU Retinotopy subjects: (instantiated after the class def)
    subjects = None
    # The featuers we know how to make.
    _builtin_features = {
        # Functional Features first.
        'prf_polar_angle':  FlatmapFeature('prf_polar_angle', 'nearest'),
        'prf_eccentricity': FlatmapFeature('prf_eccentricity', 'linear'),
        'prf_cod':          FlatmapFeature('prf_variance_explained', 'linear'),
        'prf_sigma':        FlatmapFeature('prf_radius', 'linear'),
        'prf_x':            FlatmapFeature('prf_x', 'linear'),
        'prf_y':            FlatmapFeature('prf_y', 'linear'),
        'prf_scaled_eccentricity': FlatmapFeature(
            'prf_scaled_eccentricity', 'linear'),
        'prf_scaled_x':            FlatmapFeature('prf_scaled_x', 'linear'),
        'prf_scaled_y':            FlatmapFeature('prf_scaled_y', 'linear'),
        # The vertex coordinates themselves; we add these in.
        'x': FlatmapFeature('midgray_x', 'linear'),
        'y': FlatmapFeature('midgray_y', 'linear'),
        'z': FlatmapFeature('midgray_z', 'linear'),
        # The visual area and visual sector-based features.
        'V1':  LabelFeature('visual_area:1', 'nearest'),
        'V2':  LabelFeature('visual_area:2', 'nearest'),
        'V3':  LabelFeature('visual_area:3', 'nearest')
    }
    @classmethod
    def builtin_features(cls):
        fs = NYUImageCache._builtin_features
        return dict(BilateralFlatmapImageCache.builtin_features(), **fs)
    @classmethod
    def unpack_target(cls, target):
        if isinstance(target, Mapping):
            sid = target['subject']
        else:
            sid = target
        return (sid,)
    def cache_filename(self, target, feature):
        return os.path.join(feature, f"{target['subject']}.pt")
    def make_flatmap(self, target, view=None):
        # We may have been given (rater, sid, h) or ((rater, sid), h):
        (sid,) = self.unpack_target(target)
        if view is None:
            raise ValueError("NYUImageCache requires a view")
        h = view['hemisphere']
        # Get the subject and hemi.
        sub = NYUImageCache.subjects[sid]
        hem = sub.hemis[h]
        # Fix the properties now, if needed; note that we need to align
        # the midgray coordinates to the fsaverage surface's midgray.
        fsa = ny.freesurfer_subject('fsaverage')
        fsahem = fsa.hemis[h]
        midgray = rigid_align_cortices(hem, fsahem, 'midgray')
        (x,y,z) = midgray.coordinates
        # FreeSurfer subjects store their sulcal depths (often called convexity)
        # in mm; the HCP stored theirs in cm, so we need to scale the FreeSurfer
        # convexities to match the HCP's.
        convex = hem.prop('convexity') / 10.0
        hem = hem.with_prop(
            midgray_x=x,
            midgray_y=y,
            midgray_z=z,
            convexity=convex)
        # Make the flatmap:
        fmap = ny.to_flatmap('occipital_pole', hem, radius=np.pi/2)
        fmap = fmap.with_meta(subject_id=sid, hemisphere=h)
        # And return!
        return fmap
    # We overload fill_image so that we can call down then turn NaNs into 0s.
    def fill_image(self, target, feature, im):
        super().fill_image(target, feature, im)
        im[torch.isnan(im)] = 0
        return im
NYUImageCache.subjects = pimms.lazy_map(
    {s: ny.util.curry(NYUImageCache.load_subject, s)
     for s in NYUImageCache.subject_list})

class NYUDataset(ImageCacheDataset):
    """A PyTorch Dataset object that encapsulates the NYU retinotopy dataset.

    The `NYUDataset` is a PyTorch dataset for use with image-based models such
    as CNNs. The dataset may be configured to use any of a number of known
    features, including features based on the hand-drawn annotations. For a full
    list of possible features, check the `NYUImageCache` type and the results of
    the `NYUImageCache.builtin_features()` method.
    """
    __slots__ = ()
    def __init__(self, inputs, outputs,
                 sids=Ellipsis,
                 image_size=Ellipsis,
                 cache_path=Ellipsis,
                 transform=None,
                 input_transform=None,
                 output_transform=None,
                 hemis='lr',
                 cache_image_size=Ellipsis,
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
        # If the auto-cache_path is requested, use it.
        if cache_path is Ellipsis:
            from ..config import dataset_cache_path
            if dataset_cache_path is not None:
                dataset_cache_path = os.path.join(dataset_cache_path, 'NYU')
            cache_path = os.path.join(dataset_cache_path, 'NYU')
        # Make an HCPLines Occipital Image Cache object first.
        imcache = NYUImageCache(
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
        sids = NYUImageCache.subject_list
        if sids is Ellipsis:
            from ..config import nyu_sids
            sids = nyu_sids
        # Make the targets dicts
        targets = tuple({'subject': s} for s in sids)
        # If we have been given an alias string for the inputs or outputs,
        # translate those now based on the table in _core.py.
        if isinstance(inputs, str):
            from ._core import input_properties as ps
            inputs = ps.get(inputs, (inputs,))
        if isinstance(outputs, str):
            from ._core import output_properties as ps
            outputs = ps.get(outputs, (outputs,))
        # Now go ahead and initialize our superclass using it.
        super().__init__(
            imcache, inputs, outputs, targets,
            image_size=image_size,
            transform=transform,
            input_transform=input_transform,
            output_transform=output_transform)

def make_datasets(in_features, out_features,
                  features=None,
                  partition=Ellipsis,
                  sids=Ellipsis,
                  image_size=Ellipsis,
                  transform=None,
                  input_transform=None,
                  output_transform=None,
                  hemis='lr',
                  cache_image_size=Ellipsis,
                  cache_path=Ellipsis,
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
        the subject list `visual_autolabel.benson2024.config.nyu_sids` is used.
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
    if sids is Ellipsis:
        from ..config import nyu_sids
        sids = nyu_sids
    if partition is Ellipsis:
        from ...config import default_partition
        partition = default_partition
    (trn_sids, val_sids) = make_partition(sids, how=partition)
    def curry_fn(sids):
        return lambda:NYUDataset(
            in_features, out_features,
            sids=sids,
            features=features,
            cache_path=cache_path,
            image_size=image_size,
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
    return pimms.lmap({'trn': curry_fn(trn_sids),
                       'val': curry_fn(val_sids)})
def make_dataloaders(in_features, out_features,
                     features=None,
                     partition=Ellipsis,
                     sids=Ellipsis,
                     image_size=Ellipsis,
                     transform=None,
                     input_transform=None,
                     output_transform=None,
                     hemis='lr',
                     cache_image_size=Ellipsis,
                     cache_path=Ellipsis,
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
    """Returns a pair of PyTorch dataloaders for the NYU dataset as a dict.

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
        the subject list is imported from
        `visual_autolabel.image.NYUImageCache.subject_list`.
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
        datasets = make_datasets(
            in_features, out_features,
            sids=sids,
            features=features,
            partition=partition,
            cache_path=cache_path,
            image_size=image_size,
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
