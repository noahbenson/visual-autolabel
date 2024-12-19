# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/cmd/_apply.py
# Code to for applying the Benson, Song, et al. (2025) models to an individual
# subject.


#===============================================================================
# Dependencies

import os
from pathlib import Path

import torch
import numpy as np
import neuropythy as ny
import pimms

from ..benson2025 import unet
from ..benson2025.hcp import (input_properties, output_properties)
from ..image import (
    BilateralFlatmapImageCache,
    ImageCacheDataset,
    FlatmapFeature)
from ..util import rigid_align_cortices

#===============================================================================
# Code

class NeuropythySubjectImageCache(BilateralFlatmapImageCache):
    """An image cache type for a single neuropythy subject.

    The idea behind this class is that the user will provide a Neuropythy
    subject object. We will set up an ImageCache object that handles just that
    subject. The ImageCache will work only with anatomical data (T1) because the
    pRF models require additional training still. Note that because we don't
    actually care about caching the data for a single subject, this is just a
    convenient way to make the machinery work and not a typical use of the
    ImageCache class.
    """
    
    def __init__(self, subject):
        # We want to initialize using the parameters that were used by Benson,
        # Song, et al. (2025) in order to make images that are compatible with
        # that model.
        super().__init__(
            hemis='lr',
            image_size=Ellipsis,
            cache_path=None,
            overwrite=False,
            mkdirs=True,
            mkdir_mode=0o775,
            multiproc=False,
            timeout=None,
            dtype='float32',
            memcache=False,
            normalization=None,
            features=None,
            flatmap_cache=True)
        sid = str(id(subject))
        # Okay, we should now have a subject.
        self.subject = subject
        self.sid = sid
    @classmethod
    def builtin_features(cls):
        return dict(
            BilateralFlatmapImageCache.builtin_features(),
            x=FlatmapFeature('midgray_x', 'linear'),
            y=FlatmapFeature('midgray_y', 'linear'),
            z=FlatmapFeature('midgray_z', 'linear'))
    def unpack_target(cls, target):
        return (target['subject'],)
    def cache_filename(self, target, feature):
        return os.path.join(feature, f"{target['subject']}.pt")
    def make_flatmap(self, target, view=None):
        (sid,) = self.unpack_target(target)
        if view is None:
            raise ValueError("GeneralImageCache requires a view")
        h = view['hemisphere']
        sub = self.subject
        hem = sub.hemis[h]
        fsa = ny.freesurfer_subject('fsaverage')
        fsahem = fsa.hemis[h]
        midgray = rigid_align_cortices(hem, fsahem, 'midgray')
        (x, y, z) = midgray.coordinates
        convex = hem.prop('convexity') / 10.0
        null = np.zeros_like(x)
        hem = hem.with_prop(
            midgray_x=x,
            midgray_y=y,
            midgray_z=z,
            convexity=convex,
            V1=null, V2=null, V3=null,
            E0=null, E1=null, E2=null, E3=null, E4=null)
        fmap = ny.to_flatmap('occipital_pole', hem, radius=np.pi/2)
        fmap = fmap.with_meta(subject_id=sid, hemisphere=h)
        return fmap
    def fill_image(self, target, feature, im):
        super().fill_image(target, feature, im)
        im[torch.isnan(im)] = 0
        return im

class NeuropythySubjectDataset(ImageCacheDataset):
    __slots__ = ()
    def __init__(self, subject, outputs):
        imcache = NeuropythySubjectImageCache(subject)
        sid = imcache.sid
        targets = ({'subject': sid},)
        inputs = input_properties['anat']
        if outputs != 'area' and outputs != 'ring':
            raise ValueError("outputs must be 'area' or 'ring'")
        outputs = output_properties[outputs]
        super().__init__(imcache, inputs, outputs, targets)

def apply_benson2025(subject, outputs):
    """Apply one of the CNN model from Benson, Song, et al. (2025) to a
    subject.

    The `subject` argument must be a neuropythy subject object. The `outputs`
    argument should be either `'area'` or `'ring'` for predicting visual areas
    or for iso-eccentric regions, respectively.

    The return value is tuple of vectors of `(lh_labels, rh_labels)`.
    """
    if outputs != 'area' and outputs != 'ring':
        raise ValueError("outputs must be 'area' or 'ring'")
    # Get the model!
    mdl = unet('anat', outputs)
    # Make a dataset for this subject:
    ds = NeuropythySubjectDataset(subject, outputs)
    sub = ds.image_cache.subject
    # Predict their labels!
    lbls = []
    for h in ('lh','rh'):
        hem = sub.hemis[h]
        view = {'hemisphere': h}
        fmap = ds.image_cache.get_flatmap(ds.targets[0], view=view)
        fmlbl = ds.predlabels(0, mdl, view=view)
        hlbl = np.zeros(hem.vertex_count, dtype=fmlbl.dtype)
        hlbl[fmap.labels] = fmlbl
        lbls.append(hlbl)
    return tuple(lbls)
    
    
