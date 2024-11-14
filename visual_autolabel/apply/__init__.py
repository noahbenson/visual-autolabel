################################################################################
# visual_autolabel/apply/__init__.py

"""Code for applying the visual-autolabel model to FreeSurfer subjects.
"""


# Dependencies #################################################################

import os, sys
from pathlib import Path

import torch
import numpy as np
import neuropythy as ny

from ..config import (
    default_partition,
    default_image_size,
    saved_image_size
)
from ..util import (
    partition_id,
    partition as make_partition,
    is_partition,
    trndata,
    valdata,
    convrelu
)
from ..image import (
    BilateralFlatmapImageCache,
    ImageCacheDataset
)


# Classes ######################################################################

class FreeSurferImageCache(BilateralFlatmapImageCache):
    """An image caching class for FreeSurfer based on
    BilateralFlatmapImageCache.
    """
    __slots__ = ('subject_path',)
    def __init__(self, subject_path,
                 hemis='lr',
                 image_size=Ellipsis,
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
        super().__init__(
            hemis=hemis,
            image_size=image_size,
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
        self.subject_path = ny.util.pseudo_path(subject_path)
    def load_subject(subj_id):
        return ny.freesurfer_subject(subj_id)
    @classmethod
    def builtin_features(cls):
        fs = {
            'x': FlatmapFeature('midgray_x', 'linear'),
            'y': FlatmapFeature('midgray_y', 'linear'),
            'z': FlatmapFeature('midgray_z', 'linear'),
        }
        return dict(BilateralFlatmapImageCache.builtin_features(), **fs)
    def unpack_target(cls, target):
        if isinstance(target, Mapping):
            sid = target['subject']
        else:
            sid = target
        return (sid,)
    def cache_filename(self, target, feature):
        return os.path.join(feature, f"{target['subject']}.pt")
    def make_flatmap(self, target, view=None):
        (sid,) = self.unpack_target(target)
        if view is None:
            raise ValueError("GeneralImageCache requires a view")
        h = view['hemisphere']
        sub = self.subjects[sid]
        hem = sub.hemis[h]
        fsa = ny.freesurfer_subject('fsaverage')
        fsahem = fsa.hemis[h]
        midgray = rigid_align_cortices(hem, fsahem, 'midgray')
        (x, y, z) = midgray.coordinates
        convex = hem.prop('convexity') / 10.0
        hem = hem.with_prop(midgray_x=x, 
                            midgray_y=y, 
                            midgray_z=z, 
                            convexity=convex)
        fmap = ny.to_flatmap('occipital_pole', hem, radius=np.pi/2)
        fmap = fmap.with_meta(subject_id=sid, hemisphere=h)
        return fmap
    def fill_image(self, target, feature, im):
        super().fill_image(target, feature, im)
        im[torch.isnan(im)] = 0
        return im

class FreeSurferDataset(ImageCacheDataset):
    __slots__ = ()
    def __init__(self, subject_path, outputs,
                 image_size=Ellipsis, 
                 cache_path=None, 
                 transform=None, 
                 input_transform=None, 
                 output_transform=None, 
                 hemis='lr', 
                 cache_image_size=Ellipsis, 
                 overwrite=False, 
                 mkdirs=True, mkdir_mode=0o775, 
                 multiproc=True, timeout=None, 
                 dtype='float32', memcache=True, 
                 normalization=None, 
                 features=None, 
                 flatmap_cache=True):
        if cache_path is Ellipsis:
            from ..config import dataset_cache_path
            if dataset_cache_path is not None:
                dataset_cache_path = os.path.join(dataset_cache_path, 'FreeSurfer')
            cache_path = os.path.join(dataset_cache_path, 'FreeSurfer')
        imcache = FreeSurferImageCache(
            subject_path=subject_path,
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
        targets = ({'subject': subject_path},)
        inputs = (
            'x', 'y', 'z',
            'curvature', 'convexity',
            'thickness', 'surface_area')
        if isinstance(outputs, str):
            from ..benson2024.hcp._core import output_properties as ps
            outputs = ps.get(outputs, (outputs,))
        super().__init__(
            imcache, inputs, outputs, targets,
            image_size=image_size,
            transform=transform,
            input_transform=input_transform,
            output_transform=output_transform)


# apply_model ##################################################################

def apply_model(subject_path, outputs='both', output_dir=None,
                lh_area_filename='lh.visual_area.mgz',
                lh_ring_filename='lh.visual_ring.mgz',
                rh_area_filename='rh.visual_area.mgz',
                rh_ring_filename='rh.visual_ring.mgz'):
    """Applies the CNN models from Benson, Song, & Winawer (2024) to the given
    FreeSurfer subject and returns the predicted visual area boundaries.

    `apply_model(path)` returns a tuple of `(area_predictions,ring_predictions)`
    for the FreeSurfer subject whose directory is given by the argument `path`.
    The `area_predictions` and `ring_predictions` data are each tuples of
    `(lh_data,rh_data)`, each of which is a numpy vectors of the labels of each
    vertex on the subject's cortical surfaces.

    The optional parameter `output_dir` (default: None) can be set in order to
    save the labels to a directory. The label files will be named according to
    the four optional parameters `lh_area_filename`, `lh_ring_filename`,
    `rh_area_filename`, and `rh_ring_filename`.
    """
    # TODO!  We need to create the classes above and create a unet model, load
    # the weights for the appropriate models, and apply them using
    # dataset.predlabels() then save them out.
    # We might want to put a function the loads and caches our best models in
    # the benson2024 subpackage.
    pass

