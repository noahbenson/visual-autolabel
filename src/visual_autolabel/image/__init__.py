# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/image/__init__.py
# Submodule of the visual_autolabel package that hangles image-based CNNs/data.

from ._data import (
    ImageCache,
    FlatmapFeature,
    FlatmapImageCache,
    BilateralFlatmapImageCache,
    ImageCacheDataset,
    LabelFeature,
    LabelDiffFeature,
    LabelUnionFeature,
    LabelIntersectFeature,
    NullFeature
)
from ._data3D import (
    dataset3D_cache_path_init as dataset3D_cache_path,
    noddi_data_path_init as noddi_data_path,
    _sids as sids
)
from ._model import (
    UNet2D,
    UNet3D,
    HybridUNet as UNet
)
                    
__all__ = [
    'ImageCache',
    'FlatmapFeature',
    'FlatmapImageCache',
    'BilateralFlatmapImageCache',
    'ImageCacheDataset',
    'LabelFeature',
    'LabelDiffFeature',
    'LabelUnionFeature',
    'LabelIntersectFeature',
    'NullFeature',
    'UNet'
]
