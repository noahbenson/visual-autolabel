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
from ._hcp import (
    HCPLinesImageCache,
    HCPLinesDataset,
    make_datasets,
    make_dataloaders
)
from ._nyu import (
    NYURetinotopyImageCache,
    NYURetinotopyDataset,
    make_datasets as nyu_make_datasets,
    make_dataloaders as nyu_make_dataloaders
)
from ._model import (
    UNet
)
                    
__all__ = [
    'ImageCache',
    'FlatmapFeature',
    'FlatmapImageCache',
    'BilateralFlatmapImageCache',
    'ImageCacheDataset',
    'HCPLinesImageCache',
    'HCPLinesDataset',
    'NYURetinotopyImageCache',
    'NYURetinotopyDataset',
    'make_datasets',
    'make_dataloaders',
    'UNet',
    'LabelFeature',
    'LabelDiffFeature',
    'LabelUnionFeature',
    'LabelIntersectFeature',
    'NullFeature'
]
