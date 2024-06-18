# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/config/__init__.py
# Configuration of the visual_autolabel package.

"""Global configuration variables for the `visual_autolabel` package.

The `visual_autolabel.config` package contains definitions of global variables
that are used throughout the package. See also the 
`visual_autolabel.benson2024.config` package for configuration items that were
used in the paper by Benson et al. (2024).

Attributes
----------
saved_image_size : int
    The size of images (number of image rows) that get saved to cache. It's
    relatively efficient to downsample images, so keeping this somewhat larger
    than needed is a good idea. The value used in the package is 512.
default_image_size : int
    The default size for images used in model training. This is the number of
    rows in the images.
default_partition : tuple
    The default way of partitioning the subjects into training and validation
    datasets. See also the `visual_autolabel.partition` function. The value used
    in the package is `(0.8, 0.2)`, indicating that 80% of subjects should be
    in the training dataset, and 20% of the subjects should be in the valudation
    dataset.
"""

from ._core import (
    saved_image_size,
    default_image_size,
    default_partition,
)

__all__ = [
    "saved_image_size",
    "default_image_size",
    "default_partition",
]
