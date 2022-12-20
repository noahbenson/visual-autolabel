# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/__init__.py
# Package initialization for the visual-autolabel project.

"""Automatic labeling of V1, V2, and V3 in human visual cortex.

This package encapsulates the various data-loading, modeling, and model-training
code necessary to predict the V1, V2, and V3 labels in human visual cortex. This
package uses the Human Connectome Project (HCP) and the associated retinotopy
dataset as training and validation datasets, and it uses a U-Net CNN model using
a ResNet-18 backbone to predict the visual labels.

The primary entry-point for this package is the `run_modelplan()` function, or,
alternately, the `build_model()` function. `run_modelplan()` accepts a wide
range of optional parameters that can be used to customize the model or training
paradigm used to build and train a model, as well as a list of training-steps,
represented as parameter-dictionaries, which are executed (via the `build_model`
function) to produce the trained model.

Attributes
----------
config : Python subpackage
    The `visual_autolabel.config` sub-package contains the configuration items
    used throughout the package.
sids : NumPy array of ints
    The subject IDs of the HCP subjects used in the training datasets
"""


#-------------------------------------------------------------------------------
# Configuration
# We don't import the config items themselves, but we import the config module.
from . import config
# One exception: we do import the subject-IDs.
from .config import sids

#-------------------------------------------------------------------------------
# Utilities
# We import most of the utilities.
from .util   import (partition, partition_id, is_partition, trndata, valdata,
                     dice_loss, bce_loss, loss)

#-------------------------------------------------------------------------------
# Image-based Data and Model
# And we use the image-based datasets and networks.
from .image import (HCPLinesDataset, UNet, make_datasets, make_dataloaders)

#-------------------------------------------------------------------------------
# Model Training
# Finally, we import the relevant training functions.
from .train  import (train_model, build_model, run_modelplan)


#===============================================================================
# __all__
__all__ = ["config",
           "sids",
           "partition",
           "partition_id",
           "is_partition",
           "trndata",
           "valdata",
           "dice_loss",
           "bce_loss",
           "loss",
           "HCPVisualDataset",
           "make_datasets",
           "make_dataloaders",
           "UNet",
           "train_model",
           "build_model",
           "run_modelplan"]
