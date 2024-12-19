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
from . import config

#-------------------------------------------------------------------------------
# Utilities
from . import util
from .util import (
    is_partition,
    trndata,
    valdata,
    partition_id,
    partition,
    is_logits,
    bce_loss,
    dice_loss,
    loss,
    dice_scores,
    dice_score,
    autolog)

#-------------------------------------------------------------------------------
# Image-based Data and Model
from . import image
from .image import (
    FlatmapFeature,
    LabelFeature,
    NullFeature,
    UNet)

#-------------------------------------------------------------------------------
# Model Training
from . import train
from .train import (
    train_model,
    build_model,
    run_modelplan,
    train_until,
    load_training)

#-------------------------------------------------------------------------------
# Plotting Utilities
from . import plot


#-------------------------------------------------------------------------------
# Tooling for the Benson et al. (2024) paper.
from . import benson2025


#===============================================================================
__all__ = [
    "partition",
    "partition_id",
    "is_partition",
    "trndata",
    "valdata",
    "dice_loss",
    "bce_loss",
    "loss",
    "HCPLinesDataset",
    "UNet",
    "train_model",
    "build_model",
    "run_modelplan",
    "train_until",
    "load_training",
    "plot",
    "benson2025"
]
