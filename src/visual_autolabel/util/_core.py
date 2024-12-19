# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/util/_core.py
# Implementation of general utilities for the visual_autolabel library.

#===============================================================================
# Constants / Globals

import os
from collections.abc import Mapping

import numpy as np
import torch

from ..config import (
    default_partition)


#===============================================================================
# Utility Functions

#-------------------------------------------------------------------------------
# Subject Partitions
# Code for dealing with partitions of training and validation subjects.

def is_partition(obj):
    """Returns true if the given object is a subject partition, otherwise False.

    `is_partition(x)` returns `True` if `x` is a mapping with the keys `'trn'`
    and `'val'` or is a tuple with 2 elements. Otherwise, returns `False`.

    Parameters
    ----------
    obj : object
        The object whose quality as a subject partition is to be determined.

    Returns
    -------
    boolean
        `True` if `obj` represents a subject partition and `False` otherwise.
    """
    return ((isinstance(obj, (tuple,list)) and len(obj) == 2) or
            (isinstance(obj, Mapping) and 'trn' in obj and 'val' in obj))
def trndata(obj):
    """Returns the training data of an object representing a subject partition.

    `trndata((trn_data, val_data))` returns `trn_data` (i.e., if given a tuple
    of length 2, `trndata` will return the first element).

    `trndata({'trn': trn_data, 'val': val_data})` also returns `trn_data`.

    See also: `valdata`

    Parameters
    ----------
    obj : mapping or tuple
        Either a dict-like object with the keys `'trn'` and `'val'` or a tuple
        with two elements `(trn, val)`.

    Returns
    -------
    object
        Either the first element of `obj` when `obj` is a tuple or `obj['trn']`
        when `obj` is a mapping.
    """
    if isinstance(obj, (tuple,list)):
        return obj[0]
    else:
        return obj['trn']
def valdata(obj):
    """Returns the validation data from a subject partition.

    `valdata((trn_data, val_data))` returns `val_data` (i.e., if given a tuple
    of length 2, `valdata` will return the second element).

    `valdata({'trn': trn_data, 'val': val_data})` also returns `val_data`.

    Parameters
    ----------
    obj : mapping or tuple
        Either a dict-like object with the keys `'trn'` and `'val'` or a tuple
        with two elements `(trn, val)`.

    Returns
    -------
    object
        Either the second element of `obj` when `obj` is a tuple or `obj['val']`
        when `obj` is a mapping.
    """
    if isinstance(obj, (tuple,list)):
        return obj[1]
    else:
        return obj['val']
def partition_id(obj):
    """Returns a string that uniquely represents a subject partition.

    Parameters
    ----------
    obj : tuple or mapping of a subject partition
        A mapping that contains the keys `'trn'` and `'val'` or a tuple with two
        elements, `(trn, val)`. Both `trn` and `val` must be either iterables of
        subject-ids, datasets with the attribute `sids`, or dataloaders whose
        datasets have th attribute `sids`.

    Returns
    -------
    str
        A hexadecimal string that uniquely represents the partition implied by
        the `obj` parameter.
    """
    from torch.utils.data import (DataLoader, Dataset)
    trndat = trndata(obj)
    valdat = valdata(obj)
    if isinstance(trndat, DataLoader): trndat = trndat.dataset
    if isinstance(valdat, DataLoader): valdat = valdat.dataset
    if isinstance(trndat, Dataset):    trndat = trndat.sids
    if isinstance(valdat, Dataset):    valdat = valdat.sids
    trn = [(sid,'1') for sid in obj[0]]
    val = [(sid,'0') for sid in obj[1]]
    sids = sorted(trn + val, key=lambda x:x[0])
    pid = int(''.join([x[1] for x in sids]), 2)
    return hex(pid)
def lookup_sids(dataset):
    """Returns a list of subject IDs for the given dataset name.

    The only argument, `dataset`, should be either `'hcp'` or `'nyu'`.
    """
    if dataset == 'hcp':
        from ..benson2025.config import hcp_sids
        return hcp_sids
    elif dataset == 'nyu':
        from ..benson2025.config import nyu_sids
        return nyu_sids
    else:
        raise ValueError("unrecognized dataset: {dataset}")
def partition(sids, how=default_partition):
    """Partitions a list of subject-IDs into a training and validation set.

    `partition(sids, (frac_trn, frac_val))` returns `(trn_sids, val_sids)` where
    the fraction `frac_trn` of the `sids` have been randomly placed in the
    training seet and `frac_val` of the subjects have been placed in the 
    validation set, randomly. The sum `frac_trn + frac_val` must be between 0
    and 1.

    `partition(sids, (num_trn, num_val))` where `num_trn` and `num_val` are both
    positive integers whose sum is less than or equal to `len(sids)` places
    exactly the number of subject-IDs, randomly, in each category.

    partition(sids, idstring)` where `idstring` is a hexadecimal string returned
    by `partition_id()` reproduces the original partition used to create the
    string.

    Parameters
    ----------
    sids : list-like
        A list, tuple, array, or iterable of subject identifiers. The
        identifiers may be numers or strings, but they must be sortable.
    how : tuple or str
        Either a tuple `(trn, val)` containing either the fraction of training
        and validation set members (`trn + val == 1`) or the (integer) 
        count of training and validation set members (`trn + val == len(sids)`),
        or a hexadecimal string created by `partition_id`.

    Returns
    -------
    tuple of arrays
        A tuple `(trn_sids, val_sids)` whose members are numpy arrays of the
        subject-IDs in the training and validation sets, respectively.
    """
    if isinstance(sids, str):
        sids = lookup_sids(sids)
    sids = np.asarray(sids)
    n = len(sids)
    if how is None: how = default_partition
    if isinstance(how, (tuple,list)):
        ntrn = trndata(how)
        nval = valdata(how)
        # If these are basic lists, numpy-ify them.
        if isinstance(ntrn, (tuple,list)): ntrn = np.asarray(ntrn)
        if isinstance(nval, (tuple,list)): nval = np.asarray(nval)
        # Otherwise, parse them.
        if isinstance(ntrn, float) and isinstance(nval, float):
            if ntrn < 0 or nval < 0: raise ValueError("trn and val must be > 0")
            nval = round(nval * n)
            ntrn = round(ntrn * n)
            tot = nval + ntrn
            if tot != n: raise ValueError("partition requires trn + val == 1")
        elif isinstance(ntrn, int) and isinstance(nval, int):
            if ntrn < 0 or nval < 0: raise ValueError("trn and val must be > 0")
            tot = ntrn + nval
            if tot != n: 
                raise ValueError("partition requires trn + val == len(sids)")
        elif isinstance(ntrn, np.ndarray) and isinstance(nval, np.ndarray):
            a1 = np.unique(sids)
            a2 = np.unique(np.concatenate([ntrn, nval]))
            #if np.array_equal(a1, a2) and len(a1) == len(sids):
            return (ntrn, nval)
            #else:
            #    raise ValueError("partitions must include all sids")
        else: raise ValueError("trn and val must both be integers or floats")
        val_sids = np.random.choice(sids, nval, replace=False)
        trn_sids = np.setdiff1d(sids, val_sids)
    elif isinstance(how, str):
        sids = np.sort(sids)
        how = int(how, 16)
        fmt = f'{{0:0{len(sids)}b}}'
        trn_ii = np.array([1 if s == '1' else 0 for s in fmt.format(how)],
                          dtype=bool)
        trn_sids = sids[trn_ii]
        val_sids = sids[~trn_ii]
    else:
        raise ValueError(f"invalid partition method: {how}")
    return (trn_sids, val_sids)

#-------------------------------------------------------------------------------
# Filters and PyTorch Modules
# Code for dealing with PyTorch filters and models.

def kernel_default_padding(kernel_size):
    """Returns an appropriate default padding for a kernel size.

    The returned size is `kernel_size // 2`, which will result in an output
    image the same size as the input image.

    Parameters
    ----------
    kernel_size : int or tuple of ints
        Either an integer kernel size or a tuple of `(rows, cols)`.

    Returns
    -------
    int
        If `kernel_size` is an integer, returns `kernel_size // 2`.
    tuple of ints
        If `kernel_size` is a 2-tuple of integers, returns
        `(kernel_size[0] // 2, kernel_size[1] // 2)`.
    """
    try:              return (kernel_size[0] // 2, kernel_size[1] // 2)
    except TypeError: return kernel // 2
def convrelu(in_channels, out_channels,
             kernel=3, padding=None, stride=1, bias=True, inplace=True):
    """Shortcut for creating a PyTorch 2D convolution followed by a ReLU.

    Parameters
    ----------
    in_channels : int
        The number of input channels in the convolution.
    out_channels : int
        The number of output channels in the convolution.
    kernel : int, optional
        The kernel size for the convolution (default: 3).
    padding : int or None, optional
        The padding size for the convolution; if `None` (the default), then
        chooses a padding size that attempts to maintain the image-size.
    stride : int, optional
        The stride to use in the convolution (default: 1).
    bias : boolean, optional
        Whether the convolution has a learnable bias (default: True).
    inplace : boolean, optional
        Whether to perform the ReLU operation in-place (default: True).

    Returns
    -------
    torch.nn.Sequential
        The model of a 2D-convolution followed by a ReLU operation.
    """
    if padding is None:
        padding = kernel_default_padding(kernel)
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel,
                        padding=padding, bias=bias),
        torch.nn.ReLU(inplace=inplace))
def convrelu3D(in_channels, out_channels,
             kernel=3,
             padding=None,
             stride=1,
             bias=True,
             inplace=True):
    """Shortcut for creating a PyTorch 3D convolution followed by a ReLU.

    Parameters
    ----------
    in_channels : int
        The number of input channels in the convolution.
    out_channels : int
        The number of output channels in the convolution.
    kernel : int, optional
        The kernel size for the convolution (default: 3).
    padding : int or None, optional
        The padding size for the convolution; if `None` (the default), then
        chooses a padding size that attempts to maintain the image-size.
    stride : int, optional
        The stride to use in the convolution (default: 1).
    bias : boolean, optional
        Whether the convolution has a learnable bias (default: True).
    inplace : boolean, optional
        Whether to perform the ReLU operation in-place (default: True).

    Returns
    -------
    torch.nn.Sequential
        The model of a 3D-convolution followed by a ReLU operation.
    """
    if padding is None:
        padding = kernel_default_padding(kernel)
    return torch.nn.Sequential(
        torch.nn.Conv3d(
            in_channels, out_channels, kernel,
            padding=padding,
            bias=bias),
        torch.nn.ReLU(inplace=inplace))

#-------------------------------------------------------------------------------
# Loss Functions

def is_logits(data):
    """Attempts to guess whether the given PyTorch tensor contains logits.

    If the argument `data` contains only values that are no less than 0 and no
    greater than 1, then `False` is returned; otherwise, `True` is returned.
    """
    if   (data > 1).any(): return True
    elif (data < 0).any(): return True
    else:                  return False
def dice_loss(pred, gold, logits=None, smoothing=1, graph=False, metrics=None):
    """Returns the loss based on the dice coefficient.
    
    `dice_loss(pred, gold)` returns the dice-coefficient loss between the
    tensors `pred` and `gold` which must be the same shape and which should
    represent probabilities. The first two dimensions of both `pred` and `gold`
    must represent the batch-size and the classes.

    Parameters
    ----------
    pred : tensor
        The predicted probabilities of each class.
    gold : tensor
        The gold-standard labels for each class.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    smoothing : number, optional
        The smoothing coefficient `s`. The default is `1`.
    metrics : dict or None, optional
        An optional dictionary into which the key `'dice'` should be inserted
        with the dice-loss as the value.

    Returns
    -------
    float
        The dice-coefficient loss of the prediction.
    """
    pred = pred.contiguous()
    gold = gold.contiguous()
    if logits is None: logits = is_logits(pred)
    if logits: pred = torch.sigmoid(pred)
    intersection = (pred * gold)
    pred = pred**2
    gold = gold**2
    while len(intersection.shape) > 2:
        intersection = intersection.sum(dim=-1)
        pred = pred.sum(dim=-1)
        gold = gold.sum(dim=-1)
    if smoothing is None: smoothing = 0
    loss = (1 - ((2 * intersection + smoothing) / (pred + gold + smoothing)))
    # Average the loss across classes then take the mean across batch elements.
    loss = loss.mean(dim=1).mean()
    if metrics is not None:
        if 'dice' not in metrics: metrics['dice'] = 0.0
        metrics['dice'] += loss.data.cpu().numpy() * gold.size(0)
    return loss
def bce_loss(pred, gold, logits=None, reweight=True, metrics=None):
    """Returns the loss based on the binary cross entropy.
    
    `bce_loss(pred, gold)` returns the binary cross entropy loss between the
    tensors `pred` and `gold` which must be the same shape and which should
    represent probabilities. The first two dimensions of both `pred` and `gold`
    must represent the batch-size and the classes.

    Parameters
    ----------
    pred : tensor
        The predicted probabilities of each class.
    gold : tensor
        The gold-standard labels for each class.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    reweight : boolean, optional
        Whether to reweight the classes by calculating the BCE for each class
        then calculating the mean across classes. If `False`, then the raw BCE
        across all pixels, classes, and batches is returned (the default).
    metrics : dict or None, optional
        An optional dictionary into which the key `'bce'` should be inserted
        with the dice-loss as the value.

    Returns
    -------
    float
        The binary cross entropy loss of the prediction.
    """
    if logits is None: logits = is_logits(pred)
    if logits: f = torch.nn.functional.binary_cross_entropy_with_logits
    else:      f = torch.nn.functional.binary_cross_entropy
    if reweight:
        n = pred.shape[-1] * pred.shape[-2] * pred.shape[0]
        r = 0
        for k in range(pred.shape[1]):
            (p,t) = (pred[:,[k]], gold[:,[k]])
            r += f(p, t) * (n - torch.sum(t)) / n
    else:
        r = f(pred, gold)
    if metrics is not None:
        if 'bce' not in metrics: metrics['bce'] = 0.0
        metrics['bce'] += r.data.cpu().numpy() * gold.size(0)
    return r
def loss(pred, gold,
         logits=True,
         bce_weight=0.5,
         smoothing=1,
         reweight=True,
         metrics=None):
    """Returns the weighted sum of dice-coefficient and BCE-based losses.

    `loss(pred, gold)` calculates the loss value between the given prediction
    and gold-standard labels, both of which must be the same shape and whose
    elements should represent probability values.

    Parameters
    ----------
    pred : tensor
        The predicted probabilities of each class.
    gold : tensor
        The gold-standard labels for each class.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    bce_weight : float, optional
        The weight to give the BCE-based loss; the weight for the 
        dice-coefficient loss is always `1 - bce_weight`. The default is `0.5`.
    reweight : boolean, optional
        Whether to reweight the classes by calculating the BCE for each class
        then calculating the mean across classes. If `False`, then the raw BCE
        across all pixels, classes, and batches is returned (the default).
    smoothing : number, optional
        The smoothing coefficient `s` to use with the dice-coefficient liss.
        The default is `1`.
    metrics : dict or None, optional
        An optional dictionary in which the keys `'bce'`, `'dice'`, and `'loss'`
        should be mapped to floating-point values representing the cumulative
        losses so far across samples in the epoch. The losses of this
        calculation are added to these values.

    Returns
    -------
    number
        The weighted sum of losses of the prediction.
    """
    if bce_weight < 0 or bce_weight > 1:
        raise ValueError("bce_weight must be between 0 and 1")
    else:
        dice_weight = 1 - bce_weight
    if logits is None: logits = is_logits(pred)
    bce  = bce_loss(pred, gold,
                    logits=logits,
                    reweight=reweight,
                    metrics=metrics)
    dice = dice_loss(pred, gold,
                     logits=logits,
                     smoothing=smoothing,
                     metrics=metrics)
    loss = bce * bce_weight + dice * dice_weight
    if metrics is not None:
        if 'loss' not in metrics: metrics['loss'] = 0.0
        metrics['loss'] += loss.data.cpu().numpy() * gold.size(0)
    return loss
# The dice_score* functions are like the dice_loss functions, but they operate
# on properties where each vertex has a label value and 0 indicates no label.
def dice_scores(trueprop, predprop, smooth=0, rtype=list, includezero=False):
    """Calculate the per-channel dice-score for a segmentation image.
    
    Returns a dice similarity coefficient for each label value in the `truprop`
    and `predprop` images. Unlike the `dice_loss` function, `dice_scores`
    requires a single vector of label values for each property, so it does not
    accept or understand probabilities as the loss functions do.

    Parameters
    ----------
    trueprop : iterable of labels
        The true property of labels against which the predicted property is
        compared. This must be an iterable of integer label values with zero
        rtpically indicating that no label was assigned.
    predprop : iterable of labels
        The predictede property of labels against which the true property is
        compared. This must be an iterable of integer label values with zero
        rtpically indicating that no label was assigned. The predicted and true
        label properties must have the same lengths.
    smooth : float, optional
        The smoothing factor to apply; 0 (the default) uses no smoothing.
    rtype : list or dict, optional
        The type of return value desired. This should be either the literal
        `list` or the literal `dict`.
    includezero : boolean, optional
        Whether to include the value 0 as a label or not. The default is
        `False`.

    Returns
    -------
    list
        If `rtype` is `list`, then a list of scores is returned. The scores are
        given in the same order as the values in `numpy.unique(trueprop)` with
        zero excluded depending on the `includezero` option.
    dict
        If `rtype` is `dict`, then a dictionary, whose keys are label values and
        whose values are the dice scores associated with each label, is
        returned.
    """
    if torch.is_tensor(trueprop):
        trueprop = trueprop.detach().numpy()
    else:
        trueprop = np.asarray(trueprop)
    if torch.is_tensor(predprop):
        predprop = predprop.detach().numpy()
    else:
        predprop = np.asarray(predprop)
    dice = [] if rtype is list else {}
    for ll in np.unique(trueprop):
        if ll == 0 and not includezero:
            continue
        tt = (trueprop == ll)
        pp = (predprop == ll)
        isect = np.sum(tt & pp)
        d = (2*isect + smooth) / (np.sum(tt) + np.sum(pp) + smooth)
        if rtype is list:
            dice.append(d)
        else:
            dice[ll] = d
    return dice
def dice_score(trueprop, predprop, smooth=0, includezero=False):
    """Calculates the mean-channel dice-score for a segmentation image.
    
    Returns the mean of the `dice_scores(truprop, predprop)`.

    Parameters
    ----------
    trueprop : iterable of labels
        The true property of labels against which the predicted property is
        compared. This must be an iterable of integer label values with zero
        rtpically indicating that no label was assigned.
    predprop : iterable of labels
        The predictede property of labels against which the true property is
        compared. This must be an iterable of integer label values with zero
        rtpically indicating that no label was assigned. The predicted and true
        label properties must have the same lengths.
    smooth : float, optional
        The smoothing factor to apply; 0 (the default) uses no smoothing.
    includezero : boolean, optional
        Whether to include the value 0 as a label or not. The default is
        `False`.

    Returns
    -------
    float
        The mean dice score across labels in the `trueprop`.
    """
    sc = dice_scores(trueprop, predprop, smooth=smooth, includezero=includezero)
    return np.mean(sc)

#-------------------------------------------------------------------------------
# HCP Utilities

def sectors_to_rings(visual_sector, visual_area):
    """Converts the `'visual_sectors'` property from neuropythy's `'hcp_lines'`
    dataset into a label property.

    The `'hcp_lines'` dataset uses eccentricity rings 0 through 4. The property
    returned from this function uses values 0 through 5 to represent the
    following:
     * 0: a vertex outside of the V1-V3 region;
     * 1: eccentricity ring 0 (0-0.5 deg);
     * 2: eccentricity ring 1 (0.5-1 deg);
     * 3: eccentricity ring 2 (1-2 deg);
     * 4: eccentricity ring 3 (2-4 deg);
     * 5: eccentricity ring 4 (4-7 deg).
    """
    v123 = np.isin(visual_area, (1,2,3))
    nonfov_sectors = tuple(k for k in range(2, 27) if k != 6)
    r = np.zeros(visual_sector.shape, dtype=int)
    e0 = v123 & ~np.isin(visual_sector, nonfov_sectors)
    e1 = np.isin(visual_sector, (2, 7, 11, 15, 19, 23))
    e2 = np.isin(visual_sector, (3, 8, 12, 16, 20, 24))
    e3 = np.isin(visual_sector, (4, 9, 13, 17, 21, 25))
    e4 = np.isin(visual_sector, (5, 10, 14, 18, 22, 26))
    r[e4] = 5
    r[e3] = 4
    r[e2] = 3
    r[e1] = 2
    r[e0] = 1
    return r

#-------------------------------------------------------------------------------
# Other Utilities

def autolog(filename, stdout=True, clear=False,
            mkdirs=True, mkdir_mode=0o775):
    """Returns a function that can be used as a logger for the given filename.
    
    `autolog(filename)` returns a function that acts like the `print()`
    function but that writes both to stdout and to the given file.
    
    Parameters
    ----------
    filename : str
        The name of the file to which to log any text.
    stdout : boolean, optional
        If `True` (the default), then writes any text to stdout as well as to
        the given file; otherwise, writes only to the file.
    clear : boolean, optional
        If `True`, then clears the file before returning; otherwise, the file
        is left as-is, and text is appended (the default).
    mkdirs : boolean, optional
        Whether to create cache directories that do not exist (default `True`).
    mkdir_mode : int, optional
        What mode to use when creating directories (default: `0o775`).
    """
    if mkdirs:
        (flpath,flnm) = os.path.split(filename)
        if not os.path.isdir(flpath):
            os.makedirs(flpath, mode=mkdir_mode, exist_ok=True)
    if clear and os.path.isfile(filename):
        # Truncate the file; don't remove it.
        with open(filename, 'w') as fl: pass
    def logfn(*args):
        with open(filename, 'a') as f:
            print(*args, file=f)
        if stdout:
            print(*args)
    return logfn
def centroid(a, weights=None):
    """Returns the centroid of `a`.
    
    `centroid(a)` returns the mean of the rows of `a`.
    
    `centroid(a, w)` returns `dot(a, w) / sum(w)`.
    """
    if weights is None:
        return np.mean(a, axis=1)
    else:
        total_weight = np.sum(weights)
        return np.dot(a, weights) / total_weight
def centroid_align_points(a, b, weights=None, out=None):
    """Aligns the centroid of matrix `a` to that of `b`.
    
    `centroid_align_points(a, b)` aligns the points in the matrix `a` to those
    in the matrix `b` by aligning their centroids.
    
    Parameters
    ----------
    a : matrix
        The matrix that is to be aligned to matrix `b`. The shape of a can be
        any shape `(d,n)` where `d` is the number of dimensions, and `n` is the
        number of points.
    b : matrix
        The matrix to which `a` is to be aligned. Must be the same shape as `a`.
    weights : vector or None, optional
        The weights or masses to use in calculating the center of mass. The
        default is `None`.
    out : matrix or None, optional
        Where to store the result. If `None` (the default), then a new array is
        returned. Otherwise, the result is placed in `out`.

    Returns
    -------
    matrix
        The matrix `a` aligned to `b`. If `inplace` is `True` and `a` is a numpy
        array, `a` is returned.
    """
    centroid_a = centroid(a, weights)
    centroid_b = centroid(b, weights)
    if out is None:
        out = np.array(a)
    elif out is not a:
        out = np.asarray(out)
    else:
        out = np.asarray(out)
        out[...] = a
    out += (centroid_b - centroid_a)[:,None]
    return out
def rotation_alignment_matrix(a, b, weights=None):
    """Returns the rotation matrix that, when applied to `a`, aligns `a` to `b`.
    
    `rotation_alignment_matrix(a, b)` returns the rotation matrix that aligns
    `a` to `b`; i.e. if `r = rotation_alignment_matrix(a, b)`, then `dot(r, a)`
    minimizes the difference between `a` and `b`.
    
    `rotation_alignment_matrix(a, b, w)` uses `w` as a weight matrix such that
    the return value minimizes the weighted difference between `a` and `b`.
    """
    # First calculate the covariance matrix.
    if weights is None:
        cov = np.dot(b, np.transpose(a))
    else:
        weights = np.asarray(weights)
        cov = np.dot(b * weights[None,:], np.transpose(a))
        cov /= np.sum(weights)
    # Next, calculate the singular value decomposition.
    (u,s,vt) = np.linalg.svd(cov, compute_uv=True)
    det_u = np.linalg.det(u)
    det_v = np.linalg.det(vt)
    d = np.eye(len(a), dtype=np.asarray(a).dtype)
    d[-1,-1] = np.sign(det_v * det_u)
    return np.dot(np.dot(u, d), vt)
def rotation_align_points(a, b, weights=None, out=None):
    """Aligns the centroid of matrix `a` to that of `b` using rotation.
    
    `rotation_align_points(a, b)` aligns the points in the matrix `a` to those
    in the matrix `b` by rotating `a`.
    
    Parameters
    ----------
    a : matrix
        The matrix that is to be aligned to matrix `b`. The shape of a can be
        any shape `(d,n)` where `d` is the number of dimensions, and `n` is the
        number of points.
    b : matrix
        The matrix to which `a` is to be aligned. Must be the same shape as `a`.
    weights : vector or None, optional
        The weights or masses to use in calculating the center of mass. The
        default is `None`.
    out : matrix or None, optional
        Where to store the result. If `None` (the default), then a new array is
        returned. Otherwise, the result is placed in `out`.

    Returns
    -------
    matrix
        The matrix `a` aligned to `b`. If `inplace` is `True` and `a` is a numpy
        array, `a` is returned.
    """
    rotation_matrix = rotation_alignment_matrix(a, b, weights=weights)
    if out is not None:
        out = np.ascontiguousarray(out)
    return np.dot(rotation_matrix.astype(out.dtype), a, out=out)
def rigid_align_points(a, b, weights=None, out=None):
    """Rigidly aligns the points in matrix `a` to the points in matrix `b`.
    
    `rigid_align_points(a, b)` aligns the points in the matrix `a` to those in
    the matrix `b` by first aligning the centroid of `a` to that of `b` then
    finding and applying the rotation matrix that minimizes the differences
    between `a` and `b` using the Kabsch-Umeyama algorithm.
    
    Parameters
    ----------
    a : matrix
        The matrix that is to be aligned to matrix `b`. The shape of a can be
        any shape `(d,n)` where `d` is the number of dimensions, and `n` is the
        number of points.
    b : matrix
        The matrix to which `a` is to be aligned. Must be the same shape as `a`.
    weights : vector or None, optional
        The weights or masses to use in calculating the center of mass and the
        covariance matrix. The default is `None`.
    out : matrix or None, optional
        Where to store the result. If `None` (the default), then a new array is
        returned. Otherwise, the result is placed in `out`.

    Returns
    -------
    matrix
        The matrix `a` aligned to `b`. If `inplace` is `True` and `a` is a numpy
        array, `a` is returned.
    """
    # First, find the centroids.
    centroid_a = centroid(a, weights=weights)[:,None]
    centroid_b = centroid(b, weights=weights)[:,None]
    if out is a:
        out = np.ascontiguousarray(a)
    elif out is None:
        out = np.array(a, order='C')
    else:
        out = np.ascontiguousarray(out)
        out[...] = a
    out -= centroid_a
    # Then do the rotation. Whether inplace was True or False, it is now okay to
    # write to a (because either inplace is True or because the 
    # centroid_align_points function returned a new matrix for us).
    out = rotation_align_points(out, b - centroid_b, weights=weights, out=out)
    # Recenter at b's centroid.
    out += centroid_b
    # That's it.
    return out
def rigid_align_cortices(hem1, hem2, surface='white'):
    """Rigidly aligns two cortical surfaces.
    
    `rigid_align_cortices(hem1, hem2, surf)` is roughly equivalent to the
    function rigid_align_points(s1.coordinates, s2.coordinates)` where `s1` and
    `s2` are `hem1.surface(surf)` and `hem2.surface(surf)` respectively. The
    return value is a surface.
    """
    # Get the two surfaces.
    surf1 = hem1.surface(surface)
    surf2 = hem2.surface(surface)
    # Conform surface1 to surface 2.
    surf1_as_surf2 = hem2.interpolate(hem1, surf2.coordinates)
    # Then align surface 1 with the conformed surface 1.
    surf1_on_surf2 = rigid_align_points(surf1.coordinates, surf1_as_surf2)
    # Return the surface with the new coordinates.
    return surf1.copy(coordinates=surf1_on_surf2)
def forkrun(f, *args, **kwargs):
    """Runs the given function in a child subprocess and returns the results.
    
    This function accepts a single callable objects followed by any number of
    arguments, all of which are passed to the callable. A process fork is
    first created in which the function is run, then pickle is used to pipe the
    function's return value back to the original process.
    
    This function exists primarily to work around memory leaks that occur as a
    result of libraries that cache data internally.
    """
    import os, sys, pickle
    (inp, outp) = os.pipe()
    os.set_inheritable(inp, True)
    os.set_inheritable(outp, True)
    inp = os.fdopen(inp, 'rb')
    outp = os.fdopen(outp, 'wb')
    pid = os.fork()
    if pid == 0:
        inp.close()
        try:
            r = f(*args, **kwargs)
            outp.write(pickle.dumps(r))
            outp.flush()
            outp.close()
            os._exit(0)
        except Exception as e:
            os._exit(1)
    else:
        outp.close()
        try:
            s = inp.read()
            (pid_, status) = os.waitpid(pid, 0)
            if not s:
                raise RuntimeError(f"nothing read from child process")
            rv = os.waitstatus_to_exitcode(status)
            if rv != 0:
                raise RuntimeError(f"child process returned value {rv}")
            obj = pickle.loads(s)
        finally:
            inp.close()
    return obj
def filter_options(fn, **opts):
    """Returns a dict of keyword arguments accepted by `fn` from `opts`."""
    from inspect import signature
    sig = signature(fn)
    return {k: opts[k] for (k,v) in sig.parameters.items() if k in opts}
