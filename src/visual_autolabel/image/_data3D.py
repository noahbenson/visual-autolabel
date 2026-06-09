# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/image/_data3D.py

#TODO: rewrite this comment; also this file needs to be made general for any
# 3D dataset and the HCP-specific code needs to go into a separate subpackage
# for the analyses.
"""The subpackage of visual_autolabel.image that manages the HCP data.

In order for the data in the visual_autolabel.image package to work, you must have configured
neuropythy either to know where the HCP dataset plus the HCP retinotopy dataset
reside locally on your filesystem or to be able to automatically download HCP
data. See the neuropythy wiki for more information:
https://github.com/noahbenson/neuropythy/wiki/Configuration
"""


# Dependencies #################################################################

from pathlib import Path
import os, sys, time

import numpy as np
import scipy as sp
import neuropythy as ny
import torch
import pimms


# Subject List #################################################################

sids = np.setdiff1d(
    np.sort(ny.data['hcp_lines'].subject_list),
    # Exclude subjects without DWI data.
    [150423, 186949, 239136])
_sids = sids

# Cache Path ###################################################################
# Where we will save/load the input and label data for each subject.
# This value is initially set here, but it is then imported by the core visual_autolabel.image
# package, and once this happens we will prefer the value found in visual_autolabel.image to the
# value here (hence this one is called data_cache_path_init).

dataset3D_cache_path_init = os.environ.get('DATASET3D_CACHE_PATH')
if dataset3D_cache_path_init:
    dataset3D_cache_path_init = Path(dataset3D_cache_path_init)

# We also want a default path for loading the noddi data.
noddi_data_path_init = os.environ.get('NODDI_DATA_PATH')
if noddi_data_path_init:
    noddi_data_path_init = Path(noddi_data_path_init)


# Functions for Loading Data ###################################################

def subject(sid):
    """Returns an HCP subject."""
    return ny.data['hcp_lines'].subjects[sid]
def uncache_data(sid, name, fn,
                 cache_path=None,
                 dtype=None,
                 device=None,
                 mkdir_mode=0o775):
    """Either loads a cachefile for the given data or calculates and
    caches the data, then returns it."""
    if cache_path is None:
        from visual_autolabel.image import dataset3D_cache_path as cache_path
    if cache_path is None:
        raise ValueError("no cache path given")
    cache_path = Path(cache_path)
    filepath = cache_path / name / f'{sid}.pt'
    if filepath.is_file():
        return torch.load(filepath, weights_only=True)
    data = torch.tensor(fn(), dtype=dtype, device=device)
    # Make sure the directory exists.
    filedir = filepath.parent
    if mkdir_mode is not None and not filedir.exists():
        filedir.mkdir(mkdir_mode, parents=True, exist_ok=True)
    torch.save(data, filepath)
    return data
def voxel_downsample(img):
    """Given a tensor, halves the size of the last three dimensions by averaging
    within each voxel and returns the new tensor.

    This function always downsamples the last three dimensions by a power of 2.
    If the last three dimensions are not even, an error will be raised.
    """
    return 0.125 * (
        img[..., 0::2, 0::2, 0::2] +
        img[..., 1::2, 0::2, 0::2] +
        img[..., 0::2, 1::2, 0::2] +
        img[..., 1::2, 1::2, 0::2] +
        img[..., 0::2, 0::2, 1::2] +
        img[..., 1::2, 0::2, 1::2] +
        img[..., 0::2, 1::2, 1::2] +
        img[..., 1::2, 1::2, 1::2])
def subimage(sid, imagename, dataobj=True):
    """Returns the 3D image object for a specific subject ID and image.
    
    `subimage(sid, 'T1')` returns the numpy array for the T1-weighted
    image for the subject whose subject-ID is `sid`.
    
    The optional argument `dataobj` can be set to `False` in order to
    return the Nifti1Image object instead of the numpy array.
    """
    sub = subject(sid)
    im = sub.images[imagename]
    if dataobj:
        return im.dataobj
    else:
        return im
def subgraymask(sid):
    """Returns a 3D image of a subject's gray-matter mask."""
    lh_mask = subimage(sid, 'lh_gray_mask')
    rh_mask = subimage(sid, 'rh_gray_mask')
    return (lh_mask | rh_mask)
def subwhitemask(sid):
    """Returns a 3D image of a subject's white-matter mask."""
    lh_mask = subimage(sid, 'lh_white_mask')
    rh_mask = subimage(sid, 'rh_white_mask')
    return (lh_mask | rh_mask)
def sublabels_v123(sid):
    # Because we neither want to calculate this three times in a row for the
    # same subject (this happens because the 'V1', 'V2', and 'V3' feature
    # entries in the subject_features dict below each require sublabels be
    # called for the subject) nor to keep a cache of all subjects for which
    # we've calculated this, we instead use a 1-back in-memory cache.
    if sublabels_v123.last_sid == sid:
        return sublabels_v123.last_image
    sub = subject(sid)
    template_im = ny.image_clear(sub.images['T1'])
    im = sub.cortex_to_image('visual_area', template_im, method='nearest')
    im = im.dataobj
    sublabels_v123.last_image = im
    sublabels_v123.last_sid = sid
    return im
sublabels_v123.last_sid = 0
sublabels_v123.last_image = None
def subdwi(sid, which, path=None, order=0):
    """Returns a DWI image for the given subject."""
    if path is None:
        # Use the default path.
        from visual_autolabel.image import noddi_data_path as path
    path = Path(path)
    path = path / str(sid)
    # What file?
    flnm = subdwi.filenames.get(which.lower())
    if not flnm:
        raise ValueError(f"unrecognized DWI scan type: {which}")
    # Load the image:
    flnm = path / flnm
    im0 = ny.load(str(flnm), to='image')
    # We want to resample these images to have the same shape and
    # resolution as the T1.
    im1 = subject(sid).images['T1']
    af0 = im0.affine
    af1 = im1.affine
    im = sp.ndimage.affine_transform(
        im0.dataobj, 
        np.linalg.inv(af0) @ af1,
        order=order,
        output_shape=im1.shape)
    im[np.isnan(im)] = 0
    return im
subdwi.filenames = {
    # The first three entries are the actual names of the files.
    'dwi_odi.nii': 'dwi_odi.nii',
    'dwi_ficvf.nii': 'dwi_ficvf.nii',
    'faMap.nii.gz': 'faMap.nii.gz',
    # The rest are aliases for these files.
    'dwi_odi': 'dwi_odi.nii',
    'odi': 'dwi_odi.nii',
    'dwi_ficvf': 'dwi_ficvf.nii',
    'ficvf': 'dwi_ficvf.nii',
    'faMap.nii': 'faMap.nii.gz',
    'faMap': 'faMap.nii.gz',
    'fa': 'faMap.nii.gz'}
def subprf(sid, name):
    """Loads and returns 3d images of the subject's PRF data.

    The first argument should be the HCP subject ID.
    The second argument should be the name of the kind of PRF data that is to be
    loaded: `'polar_angle'`, `'eccentricity'`, `'variance_explained'` or
    `'radius'`.
    """
    import neuropythy as ny
    # Get the subject from the HCP dataset:
    sub = ny.data['hcp_lines'].subjects[sid]
    # Get the correct property name:
    prop = 'prf_' + name
    # Convert the data from the cortical surfaces into a 3D image:
    template_im = ny.image_clear(sub.images['brain'])
    output_im = sub.cortex_to_image(prop, template_im)
    data = output_im.dataobj
    data[~np.isfinite(data)] = 0
    return np.array(output_im.dataobj)

def subpixelindex(sid, which):
    sub = subject(sid)
    template_im = ny.image_clear(sub.images['T1'])
    lh_prop = np.zeros(sub.hemis['lh'].vertex_count)
    rh_prop = np.zeros(sub.hemis['rh'].vertex_count)
    for (ii, h) in enumerate(('lh', 'rh')):
        hem = sub.hemis[h]
        fmap = ny.to_flatmap('occipital_pole', hem, radius=np.pi/2.25)
        (fx, fy) = fmap.coordinates
        fx = fx - np.min(fx)
        fx = fx / np.max(fx) * 512
        fy = fy - np.min(fy)
        fy = fy / np.max(fy) * 512
        if ii == 1:
            fx = fx + 512
        coord = fx if which == 'x' else fy
        full_prop = np.zeros(hem.vertex_count)
        full_prop[fmap.labels] = coord
        if h == 'lh':
            lh_prop = full_prop
        else:
            rh_prop = full_prop
    output_im = sub.cortex_to_image(
        (lh_prop, rh_prop),
        template_im)
    data = np.array(output_im.dataobj)
    data[~np.isfinite(data)] = 0
    return data

# This variable, subject_features, is a dictionary whose keys are the names
# of input or output features for the CNN. A feature is any 3D image (volume)
# that might be used for training or that might be predicted as a
# label/segmentation. The values in the dictionary are functions that can
# load the given volume given a subject ID.
subject_features = {
    'graymask':     lambda sid: subgraymask(sid),
    'whitemask':    lambda sid: subwhitemask(sid),
    'T1':           lambda sid: subimage(sid, 'T1'),
    'T2':           lambda sid: subimage(sid, 'T2'),
    'V1':           lambda sid: sublabels_v123(sid) == 1,
    'V2':           lambda sid: sublabels_v123(sid) == 2,
    'V3':           lambda sid: sublabels_v123(sid) == 3,
    'ODI':          lambda sid: subdwi(sid, 'odi'),
    'FICVF':        lambda sid: subdwi(sid, 'ficvf'),
    'FA':           lambda sid: subdwi(sid, 'fa'),
    'polar_angle':  lambda sid: subprf(sid, 'polar_angle'),
    'eccentricity': lambda sid: subprf(sid, 'eccentricity'),
    'radius':       lambda sid: subprf(sid, 'radius'),
    'cod':          lambda sid: subprf(sid, 'variance_explained'),
    'pixel_x':      lambda sid: subpixelindex(sid, 'x'),
    'pixel_y':      lambda sid: subpixelindex(sid, 'y')
}
def load_subject_data(sid, 
                      inputs=('graymask', 'T1', 'T2'),
                      outputs=('V1', 'V2', 'V3'),
                      cache_path=None,
                      dtype=None,
                      device=None,
                      mkdir_mode=0o775,
                      forget=True,
                      subindex=None,
                      zoom=None):
    """Given an HCP subject ID, returns CNN training data.
    
    The return value of this function is a tuple of 2 values; the
    first value is the input data and the second value is the label
    data for V1, V2, and V3.
    
    The features returned for the inputs and outputs can be changed
    using the `inputs` and `outputs` optional arguments. These must
    be lists of strings that name entries in the `subject_features`
    dictionary. The default values are `inputs=('graymask','T1','T2')`
    and `outputs=('V1','V2','V3')`.
    """
    kwargs = dict(
        cache_path=cache_path,
        dtype=dtype,
        device=device,
        mkdir_mode=mkdir_mode)
    # We need to start by uncaching the data or calculating it if it
    # hasn't already been cached.
    input_keys = inputs
    output_keys = outputs
    inputs = []
    outputs = []
    for (vals, val_keys) in [(inputs, input_keys), (outputs, output_keys)]:
        for key in val_keys:
            fn = subject_features.get(key)
            if fn is None:
                raise ValueError(f"unrecognized feature key: {key}")
            im = uncache_data(sid, key, lambda:fn(sid), **kwargs)
            vals.append(im)
    # We want the results to have a shape that matches pytorch's preferred
    # training input shape: (BATCH-SIZE, CHANNELS, ROWS, COLS, SLICES).
    # Because we are returning a single input, we don't include the batch-size.
    inputs = torch.stack(inputs).to(device=device, dtype=dtype)
    # After the above line, inputs will have a shape of:
    # channels x rows x cols x slices.
    outputs = torch.stack(outputs).to(device=device, dtype=dtype)
    # If requested (the default), we forget all HCP subjects to avoid memory
    # buildup. (This is largely a hack to remove the caching of the hcp_lines
    # dataset.)
    if forget:
        from pyrsistent import pmap
        subs = ny.data['hcp_lines'].subjects
        object.__setattr__(subs, '_memoized', pmap())
        ny.hcp.forget_all()
    if subindex is not None:
        ii = (slice(0,None),) + tuple(subindex)
        inputs = inputs[ii]
        outputs = outputs[ii]
    if zoom is not None and zoom > 0:
        while zoom < 1:
            inputs = voxel_downsample(inputs)
            outputs = voxel_downsample(outputs)
            zoom *= 2
        inputs = inputs.to(device=device, dtype=dtype)
        outputs = outputs.to(device=device, dtype=dtype)
        if zoom > 1:
            raise ValueError("zoom must be a negative power of 2")
    return (inputs, outputs)

class HCPDataset3D(torch.utils.data.Dataset):
    """A PyTorch Dataset object that manages the input and label images.
    
    The `MRImageDataset` class is an instance of the `torch.utils.data.Dataset`
    class that loads in the input 3D images and the V1-V2-V3 label images.
    For more information on Dataset objects, see:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    
    To use the dataset:
    
    >>> dataset = HCPVolumeDataset()

    >>> len(dataset)
    181

    >>> (input_images, label_images) = dataset[10]
    
    >>> input_images.shape
    torch.Size([3, 310, 280, 310])
    """
    sids = _sids
    def __init__(self,
                 sids=None,
                 inputs=('graymask', 'T1', 'T2'),
                 outputs=('V1', 'V2', 'V3'),
                 cache_path=None,
                 dtype=None,
                 device=None,
                 mkdir_mode=0o775,
                 subindex=(slice(2,-2), slice(8, 256+8), slice(2,-2)),
                 zoom=1/2):
        kw = dict(
            inputs=inputs,
            outputs=outputs,
            cache_path=cache_path,
            dtype=dtype,
            device=device,
            mkdir_mode=mkdir_mode,
            subindex=subindex,
            zoom=zoom)
        if sids is None:
            sids = HCPVolumeDataset.sids
        self.sids = np.sort(sids)
        self.data = pimms.lazy_map(
            {sid: ny.util.curry(lambda sid:load_subject_data(sid, **kw), sid)
             for sid in sids})
        self.input_images = pimms.lmap(
            {sid: ny.util.curry(lambda sid: self.data[sid][0], sid)
             for sid in sids})
        self.output_images = pimms.lmap(
            {sid: ny.util.curry(lambda sid: self.data[sid][1], sid)
             for sid in sids})
        self.input_features = inputs
        self.output_features = outputs
    def __len__(self):
        return len(self.sids)
    def __getitem__(self, k):
        sid = self.sids[k]
        return self.data[sid]
    def predict_index(self, ii, model):
        return self.predict(self.sids[ii], model)
    def predict(self, sid, model):
        """Predicts the visual area labels for the given subject id."""
        (inp, outp) = self.data[sid]
        return model(inp[None,...])
    def dice_index(self, ii, model, smoothing=1):
        return self.dice(self.sids[ii], model, smoothing=smoothing)
    def dice(self, sid, model, smoothing=1):
        """Returns the dice score (between 0 and 1) for the given subject id."""
        from . import dice_loss
        (feat, gold) = self.data[sid]
        pred = model(feat[None,...])
        loss = dice_loss(pred, gold, logits=model.logits, smoothing=smoothing)
        return 1.0 - loss

def make_dataloaders(sids=None,
                     batch_size=5,
                     shuffle=False,
                     partition=(0.8, 0.2),
                     **dataset_options):
    """Creates and returns a pair of training and validation dataloader objects.
    
    `make_dataloaders()` returns a pair of training and test (or validation)
    dataloader objects. The specific parameters of the underlying datasets and
    the dataloaders that are returned by them can be controlled by the optional
    named parameters.
    """
    if sids is None:
        from visual_autolabel.image import sids
    # If split is a 2-tuple of two lists of subject IDs (i.e., the test sids
    # and the training sids), then we already have the partition; otherwise,
    # we need to randomly make the partition.
    (trn, val) = partition
    if len(np.shape(trn)) == 1 and len(np.shape(val)) == 1:
        trn = np.sort(trn)
        val = np.sort(val)
    else:
        ntrn = int(np.floor(len(sids) * trn))
        nval = len(sids) - ntrn
        trn = np.random.choice(sids, ntrn, replace=False)
        val = np.setdiff1d(sids, trn)
    datasets = (
        HCPVolumeDataset(sids=trn, **dataset_options),
        HCPVolumeDataset(sids=val, **dataset_options))
    dataload_opts = dict(shuffle=shuffle, batch_size=batch_size)
    return (
        torch.utils.data.DataLoader(datasets[0], **dataload_opts),
        torch.utils.data.DataLoader(datasets[1], **dataload_opts))
