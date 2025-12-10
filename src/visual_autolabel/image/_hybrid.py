# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/image/_hybrid.py
# Submodule of the visual_autolabel package that handles hybrid 2D/3D
#  image-based CNNs/data.

#===============================================================================
# Dependencies

from pathlib import Path

import numpy as np
import scipy.sparse as sps
import torch
from torch.utils.data import Dataset

from ._data3D import HCPDataset3D


# Image3DTo2DDataset ####################################################

class Image3DTo2DDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset class that manages sparse matrices that transform data
    from a 3D MR Image to a 2D flattened image.
    """
    def __init__(self, sids, cache_path, memcache=True):
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_dir():
            raise RuntimeError(
                f"Given cache path is not a directory: {cache_path}")
        self.sids = sids
        if memcache:
            self.data = {}
        else:
            self.data = None
    def __len__(self):
        return len(self.sids)
    def __getitem__(self, k):
        sid = self.sids[k]
        if self.data is not None and sid in self.data:
            return self.data[sid]
        matrix = torch.load(self.cache_path / f"{sid}.pt", weights_only=False)
        (row, col, val) = sps.find(matrix)
        matrix = torch.sparse_coo_tensor(
            torch.as_tensor(np.array([row, col])),
            torch.as_tensor(val),
            matrix.shape,
            dtype=torch.float32)
        if self.data is not None:
            self.data[sid] = matrix
        return matrix


# HCPHybridDataset ############################################################

class HCPHybridDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sids,
                 inputs2D,
                 inputs3D,
                 outputs=('V1', 'V2', 'V3'),
                 cache_path_2D=None,
                 cache_path_3D=None,
                 transform_cache_path=None,
                 dtype=None,
                 device=None,
                 mkdir_mode=509,
                 subindex=(slice(2, -2, None), slice(8, 264, None), slice(2, -2, None)),
                 zoom=0.5):
        from ..benson2025.hcp import HCPDataset
        self.transform_dataset = Image3DTo2DDataset(
            sids,
            cache_path=transform_cache_path)
        self.dataset3D = HCPDataset3D(
            sids=sids,
            inputs=inputs3D,
            outputs=outputs,
            cache_path=cache_path_3D,
            dtype=dtype,
            device=device,
            mkdir_mode=mkdir_mode,
            subindex=subindex,
            zoom=zoom)
        self.dataset2D = HCPDataset(
            inputs2D,
            outputs,
            sids=sids,
            cache_path=cache_path_2D)
        self.sids = sids
    def __len__(self):
        return len(self.sids)
    def __getitem__(self, k):
        inputdata3D, _ = self.dataset3D[k]
        transformdata3D = self.transform_dataset[k]
        inputdata2D, outputdata2D = self.dataset2D[k]
        return (inputdata3D, transformdata3D, inputdata2D, outputdata2D)
