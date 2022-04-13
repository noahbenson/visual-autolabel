# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/_graph.py
# Training / validation data based on mesh graphs of cortex.

#===============================================================================
# Dependencies

#-------------------------------------------------------------------------------
# External Libries
import os, sys, time, copy, pimms PIL, cv2, warnings, torch
import numpy as np
import scipy as sp
import nibabel as nib
import pyrsistent as pyr
import neuropythy as ny

#-------------------------------------------------------------------------------
# Internal Tools
from .util import (partition_id, partition)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


#===============================================================================
# HCPAnatomyDataset
# The PyTorch dataset class for the HCP.

class HCPAnatomyDataset(Dataset):
    """Graph-based CNN Training/Test Dataset.
    """
    def __init__(self, sids, h, graph_hemi='LR59k_MSMSulc', cache_path=None,
                 surface='midgray', columns=None):
        torch.utils.data.Dataset.__init__(self)
        self.sids = np.array(sids)
        self.hemi = h
        self._cache = [None]*len(self.sids)
        self.cache_path = cache_path
        self.graph_hemi = graph_hemi
        self.surface = surface
        self.columns = columns
        self._graph = None
    def __len__(self):
        return len(self.sids)
    def __getitem__(self, k):
        c = self._cache[k] 
        if c is None:
            c = self.loadsub(self.sids[k], self.hemi,
                             cache_path=self.cache_path,
                             graph_hemi=self.graph_hemi,
                             surface=self.surface)
            c[~np.isfinite(c)] = 0
            (feat,tmp) = (c[:,:-3], c[:,-3:])
            lbl = np.zeros((tmp.shape[0], tmp.shape[1]+1))
            lbl[:,1:] = tmp
            lbl[:,0] = (np.sum(lbl, axis=1) == 0)
            if self.columns is not None:
                feat = feat[:,self.columns]
            c = (torch.tensor(feat.astype(np.float32)),
                 torch.tensor(lbl.astype(np.float32)))
            self._cache[k] = c
        return c
    def graph(self, sid=None):
        if self._graph is None:
            if sid is None: sid = self.sids[0]
            h = self.hemi
            sub = data.subjects[sid]
            hem = sub.hemis[h + '_' + self.graph_hemi]
            # Make the initial (uni-directed) graph.
            G = dgl.graph(tuple(surf.tess.edges))
            # And make it bidirectional.
            self._graph = dgl.to_bidirected(G)
        return self._graph
    @classmethod
    def loadsub(self, sid, h, cache_path=None, graph_hemi='LR59k_MSMSulc',
                surface='midgray'):
        if cache_path is not None:
            # Try to load it first
            flnm = os.path.join(cache_path, f'{sid}_{h}.mgz')
            if os.path.isfile(flnm):
                return ny.load(flnm)
        # Load and prep the data.
        sub = data.subjects[sid]
        hem = sub.hemis[h]
        # Convert polar angle and eccen into x/y
        ang = (90 - hem.prop('prf_polar_angle')) * np.pi/180
        ecc = hem.prop('prf_eccentricity')
        (x,y) = (ecc*np.cos(ang), ecc*np.sin(ang))
        vas = hem.prop('visual_area')
        ii = np.where(vas > 0)[0]
        lbls = np.zeros((hem.vertex_count, 3))
        lbls[(ii, vas[ii] - 1)] = 1
        feat = [hem.prop('curvature'),
                hem.prop('convexity'),
                hem.prop('thickness'),
                hem.prop('white_surface_area'),
                hem.prop('midgray_surface_area'),
                hem.prop('pial_surface_area'),
                x, y,
                hem.prop('prf_radius'),
                hem.prop('prf_variance_explained'),
                lbls[:,0], lbls[:,1], lbls[:,2]]
        # Convert over to the graph surface
        ghem = sub.hemis[h + '_' + graph_hemi]
        feat = hem.interpolate(ghem, feat)
        # Add in the surface coordinates.
        feat = np.vstack([ghem.surface(surface).coordinates, feat])
        feat = feat.T
        # If there's a cache path, save this.
        if cache_path is not None:
            ny.save(flnm, feat)
        # And return the data.
        return feat
