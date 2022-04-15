# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/_graph.py
# Training / validation data based on mesh graphs of cortex.

"""Graph-based convolutions for visual autolabeling.

This package currently stores old code that was started but not completed. At
one point during development, the code below worked (it is broadly similar to
the code in _image.py) and generated an approximate, but poor, prediction of the
V1-V3 labels. It does not currently work, but it is provided as an example of
how graph convolutions might work.

The code uses the dgl library, which performs the graph convolutions.
"""

#===============================================================================
# Dependencies

#-------------------------------------------------------------------------------
# External Libries
import os, sys, time, copy, pimms PIL, cv2, warnings, torch, dgl
import numpy as np,
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

class HCPVisualDataset(Dataset):
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

# Same thing for graph datasets.
def make_datasets(partition=None, cache_path=g_cache_path):
    if partition is None: partition = (val_sids, trn_sids)
    elif partition is Ellipsis: partition = partition_sids()
    (vs, ts) = partition
    def genfn(sids, h, sl):
        return (lambda:GHCPAnatomyDataset(sids, h,
                                          cache_path=cache_path,
                                          columns=sl))
    g_trn_sets = pyr.pmap(
        {h: pimms.lmap(
             {'anat': genfn(ts, h, slice(0, 6)),
              'func': genfn(ts, h, slice(6, None)),
              'both': genfn(ts, h, slice(0, None))})
         for h in ['lh','rh']})
    g_val_sets = pyr.pmap(
        {h: pimms.lmap(
             {'anat': genfn(vs, h, slice(0, 6)),
              'func': genfn(vs, h, slice(6, None)),
              'both': genfn(vs, h, slice(0, None))})
         for h in ['lh','rh']})
    g_datasets = pyr.m(
        lh={'trn': g_trn_sets['lh'], 'val': g_val_sets['lh']},
        rh={'trn': g_trn_sets['rh'], 'val': g_val_sets['rh']})
    return g_datasets

def make_dataloaders(h, dataset='func', datasets=g_datasets,
                       shuffle=True, batch_size=2, **kw):
    '''
    make_dataloaders() is equivalent to make_dataloaders() but for
      graph-based CNNs instead of image-based CNNs.
    '''
    return {
        k: DataLoader(sets[dataset], batch_size=batch_size, shuffle=shuffle)
        for (k,sets) in g_datasets[h].items()}


#===============================================================================
# The Graph-based Model Code

class MeshConv(torch.nn.Module):
    def __init__(self, graph, in_features,
                 out_features=None,
                 extra_features=None,
                 in_nodes=None,
                 out_nodes=None,
                 norm='right',
                 weight=Ellipsis,
                 allow_zero_in_degree=True,
                 relu=True,
                 dtype=None,
                 device=None):
        torch.nn.Module.__init__(self)
        # The graph we operate over:
        self.graph = graph
        # The number of input and output features, along with extra input
        # features, for the return sweep of the UNet structure.
        self.in_features = in_features
        self.extra_features = extra_features
        if out_features is None: out_features = in_features
        self.out_features = out_features
        # Weights, if given:
        if weight is Ellipsis:
            weight = graph.edata.get('weight', None)
        elif isinstance(weight, str):
            weight = graph.edata[weight]
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=dtype, device=device)
        self.weight = weight
        # Make the graph convolution:
        if extra_features is not None: in_features += extra_features
        self.conv = dgl.nn.GraphConv(in_features, out_features,
                                     norm=norm,
                                     allow_zero_in_degree=allow_zero_in_degree)
        # Count the input/output nodes if need-be, based on edge directions.
        (u,v) = graph.edges()
        if in_nodes is None:  in_nodes  = torch.max(u) + 1
        if out_nodes is None: out_nodes = torch.max(v) + 1
        # And save these numbers.
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        # Max nodes is the size of the matrix that we need to pass to the graph
        # convolution operator.
        self.max_nodes = max(in_nodes, out_nodes)
    def forward(self, x0, extra=None):
        (nnodes, nfeats) = x0.shape[-2:]
        # First, if extra was given, we need to add it to x0.
        if extra is None:
            if self.extra_features is not None:
                if nfeats != self.in_features + self.extra_features:
                    raise RuntimeError(
                        f"{self.extra_features} extra features required")
            x = x0
        else:
            if self.extra_features is None:
                raise RuntimeError(f"{extra.shape[0]} extra features given" +
                                   " to node that requires none")
            elif extra.shape[1] != self.extra_features:
                raise ValueError(f"{extra.shape[0]} extra features given but" +
                                 f" {self.extra_features} expected")
            x = torch.cat([x, extra], dim=1)
        # Extend x0 if needed by the transformation.
        n = self.max_nodes
        if n > x.shape[0]:
            zs = torch.zeros((n - x.shape[0],) + x.shape[1:])
            x = torch.cat([x, zs], dim=0)
        elif n < x0.shape[0]:
            warnings.warn(f"expected {n} nodes but received {x.shape[0]}")
            x = x[:n, :]
        # Do the transformation.
        x = self.conv(self.graph, x, edge_weight=self.weight)
        # Chop x down if need-be.
        n_out = self.out_nodes
        if n_out < x.shape[0]:
            x = x[:n_out, :]
        elif n_out != x.shape[0]:
            raise RuntimeError(f"output nodes ({x.shape[0]}) less than" +
                               f" expected ({n_out})")
        # Return x.
        return x
class GraphResidual(torch.nn.Module):
    def __init__(self, graph, in_features,
                 out_features=None,
                 norm='right',
                 weight=Ellipsis,
                 allow_zero_in_degree=True,
                 dtype=None,
                 device=None):
        torch.nn.Module.__init__(self)
        if out_features is None: out_features = in_features
        kw = dict(norm=norm, weight=weight, dtype=dtype, device=device,
                  allow_zero_in_degree=allow_zero_in_degree)
        # Basically, we want a mesh convolution, but we assume that the graph
        # is the same for input and output.
        self.conv1 = MeshConv(graph, in_features,
                              out_features=out_features,
                              **kw)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = MeshConv(graph, out_features,
                              out_features=out_features,
                              **kw)
        if in_features != out_features:
            nnodes = graph.num_nodes()
            self.graph_1x1 = dgl.graph((np.arange(nnodes), np.arange(nnodes)))
            self.conv_1x1 = MeshConv(self.graph_1x1, in_features,
                                     out_features=out_features,
                                     **kw)
        else:
            self.graph_1x1 = None
            self.conv_1x1 = None
        self.relu2 = torch.nn.ReLU()
    def forward(self, x0):
        x = self.conv1(x0)
        x = self.relu1(x)
        x = self.conv2(x)
        if self.conv_1x1 is not None:
            x0 = self.conv_1x1(x0)
        x += x0
        return self.relu2(x)
class UNet(torch.nn.Module):
    def __init__(self, h, in_classes, out_classes=4, res=59, align='MSMSulc',
                 iterations=8, depth=None, sample_sid=111312,
                 middle_strategy='residual',
                 dtype=None, device=None):
        torch.nn.Module.__init__(self)
        # Grab some external data for this model.
        sub = data.subjects[sample_sid]
        self.hemi = hemi = sub.hemis[f'{h}_LR{res}k_{align}']
        reg = hemi.registrations['fs_LR']
        rad = np.mean(np.sqrt(np.sum(reg.coordinates**2, axis=0)))
        n = hemi.vertex_count
        # We can go ahead and build the input/output graph.
        u = reg.tess.faces.flatten()
        v = np.roll(reg.tess.faces, -1, axis=0).flatten()
        self.io_graph = g0 = dgl.graph((u,v))
        # Now we make the stack of sampled graphs.
        (points, faces, fwds, invs) = graph_spheres(
            iterations=iterations, radius=rad,
            dtype=dtype, device=device)
        if depth is not None:
            fwds = fwds[:depth]
            invs = invs[:depth]
            faces = faces[:depth+1]
        # Convert all the faces into edges and then to bidirected graphs.
        graphs = [g0]
        for fs in faces:
            (a,b,c) = fs.T
            g = dgl.graph((torch.cat([a,b,c]),
                           torch.cat([b,c,a])))
            graphs.append(g)
        # We are basically doing a set of graph convolution steps.
        steps = []
        # We start with a convolution that samples from g0 (the input graph)
        # to the most dense sphere graph (made above).
        mesh = ny.mesh(faces[0].detach().numpy().T,
                       points.detach().numpy().T)
        addr = reg.address(mesh.coordinates)
        ii = np.arange(mesh.vertex_count)
        to_nodes = np.concatenate([ii,ii,ii])
        from_nodes = addr['faces'].flatten()
        wgts = addr['coordinates']
        wgts = np.vstack([wgts, [1 - np.sum(wgts, axis=0)]])
        wgts = torch.tensor(wgts.flatten(), dtype=dtype, device=device)
        g_init_resample = dgl.graph((from_nodes, to_nodes))
        g_init_resample.edata['weight'] = wgts
        start_layer = MeshConv(g_init_resample, in_classes,
                               out_features=in_classes,
                               dtype=dtype, device=device)
        # We also need the reverse--a layer to resample from the high-res
        # output graph to the original output graph.
        addr = mesh.address(reg.coordinates)
        ii = np.arange(reg.vertex_count)
        to_nodes = np.concatenate([ii,ii,ii])
        from_nodes = addr['faces'].flatten()
        wgts = addr['coordinates']
        wgts = np.vstack([wgts, [1 - np.sum(wgts, axis=0)]])
        wgts = torch.tensor(wgts.flatten(), dtype=dtype, device=device)
        g_final_resample = dgl.graph((from_nodes, to_nodes))
        g_final_resample.edata['weight'] = wgts
        final_layer = MeshConv(g_final_resample, out_classes,
                               out_features=out_classes,
                               dtype=dtype, device=device)
        # We now build the forward layers. Each of these is itself a
        # sequential block whose output needs to be saved for the inverse
        # upsampling steps.
        fwd_layers = []
        nstep = len(fwds)
        for (ii,fwd,g) in zip(range(nstep), fwds, graphs[2:]):
            infeat  = out_classes * 2**ii if ii > 0 else in_classes
            outfeat = out_classes * 2**(ii+1)
            # First, we do a resample step.
            fsamp = MeshConv(fwd, infeat,
                             out_features=outfeat,
                             dtype=dtype, device=device)
            # Then we do the residual step.
            res1 = GraphResidual(g, outfeat, outfeat,
                                 dtype=dtype, device=device)
            res2 = GraphResidual(g, outfeat, outfeat,
                                 dtype=dtype, device=device)
            # And max pooling
            seq = torch.nn.Sequential(fsamp, res1, res2)
            fwd_layers.append(seq)
        # And the inverse layers.
        inv_layers = []
        for (ii,inv,g) in zip(range(nstep), invs, graphs[1:]):
            infeat  = out_classes * 2**(ii+1)
            exfeat  = out_classes * 2**(ii+1)
            outfeat = out_classes * 2**ii
            # First, the upsampling step
            fsamp = MeshConv(inv, infeat,
                             out_features=outfeat,
                             extra_features=exfeat,
                             dtype=dtype, device=device)
            # Now, the residual step.
            res1 = GraphResidual(g, outfeat, outfeat,
                                 dtype=dtype, device=device)
            res2 = GraphResidual(g, outfeat, outfeat,
                                 dtype=dtype, device=device)
            seq = torch.nn.Sequential(fsamp, res1, res2)
            inv_layers.append(seq)
        # Finally, we have the middle layers.
        g_mid = graphs[-1]
        in_midfeats  = out_classes * 2**(nstep)
        out_midfeats = out_classes * 2**(nstep)
        mid_layers = torch.nn.Sequential(
            GraphResidual(graphs[-1], in_midfeats, in_midfeats,
                          dtype=dtype, device=device),
            GraphResidual(graphs[-1], in_midfeats, in_midfeats,
                          dtype=dtype, device=device),
            GraphResidual(graphs[-1], in_midfeats, out_midfeats,
                          dtype=dtype, device=device),
            GraphResidual(graphs[-1], out_midfeats, out_midfeats,
                          dtype=dtype, device=device))
        # Stack all these up into a single set of layers.
        self.start_layer = start_layer
        self.fwd_layers = tuple(fwd_layers)
        self.mid_layers = mid_layers
        self.inv_layers = tuple(reversed(inv_layers))
        self.final_layer = final_layer
        # Also save the various parameters.
        self.build_params = pyr.m(res=res, align=align,
                                  iterations=iterations,
                                  sample_sid=sample_sid,
                                  dtype=dtype, device=device)
    def forward(self, x):
        # We don't deal with batch sizes: just calculate them separately.
        if len(x.shape) == 3:
            return torch.cat([self.forward(u)[None,:,:] for u in x],
                             dim=0)
        # First, we run the starting layer.
        x = self.start_layer(x)
        # The forward filter; save the outputs as we go.
        dat = []
        for step in self.fwd_layers:
            x = step(x)
            dat.append(x)
        # Now the middle filters:
        x = self.mid_layers(x)
        # And the inverse layers:
        for (step,extra) in zip(self.inv_layers, reversed(dat)):
            x = torch.cat([x, extra], dim=1)
            x = step(x)
        x = self.final_layer(x)
        x = torch.sigmoid(x)
        x = x / torch.sum(x, dim=1)[:,None]
        return x
