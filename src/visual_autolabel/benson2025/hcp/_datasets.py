# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2025/hcp/_datasets.py
# Code to manage the HCP-specific aspects of the AutoLabeler dataset.


#===============================================================================
# Dependencies

# External Libries -------------------------------------------------------------

import pyrsistent as pyr
import os,six,warnings,re
from collections.abc import (Sequence, Mapping)

import numpy as np
import neuropythy as ny
import pimms, torch
from torch.utils.data import (Dataset, DataLoader)

# Internal Tools ---------------------------------------------------------------

from ...util import (
    partition as make_partition,
    is_partition,
    trndata,
    valdata,
    convrelu
)
from ...image import (
    ImageCacheDataset,
    BilateralFlatmapImageCache,
    FlatmapFeature,
    LabelFeature,
    LabelDiffFeature,
    LabelUnionFeature,
    LabelIntersectFeature
)




#===============================================================================
# HCP Datasets
from neuropythy.datasets.hcp import (HCPMetaDataset, to_boolean)
from neuropythy.datasets.hcp_lines import HCPLinesDataset,mapsmerge
from neuropythy.datasets.core import add_dataset
from neuropythy.util import (config, curry, auto_dict, address_data, pseudo_path)

from ._core import (caonly_properties,vaonly_properties,daonly_properties,visual_area_neighbors,visual_area_label_key,input_properties,output_properties,subject_list,partition)


config.declare_dir('hcp_vd_lines_path',
                   environ_name='HCP_VD_LINES_PATH', default_value=None)
# TODO change back after osf
config.declare('hcp_vd_lines_auto_download', environ_name='HCP_VD_LINES_AUTO_DOWNLOAD',
               filter=to_boolean, default_value=False)



@pimms.immutable
class CVDLinesDataset(HCPMetaDataset):
    """
    Central Ventral Dorsal Lines Dataset
    NOTE:  an abstract class
    """

    # TODO
    osf_path = 'osf://gqnp8/'

    subject_list=subject_list

    anatomist2rater ={
        'A11':'R1',

        'A7': 'R3',
        'A8': 'R2',
        'A12':'R4',
        'A13':'R5',

        'A5': 'R6',
        'A6': 'R8',
        'A9': 'R7',
        'A10':'R9'
    }

    central_raters = ('A1', 'A2', 'A3', 'A4')
    ventral_raters = ('A7', 'A8', 'A11', 'A12', 'A13')
    dorsal_raters  = ('A5', 'A6', 'A9', 'A10', 'A11')
    region_raters  = {'central':central_raters,'ventral':ventral_raters, 'dorsal':dorsal_raters}

    _vd_raters=ventral_raters + dorsal_raters
    _properties=['labels']
    _vd=('ventral','dorsal')

    mean_anatomist_name = 'mean'
    mean_subject_name   = 999999
    mean_sampling_resolution = 500
    anatomist_list=tuple(sorted(set(ventral_raters + dorsal_raters)))
    full_anatomist_list = anatomist_list + (mean_anatomist_name,)

    def __init__(self,
                 cache_directory=Ellipsis, create_mode=0o775, create_directories=True,
                 metadata_path=None, genetic_path=None, behavioral_path=None, meta_data=None,):
        cdir = cache_directory
        self.cls=self.__class__

        # get source path
        if cdir is Ellipsis: cdir = config['hcp_vd_lines_path']

        # If we're configured for auto-downloading, do it!
        if config['hcp_vd_lines_auto_download']:
            self.source_path = self.cls.osf_path
        elif cdir is None:
            raise ValueError('No HCP VD lines path given and auto-download is disabled')
        elif not os.path.exists(cdir):
            raise ValueError('HCP VD lines path does not exist and auto-download is disabled')
        elif not os.path.isdir(cdir):
            raise ValueError('HCP VD lines path is not a directory')
        else:
            self.source_path = cdir
            cdir = None


        HCPMetaDataset.__init__(self, 'hcp_vd_lines',
                                metadata_path=metadata_path, genetic_path=genetic_path,
                                behavioral_path=behavioral_path,
                                cache_directory=cdir,
                                create_directories=create_directories,
                                create_mode=create_mode,
                                meta_data=meta_data,
                                cache_required=True)
    @pimms.value
    def raters_list(cls):
        return (cls.anatomist2rater[item] for item in cls.anatomist_list)

    @pimms.param
    def source_path(sp):
        '''
        hcplines.source_path is the source (input) path of the HCP-lines data.
        '''
        return sp

    @pimms.value
    def pseudo_path(cls,source_path, cache_directory):
        '''
        hcplines.pseudo_path is the pseudo-path object that handles the loading and caching of the
        HCP-lines raw data.
        '''
        pp = pseudo_path(source_path, cache_path=cache_directory)
        # If the source path is the known OSF path, we can drastically speed things up by manually
        # loading in the OSF tree.
        if source_path == cls.osf_path:
            try:
                tree = ny.load(os.path.join(ny.library_path(), 'data', 'hcp_vd_lines_osftree.json.gz'))
                object.__setattr__(pp._path_data.pathmod, 'osf_tree', tree)
            except:
                s = "Could not pre-load OSF-tree; initial loading of data from the OSF may be slow"
                warnings.warn(s)
        return pp


    @pimms.value
    def _cached_data(cls,pseudo_path):
        '''
        _cached_data is a pimms lazy-map of all the cached data that is found in the HCP-lines
        dataset. Any cached data that is not found is automatically generated and saved when
        requested.
        '''
        # see what anatomist directories are there
        anatomists = cls.full_anatomist_list
        #anatomists = tuple([anat for anat in anatomists if pseudo_path.find('labels', cls._r2a(anat)) is not None])
        # we just build up lazy-maps that load in the content as requested

        c={vd:
           {p:
            {a:
             {s: pimms.lazy_map(
              {h:
                 curry(cls.load_properties,pseudo_path,vd,a,s,h,p)
               for h in ('lh','rh')})
              for s in cls.subject_list}
             for a in anatomists}
            for p in cls._properties}
           for vd in cls._vd}
        if len(cls._vd)==1:
            c=c[cls._vd[0]]

        #c = {k:pyr.pmap(v) for p in c for (k,v) in six.iteritems(p)}

        return pimms.lazy_map(c)
    @classmethod
    def load_properties(cls,pseudo_path,vd, anat, sid,h, name):
        '''
        load_properties(pd, anat, sid, name) loads the properties cache for the given anatomist,
          subject, and data-name, which must be either 'labels' or 'distances'. If no cache data for
          the given anatomist and sid are found, then None is returned. The data is loaded from the
          given pseudo_path pd.
        '''
        rater=cls.anatomist2rater[anat]
        name = name.lower()
        if name[-1]=='s':
            lname=name[:-1]
        else:
            lname=name
        if name not in cls._properties:
            raise ValueError('Property name must be ' + ', '.join(cls._properties))
        pp = cls.cache_path(pseudo_path,name,rater, sid, '%s.%s_%s.mgz' % (h,vd,lname),
                                        create_directories=False)
        if pp is None or not os.path.isfile(pp): return None
        va=ny.load(pp)
        # XXX?
        vs=np.zeros_like(va)
        return {'visual_area':va,'visual_sector':vs}

    @classmethod
    def cache_path(cls,pseudo_path, *drs, **kw):
        '''
        cache_path(pd, dirparts...) is like os.path.join(dirparts...) except that it finds the given
          cache path inside the pseudo_path directory given by pd.

        The following optional arguments may be given:
          * create_directories (default: True) specifies whether directories should be created if
            they do not already exist.
          * create_mode (default 0o755) specifies the mode for creating directories.
          * prepend_cache_directory (default: True) indicates whether the function should
            automatically prepend the name of the normalized-data directory to the path.
        '''
        k = 0
        if 'create_directories' in kw:
            create_directories = kw['create_directories']
            k += 1
        else: create_directories = True
        if 'create_mode' in kw:
            create_mode = kw['create_mode']
            k += 1
        else: create_mode = 0o755
        if 'prepend_cache_directory' in kw:
            prepend_cache_directory = kw['prepend_cache_directory']
            k += 1
        else: prepend_cache_directory = True
        if k != len(kw): raise ValueError('Unrecognized keyword arguments given to cache_path')

        #if prepend_cache_directory: drs = (cls.normalized_directory_name,) + drs
        drs = [str(dd) for dd in drs]
        try: pth = pseudo_path.local_path(*drs)
        except Exception: pth = None
        if pth is None:
            pth = pseudo_path.local_cache_path(*drs)
            if create_directories:
                pdir = os.path.dirname(pth)
                if not os.path.isdir(pdir): os.makedirs(pdir, mode=create_mode)
        return pth


    @pimms.value
    def subject_labels(cls,_cached_data):
        '''
        subject_labels is a dict-structure of the labels granted to each subject; for visual areas,
        this is <anatomist>_visal_area or just visual_area for the mean, with 1, 2, and 3 labeled.
        For sectors, this is according the HCPLinesDataset.sector_labels and sector_label_index.
        '''

        return _cached_data.get('labels', {})

@pimms.immutable
class VentralLinesDataset(CVDLinesDataset):
    anatomist_list=CVDLinesDataset.ventral_raters
    full_anatomist_list = anatomist_list + (CVDLinesDataset.mean_anatomist_name,)
    _vd=('ventral',)

@pimms.immutable
class DorsalLinesDataset(CVDLinesDataset):
    anatomist_list=CVDLinesDataset.dorsal_raters
    full_anatomist_list = anatomist_list + (CVDLinesDataset.mean_anatomist_name,)
    _vd=('dorsal',)

add_dataset('hcp_cvd_lines',      lambda:CVDLinesDataset().persist())
add_dataset('hcp_ventral_lines', lambda:VentralLinesDataset().persist())
add_dataset('hcp_dorsal_lines',  lambda:DorsalLinesDataset().persist())

#===============================================================================
# The HCP Feature Cache

class HCPImageCache(BilateralFlatmapImageCache):
    """An ImageCache subclass that handles features of the HCP Occipital Pole.

    The `HCPImageCache` type is a simple overload of the `ImageCache` type that
    includes instructions for plotting most occipital pole features. The
    `target_id` parameters for the class's methods are always a tuple of
    `(subject_ID, hemi)` where `hemi` is either `'lh'` or `'rh'` or one of the
    other HCP hemispheres such as `'lh_LR32k'`.
    """
    # We include an init function so that we can handle default options like
    # cache_path.
    def __init__(self,
                 hemis='lr',
                 image_size=Ellipsis,
                 cache_path=Ellipsis,
                 overwrite=False,
                 mkdirs=True,
                 mkdir_mode=0o775,
                 multiproc=True,
                 timeout=None,
                 dtype='float32',
                 memcache=True,
                 normalization=None,
                 features=None,
                 flatmap_cache=True):
        # If we are given an Ellipsis for the cache_path, we import from config.
        if cache_path is Ellipsis:
            from ..config import dataset_cache_path
            if dataset_cache_path is not None:
                dataset_cache_path = os.path.join(dataset_cache_path, 'HCP')
            cache_path = dataset_cache_path
        BilateralFlatmapImageCache.__init__(
            self,
            hemis=hemis,
            image_size=image_size,
            cache_path=cache_path,
            overwrite=overwrite,
            mkdirs=mkdirs,
            mkdir_mode=mkdir_mode,
            multiproc=multiproc,
            timeout=timeout,
            dtype=dtype,
            memcache=memcache,
            normalization=normalization,
            features=features,
            flatmap_cache=flatmap_cache)
    # The featuers we know how to make.
    _builtin_features = {
        # Functional Features first.
        'prf_polar_angle':  FlatmapFeature('prf_polar_angle', 'nearest'),
        'prf_eccentricity': FlatmapFeature('prf_eccentricity', 'linear'),
        'prf_cod':          FlatmapFeature('prf_variance_explained', 'linear'),
        'prf_sigma':        FlatmapFeature('prf_radius', 'linear'),
        'prf_x':            FlatmapFeature('prf_x', 'linear'),
        'prf_y':            FlatmapFeature('prf_y', 'linear'),
        # The visual area and visual sector-based features.
        'V1':    LabelFeature('visual_area:1',   'nearest'),
        'V2':    LabelFeature('visual_area:2',   'nearest'),
        'V3':    LabelFeature('visual_area:3',   'nearest'),
        'SV1d0': LabelFeature('visual_sector:1', 'nearest'),
        'SV1d1': LabelFeature('visual_sector:2', 'nearest'),
        'SV1d2': LabelFeature('visual_sector:3', 'nearest'),
        'SV1d3': LabelFeature('visual_sector:4', 'nearest'),
        'SV1d4': LabelFeature('visual_sector:5', 'nearest'),
        'SV1v0': LabelFeature('visual_sector:6', 'nearest'),
        'SV1v1': LabelFeature('visual_sector:7', 'nearest'),
        'SV1v2': LabelFeature('visual_sector:8', 'nearest'),
        'SV1v3': LabelFeature('visual_sector:9', 'nearest'),
        'SV1v4': LabelFeature('visual_sector:10', 'nearest'),
        'SV2d1': LabelFeature('visual_sector:11', 'nearest'),
        'SV2d2': LabelFeature('visual_sector:12', 'nearest'),
        'SV2d3': LabelFeature('visual_sector:13', 'nearest'),
        'SV2d4': LabelFeature('visual_sector:14', 'nearest'),
        'SV2v1': LabelFeature('visual_sector:15', 'nearest'),
        'SV2v2': LabelFeature('visual_sector:16', 'nearest'),
        'SV2v3': LabelFeature('visual_sector:17', 'nearest'),
        'SV2v4': LabelFeature('visual_sector:18', 'nearest'),
        'SV3d1': LabelFeature('visual_sector:19', 'nearest'),
        'SV3d2': LabelFeature('visual_sector:20', 'nearest'),
        'SV3d3': LabelFeature('visual_sector:21', 'nearest'),
        'SV3d4': LabelFeature('visual_sector:22', 'nearest'),
        'SV3v1': LabelFeature('visual_sector:23', 'nearest'),
        'SV3v2': LabelFeature('visual_sector:24', 'nearest'),
        'SV3v3': LabelFeature('visual_sector:25', 'nearest'),
        'SV3v4': LabelFeature('visual_sector:26', 'nearest'),
        # These get a bit complex as they subtract or union pieces together.
        'SV1fov': LabelFeature('visual_sector:1 6'),
        'SV2fov': LabelDiffFeature(
            'visual_area:2--visual_sector:11 12 13 14 15 16 17 18',
            'nearest'),
        'SV3fov': LabelDiffFeature(
            'visual_area:3--visual_sector:19 20 21 22 23 24 25 26',
            'nearest'),
        # Ventral and Dorsal visual area labels.
        'hV4':  LabelFeature('visual_area:4',  'nearest'),
        'VO1':  LabelFeature('visual_area:5',  'nearest'),
        'VO2':  LabelFeature('visual_area:6',  'nearest'),
        'V3a':  LabelFeature('visual_area:7',  'nearest'),
        'V3b':  LabelFeature('visual_area:8',  'nearest'),
        'IPS0': LabelFeature('visual_area:9',  'nearest'),
        'LO1':  LabelFeature('visual_area:10', 'nearest'),
        # Eccentricity regions.
        'E0': LabelDiffFeature(
            'visual_area:1 2 3--visual_sector:2 3 4 5 7 8 9 10 11 12 13 14 15'
            ' 16 17 18 19 20 21 22 23 24 25 26',
            'nearest'),
        'E1': LabelFeature('visual_sector:2 7 11 15 19 23',  'nearest'),
        'E2': LabelFeature('visual_sector:3 8 12 16 20 24',  'nearest'),
        'E3': LabelFeature('visual_sector:4 9 13 17 21 25',  'nearest'),
        'E4': LabelFeature('visual_sector:5 10 14 18 22 26', 'nearest'),
        # Anatomical Features.
        'myelin': FlatmapFeature('myelin', 'linear'),
        # The vertex coordinates themselves; we add these in.
        'x': FlatmapFeature('midgray_x', 'linear'),
        'y': FlatmapFeature('midgray_y', 'linear'),
        'z': FlatmapFeature('midgray_z', 'linear')
    }
    @classmethod
    def builtin_features(cls):
        fs = cls._builtin_features
        return dict(BilateralFlatmapImageCache.builtin_features(), **fs)
    @classmethod
    def unpack_target(cls, target):
        if len(target) == 2:
            if isinstance(target, Mapping):
                rater = target['rater']
                sid = target['subject']
            else:
                (rater, sid) = target
        else:
            raise ValueError(
                f"target for {type(self)}.make_flatmap must be one of: "
                "(rater,sid), {'rater':rater, 'subject':sid}")
        return (rater, sid)

    def __getitem__(self, targ_feat):
        (target_id, feature_name) = targ_feat
        skey=tuple("S"+item for item in visual_area_label_key)
        if not (feature_name in visual_area_label_key or bool(re.fullmatch(r'E[0-9]+',feature_name)) or feature_name.startswith(skey)):
        #    target_id['rater']='A1'
        #if feature_name in ('myelin'):
            id=target_id.copy()
            id['rater']='A1'
        else:
            id=target_id
        return self.get(id, feature_name)


    def cache_filename(self, target, feature, view=None):
        rater = target['rater']
        subject = target['subject']

        #cr=CVDLinesDataset.central_raters
        #br=CVDLinesDataset.dorsal_raters + CVDLinesDataset.dorsal_raters
        #ba=vaonly_properties + daonly_properties
        #if feature in ba and rater in cr and rater not in br:
        #    #feature=visual_area_neighbors[feature]
        #    feature='V3'

        if view is not None:
            raise ValueError(f'{self.__class__}.cache_filename does not use `view`')
        return os.path.join(feature, f"{rater}_{subject}.pt")

    def make_flatmap(self, target, view=None):
        # We may have been given (rater, sid, h) or ((rater, sid), h):
        (rater, sid) = self.unpack_target(target)
        if view is None:
            raise ValueError("HCPImageCache requires a view")
        h = view['hemisphere']
        # Get the subject and hemi.
        sub = ny.data['hcp_lines'].subjects[sid]
        hem = sub.hemis[h]
        # Fix the properties now, if needed:
        (x,y,z) = hem.surface('midgray').coordinates
        hem = hem.with_prop(midgray_x=x, midgray_y=y, midgray_z=z)
        print(1)
        if rater is not None and rater != 'mean':
            # Get the appropriate data from the dataset.

            # Get the appropriate data from the dataset.
            #if rater in ('A1', 'A2', 'A3', 'A4'):
            #    dat = ny.data['hcp_lines'].subject_labels[rater][sid][h]
            #    va = dat['visual_area']
            #    vs = dat['visual_sector']
            #else:
            #    va = np.zeros(hem.vertex_count, dtype=int)
            #    vs = np.zeros(hem.vertex_count, dtype=int)
            ## We also need the ventral and dorsal labels.
            #extra_labels = self._get_vd_labels(rater, sid, h)
            #if extra_labels is not None:
            #    va = np.array(va)
            #    ii = (va == 0)
            #    va[ii] = extra_labels[ii]

            dat=[None,None,None]
            va = np.zeros((hem.vertex_count,3), dtype=int)
            vs = np.zeros((hem.vertex_count,3), dtype=int)
            for i,db in enumerate(('hcp_lines','hcp_ventral_lines','hcp_dorsal_lines')):
                if rater in ny.data[db].subject_labels:
                    dat = ny.data[db].subject_labels[rater][sid][h]
                    va[:,i]  = dat['visual_area']
                    vs[:,i]  = dat['visual_sector']

            # take minimum non-zero element
            min=np.where(va==0,np.inf,va)
            idx = np.where(np.isfinite(min).any(axis=1), min.argmin(axis=1), 0)

            va=va[np.arange(va.shape[0]),idx]
            vs=vs[np.arange(vs.shape[0]),idx]
            print(sum(va))

            hem = hem.with_prop(
                visual_area=va,
                visual_sector=vs)
        # Make the flatmap:
        fmap = ny.to_flatmap('occipital_pole', hem, radius=np.pi/2.25)
        # XXX if not rater?
        fmap = fmap.with_meta(subject_id=sid, rater=rater, hemisphere=h)
        return fmap
    # We overload fill_image so that we can call down then turn NaNs into 0s.
    def fill_image(self, target, feature, im):
        super().fill_image(target, feature, im)
        im[torch.isnan(im)] = 0
        return im

class HCPDataset(ImageCacheDataset):
    """A PyTorch Dataset object that encapsulates the HCP lines dataset.

    The `HCPDataset` is a PyTorch dataset for use with image-based models such
    as CNNs. The dataset may be configured to use any of a number of known
    features, including features based on the hand-drawn annotations. For a full
    list of possible features, check the `HCPImageCache` type and the results of
    the `HCPImageCache.builtin_features()` method.
    """
    __slots__ = ()


    @staticmethod
    def load_plan(plan_filename,exit_on_error=False):
        import sys,json
        try:
            with open(plan_filename, 'rt') as fl:
                plan = json.load(fl)
        except Exception as e:
            if exit_on_error:
                print(f"Error reading plan file ({plan_filename})\n{str(e)}",
                      file=sys.stderr)
                sys.exit(2)
            else:
                raise
        return plan

    @staticmethod
    def load_opts(opts_filename,exit_on_error=False,parse=True):
        import sys,json
        try:
            with open(opts_filename, 'rt') as fl:
                opts = json.load(fl)
        except Exception as e:
            if exit_on_error:
                print(f"Error reading options file ({opts_filename})\n{str(e)}",
                      file=sys.stderr)
                sys.exit(2)
            else:
                raise

        if parse:
            return HCPDataset.parse_opts(opts)
        else:
            return opts


    @staticmethod
    def parse_opts(opts):
        # pri_args: arguments obtained from elsewhere that receive priority

        # inputs
        inputs = opts.pop('inputs', None)
        if inputs is None:
            inputs = input_properties
            inputs = dict(inputs)
            del inputs['null']
        elif isinstance(inputs, str):
            if inputs == 'all':
                inputs = input_properties
            elif inputs in input_properties:
                inputs = {inputs: input_properties[inputs]}
            else:
                # Otherwise parse them and then leave them as-is! A dictionary or a
                # list could be provided.
                from ast import literal_eval
                try:
                    inputs = literal_eval(inputs)
                except ValueError:
                    pass
        # outptus
        if 'output' in opts or 'outputs' in opts:
            outputs = opts.pop('output', 'outputs')
        else:
            outputs = opts.pop('prediction', 'area')
        if isinstance(outputs, str):
            outputs = output_properties[outputs]
        # Check if the partition is set to use the default HCP partition.
        if opts.get('partition') == 'default':
            opts['partition'] = partition()

        return (inputs,outputs,opts)



    @staticmethod
    def from_file(opts_filename,exit_on_error=False,hcp_restricted_path=None,**kwargs):

        (inputs,outputs,opts)=HCPDataset.load_opts(opts_filename,exit_on_error=exit_on_error,hcp_restricted_path=hcp_restricted_path)

        to_rm=['until','base_model','model_cache_path','partition']
        opts = {k: v for k, v in opts.items() if k not in to_rm}

        to_mv={'dataset_cache_path':'cache_path'}
        for k,v in to_mv.items():
            if k in opts:
                opts[v]=opts.pop(k)

        opts = {k: v for k, v in opts.items() if k not in kwargs}
        return HCPDataset(inputs,outputs,**opts,**kwargs)



    def __init__(self, inputs, outputs,
                 raters=('A1', 'A2', 'A3', 'A4'),
                 sids=Ellipsis,
                 image_size=Ellipsis,
                 transform=None,
                 input_transform=None,
                 output_transform=None,
                 hemis='lr',
                 cache_image_size=Ellipsis,
                 cache_path=Ellipsis,
                 overwrite=False,
                 mkdirs=True,
                 mkdir_mode=0o775,
                 multiproc=True,
                 timeout=None,
                 dtype='float32',
                 memcache=True,
                 normalization=None,
                 features=None,
                 flatmap_cache=True):

        # XXX move to highest superclass?
        if multiproc != False:
            try:
                isjupytr = get_ipython().__class__.__name__ in ['ZMQInteractiveShell', 'TerminalInteractiveShell']
            except:
                isjupytr=False
            if isjupytr:
                if multiproc == True:
                    warnings.warn("Running multiproc=True in a jupyter notebook will cause processing to hang!")
                else:
                    multiproc=False


        # Make an HCP Occipital Image Cache object first.
        imcache = HCPImageCache(
            hemis=hemis,
            image_size=cache_image_size,
            cache_path=cache_path,
            overwrite=overwrite,
            mkdirs=mkdirs,
            mkdir_mode=mkdir_mode,
            multiproc=multiproc,
            timeout=timeout,
            dtype=dtype,
            memcache=memcache,
            normalization=normalization,
            features=features,
            flatmap_cache=flatmap_cache)

        dset = ny.data['hcp_lines']
        vddset = ny.data['hcp_cvd_lines']

        # Figure out the targets dicts of rater and subject.
        if sids is Ellipsis:
            from ..config import hcp_sids
            sids = hcp_sids

        if 'ventral' in raters:
            raters = tuple(x for x in raters if x != 'ventral')
            raters=raters + vddset.ventral_raters
        if 'dorsal' in raters:
            raters = tuple(x for x in raters if x != 'dorsal')
            raters=raters + vddset.dorsal_raters
        if 'central' in raters:
            raters = tuple(x for x in raters if x != 'central')
            raters=raters + vddset.central_raters


        # Figure out the exclusions next.
        # Step through these and process from (rater, sid, h) into (rater, sid)
        # when necessary.
        exclusions = dset.exclusions
        tmp = exclusions
        exclusions = set([])
        for excl in tmp:
            if isinstance(excl, tuple) and len(excl) == 1:
                excl = excl[0]
            if isinstance(excl, str):
                if excl in raters:
                    for s in subjects:  # XXX
                        exclusions.add((excl, s))
            elif isinstance(excl, int):
                if excl in subjects:  # XXX
                    for r in raters:
                        exclusions.add((r, excl))
            elif len(excl) == 3:
                (r,s,h) = excl
                exclusions.add((r,s))
            elif len(excl) == 2:
                exclusions.add(excl)
            else:
                raise ValueError(f"invalid exclusion: {excl}")

        # Make the target list.
        targets = tuple(
            [{'rater':r, 'subject':s}
             for r in raters for s in sids
             if (r,s) not in exclusions])

        # If we have been given an alias string for the inputs or outputs,
        # translate those now based on the table in _core.py.
        if isinstance(inputs, str):
            inputs = input_properties.get(inputs, (inputs,))
        if isinstance(outputs, str):
            outputs = output_properties.get(outputs, (outputs,))
        # Now go ahead and initialize our superclass using it.
        super().__init__(
            imcache, inputs, outputs, targets,
            image_size=image_size,
            transform=transform,
            input_transform=input_transform,
            output_transform=output_transform)

def make_datasets_from_file(opts_filename,exit_on_error=False,hcp_restricted_path=None,**kwargs):

    (inputs,outputs,opts)=HCPDataset.load_opts(opts_filename,exit_on_error=exit_on_error,hcp_restricted_path=hcp_restricted_path)

    to_rm=['until','base_model','model_cache_path']
    opts = {k: v for k, v in opts.items() if k not in to_rm}

    to_mv={'dataset_cache_path':'cache_path'}
    for k,v in to_mv.items():
        if k in opts:
            opts[v]=opts.pop(k)

    opts = {k: v for k, v in opts.items() if k not in kwargs}

    return make_datasets(inputs,outputs,**opts,**kwargs)


def make_datasets(in_features, out_features,
                  features=None,
                  partition=Ellipsis,
                  raters=Ellipsis,
                  sids=Ellipsis,
                  image_size=Ellipsis,
                  transform=None,
                  input_transform=None,
                  output_transform=None,
                  hemis='lr',
                  cache_image_size=Ellipsis,
                  cache_path=Ellipsis,
                  overwrite=False,
                  mkdirs=True,
                  mkdir_mode=0o775,
                  multiproc=True,
                  timeout=None,
                  dtype='float32',
                  memcache=True,
                  normalization=None,
                  flatmap_cache=True):
    """Returns a mapping of training and validation datasets.

    The mapping returned by `make_datasets()` contains, at the top level, the
    keys `'trn'` and `'val'` whose keys are the training and validation
    datasets, respectively. At the next level, the keys are `'anat'`, `'func'`,
    and `'both'` for the dataset input image type. The second level of maps are
    lazy.

    Parameters
    ----------
    features : 'func' or 'anat' or 'both' or None
        The type of input images that the dataset uses: functional data
        (`'func'`), anatomical data (`'anat'`), or both (`'both'`). If `None`
        (the default), then a mapping is returned with each input dataset type
        as values and with `'func'`, `'anat'`, and `'both'` as keys.
    sids : list-like, optional
        An iterable of subject-IDs to be included in the datasets. By default,
        the subject list `visual_autolabel.util.sids` is used.
    partition : partition-like
        How to make the partition of sujbect-IDs; the partition is made using
        `visual_autolabel.utils.partitoin(sids, how=partition)`.
    image_size : int, optional
        The width of the training images, in pixels (default: 512).
    cache_path : str or None, optional
        The path in which the dataset will be cached, or None if no cache is to
        be used (the default).

    Returns
    -------
    nested mapping of HCPDataset objects
        A nested dictionary structure whose values at the bottom are datasets
        for training and validation partitions and for anatomy, function, and
        both. If `features` is `None`, then the return value is equivalent to
        `{f: make_datasets(f) for f in ['anat', 'func', 'both']}`.
    """
    # If we are given an Ellipsis for the sids or cache_path, we import them
    # from the benson2025.config namespace.
    if raters is Ellipsis:
        raters=CVDLinesDataset.central_raters
    if cache_path is Ellipsis:
        from ..config import dataset_cache_path
        if dataset_cache_path is not None:
            dataset_cache_path = os.path.join(dataset_cache_path, 'HCP')
        cache_path = dataset_cache_path
    if sids is Ellipsis:
        from ..config import hcp_sids
        sids = hcp_sids
    if partition is Ellipsis:
        from ...config import default_partition
        partition = default_partition
    (trn_sids, val_sids) = make_partition(sids, how=partition)
    def curry_fn(sids):
        return lambda:HCPDataset(
            in_features, out_features,
            sids=sids,
            raters=raters,
            features=features,
            cache_path=cache_path,
            image_size=image_size,
            transform=transform,
            input_transform=input_transform,
            output_transform=output_transform,
            hemis=hemis,
            cache_image_size=cache_image_size,
            overwrite=overwrite,
            mkdirs=mkdirs,
            mkdir_mode=mkdir_mode,
            multiproc=multiproc,
            timeout=timeout,
            dtype=dtype,
            memcache=memcache,
            normalization=normalization,
            flatmap_cache=flatmap_cache)
    return pimms.lmap({'trn': curry_fn(trn_sids),
                       'val': curry_fn(val_sids)})
def make_dataloaders(in_features, out_features,
                     features=None,
                     partition=Ellipsis,
                     raters=Ellipsis,
                     sids=Ellipsis,
                     image_size=Ellipsis,
                     transform=None,
                     input_transform=None,
                     output_transform=None,
                     hemis='lr',
                     cache_image_size=Ellipsis,
                     cache_path=Ellipsis,
                     overwrite=False,
                     mkdirs=True,
                     mkdir_mode=0o775,
                     multiproc=True,
                     timeout=None,
                     dtype='float32',
                     memcache=True,
                     normalization=None,
                     flatmap_cache=True,
                     datasets=None,
                     shuffle=True,
                     batch_size=5):
    """Returns a pair of PyTorch dataloaders as a dictionary.

    `make_dataloaders('func')` returns training and validation dataloaders (in a
    dictionary whose keys are `'trn'` and `'val'`) for the functional data of
    HCP. The dataloaders and datasets can be modified with the optional
    arguments.

    Parameters
    ----------
    in_features : list-like of feature names
        A list or tuple of the feature names that are to be used as input.
    out_features : list-like of feature names
        A list or tuple of the feature names that are to be used as outputs.
    features : dict of features, optional
        A dictionary of features to be used when creating the datasets.
    sids : list-like, optional
        An iterable of subject-IDs to be included in the datasets. By default,
        the subject list `visual_autolabel.util.sids` is used.
    partition : partition-like, optional
        How to make the partition of sujbect-IDs; the partition is made using
        `visual_autolabel.utils.partitoin(sids, how=partition)`.
    image_size : int or None, optional
        The width of the training images, in pixels; if `None`, then 512 is
        used (default: `None`).
    cache_path : str or None, optional
        The path in which the dataset will be cached, or None if no cache is to
        be used (the default).
    datasets : None or mapping of datasets, optional
        A mapping of datasets that should be used. If the keys of this mapping
        are `'trn'` and `'val'` then all of the above arguments are ignored and
        these datasets are used for the dataloaders. Otherwise, if `features` is
        a key in `datasets`, then `datasets[features]` is used and the other
        options above are ignored. Otherwise, if `datasets` is `None` (the
        default), then the datasets are created using the above options.
    shuffle : boolean, optional
        Whether to shuffle the IDs when loading samples (default: `True`).
    batch_size : int, optional
        The batch size for samples from the dataloader (default: 5).

    Returns
    -------
    nested mapping of PyTorch DataLoader objects
        A nested dictionary structure whose values at the bottom are PyTorch
        data-loader objects for training and validation partitions and for
        anatomy, function, and both. If `features` is `None`, then the return
        value is equivalent to
        `{f: make_dataloader(f, **kw) for f in ['anat', 'func', 'both']}`.
    """

    if raters is Ellipsis:
        raters=CVDLinesDataset.central_raters
    # What were we given for datasets?
    if datasets is None:
        # We need to make the datasets using the other options.
        datasets = make_datasets(
            in_features, out_features,
            sids=sids,
            raters=raters,
            features=features,
            partition=partition,
            cache_path=cache_path,
            image_size=image_size,
            transform=transform,
            input_transform=input_transform,
            output_transform=output_transform,
            hemis=hemis,
            cache_image_size=cache_image_size,
            overwrite=overwrite,
            mkdirs=mkdirs,
            mkdir_mode=mkdir_mode,
            multiproc=multiproc,
            timeout=timeout,
            dtype=dtype,
            memcache=memcache,
            normalization=normalization,
            flatmap_cache=flatmap_cache)
    # At this point, datasets must have 'trn' and 'val' entries in order to be
    # valid, or it must be a 2-tuple.
    if not is_partition(datasets):
        raise ValueError("make_dataloaders(): provided datasets are not valid")
    # Okay, now we can make the data-loaders using these datasets.
    trn = trndata(datasets) # simple selection from list/dictionary
    val = valdata(datasets)
    return dict(
        trn=DataLoader(trn, batch_size=batch_size, shuffle=shuffle),
        val=DataLoader(val, batch_size=batch_size, shuffle=shuffle))
