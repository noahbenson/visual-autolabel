class GeneralImageCache(BilateralFlatmapImageCache):
    
    def __init__(self, dataset_path, subject_list, hemis='lr', image_size=Ellipsis, cache_path=None, overwrite=False, mkdirs=True, mkdir_mode=0o775, multiproc=True, timeout=None, dtype='float32', memcache=True, normalization=None, features=None, flatmap_cache=True):
        super().__init__(hemis, image_size, cache_path, overwrite, mkdirs, mkdir_mode, multiproc, timeout, dtype, memcache, normalization, features, flatmap_cache)
        self.dataset_path = ny.util.pseudo_path(dataset_path)
        self.subject_list = subject_list
        self.subjects = None


    def load_subject(subj_id, path, max_eccen=12.4):
        # Freesurfer data path?
        subpp = path.subpath(f'derivatives/freesurfer/{subj_id}')
        sub = ny.freesurfer_subject(subpp)
        
        # Load data ? 
        prfpp = path.subpath(f'derivatives/prfanalyze-vista/{subj_id}/ses-nyu3t01/')
        labpp = path.subpath(f'derivatives/ROIs/{subj_id}/')
        
        for h in ['lh', 'rh']:
            hem = sub.hemis[h]

            prfs = dict(
                prf_polar_angle=ny.load(prfpp.local_path(f'{h}.angle_adj.mgz')),
                prf_eccentricity=ny.load(prfpp.local_path(f'{h}.eccen.mgz')),
                prf_variance_explained=ny.load(prfpp.local_path(f'{h}.vexpl.mgz')),
                prf_radius=ny.load(prfpp.local_path(f'{h}.sigma.mgz')),
                prf_x=ny.load(prfpp.local_path(f'{h}.x.mgz')),
                prf_y=ny.load(prfpp.local_path(f'{h}.y.mgz')),
                visual_area=ny.load(labpp.local_path(f'{h}.ROIs_V1-4.mgz')))
            prfs['prf_scaled_eccentricity'] = prfs['prf_eccentricity'] / max_eccen * 8
            prfs['prf_scaled_x'] = prfs['prf_eccentricity'] / max_eccen * 8
            prfs['prf_scaled_y'] = prfs['prf_eccentricity'] / max_eccen * 8
            hem = hem.with_prop(prfs)
            sub = sub.with_hemi({h: hem})
        return sub

    
    self.subjects = pimms.lazy_map({s: ny.util.curry(self.load_subject, s, self.dataset_path) for s in self.subject_list})

    @classmethod
    def builtin_features(cls):
        fs = {
            'prf_polar_angle': FlatmapFeature('prf_polar_angle', 'nearest'),
            'prf_eccentricity': FlatmapFeature('prf_eccentricity', 'linear'),
            'prf_cod': FlatmapFeature('prf_variance_explained', 'linear'),
            'prf_sigma': FlatmapFeature('prf_radius', 'linear'),
            'prf_x': FlatmapFeature('prf_x', 'linear'),
            'prf_y': FlatmapFeature('prf_y', 'linear'),
            'prf_scaled_eccentricity': FlatmapFeature('prf_scaled_eccentricity', 'linear'),
            'prf_scaled_x': FlatmapFeature('prf_scaled_x', 'linear'),
            'prf_scaled_y': FlatmapFeature('prf_scaled_y', 'linear'),
            'x': FlatmapFeature('midgray_x', 'linear'),
            'y': FlatmapFeature('midgray_y', 'linear'),
            'z': FlatmapFeature('midgray_z', 'linear'),
            'V1': LabelFeature('visual_area:1', 'nearest'),
            'V2': LabelFeature('visual_area:2', 'nearest'),
            'V3': LabelFeature('visual_area:3', 'nearest')
        }
        return dict(BilateralFlatmapImageCache.builtin_features(), **fs)
    def unpack_target(cls, target):
        if isinstance(target, Mapping):
            sid = target['subject']
        else:
            sid = target
        return (sid,)
    
    def cache_filename(self, target, feature):
        return os.path.join(feature, f"{target['subject']}.pt")
    
    def make_flatmap(self, target, view=None):
        (sid,) = self.unpack_target(target)
        if view is None:
            raise ValueError("GeneralImageCache requires a view")
        h = view['hemisphere']
        sub = self.subjects[sid]
        hem = sub.hemis[h]
        fsa = ny.freesurfer_subject('fsaverage')
        fsahem = fsa.hemis[h]
        midgray = rigid_align_cortices(hem, fsahem, 'midgray')
        (x, y, z) = midgray.coordinates
        convex = hem.prop('convexity') / 10.0
        hem = hem.with_prop(midgray_x=x, 
                            midgray_y=y, 
                            midgray_z=z, 
                            convexity=convex)
        fmap = ny.to_flatmap('occipital_pole', hem, radius=np.pi/2)
        fmap = fmap.with_meta(subject_id=sid, hemisphere=h)
        return fmap

    def fill_image(self, target, feature, im):
        super().fill_image(target, feature, im)
        im[torch.isnan(im)] = 0
        return im
    


class GeneralDataset(ImageCacheDataset):
    __slots__ = ()
    
    def __init__(self, dataset_path, inputs, 
                 outputs, subject_list=Ellipsis, 
                 sids=Ellipsis, image_size=Ellipsis, 
                 cache_path=Ellipsis, 
                 transform=None, 
                 input_transform=None, 
                 output_transform=None, 
                 hemis='lr', 
                 cache_image_size=Ellipsis, 
                 overwrite=False, 
                 mkdirs=True, mkdir_mode=0o775, 
                 multiproc=True, timeout=None, 
                 dtype='float32', memcache=True, 
                 normalization=None, 
                 features=None, 
                 flatmap_cache=True):
        

        if cache_path is Ellipsis:
            from ..config import dataset_cache_path
            if dataset_cache_path is not None:
                dataset_cache_path = os.path.join(dataset_cache_path, 'Data')
            cache_path = os.path.join(dataset_cache_path, 'Data')
        

        imcache = GeneralImageCache(
            dataset_path=dataset_path,
            subject_list=subject_list,
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
        

        if sids is Ellipsis:
            sids = subject_list
        

        targets = tuple({'subject': s} for s in sids)
        

        if isinstance(inputs, str):
            from ._core import input_properties as ps
            inputs = ps.get(inputs, (inputs,))
        if isinstance(outputs, str):
            from ._core import output_properties as ps
            outputs = ps.get(outputs, (outputs,))
        

        super().__init__(
            imcache, inputs, outputs, targets,
            image_size=image_size,
            transform=transform,
            input_transform=input_transform,
            output_transform=output_transform)
 
    