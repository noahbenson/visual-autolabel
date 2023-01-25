# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/image/_data.py
# Code to manage the training / validation data based on images of cortex.

#===============================================================================
# Dependencies

# External Libries

import os, sys, time, copy, warnings
from collections import (namedtuple, Mapping, Sequence)

import numpy as np
import scipy as sp
import nibabel as nib
import pyrsistent as pyr
import neuropythy as ny
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch, pimms
from torch.utils.data import (Dataset, DataLoader)
from torchvision.transforms import Resize
from pathlib import Path

# Internal Tools
from ..config import (
    sids,
    default_partition,
    default_image_size,
    saved_image_size
)
from ..util import (
    partition_id,
    partition as make_partition,
    is_partition,
    trndata,
    valdata,
    convrelu
)


#===============================================================================
# Image Generation and Caching

ImageCacheOptions = namedtuple(
    'ImageCacheOptions',
    ('image_size', 
     'cache_path',
     'overwrite',
     'mkdirs',
     'mkdir_mode',
     'multiproc',
     'timeout',
     'dtype',
     'memcache'))
class ImageCache:
    """A class for managing a set of images that must be generated for a CNN.

    The `ImageCache` class manages the generation of feature and label images
    that are to be stacked into an input or output tensor for the training and
    evaluation of a PyTorch model such as a CNN.

    The intended use of `ImageCache` is to overload the class with the methods
    `fill_image()` and `cache_filename()` defined. Whenever a `ImageCache`
    object's `get(record)` method is called, it converts the passed `record`
    argument into a filename by calling the `cache_filename(record)`. If that
    filename is found then it is loaded and returned; otherwise the
    `fill_image(record, ...)` method is called and the resulting feature matrix
    is saved to the cache directory. If the `fill_image` method is not defined,
    then an image is generated using the `plot_image` method instead.

    The `ImageCache` type uses views to augment the way that targets get painted
    onto a training image. Views are most frequently used for specifying that
    both left and right hemispheres should be painted into the same image. The
    `views` parameter can be specified in a number of ways.  First, if `views`
    is a matrix (e.g., list of lists or tuple of tuples) whose cells are tuples
    of `(view_params_dict, (x0, y0, width, height))` where the
    `view_params_dict` is the dictionary that specifies how the view is related
    to the target (for example, `{'hemisphere': 'lh'}`) and where the values
    `(x0, y0, width, height)` specify the rectangular subset of the image that
    should be painted with this particular view. These values are always real
    numbers between 0 and 1 (i.e., scaled image coordinates). Alternately, a
    single dict may be given. In this case, each value of the dict must be a
    matrix of values, and all of the matrices must be the same shape. The image
    is split up into sub-images matching the matrices shape, where all the
    sub-images are (approximately) the same size; a view is established for each
    based on the cells in the value matrices. Additionally, `None` is a value
    value for the `views` if there is only one view, or if the view is entirely
    encapsulated in the `target` data.

    Parameters
    ----------
    image_size : int or 2-tuple of ints, optional
        The image size in pixels; if an integer is given, then that integer is
        used as the number of pixels for both the width and height of each
        feature image.  Otherwise, a tuple of `(width_px, height_px)` must be
        given. The default is 512.
    cache_path : str or path-like, optional
        The directory in which the local cache for the features should be
        loaded and/or stored when features are generated. If `None` is given,
        then no cache is used; this is the default.
    overwrite : bool, optional
        Whether the `ImageCache` object should overwrite existing feature
        files or not. The default value is `False`.
    mkdirs : bool, optional
        Whether the `ImageCache` object should create directories that do not
        exist in order to save cache data. If `True`, then the directories are
        automatically created, and if `False`, then errors are raised instead.
        The default is `True`.
    mkdir_mode : int, optional
        The mode used to create directories when needed; the default is `0o775`.
    multiproc : bool, optional
        Whether the feature generation should be separated by forking the image
        generation and caching code into another process. By default this is
        `False`, which means that the image generation is performed in the
        current process. The main reason for using the value `True` here is if
        the generation of images may require large amounts of memory that are
        not immediately freed (for example, if generating the images requires
        loading data that is then cached in-memory and not immediately
        released). This generally should not be necessary.
    timeout : positive real or None, optional
        If running the image generation in a background process (i.e., if
        `multiproc` is `True`), the `timeout` parameter is the amount of time
        in seconds to wait for the child process to finish before givine up. The
        default value `None` indicates that the method should wait forever.
    dtype : 'float32' or 'float64', optional
        The name of the PyTorch dtype to use. The default is `'float64'`.
    cache : bool, optional
        Whether to use an in-memory cache of the features in addition to the
        disk-based caching. The default is `True`.

    """
    __slots__ = ('options', 'cache', 'views')
    # utility methods for initializing a set of views.
    @staticmethod
    def _matrix_shape(obj):
        if not isinstance(obj, Sequence):
            raise ValueError("value must be a matrix but is not a sequence")
        if not all(isinstance(r, Sequence) for r in obj):
            raise ValueError("value must be a matrix but has non-sequence row")
        # We allow () to stand for a 0x0 matrix.
        if len(obj) == 0:
            return (0, 0)
        m = len(obj[0])
        if not all(len(r) == m for r in obj):
            raise ValueError("value must be a matrix but has ragged rows")
        else:
            return (len(obj), m)
    @staticmethod
    def _matrix_tile(n, m):
        xs = np.linspace(0,1,m+1)
        ys = np.linspace(0,1,n+1)
        (xs,ys) = np.meshgrid(xs, ys)
        x0s = xs[:-1, :-1]
        y0s = ys[:-1, :-1]
        ws = xs[:-1, 1:] - xs[:-1, :-1]
        hs = ys[1:, :-1] - ys[:-1, :-1]
        return (x0s, y0s, ws, hs)
    @classmethod
    def _init_views(cls, views):
        if views is None:
            return None
        elif isinstance(views, Mapping):
            sh = None
            for (k,v) in views.items():
                vsh = ImageCache._matrix_shape(v)
                if sh is None: sh = vsh
                elif sh != vsh:
                    raise ValueError("views values must have equal shapes")
            if sh is None: # views == {}
                return None
            (n,m) = sh
            # Setup the (evenly-spaced) rectangles.
            (x0s, y0s, ws, hs) = ImageCache._matrix_tile(n, m)
            res = []
            for ri in range(n):
                for ci in range(m):
                    d = {k: v[ri][ci] for (k,v) in views.items()}
                    r = (x0s[ri,ci], y0s[ri,ci], ws[ri,ci], hs[ri,ci])
                    res.append((d, r))
            return res
        if isinstance(views, Sequence):
            (n,m) = ImageCache._matrix_shape(views)
            if n == 0 or m == 0:
                return None
            # We could be given dicts at each point or tuples of (dict,rect)
            # at each point. 
            if (m == 2 and
                all(isinstance(r[0], Mapping) for r in views) and
                all(isinstance(r[1], Sequence) for r in views) and 
                all(len(r[1]) == 4 for r in views)):
                # We have a list of (view, rect) specifications alread.
                return views
            elif all(all(isinstance(u, Mapping) for u in r) for r in views):
                # We have been given a matrix of views that should be evenly
                # tiled over the image.
                (x0s, y0s, ws, hs) = ImageCache._matrix_tile(n, m)
                res = []
                for ri in range(n):
                    for ci in range(m):
                        rect = (x0s[ri,ci], y0s[ri,ci], ws[ri,ci], hs[ri,ci])
                        view = views[ri][ci]
                        res.append((view, rect))
                return res
            else:
                raise ValueError(
                    "views must be a list of (dict,(x0,y0,w,h))"
                    " or a matrix of dicts")
        else:
            raise ValueError("views must be a mapping or matrix")
    # Construction.
    def __init__(self,
                 views=None,
                 image_size=Ellipsis,
                 cache_path=None,
                 overwrite=False,
                 mkdirs=True,
                 mkdir_mode=0o775,
                 multiproc=False,
                 timeout=None,
                 dtype='float32',
                 memcache=True):
        self.views = self._init_views(views)
        # The arguments all go into the options namedtuple.
        if image_size is Ellipsis:
            from ..config import saved_image_size as image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            image_size = tuple(image_size)
            if len(image_size) == 1:
                image_size = image_size * 2
        if len(image_size) != 2:
            raise TypeError("image_size must be an int or 2-tuple")
        if not all(isinstance(sz, int) and sz > 0 for sz in image_size):
            raise TypeError("image_size must contain positive integers")
        if cache_path is not None:
            cache_path = Path(cache_path)
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean value")
        if not isinstance(mkdirs, bool):
            raise TypeError("mkdirs must be a boolean value")
        if not isinstance(mkdir_mode, int):
            raise TypeError("mkdir_mode must be an integer")
        if not isinstance(multiproc, bool):
            raise TypeError("multiproc must be a boolean value")
        if timeout is not None:
            timeout = float(timeout)
            if timeout <= 0:
                raise ValueError("timeout must be postiive or None")
        if not isinstance(memcache, bool):
            raise ValueError("memcache must be True or False")
        self.options = ImageCacheOptions(image_size=image_size,
                                         cache_path=cache_path,
                                         overwrite=overwrite,
                                         mkdirs=mkdirs,
                                         mkdir_mode=mkdir_mode,
                                         multiproc=multiproc,
                                         timeout=timeout,
                                         dtype=dtype,
                                         memcache=memcache)
        # Now, initialize our cache (which is initially empty).
        self.cache = {}
    # The methods that are intended to be overloaded.
    def plot_image(self, target, feature, axes, view=None):
        """Plots a feature in grayscale on the given axes.

        The `plot_image` method must be overloaded by subclasses of the
        `ImageCache` type. The method is always passed the target ID and
        feature name of the feature that is to be generated and a set of
        matplotlib axes on which the feature should be plotted.

        The `plot_image` method is an alternative to the `make_feature`
        method. In a given subclass of `ImageCache`, only one must be
        implemented for a given target ID and feature name; if one method raises
        a `NotImplementedError` then the other method is attempted.
        """
        #TODO: Either delete this method/interface or make it work with views.
        raise NotImplementedError(
            f"plot_image not implemented for type {type(self)}")
    def fill_view(self, target, feature, image, view):
        """Paints the given feature and view into the given image array.

        The `fill_view` or method must be overloaded by subclasses of the
        `ImageCache` type that use views to manage multi-panel or multi-part
        images. The method is always passed the target ID, the feature name of
        the feature that is to be generated, the view for which is being filled,
        and a 3D PyTorch tensor corresponding to that view into which the image
        should be painted. The tensor has a size of `(rows, cols)`, which is
        always the requested image size mediated by the `views` class parameter.

        If the `fill_view` method is not overloaded, then the `fill_image`
        method must be overloaded instead.
        """
        raise NotImplementedError(
            f"fill_view not implemented for type {type(self)}")
    @staticmethod
    def view_slices(imshape, viewrect):
        """Returns `(row_slice, col_slice)` for the given image shape tuple and
        view rectangle specification.
        """
        from math import floor, ceil
        (x0, y0, w, h) = viewrect
        (rs, cs) = imshape
        cii = slice(floor(x0*cs), floor((x0 + w)*cs))
        rii = slice(rs - floor((y0 + h)*rs), rs - floor(y0*rs))
        return (rii, cii)
    @staticmethod
    def view_subimage(im, viewrect, rcfirst=True):
        """Extracts a subimage from an image matrix and returns it.

        `view_subimage(im, (x0, y0, w, h))` extracts the sub-image of the image
        matrix `im` according to the rectangle specification `(x0, y0, w, h)`.
        All the values in the rectangle specification must be between 0 and 1
        (i.e., they are specified in scaled image coordinates).
        """
        if rcfirst:
            (rii, cii) = ImageCache.view_slices(np.shape(im)[:2], viewrect)
            return im[rii, cii]
        else:
            (rii, cii) = ImageCache.view_slices(np.shape(im)[-2:], viewrect)
            return im[..., rii, cii]
    def view_rectangle(self, view):
        """Returns the rectangle spec for the view with the given meta-data.

        Views are specified as a comination of meta-data specifying the view and
        a sub-image rectangle. This converts the former into the latter. If no
        such view can be found, `None` is returned.
        """
        for (spec, rect) in self.views:
            if spec == view:
                return rect
        return None
    def image_size(self, view=None):
        """Returns the size of an image as used by the cache.

        This method is used whenever there is a view that may change the size of
        the image used for a specific flatmap/view. If the optional `view`
        parameter is either `None` or not provided, then the
        `options.image_size` is returned; otherwise, the size of the specific
        view is returned.
        """
        imshape = self.options.image_size
        if view is None:
            return imshape
        if self.views is None:
            raise RuntimeError("view image_size requested but views is None")
        # The view must be a viewdict, so we look for it.
        for (spec, rect) in self.views:
            if spec == view:
                (rii, cii) = ImageCache.view_slices(imshape, rect)
                rii = range(*rii.indices(imshape[0]))
                cii = range(*cii.indices(imshape[1]))
                return (len(rii), len(cii))
        # If we reach this point, the view was not found!
        raise ValueError(f"given view was not found: {view}")
    def fill_image(self, target, feature, im):
        """Paints the given feature into the given image array.

        The `fill_image` method must be overloaded by subclasses of the
        `ImageCache` type that do not use views to manage their images. The
        method is always passed the target ID and feature name of the feature
        that is to be generated and a 3D PyTorch tensor into which the image
        should be painted. The tensor has a size of `(rows, cols)`, which is
        always the class's `image_size` parameter. If the class uses views, then
        the `fill_view` method should be overloaded instead.

        The `fill_image` method is an alternative to the `plot_image` method. In
        a given subclass of `ImageCache`, only one must be implemented for a
        given target ID and feature name; if one method raises a
        `NotImplementedError` then the other method is attempted.

        The primary advantage of implementing the `fill_image` method directly
        instead of `plot_image` is that it allows one to give the pixels values
        more precise than 0-255.  If one uses `plot_image` then the resulting
        image is rendered in RGB, and since the feature images must individually
        be grayscale, this clips the pixel values to the range 0-255.
        """
        # If there are no views, then this is straightforward:
        if self.views is None:
            self.fill_view(target, feature, im, None)
        else:
            # If there are views, we make this a bit more complicated.
            for (spec, rect) in self.views:
                subim = ImageCache.view_subimage(im, rect)
                self.fill_view(target, feature, subim, spec)
        return None
    def cache_filename(self, target, feature):
        """Returns a relative filename for the appropriate cache file.

        The `cache_filename` method must be overloaded by subclasses of the
        `ImageCache` type. The method is always passed a target ID and a
        feature name, and it must return a unique cache filename for the given
        target and feature image. The returned filename should be a relative
        filename, and the `cache_path` of the `ImageCache` object is
        automatically prepended to the cache filename prior to reading or
        writing the cache.
        """
        raise TypeError(f"cache_filename not implemented for type {type(self)}")
    # The private methods.
    def _cachepath(self, target, feature, ending=".pt"):
        filename = self.cache_filename(target, feature)
        if ending is not None and not filename.lower().endswith(ending):
            from warnings import warn
            warn(f"filename does not end with '{ending}'; appending it")
            filename = filename + ending
        return filename
    def _dtype(self):
        "Returns the PyTorch dtype object."
        return getattr(torch, self.options.dtype)
    def _generate_feature(self, target_id, feature_name, filename,
                          return_image=False):
        """Private static method to actually generate a feature."""
        # Fairly straighforward approach here. Generate the figure, save it as
        # an image to the path filename. We'll put the image-data into this
        # tensor:
        im = torch.zeros(self.options.image_size, dtype=self._dtype())
        # First try to make the image directly, without using pyplot.
        try:
            r = self.fill_image(target_id, feature_name, im)
            # We allow a NotImplemented return val instead of raising the error.
            okay = r is not NotImplemented
        except NotImplementedError:
            okay = False
        if not okay:
            # The fill_image approach failed, so we need to use pyplot and the
            # plot_figure method.
            import matplitlib.pyplot as plt
            # Start by making a figure and axes for the plots.
            figsize = tuple(px / 72.0 for px in self.options.image_size)
            dpi = 72
            (fig,ax) = plt.subplots(1,1, figsize=figsize, dpi=dpi)
            # Run the method that actually draws the figure.
            self.plot_image(target_id, feature_name, ax)
            # Tidy things up for image saving.
            ax.axis('off')
            fig.subplots_adjust(0,0,1,1,0,0)
            fig.tight_layout(pad=0)
            # Go ahead and draw the figure:
            fig.canvas.draw()
            # We can get the image data directly:
            rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            rgb = np.reshape(rgb, self.options.image_size + (3,))
            im[:,:] = np.mean(rgb, axis=2)
            im /= 255.0
        # The image has been generated; if we have a cache path, we can save the
        # file to disk.
        if self.options.cache_path is not None and filename is not None:
            path = os.path.join(self.options.cache_path, filename)
            # Make sure there is a directory.
            (dirname, filename) = os.path.split(path)
            if not os.path.exists(dirname):
                if self.options.mkdirs:
                    os.makedirs(dirname, mode=self.options.mkdir_mode)
                else:
                    raise RuntimeError(f"no cache dir for feature: {dirname}")
            torch.save(im, path)
        # And return the image itself, if requested.
        return im if return_image else None
    def _mpcall(self, fn, args,
                multiproc=Ellipsis,
                timeout=Ellipsis):
        if multiproc is Ellipsis:
            multiproc = self.options.multiproc
        if timeout is Ellipsis:
            timeout = self.options.timeout
        if multiproc:
            from multiprocessing import Process
            p = Process(target=fn, args=args)
            p.start()
            p.join(timeout)
            ec = p.exitcode
            if ec is None:
                p.kill()
            p.close()
            if ec != 0:
                raise RuntimeError(f"mpcall return value of {ec}")
        else:
            fn(*args)
        return None
    def _get_feature(self, target_id, feature_name, filename,
                     overwrite=Ellipsis, multiproc=Ellipsis, timeout=Ellipsis):
        """Private method that generates and caches a feature image."""
        if overwrite is Ellipsis:
            overwrite = self.options.overwrite
        if multiproc is Ellipsis:
            multiproc = self.options.multiproc
        im = None
        # First, see if there is already a file, assuming overwrite isn't set.
        if not overwrite:
            im = self.load_image(filename)
            if im is not None:
                return im.to(self._dtype())
        # The main task here is to call down to the static method, either in a
        # separate process or in this one.
        fn = self._generate_feature
        if multiproc:
            if self.options.cache_path is None:
                raise RuntimeError("background feature generation requires that"
                                   " the ImageCache use filesystem caching")
            # We generate the feature in a separate process that saves it and
            # returns the filename.
            self._mpcall(fn, (target_id, feature_name, filename),
                         multiproc=multiproc,
                         timeout=timeout)
            # At this point, if the cache image has not been generated, there's
            # been a problem/bug.
            im = self.load_image(filename)
            if im is None:
                raise RuntimeError("feature generated in background,"
                                   " but could not load file")
        else:
            im = fn(target_id, feature_name, filename, return_image=True)
        return im.to(self._dtype())
    # The methods that are intended to provide an API for the feature images.
    def load_image(self, filename, feature_name=None):
        """Loads the requested feature image and returns it.

        `fdata.load_image(filename)` loads the image with the given filename,
        which can be determined using the `fdata.cache_filename(target_id,
        feature_name)` method.

        `fdata.load_image(target_id, feature_name)` is equivalent to calling
        `fdata.load_image(fdata.cache_filename(target_id, feature_name))`.
        """
        if feature_name is not None:
            filename = self._cachepath(filename, feature_name)
        # We need to prepend the cache path.
        cp = str(self.options.cache_path)
        if not filename.startswith(cp) and not os.path.isabs(filename):
            filename = os.path.join(cp, filename)
        try:
            return torch.load(filename)
        except Exception:
            return None
    def get(self, target_id, feature_name,
            overwrite=Ellipsis,
            multiproc=Ellipsis,
            timeout=Ellipsis):
        """Either loads or generates then caches a feature image and returns it.

        `feature_cache.get(target_id, feature_name)` returns the feature matrix
        for the given `target_id` and `feature_name` as a PyTorch tensor of
        floating-point values.

        Parameters
        ----------
        target_id : object
            The object that identifies the target of the feature image
            requested. The `target_id` may be any object, but it must be
            understood by the `plot_image` and `cache_filename` methods.
        feature_name : str
            The name of the feature that is to be generated. The feature name
            may be any non-empty string, but it must be understood by the
            `make_image`, `plot_image`, and `cache_filename` methods.
        overwrite : bool or Ellipsis, optional
            The optional argument `overwrite` may be set to `True` to force the
            image to be generated even if the cache for it already exists. The
            value `False` indicates that the cached image should be used
            whenever possible. The default value, `Ellipsis`, indicates that
            whatever value is stored in the `ImageCache` object's `overwrite`
            option should be used.
        multiproc : bool or Ellipsis, optional
            Whether to run the image generation in a background process. The
            default value, `Ellipsis`, defers to the `multiproc` option of the
            `ImageCache` object.
        timeout : positive real or None or Ellipsis, optional
            If running the image generation in a background process (i.e., if
            `background` is `True`), the `timeout` parameter is the amount of
            time in seconds to wait for the child process to finish before
            givine up. The value `None` indicates that the method should wait
            forever. The default value, `Ellipsis`, indicates that the method
            should defer to the `ImageCache` object's `timeout` option.
        """
        # First, get the filename for this particular target/feature.
        filename = self._cachepath(target_id, feature_name)
        # See if this file is in our cache, assuming that we do not have the
        # overwrite flat set to True.
        if overwrite is Ellipsis:
            overwrite = self.options.overwrite
        # Check the cache first.
        feature = None if overwrite else self.cache.get(filename, None)
        if feature is None:
            # We need to generate the feature.
            feature = self._get_feature(target_id, feature_name, filename,
                                        multiproc=multiproc,
                                        timeout=timeout)
            # Then we put it in the cache. We only do this if we're actualy
            # configured to use a cache.
            if filename is not None and self.options.memcache:
                self.cache[filename] = feature
        return feature
    def __getitem__(self, targ_feat):
        (target_id, feature_name) = targ_feat
        return self.get(target_id, feature_name)
    def _precache_features(self, target, features, overwrite, timeout):
        # We just call _get_feature without multiproc over all the features.
        # Step through each feature.
        for f in features:
            self.get(target, f, multiproc=False, overwrite=overwrite)
        return None
    def precache_features(self, target_id, features,
                          overwrite=Ellipsis,
                          multiproc=True,
                          timeout=Ellipsis):
        """Pre-caches a set of features for a target ID.

        `cache.precache_features(target, (f1, f2, f3...))` ensures that cache
        files exist for all of the features (`f1`, `f2`, `f3`...). It always
        returns `None` unless it raises an exception.

        Note that the `multiproc` defaults to `True` instead of `Ellipsis`.
        """
        if multiproc is Ellipsis:
            multiproc = self.options.multiproc
        if multiproc:
            if self.options.cache_path is None:
                raise RuntimeError("multiproc feature generation requires that"
                                   " the ImageCache use filesystem caching")
        # We're just going to call the private version of this function, either
        # in another process or not.
        fn = self._precache_features
        # We generate the feature in a separate process that saves it.
        self._mpcall(fn, (target_id, features, overwrite, timeout),
                     multiproc=multiproc,
                     timeout=timeout)
        return None

    
#===============================================================================
# Flatmaps

# The ImageCache code for flatmap-based features (which are probably most
# features these classes will get used for).

class FlatmapFeature:
    """A type for storing information about flatmap properties.
    """
    __slots__ = ('property', 'interp_method')
    def __init__(self, property, interp_method=None):
        # Check property.
        if   property is None: pass
        elif isinstance(property, str): pass
        elif callable(property): pass
        else: raise ValueError("property must be a string or callable")
        # Check interp_method.
        if   interp_method is None: pass
        elif interp_method == 'linear': pass
        elif interp_method == 'nearest': pass
        else: raise ValueError(f"bad interpolation name: {self.interp_method}")
        # Assign them.
        self.property = property
        self.interp_method = interp_method
    def __call__(self, fmap, addrs, target, view=None):
        if view is None:
            view = {}
        p = self.get_property(fmap, target, view=view)
        pp = ny.util.address_interpolate(addrs, p, method=self.interp_method)
        return pp
    def get_property(self, fmap, target, view={}):
        """Optional method to extract a property from a flatmap.

        This method may be overloaded if a property must be calculated from a
        flatmap. The method normally just extracts the property by name, so
        overloading it will result in the property being extracted by the
        overloaded version.

        The optional argument `view` is used to indicate that the property
        being extracted is a view of the `target` that requires additional
        target meta-data to identify. For example, when creating a flatmap
        feature for a dataset that uses a flatmap of only one hemisphere per
        feature-image, the hemisphere label is included in the target data, but
        when both left and right hemispheres are included in the same image, as
        with the `BilateralFlatmapImageCache` class, the `view` parameter would
        typically contain either `{'hemisphere': 'lh'}` or
        `{'hemisphere': 'rh'}`. The `view` must be a dict with an empty dict
        indicating a lack of view data. The value `None` is permitted when the
        `FlatmapFeature`'s `__call__` method is run, however.
        """
        if isinstance(self.property, str):
            return fmap.prop(self.property)
        else:
            return self.property(fmap)
           
class FlatmapImageCache(ImageCache):
    """A ImageCache type for features made from flatmaps of hemispheres.

    The `FlatmapImageCache` class inherits from the `ImageCache` class and
    is meant to be used whenever the features for a dataset are derived from
    Neuropythy's flatmap projections of the cortical surface. In such a case,
    the user may overload the `FlatmapImageCache` class and redefine only the
    `make_flatmap(target)`, and `cache_filename(target, feature)` methods.
    """
    __slots__ = ('normalization', 'features', 'flatmap_cache')
    def __init__(self,
                 views=None,
                 image_size=Ellipsis,
                 cache_path=None,
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
        assert normalization in ('normalize', 'standardize', None), \
            f"invalid normalization: {normalization}"
        self.normalization = normalization
        if features is None: features = {}
        assert isinstance(features, Mapping), \
            f"features must be None or a dict-like"
        self.features = dict(self.builtin_features(), **features)
        if flatmap_cache is None or flatmap_cache is False:
            flatmap_cache = None
        elif isinstance(flatmap_cache, Mapping):
            flatmap_cache = dict(flatmap_cache)
        elif flatmap_cache is True:
            flatmap_cache = {}
        else:
            raise ValueError("flatmap_cache must be dict-like or None")
        self.flatmap_cache = flatmap_cache
        super().__init__(
            views=views,
            image_size=image_size,
            cache_path=cache_path,
            overwrite=overwrite,
            mkdirs=mkdirs,
            mkdir_mode=mkdir_mode,
            multiproc=multiproc,
            timeout=timeout,
            dtype=dtype,
            memcache=memcache)
    def make_flatmap(self, target, view=None):
        """Returns the flatmap for the given target.

        The `make_flatmap` method must be overloaded by subclasses of the
        `FlatmapImageCache` type. The method is always passed the target ID of
        the flatmap that is to be generated and must return a Neuropythy `Mesh`
        object with 2D coordinates (a flatmap).

        Note that in order for the `FlatmapImageCache` workflow to function
        correctly, the flatmap must have properties whose names match those of
        the features being generated (so, if you want a feature named
        `'thickness'`, be sure there is a flatmap property named `'thickness'`
        in the returned flatmap). The alternative to this is to overload and
        explicitly provide an overloaded `FlatmapFeature` object that extracts a
        feature in some non-typical way (however, this should not be necessary
        in most situations: just perform the calculations in the make_flatmap
        method instead and attach the needed properties to the flatmap using
        the `with_prop` method).

        The optional parameter `view` is used to communicate that the flatmap
        being generated is a single view of the target whose additional view
        parameters are specified in a dictionary. The `view` value must be
        either a `Mapping` (like a `dict`) or `None`. Typically, the `view`
        is `None` if all necessary parameters for the flatmap are in the target
        and will be something like `{'hemisphere': 'lh'}` when multiple views of
        a subject appear in a single training image.
        """
        raise NotImplementedError(
            f"make_flatmap not implemented for type {type(self)}")
    @classmethod
    def _to_hashable(cls, obj):
        "Turns nested lists/tuples/dicts into hashable equivalents."
        if isinstance(obj, dict):
            return frozenset(map(cls._to_hashable, obj.items()))
        elif isinstance(obj, (list, tuple)):
            return tuple(map(cls._to_hashable, obj))
        else:
            return obj
    def get_flatmap(self, target, view=None):
        """Returns the flatmap for the given target ID.

        The `get_flatmap` method is a simple wrapper around the `make_flatmap`
        method that handles flatmap caching. In general, you should always
        access flatmaps using the `get_flatmap` method instead of `make_flatmap`
        for this reason.
        """
        if self.flatmap_cache is None:
            return self.make_flatmap(target, view=view)
        # We need to sanitize the target into something hashable.
        hashtarg = self._to_hashable((target, view))
        fmap = self.flatmap_cache.get(hashtarg, None)
        if fmap is None:
            fmap = self.make_flatmap(target, view=view)
            # Since we are caching the flatmap, we want to add the addresses
            # at this point--that way they need only be calculated once.
            from functools import partial
            fn = partial(lambda s,fm,v: s.flatmap_imaddrs(fm, view=v),
                         self, fmap, view)
            fmap = fmap.with_meta(pimms.lmap({'addresses': fn}))
            self.flatmap_cache[hashtarg] = fmap
        return fmap
    def cache_filename(self, target, feature, view=None):
        """Returns a relative filename for the appropriate cache file.

        The `cache_filename` method must be overloaded by subclasses of the
        `ImageCache` type. The method is always passed a target ID and a
        feature name, and it must return a unique cache filename for the given
        target and feature image. The returned filename should be a relative
        filename, and the `cache_path` of the `ImageCache` object is
        automatically prepended to the cache filename prior to reading or
        writing the cache.
        """
        raise TypeError(f"cache_filename not implemented for type {type(self)}")
    def flatmap_imaddrs(self, fmap, im=None, view=None, recalc=False):
        """Given a flatmap and an image, returns the addresses of the pixels.

        `flatmap_imaddrs(flatmap, im)` returns an addresses dictionary for the
        pixels in the given image `im`, where the addresses are for the given
        `flatmap`. The method both indexes the addresses before returning, and
        it excludes any image pixels that do not land in the flatmap. The pixels
        included in the addresses are detailed in the `'pixel_indices'` key of
        the returned addresses dictionary. If `im` is excluded or `None`, then
        the image cache's default image size is used.

        The optional argument `recalc` (default `False`) specifies whether the
        addresses should be recalculated by the method; this only applies when
        existing addresses are stored in the flatmap's `meta_data` field under
        the `'addresses'` key. If these exist and `recalc` is `False`, then
        these addresses are returned.
        """
        # This method gets the addresses of the image pixels in the flatmap.
        # Start by checking for existing addresses.
        addrs = fmap.meta_data.get('addresses')
        if not recalc and addrs is not None:
            return addrs
        # Start by prepping the image pixels for addressing.
        (xmin,ymin) = np.min(fmap.coordinates, axis=1)
        (xmax,ymax) = np.max(fmap.coordinates, axis=1)
        (ynum,xnum) = self.image_size(view) if im is None else im.shape
        # Note here that we invert the y-axis, just by convention.
        (x,y) = torch.meshgrid(torch.linspace(xmin, xmax, xnum),
                               torch.linspace(ymax, ymin, ynum),
                               indexing='xy')
        # Figure out which of these are likely to be in the flatmap.
        (xmu, ymu) = ((ymax + ymin)/2, (xmax + xmin)/2)
        r = np.max([xmax - xmu, ymax - ymu])
        ii = torch.sqrt(x**2 + y**2) < r
        # Get the addresses.
        xy = [x[ii].detach().numpy(), y[ii].detach().numpy()]
        addrs = fmap.address(xy)
        # Index the faces for the flatmap.
        addrs['faces'] = fmap.tess.index(addrs['faces'])
        # Trim out the addresses that weren't actually in the flatmap.
        jj = np.isfinite(addrs['coordinates'][0])
        addrs = {k: v[:,jj] for (k,v) in addrs.items()}
        jj = ~jj
        jj = tuple(kk[jj] for kk in np.where(ii))
        ii[jj] = False
        # We add these ii to the addresses.
        addrs['pixel_indices'] = np.where(ii)
        # That's it.
        return addrs
    def fill_view(self, target, feature, im, view=None):
        # We requires the flatmap.
        fmap = self.get_flatmap(target, view=view)
        # The view option is passed along to feats to indicate when the target
        # has an additional view data due to how the original image was split
        # up; usually this is is used for specifying the hemisphere.  Start by
        # getting the relevant feature data.
        if self.features is not None:
            feat = self.features.get(feature, None)
        else:
            feat = None
        if feat is None:
            feat = FlatmapFeature(feature, None)
        # Get the addresses.
        addrs = self.flatmap_imaddrs(fmap, im, view=view)
        ii = addrs['pixel_indices']
        # Set everything outside the flatmap to 0.
        im[...] = 0.0
        # And set everything inside it to the appropriate values.
        z = feat(fmap, addrs, target, view=view)
        # Normalize if needed.
        if self.normalization == 'normalize':
            z /= np.sqrt(np.sum(z**2))
        elif self.normalization == 'standardize':
            z -= np.mean(z)
            z /= np.std(z)
        # And paint the image.
        im[ii] = torch.as_tensor(z, dtype=im.dtype)
        return None
    _builtin_features = {
        # A few properties are basically universal.
        'curvature':    FlatmapFeature('curvature',    'linear'),
        'convexity':    FlatmapFeature('convexity',    'linear'),
        'surface_area': FlatmapFeature('surface_area', 'linear'),
        'thickness':    FlatmapFeature('thickness',    'linear'),
    }
    @classmethod
    def builtin_features(cls):
        """Returns a dictionary of builtin features for the class.

        The return value of this classmethod must be a dictionary (or dict-like
        object) whose keys are feature names and whose values are
        `FlatmapFeature` objects.
        """
        return FlatmapImageCache._builtin_features
    def inv(self, target, image, view=None, null=np.nan, dtype=None):
        """Returns the the given image data interpolate onto target's flatmap.

        This function interpolates image data from `image`, which must have a
        shape matching the image cache's `image_size`, onto the vertices of the
        flatmap for the given `target` and returns these values as a property.

        Parameters
        ----------
        target : object
            The target ID of the flatmap onto which the image data is to be
            interpolated.
        image : numpy matrix
            The image array from which the values should be interpolated.
        view : None or dict, optional
            The view-data specific to the target for the specified image. For
            example, an image cache tracking both hemispheres might use the view
            data `{'hemisphere': 'lh'}` for the left hemisphere part of the
            image. The default is `None`, which indicates that all relevant view
            data is included in the target or not needed.
        null : object, optional
            The value to give to any vertices that fall outside of the image.
            The default value is `nan`.
        dtype : None or dtype-like, optional
            The NumPy dtype to use for the returned array. The default value,
            `None`, uses whatever type is extracted from the image.

        Returns
        -------
        numpy array
            A property array for the target's flatmap produced from the given
            image.
        """
        # Get the flatmap. If target is itself a flatmap, just use it.
        if ny.is_mesh(target):
            fmap = target
        else:
            fmap = self.get_flatmap(target, view=view)
        # If view is not None, we need to extract a sub-image.
        if view is not None:
            rect = self.view_rectangle(view)
            if rect is None:
                raise ValueError(f"view not found: {view}")
            image = ImageCache.view_subimage(image, rect, rcfirst=False)
        # Get a numpy matrix for the image. We allow tuples/lists of images or
        # 3D images whose first dimension is channels, and we return a similar
        # object of properties.
        from collections.abc import Mapping
        if isinstance(image, Mapping):
            return {k: self.inv(fmap, v, view=view, null=null, dtype=dtype)
                    for (k,v) in image.items()}
        elif torch.is_tensor(image):
            im = image.detach().numpy()
        else:
            im = np.asarray(image)
        imsz = self.image_size(view)
        if im.shape[-2]/im.shape[-1] != imsz[0]/imsz[1]:
            raise ValueError(
                f"given image shape {im.shape} must match {imsz}")
        if im.shape[-2] != imsz[0]:
            im = Resize(imsz)(torch.from_numpy(im)[None,...])[0]
            im = im.detach().numpy()
        # Okay, let's orient ourselves with respect to the image and flatmap.
        (xmin,ymin) = np.min(fmap.coordinates, axis=1)
        (xmax,ymax) = np.max(fmap.coordinates, axis=1)
        (ynum,xnum) = imsz
        # How to transform from (x,y) in the flatmp to (r,c) in the image?
        # The image was made using coordinates X = linspace(xmin, xmax, xnum)
        # and Y = linspace(ymax, ymin, ynum) (note the y-axis reversal). This
        # means that means that:
        #  - for rows, y(r) = ymax - (ymax-ymin)/(ynum-1) * r
        #  - for cols, x(c) = xmin + (xmax-xmin)/(xnum-1) * c
        # Therefore,
        #  - for x, c(x) = (xnum-1) * (x-xmin)/(xmax-xmin)
        #  - for y, r(y) = (ynum-1) * (y-ymax)/(ymin-ymax)
        # We can just convert into rows/columns then round to the nearest int
        # in order to find the pixels from which we interpolate for each vertex.
        (x,y) = fmap.coordinates
        c = ((xnum-1) * (x-xmin)/(xmax-xmin)).round().astype(int)
        r = ((ynum-1) * (y-ymax)/(ymin-ymax)).round().astype(int)
        # Mask out anything falling outside of the image.
        jj = (c < 0) | (c >= xnum) | (r < 0) | (r >= ynum)
        if jj.sum() > 0:
            c[jj] = 0
            r[jj] = 0
        else:
            jj = ()
        # Okay, extract the property values.
        prop = im[..., r, c]
        if dtype is not None:
            prop = prop.astype(dtype)
        # Set the nulls.
        if len(jj) > 0:
            prop[..., jj] = null
        # That's it; return the property.
        return prop
    def invlabels(self, target, image, view=None, labelsets=None):
        """Returns a label property for the the given image stack.

        This function first runs the `inv` method using the same arguments, then
        converts the return-value into a property array in which each value in
        the property is 1 plus the index of the image channel with the highest
        probability, or 0 if the sum of probabilities across the channels is
        less than 1 minus the highest probability across channels.

        Parameters
        ----------
        target : object
            The target ID of the flatmap onto which the image data is to be
            interpolated.
        image : numpy matrix
            The image array from which the values should be interpolated.
        view : None or dict, optional
            The view-data specific to the target for the specified image. For
            example, an image cache tracking both hemispheres might use the view
            data `{'hemisphere': 'lh'}` for the left hemisphere part of the
            image. The default is `None`, which indicates that all relevant view
            data is included in the target or not needed.

        Returns
        -------
        numpy array
            A property label array for the target's flatmap produced from the
            given image channels.
        """
        # Parse the labelsets parameter.
        if labelsets is None:
            lsets = [slice(0, ps.shape[0], 0)]
        elif isinstance(labelsets, slice):
            lsets = [labelsets]
        elif isinstance(labelsets, Mapping):
            lsets = list(labelsets.values())
        else:
            lsets = labelsets
        # Invert the image(s)
        invs = self.inv(target, image, view=view, null=0)
        res = []
        # Convert the labelsets into
        for ls in lsets:
            ps = invs[ls, ...]
            psum = np.sum(ps, axis=0)
            pmax = np.max(ps, axis=0)
            lbl = np.zeros(ps.shape[-1], dtype=int)
            ii = pmax > 1 - psum
            lbl[ii] = np.argmax(ps[:, ii], axis=0) + 1
            res.append(lbl)
        if lsets is labelsets:
            return res
        elif isinstance(labelsets, Mapping):
            return {k:v for (k,v) in zip(labelsets.keys(), res)}
        else:
            return res[0]
class BilateralFlatmapImageCache(FlatmapImageCache):
    """A ImageCache type for features made from flatmaps of both hemispheres.

    The `BilateralFlatmapImageCache` class inherits from the
    `FlatmapImageCache` class and is meant to be used whenever the features
    for a dataset are derived from Neuropythy's flatmap projections of the
    cortical surface of both hemispheres side-by-side in the resulting image. In
    such a case, the user may overload the `BilateralFlatmapImageCache` class
    and redefine only the `make_flatmap(target)`, `flatmap_feature(target,
    feature)`, and `cache_filename(target, feature)` methods, just like in the
    `FlatmapImageCache` class. The critical difference between these classes
    is that in the `BilateralFlatmapFeatureClass`, the `target` that is passed
    to these methods is always a tuple `(target, hemi)` where `hemi` is either
    the hemisphere name (typically `'lh'` or `'rh'`) and `target` is the actual
    target ID that was passed to the `ImageCache`.

    The `BilateralFlatmapImageCache` type uses the view parameter `'hemisphere'`
    to pass the hemisphere along to features.
    """
    __slots__ = ('hemis')
    def __init__(self,
                 hemis='lr',
                 image_size=Ellipsis,
                 cache_path=None,
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
        if hemis is Ellipsis:
            hemis = 'lr'
        if isinstance(hemis, str):
            hemis = ny.to_hemi_str(hemis)
            if hemis == 'lr':
                hemis = ('lh', 'rh')
            else:
                hemis = (hemis,)
        else:
            hemis = tuple(hemis)
        self.hemis = hemis
        views = [[{'hemisphere': h} for h in hemis]]
        # If the image size is Ellipsis or an int, we want to fix it to have
        # an equal aspect ratio for each sub-image.
        if image_size is Ellipsis:
            from ..config import saved_image_size as image_size
        if isinstance(image_size, int):
            image_size = (image_size*len(views), image_size*len(views[0]))
        super().__init__(
            views=views,
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
    def inv(self, target, image, view=None, null=np.nan, dtype=None):
        if isinstance(view, str): view = {'hemisphere': view}
        return super().inv(target, image, view=view, null=null, dtype=dtype)


#===============================================================================
# ImageCache-based Datasets.

_ImageCacheDatasetOptionsBase = namedtuple(
    '_ImageCacheDatasetOptionsBase',
    ('resize',
     'transform',
     'input_transform',
     'output_transform',
     'target_cache_key',
     'feature_cache_fn'))
class ImageCacheDatasetOptions(_ImageCacheDatasetOptionsBase):
    @property
    def image_size(self):
        if self.resize is None:
            return None
        else:
            return self.resize.size
class ImageCacheDataset(Dataset):
    """A PyTorch dataset that works with ImageCache types.

    The ImageCacheDataset
    """
    __slots__ = ('image_cache',
                 'input_layers',
                 'output_layers',
                 'targets',
                 'options',
                 'cache')
    default_options = ImageCacheDatasetOptions(
        resize=Resize(size=default_image_size),
        transform=None,
        input_transform=None,
        output_transform=None,
        target_cache_key=None,
        feature_cache_fn=None)
    def __init__(self, image_cache, inputs, outputs, targets,
                 options=None,
                 image_size=Ellipsis,
                 transform=Ellipsis,
                 input_transform=Ellipsis,
                 output_transform=Ellipsis,
                 target_cache_key=Ellipsis,
                 feature_cache_fn=Ellipsis,
                 memcache=True):
        # Start by making sure the base options are valid.
        if options is None or options is Ellipsis:
            options = ImageCacheDataset.default_options
        # Now process the args and options.
        inputs  = (inputs,)  if isinstance(inputs,  str) else tuple(inputs)
        outputs = (outputs,) if isinstance(outputs, str) else tuple(outputs)
        if image_size is Ellipsis:
            from ..config import default_image_size as image_size
        resize = Resize(size=image_size)
        if transform is Ellipsis:
            transform = options.transform
        if input_transform is Ellipsis:
            input_transform = options.input_transform
        if output_transform is Ellipsis:
            output_transform = options.output_transform
        if target_cache_key is Ellipsis:
            target_cache_key = options.target_cache_key
        if feature_cache_fn is Ellipsis:
            feature_cache_fn = options.feature_cache_fn
        options = ImageCacheDatasetOptions(
            resize=resize,
            transform=transform,
            input_transform=input_transform,
            output_transform=output_transform,
            target_cache_key=target_cache_key,
            feature_cache_fn=feature_cache_fn)
        targets = tuple(targets)
        # Set all the fields.
        self.image_cache   = image_cache
        self.input_layers  = inputs
        self.output_layers = outputs
        self.targets       = targets
        self.options       = options
        self.cache         = {} if memcache else None
    @property
    def feature_count(self):
        """Returns the number of features in the dataset's input images.
        
        The feature count of `d` is the length of the `d.input_layers`.
        """
        return len(self.input_layers)
    @property
    def segment_count(self):
        """Returns the number of segments (output labels) in the dataset.

        The segment count of `d` is the length of the `d.output_layers`.
        """
        return len(self.output_layers)
    def target_to_cache(self, target):
        """Returns the version of the target ID appropriate for the cache.

        Because an `ImageCacheDataset` object may use a different set of targets
        than its `ImageCache` object, a key is provided to translate between the
        two. An example of this is when the `ImageCache` tracks the images for a
        set of subjects, each of which may have been labeled by multiple
        raters. The targets tracked by the dataset, in this case, might be
        tuples of `(rater, subject)` while the `ImageCache` object tracks
        targets of just the `subject`. The `target_cache_key` in this case might
        be either `1` or `lambda target: target[1]`.
        """
        tck = self.options.target_cache_key
        if tck is None:
            return target
        elif callable(tck):
            return tck(target)
        else:
            return target[tck]
    def feature_to_cache(self, target, feature):
        """Returns the version of the feature appropriate for the cache.

        Because an `ImageCacheDataset` object may use the feature parameter in a
        different way than the `ImageCache` object, the `feature_cache_fn`
        option is provided to translate between the two. The option must be a
        function that is called as `feature_cache_fn(target, feature)` where
        both `target` and `feature` are the version of the parameters used by
        the dataset (and not the versions that are translated for the cache by
        the `target_to_cache` and `feature_to_cache` methods). The return value
        is the feature name passed to the image cache.
        """
        fcf = self.options.feature_cache_fn
        if fcf is None:
            return feature
        else:
            return fcf(target, feature)
    def _to_target_index(self, k):
        if isinstance(k, int):
            return (self.targets[k], k)
        for (ii,targ) in enumerate(self.targets):
            if k == targ:
                return (targ, ii)
        raise KeyError(k)
    def __getitem__(self, k):
        ims = None if self.cache is None else self.cache.get(k)
        if ims is not None:
            return ims
        (target, k) = self._to_target_index(k)
        imcache = self.image_cache
        tt = self.target_to_cache(target)
        ifeats = [self.feature_to_cache(target, f) for f in self.input_layers]
        ofeats = [self.feature_to_cache(target, f) for f in self.output_layers]
        input_im  = torch.stack([imcache[tt, ff] for ff in ifeats], axis=0)
        output_im = torch.stack([imcache[tt, ff] for ff in ofeats], axis=0)
        if self.options.resize is not None:
            input_im  = self.options.resize(input_im)
            output_im = self.options.resize(output_im)
        # Now transpose back to (rows, cols, channels):
        if self.options.transform is not None:
            input_im  = self.options.transform(input_im)
            output_im = self.options.transform(output_im)
        if self.options.input_transform is not None:
            input_im  = self.options.input_transform(input_im)
        if self.options.output_transform is not None:
            output_im = self.options.output_transform(output_im)
        # Add to the cache.
        if self.cache is not None:
            self.cache[k] = (input_im, output_im)
        return (input_im, output_im)
    def __len__(self):
        return len(self.targets)
    def predlabels(self, k, model, view=Ellipsis, labelsets=None):
        """
        """
        #TODO write docs
        (target, k) = self._to_target_index(k)
        (inp, outp) = self[k]
        inp = inp[None, ...]
        pred = model(inp)[0]
        # If the model is setup to return logits, we fix those.
        if model.logits:
            pred = torch.sigmoid(pred)
        # Convert the pred into labels.
        imcache = self.image_cache
        views = imcache.views
        if view is Ellipsis:
            if views is None:
                return imcache.invlabels(target, pred, labelsets=labelsets)
            ps = [imcache.invlabels(target, pred, view=v, labelsets=labelsets)
                  for (v,rect) in views]
            return tuple(ps)
        return imcache.invlabels(target, pred, view=view, labelsets=labelsets)

#-------------------------------------------------------------------------------
# Additional Flatmap Features

class LabelFeature(FlatmapFeature):
    """A Feature class that extracts individual labels from properties."""
    @classmethod
    def _get_property(cls, property, fmap, view={}):
        (prop, els) = property.split(':')
        els = tuple(map(int, els.split(' ')))
        lbls = fmap.property(prop)
        return np.isin(lbls, els)
    def get_property(self, fmap, target, view={}):
        return self._get_property(self.property, fmap)
class LabelDiffFeature(LabelFeature):
    """A Feature class that represents the difference between labels."""
    def get_property(self, fmap, target, view={}):
        (prop1, prop2) = self.property.split('--')
        m1 = self._get_property(prop1, fmap, view=view)
        m2 = self._get_property(prop2, fmap, view=view)
        return m1 & ~m2
class LabelUnionFeature(LabelFeature):
    """A Feature class that represents the union of labels."""
    def get_property(self, fmap, target, view={}):
        (prop1, prop2) = self.property.split('||')
        m1 = self._get_property(prop1, fmap, view=view)
        m2 = self._get_property(prop2, fmap, view=view)
        return m1 | m2
class LabelIntersectFeature(LabelFeature):
    """A Feature class that represents the intersection of labels."""
    def get_property(self, fmap, target, view={}):
        (prop1, prop2) = self.property.split('&&')
        m1 = self._get_property(prop1, fmap, view=view)
        m2 = self._get_property(prop2, fmap, view=view)
        return m1 & m2
class NullFeature(FlatmapFeature):
    """A Feature class that creates a blank set of values."""
    def get_property(self, fmap, target, view={}):
        return np.zeros(fmap.vertex_count)
