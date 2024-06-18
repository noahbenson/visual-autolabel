# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/plot/_core.py
# Implementation of the plotting utilities for the visual_autolabel library.


#===============================================================================
# Constants / Globals

import os
from statistics import NormalDist

import torch
import numpy as np
import neuropythy as ny

from ..util import sectors_to_rings


#===============================================================================
# Utility Functions

def calc_visual_ring(data, rings=((0,0.5), (0.5,1), (1,2), (2,4), (4,7))):
    """Returns a `'visual_ring'` property of eccentricity bins for `data`."""
    ecc = next(v for (k,v) in data.items() if k.endswith('eccentricity'))
    lbl = next(v for (k,v) in data.items() if k.endswith('visual_area_init'))
    ii = np.isin(lbl, (1,2,3))
    rng = np.zeros(lbl.shape, dtype=lbl.dtype)
    for (ll,(emn,emx)) in enumerate(rings):
        rng[ii & (ecc >= emn) & (ecc < emx)] = ll + 1
    return rng
def calc_visual_area(data, eccmin=0, eccmax=7):
    """Returns a `'visual_ring'` property of eccentricity bins for `data`."""
    ecc = next(v for (k,v) in data.items() if k.endswith('eccentricity'))
    lbl = next(v for (k,v) in data.items() if k.endswith('visual_area_init'))
    ii = np.isin(lbl, (1,2,3)) & (ecc >= eccmin) & (ecc < eccmax)
    are = np.zeros(lbl.shape, dtype=lbl.dtype)
    are[ii] = lbl[ii]
    return are
def load_inferred(sid, h, path=None, prefix=None):
    """Loads the Bayesian inferred maps for an HCP subject.
    
    This function will load the maps from the given path if it is
    provided, but if not it will load them from the visual
    performance fields OSF database.
    """
    tr = {'polar_angle':'angle',
          'eccentricity': 'eccen',
          'radius': 'sigma',
          'visual_area': 'varea'}
    if path is None:
        pp = ny.data['visual_performance_fields'].pseudo_path
        path = 'inferred_maps'
    else:
        pp = None
    r = {}
    for (k,v) in tr.items():
        flnm = f'{sid}.{h}.inf-MSMAll_{v}.native32k.mgz'
        flnm = os.path.join(path, flnm)
        if pp is not None: flnm = pp.local_path(flnm)
        r[k] = ny.load(flnm)
    if prefix is not None:
        r = {(prefix + k):v for (k,v) in r.items()}
    return r
def add_inferred(sub, path=None, prefix='inf_'):
    """Given an HCP subject, returns a copy with inferred maps.
    
    This function loads the inferred maps for the subject and
    adds them to the left and right hemispheres, then returns
    the subject with those added maps.
    """
    sid = int(sub.name)
    lhdat = load_inferred(sid, 'lh', path=path, prefix=prefix)
    rhdat = load_inferred(sid, 'rh', path=path, prefix=prefix)
    lhdat[f'{prefix}visual_area_init'] = lhdat[f'{prefix}visual_area']
    rhdat[f'{prefix}visual_area_init'] = rhdat[f'{prefix}visual_area']
    lhdat[f'{prefix}visual_ring'] = calc_visual_ring(lhdat)
    rhdat[f'{prefix}visual_ring'] = calc_visual_ring(rhdat)
    lhdat[f'{prefix}visual_area'] = calc_visual_area(lhdat)
    rhdat[f'{prefix}visual_area'] = calc_visual_area(rhdat)
    return sub.with_hemi(lh=sub.lh.with_prop(lhdat),
                         rh=sub.rh.with_prop(rhdat))
def add_prior(sub, prefix='prior_'):
    """Given a subject, adds the retinotopic prior and returns it.
    
    This function calculates the retinotopic prior for the subejct,
    adds the prior maps to the subject's hemispheres, and returns
    the new subject.
    """
    tr = {'polar_angle':'angle',
          'eccentricity': 'eccen',
          'radius': 'sigma',
          'visual_area_init': 'varea'}
    lhdat = ny.vision.predict_retinotopy(sub.lh)
    rhdat = ny.vision.predict_retinotopy(sub.rh)
    (lhdat,rhdat) = [{k:dat[v] for (k,v) in tr.items()}
                     for dat in (lhdat,rhdat)]
    if prefix is not None:
        (lhdat,rhdat) = [
            {(prefix+k):v for (k,v) in dat.items()}
            for dat in (lhdat,rhdat)]
    # We need to add in the visual_ring and visual_area data.
    lhdat[f'{prefix}visual_ring'] = calc_visual_ring(lhdat)
    rhdat[f'{prefix}visual_ring'] = calc_visual_ring(rhdat)
    lhdat[f'{prefix}visual_area'] = calc_visual_area(lhdat)
    rhdat[f'{prefix}visual_area'] = calc_visual_area(rhdat)
    return sub.with_hemi(lh=sub.lh.with_prop(lhdat),
                         rh=sub.rh.with_prop(rhdat))
def add_raterlabels(sub):
    """Returns a copy of the given subject with labels as drawn by raters.
    """
    lbldata = ny.data['hcp_lines'].subject_labels
    sid = int(sub.name)
    lbls = {}
    for h in ['lh','rh']:
        ll = {}
        for anat in ['A1','A2','A3','A4']:
            lbldat = lbldata[anat].get(sid)
            if lbldat is None: lbldat = {}
            lbldat = lbldat.get(h)
            if lbldat is None: lbldat = {}
            areadat = lbldat.get('visual_area', None)
            if areadat is None: continue
            ll[f'{anat}_visual_area'] = areadat
            # We also want to try to turn the sectors into eccentricity
            # labels also.
            sctdat = lbldat.get('visual_sector', None)
            if sctdat is None: continue
            eccdat = sectors_to_rings(sctdat, areadat)
            ll[f'{anat}_visual_ring'] = eccdat
        lbls[h] = ll
    return sub.with_hemi(lh=sub.lh.with_prop(lbls['lh']),
                         rh=sub.rh.with_prop(lbls['rh']))

def summarize_dist(x, middle='mean', extent='std', ci=0.6826895, bfcount=None):
    """Returns summary values for the distribution of the given points.

    `summarize_dist(x)` returns the tuple `(mu-sig, mu, mu+sig)` for the mean
    `mu` and the standard deviation `sig` of the values in `x`. The function
    always returns three values representing a `(min, mid, max)` representation
    of the distribution; however, what values precisely are returned can be
    tweaked using the `middle` and `extent` options.

    Parameters
    ----------
    x : iterable of reals
        The values whose distribution is to be summarized.
    middle : 'mean' | 'median' | percentile str, optional
        The value to return representing the midpoint. This may be `'mean'` or
        `'median'` for the mean or median of `x`, or a percentile such as
        `'25%'` to indicate an explicit percentile. The default is `'mean'`.
    extent : 'std' | 'ste' | 'iqr' | 'all' | percentile str | tuple, optional
        How to calculate the minimum and maximum values returned as part of the
        representation of the distribution. This may be be a single percentile
        string such as `'50%'`, any of the special strings `('std', 'ste',
        'iqr', 'all')`, or a tuple of `(lower, upper)`. If a tuple is given,
        then lower and upper must be percentile strings, such as `('25%',
        '75%')`. The strings `'std'` for the standard deviation and `'ste'` for
        the standard error are also accepted (note that these calculate the
        deviation or error about the midpoint as defined by `middle`, whether or
        not that is the mean), and `'iqr'` is an alias for `('25%',
        '75%')`. Finally, a percentile string such as `'50%'` or `'95%` is
        always converted to the real-valued percentile argument `p` (in these
        cases, `p = 50` or `p = 95`), then is reinterpreted as
        `(f"{(100-p)/2}%", f"{100 - (100-p)/2}%")`. In other words, a single
        percentile string is interpreted as an instruction to return min and max
        values that delineate the `p`% of the y-values that are closest to the
        midpoint.
    ci : float, optional
        The confidence-interval fraction to use if the `extent` option is set to
        `'ste'`; otherwise this parameter is ignored. The default confidence
        interval is approximately 0.68, corresponding to a single standard
        deviation.
    bfcount : None or positive integer, optional
        The count of confidence intervals for use with Bonferroni correction if
        the `'ste'` option is chosen for `extent`. If `'ste'` is not the value
        of `extent`, then this value is ignored.
    """
    if torch.is_tensor(x):
        x = x.detach().numpy()
    else:
        x = np.asarray(x)
    n = len(x)
    # Find the midpoint.
    if middle == 'mean':
        xmid = np.mean(x)
    elif middle == 'median':
        xmid = np.median(x)
    elif mid.endswith('%'):
        xmid = np.percentile(mid, float(mid[:-1]))
    else:
        raise ValueError(f"unrecognized middle parameter: {middle}")
    # Calculate the min and max.
    # For simplicity, 'iqr' is just an alias for '50%'.
    if isinstance(extent, str):
        if extent == 'iqr':
            extent = '50%'
        if extent == 'std':
            std = np.sqrt(np.sum((x - xmid)**2) / n)
            return (xmid - std, xmid, xmid + std)
        elif extent == 'ste':
            if bfcount is None:
                bfcount = 1
            ste = np.sqrt(np.sum((x - xmid)**2)) / n
            # figure out what scaling to apply to the ste based on the requested
            # confidence interval and bonferroni correction count.
            alpha = 1 - ci
            lev = 1 - alpha/bfcount
            scale = NormalDist().inv_cdf(lev + (1-lev)/2)
            return (xmid - ste*scale, xmid, xmid + ste*scale)
        elif extent == 'all':
            return (np.min(x), xmid, np.max(x))
        elif extent.endswith('%'):
            p = float(extent[:-1])
            pmin = (100 - p) / 2
            pmax = 100 - pmin
            (xmin, xmax) = np.percentile(x, [pmin, pmax])
            return (xmin, xmid, xmax)
    if isinstance(extent, tuple) and len(extent) == 2:
        if not all(isinstance(p, str) and p[-1] == '%' for p in extent):
            raise ValueError("extent tuples must contain 2 percentile strings")
        ps = [float(p[:-1]) for p in extent]
        (xmin, xmax) = np.percentile(x, np.sort(ps))
        return (xmin, xmid, xmax)
    else:
        raise ValueError(f"unrecognized extent parameter: {extent}")
def plot_distbars(x, y,
                  middle='mean', extent='std',
                  ci=0.6826895, bfcount=None,
                  axes=None,
                  fw=None,
                  lw=None, ms=None,
                  lc='r', mc='r',
                  zorder=2):
    """Plots a vertical error-bar based on the distribution of y values.

    `plot_vdistbar(x, ys)` accepts a real-valued `x` and an iterable of
    real-valued `ys`. The function plots an error-bar with a dot at the mean and
    with the error-bar marking the standard deviation.

    Parameters
    ----------
    x : real or iterable of reals
        The x-value(s) of the points that are being summarized.
    y : real or iterable of reals
        The y-value(s) of the points that are being summarized.
    midpoint : 'mean' | 'median' | real | None | percentile str, optional
        The value at which to plot the midpoint. This may be an explicit real
        number, `None` to indicate that no dot should be plotted, `'mean'` or
        `'median'` for the mean or median of `ys`, or a percentile such as 
        `'25%'` to indicate an explicit percentile. The default is `'mean'`.
    extent : 'std' | 'ste' | 'iqr' | real | None | percentile str, optional
        The length that the bars should extend from the midpoint. This may be
        either a single value such as a real number, or it may be a tuple of
        `(lower, upper)`. If a tuple is given, then lower and upper must be
        either explicit real numbers or percentile strings, such as `('25%',
        '75%')`. The additional strings `'std'` for the standard deviation and
        `'ste'` for the standard error are also accepted (note that these
        calculate the deviation or error about the midpoint, not the mean), and
        `'iqr'` is an alias for `('25%', '75%')`. Finally, a percentile string
        such as `'50%'` or `'95%` is always converted to the real-valued
        percentile argument `p` (in these cases, `p = 50` or `p = 95`), then is
        reinterpreted as `(f"{(100-p)/2}%", f"{100 - (100-p)/2}%")`. In other
        words, a single percentile string is interpreted as an instruction to
        place the error bars such that they delineate the `p`% of the y-values
        that are closest to the midpoint.
    bfcount : None or positive integer, optional
        The count of confidence intervals for use with Bonferroni correction if
        the `'ste'` option is chosen for `extent`. If `'ste'` is not the value
        of `extent`, then this value is ignored.
    axes : matplotlib axes or None, optional
        The axes on which to plot the error bars; the default value of `None`
        uses the current axes.
    fw : positive real or None, optional
        The foot width (f.w.) of the error-bar's feet in plot coordinates. If
        the value is 0 or `None` (the default) then no feet are drawn.
    lw : positive real or None, optional
        The line-width of the error-bars in printer-points, which is passed
        along to matplotlib's `Axes.plot` method.
    lc : color or None, optional
        The line-color of the error-bars, which is passed along to matplotlib's
       `Axes.plot` method. The default is `'r'`.
    ms : positive real or None, optional
        The marker-size of the midpoint in printer-points, which is passed
        along to matplotlib's `Axes.plot` method.
    mc : color or None, optional
        The line-color of the error-bars, which is passed along to matplotlib's
       `Axes.plot` method. The default is `'r'`.
    zorder : real or None, optional
        The z-order parameter of matplotlib's `Axes.plot` method. The default is
        `None`.

    Returns
    -------
    list
        The plot-objects that were created by the function, which may vary
        depending on the function call. The objects always appear in the
        following order:
    """
    from numbers import Real
    # Get the axes option ready.
    ax = plt.gca() if axes is None else axes
    # We allow an (error-bar, midpoint) zorder parameter.
    if not isinstance(zorder, tuple):
        zorder = (zorder, zorder)
    if fw is None:
        fw = 0
    # Options we use for drawing lines and points.
    lnopts = dict(lw=lw, c=lc, zorder=zorder[0])
    ptopts = dict(ms=ms, c=mc, zorder=zorder[1])
    # Summarize the points; if we have multiple dimensions, we find the relevant
    # summary of each then call down with a single x and single y.
    if isinstance(x, Real):
        if isinstance(y, Real):
            # A single point.
            res = ax.plot(x, y, '.', **ptopts)
        else:
            # A vertical error-bar.
            (ymin, ymid, ymax) = summarize_dist(y, middle, extent,
                                                ci=ci, bfcount=bfcount)
            r1 = ax.plot([x,x], [ymin,ymax], '-', **lnopts)
            r2 = ax.plot(x, ymid, '.', **ptopts)
            res = r1 + r2
            if fw > 0:
                r1 = ax.plot([x-fw,x+fw], [ymin,ymin], '-', **lnopts)
                r2 = ax.plot([x-fw,x+fw], [ymax,ymax], '-', **lnopts)
                res = res + r1 + r2
    elif isinstance(y, Real):
        (xmin, xmid, xmax) = summarize_dist(x, middle, extent)
        r1 = ax.plot([xmin,xmax], [y,y],  '-', **lnopts)
        r2 = ax.plot(xmid, y, '.', **ptopts)
        res = r1 + r2
        if fw > 0:
            r1 = ax.plot([xmin,xmin], [y-fw,y+fw], '-', **lnopts)
            r2 = ax.plot([xmax,xmax], [y-fw,y+fw], '-', **lnopts)
            res = res + r1 + r2
    else:
        (xmin, xmid, xmax) = summarize_dist(x, middle, extent)
        (ymin, ymid, ymax) = summarize_dist(y, middle, extent)
        rx = plot_distbars(xmid, y,
                           middle=middle, extent=extent,
                           axes=ax, fw=fw, lw=lw, lc=lc, ms=ms, mc=mc,
                           zorder=zorder)
        ry = plot_distbars(x, ymid,
                           middle=middle, extent=extent,
                           axes=ax, fw=fw, lw=lw, lc=lc, ms=ms, mc=mc,
                           zorder=zorder)
        res = rx + ry
    return list(res)

def plot_prediction(dataset, k, model,
                    axes=None, figsize=(6,1), dpi=72*4,
                    min_alpha=0.5,
                    channels=(0,1,4,5),
                    round_labels=True):
    """Plots the data, true label, and predicted label (by model) of a dataset.
    
    `plot_prediction(dataset, k, model)` creates a `matplotlib` figure for
    `dataset[k]` (i.e., the `k`th subject/image in `dataset`). The `axes` are
    always returned.
    
    Parameters
    ----------
    dataset : HCPVisualDataset
        The dataset used to plot the predictions. This may alternately be a
        PyTorch dataloader, in which case the dataloader's dataset must be an
        `HCPVisualDataset` object.
    k : int
        The sample number or subject ID to plot. A sample number is just the
        index number for the subject in the dataset; if a number less than 1000
        is given, then it is assumed to eb a subject index, while if it is over
        1000, it is assumed to be a subject ID.
    model : PyTorch Module
        A UNet model or other PyTorch model that makes a segmentation
        of the images from the given `dataset`.
    axes : MatPlotLib axes or `None`, optional
        A set of axes onto which to plot the predictions. Must have a total
        flattened length of 3.
    figsize : tuple, optional
        A tuple of `(width, height)` in inches to use for the figure size. This
        is ignored if `axes` is provided. The default is `(6, 1)`.
    dpi : int, optional
        The number of dots per inch in the output image. If `axes` is provided,
        this option is ignored. The default is `72 * 4`.
    min_alpha : float, optional
        The minimum alpha value to show in the alpha channel of the image.
        Values below this level are replaced by the formula
        `adjusted_value = value * (1 - min_alpha) + min_alpha`. The default is
        `0.5`.
    channels : iterable of ints, optional
        When a dataset whose input images have more than 4 image channels is
        provided (i.e., the `'both'` datasets, which have 4 anatomical and 4
        functional image layers), then this list of 4 channels is used. By
        default this is `(0,1,4,5)`. This option is ignored if the dataset
        contains images with only 4 channels.
    round_labels : boolean, optional
        Whether to round every channel to either 1 or 0 before plotting the
        labels. The default is `True`.
    """
    if k > 1000:
        # We have a subject-ID instead of an index.
        k = np.where(dataset.sids == k)[0]
    (imdat, imlbl) = dataset[k]
    impre = model(imdat[None,:,:,:].float())
    if not model.apply_sigmoid:
        impre = torch.sigmoid(impre)
    impre = dataset.inv_transform(None, impre.detach()[0])
    (imdat, imlbl) = dataset.inv_transform(imdat, imlbl)
    import matplotlib.pyplot as plt
    if axes is None:
        (fig,axes) = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        made_axes = True
    else:
        made_axes = False
    try:
        # with imdat we want to adjust the alpha layer
        imdat = np.array(imdat)
        imdat[:,:,3] = imdat[:,:,3]*(1 - min_alpha) + min_alpha
        for (ax,im,rl) in zip(axes, [imdat,imlbl,impre], [0,0,round_labels]):
            if im.shape[2] > 4: im = im[:,:,:4]
            im = np.clip(im, 0, 1)
            if rl:
                z = im < 0.5
                im[z] = 0
                im[~z] = 1
            ax.imshow(im)
            ax.axis('off')
    except Exception:
        if made_axes:
            plt.close(fig)
        raise
    return axes
