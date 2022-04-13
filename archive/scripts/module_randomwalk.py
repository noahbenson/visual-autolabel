# Import some standard/utility libraries:
import os, sys, time, h5py, zipfile
import six  # six provides python 2/3 compatibility

# Import our numerical/scientific libraries, scipy and numpy:
import numpy as np
import scipy as sp

# The pimms (Python Immutables) library is a utility library that enables lazy
# computation and immutble data structures; https://github.com/noahbenson/pimms
import pimms

# The neuropythy library is a swiss-army-knife for handling MRI data, especially
# anatomical/structural data such as that produced by FreeSurfer or the HCP.
# https://github.com/noahbenson/neuropythy
import neuropythy as ny

# Import graphics libraries:
# Matplotlib/Pyplot is our 2D graphing library:·
import matplotlib as mpl
import matplotlib.pyplot as plt
# We also use the 3D graphics library ipyvolume for 3D surface rendering
import ipyvolume as ipv

import random
from numba import jit, jitclass
from numba import float32, int64
import math
#import pathos.multiprocessing as multiprocessing

import pickle
from datetime import datetime

scorefunc_type = 0
h = 'lh'

def set_up():
    # Additional matplotlib preferences:
    font_data = {'family': 'sans-serif',
                 'sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial'],
                 'size': 10,
                 'weight': 'light'}
    mpl.rc('font', **font_data)
    # we want relatively high-res images, especially when saving to disk.
    mpl.rcParams['figure.dpi'] = 72 * 2
    mpl.rcParams['savefig.dpi'] = 72 * 4

    # Check that HCP credentials were found:
    if ny.config['hcp_credentials'] is None:
        raise Exception('No valid HCP credentials were found!\n'
                        'See above instructions for remedy.')

    # Check that we can access the HCP database:
    # To do this we grab the 's3fs' object from neuropythy's 'hcp' dataset; this
    # object maintains a connection to Amazon's S3 using the hcp credentials. We use
    # it to perform a basic ls operation on the S3 filesystem. If this fails, we do
    # not have a working connection to the S3.
    try:
        files = ny.data['hcp'].s3fs.ls('hcp-openaccess')
    except Exception:
        files = None
    if files is None:
        raise Exception('Could not communicate with S3!\n'
                        'This probably indicates that your credentials are wrong'
                        ' or that you do not have an internet connection.')


def autolabel_initializition(sid, h):
    # get the subject
    sub = ny.hcp_subject(sid)
    # We will use one of the subject's hemisphere's (eigher lh or rh).
    # Get this hemisphere object from the subject object.
    hemi = sub.hemis[h]

    # We don't need the whole hemisphere; just the rear occipital hemisphere.
    fmap = ny.to_flatmap('occipital_pole', hemi)

    # We also want to extract the visual-field coordinates. These coordinates
    # are encoded on the fmap as the properties 'prf_polar_angle' and
    # 'prf_eccentricity', which we can extract using neuropythy:
    rdat = ny.retinotopy_data(fmap, 'prf_')
    # Convert the retinotopy data to (x,y) coordinates.
    (x, y) = ny.as_retinotopy(rdat, 'geographical')
    xy = np.transpose([x, y]).astype('float32')

    # We will also want the eccentricity of the vertices for convenience.
    eccen = np.sqrt(x ** 2 + y ** 2)
    # We will want to look at the angle as well.
    angle = np.arctan2(y, x)
    # We typically store polar angle as degrees of clockwise rotation
    # starting at the positive y-axis.
    angle = 90 - 180 / np.pi * angle
    angle = np.mod(angle + 180, 360) - 180
    return xy, fmap, angle, eccen, hemi

def starting_guess(hemi, fmap, h, max_eccen, sid_sg = None):

    pred = ny.vision.predict_retinotopy(hemi)

    angle0 = pred['angle']
    eccen0 = pred['eccen']
    if sid_sg is None:
        label0 = pred['varea']
    else:
        path = os.path.join('visual_area_labels', '%s.%d_varea.mgz' % (h, sid_sg))
        label0 = ny.load(path)
        sub_sg = ny.hcp_subject(sid_sg)
        hemi_sg = sub_sg.hemis[h]
        label0 = hemi_sg.interpolate(hemi, label0, method='nearest')

    # These are for the whole hemisphere, so we take the verrtices
    # that are in the fmap only.
    (angle0, eccen0, label0) = [np.array(prop[fmap.labels])
                                for prop in (angle0, eccen0, label0)]

    # We should limit the initial label predictions to the max_eccen.
    label0[eccen0 > max_eccen] = 0
    # We also want only V1, V2, and V3 labels (as well as blanks, 0).
    label0[~np.isin(label0, [0, 1, 2, 3])] = 0

    return label0


## the score function
spec1 = [('xy1', float32[:]), ('xy2', float32[:]), ('maxecc', float32),]

class scorefunc(object):

    def __init__(self, xy1, xy2, maxecc):
        self.xy1 = xy1
        self.xy2 = xy2
        self.maxecc = maxecc

    def score_xdist(self):
        pass

    def score_ydist(self):
        pass

    def score_eccdist(self):
        pass

@jitclass(spec1)
class scorefunc_Quadratic(scorefunc):
    __init__scorefunc = scorefunc.__init__

    def __init__(self, xy1, xy2, maxecc):
        #scorefunc.__init__(self, xy1, xy2, maxecc)
        self.__init__scorefunc(xy1, xy2, maxecc)

    def score_xdist(self):
        '''
        score_xdist(xy1, xy2) yields the score value for a pair of vetices that straddle
          a boundary known to correspond to the x-axis, such as the V2-V3 boundary.
        '''
        return self.xy1[1] ** 2 + self.xy2[1] ** 2

    def score_ydist(self):
        '''
        score_ydist(xy1, xy2) yields the score value for a pair of vetices that straddle
          a boundary known to correspond to the y-axis, such as the V1-V2 boundary.
        '''
        return self.xy1[0] ** 2 + self.xy2[0] ** 2

    def score_eccdist(self):
        '''
        score_eccdist(xy1, xy2, maxecc) yields the score value for a pair of vetices
          that straddle a boundary known to correspond to the max-eccentricity (or
          outer stimulus boundary) such as the V1-, V2-, and V3-peripheral boundaries.
        '''
        ecc1 = np.sqrt(self.xy1[0] ** 2 + self.xy1[1] ** 2)
        ecc2 = np.sqrt(self.xy2[0] ** 2 + self.xy2[1] ** 2)
        return ((ecc1 - self.maxecc) ** 2 + (ecc2 - self.maxecc) ** 2)

spec2 = [('h', int64)]

@jitclass(spec1+spec2)
class scorefunc_Absolute(scorefunc):
    __init__scorefunc = scorefunc.__init__

    def __init__(self, xy1, xy2, maxecc, h):
        self.__init__scorefunc(xy1, xy2, maxecc)
        self.hemi = h

    def score_xdist(self):
        '''
        score_xdist(xy1, xy2) yields the score value for a pair of vetices that straddle
          a boundary known to correspond to the x-axis, such as the V2-V3 boundary.
        '''
        return (abs(self.xy1[1])*self.xy1[1] + abs(self.xy2[1]) * self.xy2[1])*self.hemi

    def score_ydist(self):
        '''
        score_ydist(xy1, xy2) yields the score value for a pair of vetices that straddle
          a boundary known to correspond to the y-axis, such as the V1-V2 boundary.
        '''
        return (abs(self.xy1[0])*self.xy1[0] + abs(self.xy2[0]) * self.xy2[0])*self.hemi

    def score_eccdist(self):
        '''
        score_eccdist(xy1, xy2, maxecc) yields the score value for a pair of vetices
          that straddle a boundary known to correspond to the max-eccentricity (or
          outer stimulus boundary) such as the V1-, V2-, and V3-peripheral boundaries.
        '''

        ecc1 = np.sqrt(self.xy1[0] ** 2 + self.xy1[1] ** 2)-self.maxecc
        ecc2 = np.sqrt(self.xy2[0] ** 2 + self.xy2[1] ** 2)-self.maxecc
        return (abs(ecc1) * ecc1 + abs(ecc2) * ecc2)*self.hemi


@jit('float32(int64, int64, float32[:], float32[:], float32)', nopython=True)
def score_pair(lbl1, lbl2, xy1, xy2, maxecc):
    '''
    score_pair(lbl1, lbl2, xy1, xy2, maxecc) yields the score of an edge whose
      endpoints have lbl1 and lbl2 and are at visual (x,y) coordinates xy1 and xy2.
    '''
    if scorefunc_type == 0:
        sf = scorefunc_Quadratic(xy1, xy2, maxecc)
    else:
        if h == 'lh':
            hemi = 1
        elif h == 'rh':
            hemi = -1
        sf = scorefunc_Quadratic(xy1, xy2, maxecc, hemi)
    # There's no score for labels that are the same.
    if lbl1 == lbl2: return 0
    # To make things easier, make sure lbl1 <= lbl2.
    if lbl2 < lbl1: (lbl1, lbl2, xy1, xy2) = (lbl2, lbl1, xy2, xy1)
    if lbl1 == 0:
        if lbl2 == 1:
            return sf.score_eccdist()
        elif lbl2 == 2:
            return sf.score_eccdist()
        else:
            s1 = sf.score_eccdist()
            s2 = sf.score_ydist()
            return s1 if s1 < s2 else s2
    elif lbl1 == 1:
        if lbl2 == 2:
            return sf.score_ydist()
        else:
            return np.inf  # V1 and V3 should not touch
    else:
        return sf.score_xdist()  # V2-V3

@jit('Tuple((float32, int64))(int64[:], int64[:], float32[:,:], int64, int64, float32)')
def score_pair_change(nei, lbl, xy, a, newlbl, maxecc):
    '''
    score_pair_change(nei, lbls, xy, a, newlbl, maxecc) yields the change
      in score that will occur for changing vertex a from its current label,
      given by lbls[a], to newlbl.

    The change in score is yielded as (delta_score, delta_edges) where the
    delta_score is the change in the scoring function as defined by
    score_pair and the delta_edges is the change in the number of edges
    that straddle the boundaries.

    If the change in score is not valid (i.e., if the change causes V1 to
    border V3) then (inf, 0) is returned.

    The nei and xy parameters are the adjacent vertices to a and the visual
    field coordinate of a, respectively.
    '''
    # score for changing lbl[a] into newlbl
    la = lbl[a]
    if la == newlbl: return (0, 0)
    delta_sc = 0
    delta_bs = 0
    xy_a = xy[a]
    for u in nei:
        lu = lbl[u]
        xy_u = xy[u]
        if la != lu: delta_sc -= score_pair(la, lu, xy_a, xy_u, maxecc)
        tmp = score_pair(newlbl, lu, xy_a, xy_u, maxecc)
        # If the score is infinite, we know that this isn't really a valid
        # change, so return infinity to indicate this.
        if not np.isfinite(tmp): return (np.inf, 0)
        delta_sc += tmp
        if la == lu:
            # it used to have 0 score; now it has a positive one
            delta_bs += 1
        elif newlbl == lu:
            # it used to have some positive score, now has a 0
            delta_bs -= 1
    return (delta_sc, delta_bs)


def score_pair_search(nei, lbl, xy, aa, newlbls, maxecc):
    '''
    score_pair_search(nei, lbls, xy, a, newlbl, maxecc) is exactly like score_pair_change
      except that instead of accepting a single vertex a and a single new label, it accepts
      a vector of each and yields (delta_score, delta_boundary, ii) where ii is the index
      into a and newlbl of the vertex with the lowest score. Additionally, the nei
      argument must be the entire list of neighborhoods.
    '''
    min_ii = 0
    min_sc = np.inf
    min_bs = 0
    for (ii, (a, newlbl)) in enumerate(zip(aa, newlbls)):
        (dsc, dbs) = score_pair_change(nei[a], lbl, a, newlbl, maxecc)
        if dsc < min_sc:
            min_sc = dsc
            min_bd = dbs
            min_ii = ii
    return (min_sc, min_bs, min_ii)


def score_labels(u, v, labels, xys, maxecc=7.0, boundary_weight=0.0, pair_weight=1.0):
    '''
    score_labels(u, v, labels, xys) yields the total score of the given set of
      labels using the given set of edges (u,v) and the visual-field coordinates
      in the (n x 2) matrix xys.

    The optional parameters boundary_weight (default 0) and pair_weight (default
    1) can be set to change how the score considers the boundary length versus
    the sum of the pair-wise boundary-edge scores. The total score returned is
    the sum of score_pair for all edges times the pair_weight plus the length
    of the boundary (the number of edges on the boundary) times the
    boundary_weight.
    '''
    xys = np.asarray(xys).astype('float32')
    if xys.shape[1] != 2: xys = xys.T
    ii = np.where(labels[u] != labels[v])[0]
    sc = 0
    for (uu, vv) in zip(u[ii], v[ii]):
        sc += score_pair(labels[uu], labels[vv], xys[uu], xys[vv], maxecc)
    return (pair_weight * sc + boundary_weight * len(ii))


def score_change(neis, labels, xys, a, newlbl,
                 maxecc=7.0, boundary_weight=0.0, pair_weight=1.0):
    '''
    score_change(neis, labels, xys, a, newlbl) yields the change in score that
      results from changing the label for vertex a from labels[a] to newlbl.

    The argument neis must be a tuple of values (one value per vertex) such
    that each value nei[u], associated with vertex u, is itself a tuple of
    the vertices adjacent to vertex u in the vertex-edge graph. A data
    structure like this can be obtained from a cortical tesselation object
    via tess.indexed_neighborhoods.

    The optional aguments maxecc (default: 7), bounday_weight (default: 0),
    and pair_weight (default: 1) may be given and are handled as in
    score_labels, above.
    '''
    xys = np.asarray(xys).astype('float32')
    if xys.shape[1] != 2: xys = xys.T
    (delta_pair, delta_bound) = score_pair_change(np.asarray(neis[a]), labels, xys, a, newlbl, maxecc)
    return (pair_weight * delta_pair + boundary_weight * delta_bound)


## random walk

def boundary_update(nei, labels, u, newlbl, boundary):
    '''
    We keep track of the boundary during a random walk with a set object; in order to
    update that object, we call boundary_update(nei, labels, u, newlbl, boundary),
    which adds/removes all necessary edge-pairs due to the change.
    '''
    oldlbl = labels[u]
    if oldlbl == newlbl: return boundary
    for v in nei:
        lv = labels[v]
        if oldlbl == lv:
            boundary.add((u, v) if u < v else (v, u))
        elif newlbl == lv:
            boundary.remove((u, v) if u < v else (v, u))
    return boundary


def boundary_set(u, v, labels):
    '''
    boundary_set(u, v, labels) yields a new boundary-set containing all the boundary
      edges in the given set of edges defined by (u, v) for the given labels.
    '''
    ii = np.where(labels[u] != labels[v])[0]
    return set([(a, b) if a < b else (b, a) for (a, b) in zip(u[ii], v[ii])])


def autolabel_random_step(tess, labels, xys, p_good=1.0, p_bad=0.01,
                          maxecc=7, best_of=1, boundary=None,
                          boundary_weight=0, pair_weight=1):
    '''
    autolabel_random_step(tess, labels, xys) performs a single random step in
      place on the given labels which are associated with the given
      tesselation (or mesh) object tess and the given visual-field positions
      xys. The return value is the change in score if a change was made and
      None if a change was not made.

    Note that running random step often does not result in a change because
    the randomly-drawn change was subsequently rejected. Rejection is
    determined by the optional arguments p_good and p_bad (below) as well as
    the best_of option.

    The following optional arguments are accepted:
      * p_good (default: 1) determines the probability of accepting a
        randomly-generated change to the labels, given that the overall
        score function goes down due to the change. Since we are trying
        to minimize the score, this should generally be high.
      * p_bad (default: 0.01) determines the probability of accepting a
        randomly-generated change to the labels, given tha the overall
        score function goes up due to the change. Since we are trying to
        minimize the score, this should generally be low.
      * maxecc (default: 7) determines the maximum eccentricity value to
        use when scoring the labels.
      * best_of (default: 1) may be set to a positive integer n > 1 in order
        to, instead of generating just 1 random label change, generate n
        random changes and to use the one with the lowest score.
      * boundary (default: None) specifies the boundary set; passing this
        to random_step for throughout a minimization will increase speed.
      * boundary_weight (default: 0) and pair_weight (default: 1) specify
        the relative weights of the boundary-length and pair-wise score
        components of the score function.
    '''
    # in case a mesh/hemi was passed instead of a tesselation
    if not ny.is_tess(tess): tess = tess.tess
    neis = tess.indexed_neighborhoods
    # make a boundary if none exists
    if boundary is None:
        (u, v) = tess.indexed_edges
        boundary = boundary_set(u, v, labels)
    # pick some of the edges in the boundary at random
    if best_of > 1:
        es = random.sample(boundary, best_of)
        qs = np.random.rand(best_of) > 0.5
        scores = [score_change(neis, labels, xys,
                               u if q else v, labels[v if q else u],
                               maxecc=maxecc,
                               boundary_weight=boundary_weight,
                               pair_weight=pair_weight)
                  for ((u, v), q) in zip(es, qs)]
        ii = np.argmin(scores)
        (dsc, (u, v), q) = (scores[ii], es[ii], qs[ii])
        (w, z) = (u, v) if q else (v, u)
    else:
        (u, v) = random.sample(boundary, 1)[0]
        q = np.random.rand() > 0.5
        (w, z) = (u, v) if q else (v, u)
        dsc = score_change(neis, labels, xys, w, labels[z], maxecc=maxecc,
                           boundary_weight=boundary_weight, pair_weight=pair_weight)
    if not np.isfinite(dsc): return 0
    p = p_good if dsc < 0 else p_bad
    # If we fail this draw, we reject the proposed change.
    if np.random.rand() > p: return None
    # Otherwise we implement it.
    oldlbl = labels[w]
    newlbl = labels[z]
    if oldlbl == newlbl: return 0  # (shouldn't ever happen)
    boundary_update(neis[w], labels, w, newlbl, boundary)
    labels[w] = newlbl
    return dsc


def autolabel_anneal(tess, labels, xys, nsteps=250000,
                     maxecc=7, boundary_weight=0, pair_weight=1,
                     annealing_speed=8, max_best_of=5):
    '''
    autolabel_anneal(tess, labels, xys) runs a single round of simulated annealing
      on the given tesselation (or mesh) tess, the given labels, and the given
      visual-field coordiantes xys. The return value is the new score; changes to
      the labels are made in-place.

    During annearling, the "temperature" of the steps begins at a very high level
    and drops steadily. This is achieved by altering the probability of accepting
    a label-change that has a positive score-change (p_bad) while holding the
    probability of accepting a label-change that has a negative score-change
    (p_good) constant at 1. For a step k, p_bad is set to exp(-s f) where s is
    the optional parameter annealing_speed and f = k/(nsteps-1). Additionally,
    the best_of parameter to the random_label_step is gradually decreased from
    max_best_of to 1 at a similar rate

    The following options are accepted:
      * nsteps (default: 250,000) specifies the numbe of steps in the annealing
        process.
      * maxecc (default: 7) specifies the maximum eccentricity to include in the
        labels.
      * boundary_weight (default: 0) and pair_weight (default: 1) specify the
        weights of the boundary-based and pair-based component of the scoring
        function.
      * annealing_speed (default: 8) specifies the speed at which the annealing
        takes place. Higher numbers cool the simulation faster.
    '''
    # If tess is actually a mesh or hemisphere, get the tess object.
    if not ny.is_tess(tess): tess = tess.tess
    (u, v) = tess.indexed_edges
    # If there's a boundary_weight, make a boundary.
    if boundary_weight == 0:
        boundary = None
    else:
        boundary = boundary_set(u, v, labels)
    # Make sure the given parameters are reasonable.
    assert annealing_speed > 0, 'annealing_speed must be a positive number'
    assert max_best_of >= 1, 'max_best_of must be greater than 1'
    # We will keep track of the best score/labels so far.
    min_score = np.inf
    min_labels = np.array(labels)
    # Calculate the initial score.
    score = score_labels(u, v, labels, xys, maxecc=maxecc,
                         boundary_weight=boundary_weight,
                         pair_weight=pair_weight)
    # Run the steps:
    for (k, f) in enumerate(np.linspace(0, 1, nsteps)):
        # Figure out our random_label_step parameters for this step:
        pbad = np.exp(-annealing_speed * f)
        bo = np.exp(-annealing_speed * (1 - f))
        bo = int(np.ceil(max_best_of * bo))
        # Run a random step:
        dsc = autolabel_random_step(tess, labels, xys,
                                    maxecc=maxecc, boundary=boundary,
                                    p_good=1, p_bad=pbad, best_of=bo,
                                    boundary_weight=boundary_weight,
                                    pair_weight=pair_weight)
        if dsc is None: continue  # No step was taken (change rejected).
        score += dsc
        if score < min_score:
            # New lowest score!
            min_score = score
            min_labels[:] = labels
    # We've run all the steps; now just set the labels to the min-scored labels that
    # we found during the search and return the final score.
    labels[:] = min_labels
    return min_score


# # We define this function for the annealing processes to run:
# def anneal_job(boundary_weight):
#     labels = np.array(min_label)
#     score = autolabel_anneal(tess=fmap, labels=labels, xys=logxy, nsteps=nsteps,
#                              annealing_speed=annealing_speed,
#                              max_best_of=max_best_of, maxecc=logmaxecc,
#                              boundary_weight=boundary_weight,
#                              pair_weight=pair_weight)
#     return (score, labels)


def visualization(u, v, fmap, label, logxy, logmaxecc, angle, eccen, sid, h, sid_sg):
    # Plot the result of the above steps:
    boundary = label[u] != label[v]
    boundary_u = u[boundary]
    boundary_v = v[boundary]
    boundary_lbl = [tuple(sorted([label[uu], label[vv]]))
                    for (uu, vv) in zip(boundary_u, boundary_v)]
    boundary_coords = np.mean([fmap.coordinates[:, boundary_u],
                               fmap.coordinates[:, boundary_v]],
                              axis=0)

    # Setup the matplotlib/pyplot figure.
    (fig, axs) = plt.subplots(1, 2, figsize=(5.5, 5.5 / 2), dpi=144)
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    ny.cortex_plot(fmap, underlay=None, color=angle, axes=axs[0],
                   cmap='polar_angle', vmin=-180, vmax=180)
    ny.cortex_plot(fmap, underlay=None, color=eccen, axes=axs[1],
                   cmap='eccentricity', vmin=0, vmax=90)

    for (ax, name) in zip(axs, ['angle', 'eccen']):
        ax.axis('equal')
        ax.axis('off')
        ax.set_title(name)
        # color the points by their pairwise scores
        clrs = [score_pair(label[uu], label[vv], logxy[uu], logxy[vv], logmaxecc)
                for (uu, vv) in zip(boundary_u, boundary_v)]
        ax.scatter(boundary_coords[0], boundary_coords[1],
                   c=clrs, s=0.5, cmap='gray')
    plt.savefig('results/fig/' + str(sid) + '_' + str(h) + '_sg_'+str(sid_sg) + '_boundary_weight_0.05' + '.png')


# In order to compare the true labels with the predicted labels, we need
# to use a metric sometimes called the "dice coefficient" or the "bray-curtis
# dissimilarity":
# https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity
# (they are basically the same thing)
def dice_score(true_labels, pred_labels, visual_areas=[1, 2, 3]):
    '''
    dice_score(labels1, labels2) yields the Sørensen-Dice coefficient of the
      label vectors `labels1` and `labels2`. The labels must both be vectors
      of equal length, each of which contains integer labels. Labels with
      the value 0 are excluded.

    The optional argument `visual_areas` (default: `[1,2,3]`) may be given in
    order to instruct the dice_score to focus on only a subset of the labels.
    For example, to get the Dice coefficient for V1 only, you can use the
    call `dice_score(labels1, labels2, visual_areas=[1])`. By default, this
    option includes V1, V2, and V3; the return value is the the number of
    vertices in the two labelsets that are equal divided by the number of
    vertices (with vertices in both sets that are 0 excluded).
    '''
    vas = np.array([visual_areas]) if pimms.is_int(visual_areas) else np.asarray(visual_areas)
    # the included indices
    ii = np.where(np.isin(true_labels, vas) | np.isin(pred_labels, vas))[0]
    true_labels = true_labels[ii]
    pred_labels = pred_labels[ii]
    return np.sum(true_labels == pred_labels) / len(ii)


def label_prob_per_vertex(sid, h):
    '''
    calculate the probability that a particular vertex is in V1, V2, or V3 (has label 1, 2, or 3)
    '''
    #for all subjects
    sub = ny.hcp_subject(sid)
    hemi = sub.hemis[h]
    fsa = ny.freesurfer_subject('fsaverage')
    fshm = fsa.hemis[h]
    hemi.interpolate(labels, fshm, method='nearest')
    #Once we have all the labels on a common set of vertices (fsaverage),
    # we can calculate the probability of a vertex being in V1, V2, or V3

def prior_label_prob_fsa():

    # Align all subjects' labels into fsaverage
    import pandas as pd
    sids = ny.data['hcp_retinotopy'].subject_ids
    fsa = ny.freesurfer_subject('fsaverage')
    for h in ['lh', 'rh']:
        fshm = fsa.hemis[h]
        df_fsa = pd.DataFrame([], columns=range(len(fshm.labels)))
        for sid in sids:
            sub = ny.hcp_subject(sid)
            path = os.path.join('visual_area_labels', '%s.%d_varea.mgz' % (h, sid))
            if not os.path.isfile(path):
                continue
            else:
                hemi = sub.hemis[h]
                labels = ny.load(path)
                labels_interpolated = hemi.interpolate(fshm, labels, method='nearest')
                df_fsa = df_fsa.append([labels_interpolated])
                # Calculate the probability that a particular vertex is in V1, V2, or V3 (has label 1, 2, or 3)
                df_count = df_fsa.apply(pd.value_counts)
                label_prob_fsa = df_count/len(df_fsa)
                label_prob_fsa = label_prob_fsa.fillna(0)
                label_prob_fsa.to_csv(str(h) + '_label_prob_fsa.csv', header=False)