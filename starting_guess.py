import multiprocessing
import argparse
import logging
from module_randomwalk import *

# TASK: Select 8 subjects as starting guess.
# Following is the method to generating training and testing dataset.
# all_sids = ny.data['hcp_retinotopy'].subject_ids[:-3]
# trn_sids = np.random.choice(
#     np.setdiff1d(all_sids, example_sids),
#     90, replace=False)
# val_sids = np.setdiff1d(all_sids, trn_sids)
# (trn_sids, val_sids) = [np.sort(sids) for sids in (trn_sids, val_sids)]
# ny.save('training_sids.json', trn_sids)
# ny.save('validate_sids.json', val_sids)

# We define this function for the annealing processes to run:
def anneal_job(boundary_weight):
    labels = np.array(min_label)
    score = autolabel_anneal(tess=fmap, labels=labels, xys=logxy, nsteps=nsteps,
                             annealing_speed=annealing_speed,
                             max_best_of=max_best_of, maxecc=logmaxecc,
                             boundary_weight=boundary_weight,
                             pair_weight=pair_weight)
    return (score, labels)

def config(parser):
    parser.add_argument('--sub', default=0, type=int)
    parser.add_argument('--hemi', default='lh', type=str)
    parser.add_argument('--scorefunc_type', default='0', type=int)
    return parser

parser = argparse.ArgumentParser()
parser = config(parser)
args = parser.parse_args()

if __name__ == '__main__':

    #sids = ny.data['hcp_retinotopy'].subject_ids
    #sids = [100610, 128935, 111312, 115017, 102311, 104416, 108323, 111514, 105923, 134829, 156334, 165436]
    sids = ny.load('validate_sids.json')
    sid = sids[args.sub]
    global h
    global scorefunc_type
    h = args.hemi
    scorefunc_type = args.scorefunc_type
    # logger
    logging.basicConfig(
        filename= 'results/log/autolabel_sg_' + str(sid) + '_' + str(h) + '_' + str(datetime.now()) + '.log',
        level=logging.INFO)
    logging.info('Lauching the randomwalk model for sub {0} -- hemi {1}'.format(str(sid), str(h)))
    set_up()
    logging.info('Configuration appears fine!')

    # The number of annealing jobs/cpus to use.
    nprocs = multiprocessing.cpu_count()
    # The number of simulated-annealing rounds to run.
    nrounds = 4
    # The number of steps in each round.
    nsteps = 40000
    # The annealing speed.
    annealing_speed = 4
    # The max best_of value.
    max_best_of = 1
    # The boundary is ramped up over the annealing rounds to this value.
    max_boundary_weight = 0.05
    # The max eccentricity.
    # In the HCP retinotopy experiment, the stimulus was 16° wide (so 8° of
    # eccentricity, maximum); we will treat 7° of eccentricity as the max
    # we want to include because there are edge-effects approaching the 8°
    # point.
    maxecc = 7.0
    # We keep the pair_weight as 1 at all times
    pair_weight = 1

    # We are going to run a number of parallel annealing rounds.
    # Using 8 different starting guess.
    xy, fmap, angle, eccen, hemi = autolabel_initializition(sid, h)
    # Before we start, we want to convert xy to be on a log-scale; this is
    # because eccentricity is exponentially-spaced in the visual field
    # relative to its spacing on cortex, so this generally improves the
    # ability of the minimizations to deal with low-eccentricity values.
    logxy = ny.to_logeccen(xy)
    logmaxecc = ny.to_logeccen(maxecc)

    # We will keep track of the minimum label configuration we've found as we go.
    (u, v) = fmap.tess.indexed_edges

    sids_sg = [None] + [int(id) for id in np.loadtxt(str(h) + '_centers.txt')]

    dice_score_sg = dict()

    for sid_sg in sids_sg:

        logging.info('Starting guess using subject {}'.format(sid_sg))

        label0 = starting_guess(hemi, fmap, h, maxecc, sid_sg=sid_sg)

        min_score = score_labels(u, v, label0, logxy, maxecc=logmaxecc,
                                 boundary_weight=0, pair_weight=pair_weight)
        init_score = min_score
        min_label = np.array(label0)

        ## simulated annealing

        t0 = time.time()
        for roundno in range(nrounds):
            # Recalculate the boundary weight for this annealing round.
            boundary_weight = roundno * max_boundary_weight / (nrounds - 1)
            # because we've changed the boundary_weight, we need to recalculate the min score
            min_score = score_labels(u, v, min_label, logxy, maxecc=logmaxecc,
                                     boundary_weight=boundary_weight,
                                     pair_weight=pair_weight)
            score0 = min_score
            # Print a progress message.
            # logging.info('Running parallel-annealing round {} (initial score: {})...'.format(roundno + 1, min_score))
            # In parallel, we do the simulated annealing
            with multiprocessing.Pool(nprocs) as pool:
                results = pool.map(anneal_job, [boundary_weight] * nprocs)
            # Of the results, which did the best?
            for (score, labels) in results:
                if score >= min_score: continue
                min_score = score
                min_label[:] = labels
            logging.info('  {} annealing jobs finished with score change of {}'.format(
                nprocs, (min_score - score0) / score0 * 100))

        t1 = time.time()

        # Print a message about elapsed time.
        dt = t1 - t0
        m = int((dt - np.mod(dt, 60)) / 60)
        s = dt - 60 * m
        nstepstot = (nprocs * nsteps * nrounds) * 1000
        msperstep = dt / nstepstot
        logging.info('{} steps taken in {}m, {}s ({} ms / step).'.format(nstepstot, m, s, msperstep))
        dscore = min_score - init_score
        logging.info('Score Change: {} ({} per second; {} per step).'.format(dscore, dscore / dt, dscore / nstepstot))

        visualization(u, v, fmap, min_label, logxy, logmaxecc, angle, eccen, sid, h, sid_sg)
        predict_label = min_label

        # Load in the ground-truth for just the LH of the subject we've been using in
        # the minimizations above
        # (sid is the subject ID and h is the hemisphere ('lh' or 'rh'))
        true_path = os.path.join(os.getcwd() + '/visual_area_labels', '%s.%d_varea.mgz' % (h, sid))
        logging.info('Saving the results...')
        if os.path.isfile(true_path):
            true_labels = ny.load(true_path)
            true_labels = true_labels[fmap.labels]
            ds = dice_score(true_labels, predict_label)
            dice_score_sg[str(sid_sg)] = ds
            logging.info('Dice score of prediction and the ground truth is {}.'.format(ds))
        else:
            logging.info("True labels don't exist!")
            ds = None
        with open('results/label/' + str(sid) + '_' + str(h) + '_sg_'+str(sid_sg) +'_boundary_weight_0.05' + '.pickle', 'wb') as f:
            pickle.dump({'predict_label': predict_label, 'dice_score': ds}, f)
        if dice_score_sg != {}:
            key_max = max(dice_score_sg.keys(), key=(lambda k: dice_score_sg[k]))
            logging.info('Among all starting guess, result of {} as sg has the highest dice score of {}'.format(key_max, dice_score_sg[key_max]))
        logging.info('Done.')




