import argparse
import logging
from module_randomwalk import *


# TASK: Score hand-drawn labels.

def config(parser):
    parser.add_argument('--sub', default=0, type=int)
    parser.add_argument('--hemi', default='lh', type=str)
    return parser

parser = argparse.ArgumentParser()
parser = config(parser)
args = parser.parse_args()


if __name__ == '__main__':

    sids = ny.data['hcp_retinotopy'].subject_ids
    sid = sids[args.sub]
    h = args.hemi
    logging.basicConfig(filename='results/log/handlabel_'+str(sid)+'_'+str(h)+'_'+str(datetime.now())+'.log', level=logging.INFO)
    logging.info('Scoring the hand-drawn labels for sub {0} -- hemi {1}'.format(str(sid), str(h)))
    set_up()
    logging.info('Configuration appears fine!')
    sub = ny.hcp_subject(sid)
    hemi = sub.hemis[h]
    fmap = ny.to_flatmap('occipital_pole', hemi)
    #load true labels
    true_path = os.path.join(os.getcwd() + '/visual_area_labels', '%s.%d_varea.mgz' % (h, sid))
    logging.info('Loading the hand-drawn labels...')
    if not os.path.isfile(true_path):
        logging.info('Hand-drawn labels are not found...')
    else:
        true_labels = ny.load(true_path)
        true_labels = true_labels[fmap.labels]

        rdat = ny.retinotopy_data(fmap, 'prf_')
        (x, y) = ny.as_retinotopy(rdat, 'geographical')
        xy = np.transpose([x, y]).astype('float32')
        maxecc = 7.0
        logxy = ny.to_logeccen(xy)
        logmaxecc = ny.to_logeccen(maxecc)
        (u, v) = fmap.tess.indexed_edges
        score = []
        for pair_weight in [1.0, 0.8, 0.6, 0.4, 0.2]:
            score.append(score_labels(u, v, true_labels, logxy, maxecc=logmaxecc,
                                      boundary_weight=1-pair_weight, pair_weight=pair_weight))

        import csv
        label_path = 'results/score/score_hand-drawn_label.csv'
        if not os.path.isfile(true_path):
            with open(label_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sid', 'hemi', '1.0', '0.8', '0.6', '0.4', '0.2'])
        with open(label_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([str(sid), str(h), str(score[0]), str(score[1]), str(score[2]), str(score[3]), str(score[4])])
    logging.info('Done.')