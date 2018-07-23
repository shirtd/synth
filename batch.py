from ayasdi.core.api import Api
from util.args import parser
from util.afrl import *
from util.data import *
from module.predictor import LocalPredictor
from math import log as ln
import numpy as np
import requests
import json

LENS = ['Max', 'Raw Entropy', 'Approximate Kurtosis', 'Entropy', 'Mean',
        'Variance', 'Gaussian Density', 'Median', 'L1 Centrality',
        'L-Infinity Centrality', 'Neighborhood Graph Lens']

LENSPAIR = ['Neighborhood Lens', 'MDS coord', 'Metric PCA coord', 'PCA coord']

METRIC = ['Euclidean (L2)', 'Manhattan (L1)', 'Chebyshev (L-Infinity)', 'Correlation',
            'IQR Normalized Euclidean', 'Variance Normalized Euclidean'] #, 'Angle']
        # 'Norm Correlation','Norm Angle', 'Angle',
        # 'Absolute Correlation', 'Cosine', 'Categorical Cosine']
        # 'Hamming', 'Jaccard', 'Binary Jaccard']
        # 'Distance Correlation']

def get_lens(lens, res=60, equal=False, gain=2.5):
    return {'id' : lens, 'resolution': res, 'equalize': equal, 'gain': gain}

def batch(args):
    print('initializing predictor with %s and %s' % (args.train,args.test))
    pred = LocalPredictor(args.train, args.test, args.gt, args.col, args.fun, args.k, args.rows)
    base = {'column_set_id': pred.col_set['id']} #, 'lenses': lenses}
    res = {}
    for lens in LENSPAIR:
        res[lens] = {}
        lenses = [get_lens(lens+' 1'), get_lens(lens+' 2')]
        base['lenses'] = lenses
        for metric in METRIC:
            net = base.copy()
            net['metric'] = {'id' : metric}
            args.net = '_'.join([args.col, lens.replace(' ',''), metric.replace(' ','')])
            print('creating network %s' % args.net)
            try:
                pred.source.create_network(args.net, net)
            except requests.exceptions.HTTPError as err:
                print(' ! could not create network %s' % args.net)
                continue
            # print('running topological predict on %s' % (args.net))
            z = pred.topological_predict(args.net)
            a, c = pred.accuracy(z, l=[MODS[i] for i in pred.classes], verbose=False)
            s = pred.score(z)
            z['args'],z['accuracy'],z['confusion'],z['score']  = args.__dict__,a,c,s
            res[lens][metric] = z
            res[lens][metric]['probability_matrix'] = res[lens][metric]['probability_matrix'].tolist()
            res[lens][metric]['weights'] = res[lens][metric]['weights'].tolist()
            res[lens][metric]['confusion'] = res[lens][metric]['confusion'].tolist()
            res[lens][metric]['distances'] = res[lens][metric]['distances'].tolist()
        metrics = sorted(res[lens].keys(),key=lambda x: res[lens][x]['score'])
        for metric in metrics:
            print('\t%s score: %0.4f' % (metric, res[lens][metric]['score']))
    jdict = {'args' : args.__dict__, 'data' : res, 'mods' : MODS, 'mtoi' : MTOI, 'snrs' : SNRS,
            'columns' : pred.columns, 'truth' : {'mod' : pred._truth.tolist(), 'snr' : pred._snrs.tolist()}}
    dir,ftrain = os.path.split(args.train)
    ftest = os.path.splitext(os.path.split(args.train)[1])[0]
    fname = os.path.splitext(ftrain)[0] + '_' + ftest
    jout = os.path.join(dir, '_'.join([fname, args.col, args.fun])+'.json')
    print('writing to %s' % jout)
    with open(jout,'w') as f:
        json.dump(res, f)
    return res

if __name__ == '__main__':
    res = batch(parser['predict'].parse_args())

# for lens in LENSPAIR:
#     for metric in METRIC:
#         res[lens][metric]['probability_matrix'] = res[lens][metric]['probability_matrix'].tolist()
#         res[lens][metric]['weights'] = res[lens][metric]['weights'].tolist()
#         res[lens][metric]['confusion'] = res[lens][metric]['confusion'].tolist()
#         res[lens][metric]['distances'] = res[lens][metric]['distances'].tolist()
