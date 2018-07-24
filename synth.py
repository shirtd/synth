from util.data import *
from util.afrl import *
from multiprocessing import Process, Pool, cpu_count
import scipy.spatial.distance as scid
from functools import partial
from numpy import mean, var, dot
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import json, csv
import os, sys
import math

# MAIN
# parser['main'].add_argument('-c', '--chunk', type=int, default=0, help='training data chunk (0-14). default: 0')
# parser['main'].add_argument('--test', type=int, default=-1, help='test data chunk (0-14). -1 for no test. default: -1')
# parser['main'].add_argument('--snr', type=int, default=10, help='signal-to-noise ratio (+-10, +-6, +-2,). default: 10')
parser = argparse.ArgumentParser(description='construct synthetic features.')
parser.add_argument('-c', '--chunk', type=int, default=6, help='training data chunk (0-14). default: 0')
parser.add_argument('-s','--snr', type=int, nargs='+', default=None, help='signal-to-noise ratios (+-10, +-6, +-2,). default: None (all)')
parser.add_argument('-f','--file', default=None, help='filename. alternative to providing dir, chunk, and snr.')
parser.add_argument('--suff', default='', help='suffix. default: none.')
parser.add_argument('--dir', default='data', help='input data directory. default: data')
parser.add_argument('--tdir', default='data_out', help='output data directory. default: data_out')
parser.add_argument('--jdir', default='data_out_json', help='json data directory. default: data_out_json')
parser.add_argument('--fun', default='projection', help='synthetic feature function. default: projection.')
parser.add_argument('--metric', default='euclidean', help='function on centroids. only used for projection function.')
parser.add_argument('--centroids', default=None, help='centroid file.')
args = parser.parse_args()

if args.snr == None:
    args.snr = SNRS

if args.file == None:
    args.file = os.path.join(args.dir,name_file(args.chunk, args.snr, args.suff) + '.csv')

def zscore(col):
    return (col - col.mean())/col.std(ddof=0)

def euclidean(v1, v2):
    return scid.euclidean(v1, v2)

print(' > reading file %s' % args.file)
dfall = pd.read_csv(args.file)
dfgt = dfall[ICOLS]
df = dfall.drop(ICOLS, axis=1)
X = df.values
Xmods = {mod : df.loc[dfgt['MOD'] == mod].values for mod in MODS}
Xmods['AllMod'] = X
# X = df.apply(zscore).values

def activation(d, p, c):
    return d(p[c['mask']],c['centroid'])

def activations(keys, cs, d, i):
    f = partial(activation, d, X[i])
    return np.fromiter((f(cs[k]) for k in keys), np.float, count=len(keys))

def class_activation(d, p, c):
    return d(p[c['mask']],c['class_centroid'])

def class_activations(keys, cs, d, i):
    f = partial(class_activation, d, X[i])
    return np.fromiter((f(cs[k]) for k in keys), np.float, count=len(keys))

def addfeats(fun, feats, keys, i):
    x = X[i]
    f = lambda k: fun(x[feats[k]['mask']])
    return np.fromiter((f(k) for k in keys), np.float, count=len(keys))
    # return np.array(list(map(lambda k: fun(x[feats[k]['mask']]), keys)))

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle(v1, v2):
    v1_u,v2_u = unit_vector(v1), unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

functions = {
    'projection' : activations, 'class_projection' : class_activations,
     #'crossover' : crossover, 'possum' : possum, 'negsum' : negsum,
    # 'icrossover' : partial(crossover, 0, 1024),
    # 'qcrossover' : partial(crossover, 1024, 2048),
    'median' : np.median, 'mean' : np.mean, 'var' : np.var,
    'min' : min, 'max' : max, 'sum' : sum
}

metrics = {
    'euclidean' : euclidean,
    'dot' : dot, 'angle' : angle #,
    # 'zangle' : complex_angle
}

def get_means(feats, key):
    mod,mask = feats[key]['mod'], feats[key]['mask']
    return {'class_c' : [mean(Xmods[mod][:,col]) for col in mask],
            'c' : [mean(X[:,col]) for col in mask]}
    # return {'class_c' : np.average(Xmods[mod][:,mask], axis=0).tolist(),
    #         'c' : np.average(X[:,mask],axis=0).tolist()}
    # # imask,qmask = getiq(mask)
    # # zmean = [iqmean(x,icol,qcol) for icol,qcol in zip(imask,qmask)]
    # # iqy = np.vectorize(complex)(yi, yq)
    # return {'class_c' : np.mean(x[:,mask],axis=0), 'c' : }
    #         # , 'imask' : imask, 'qmask' : qmask, 'ic' : [mean(x[:,col]) for col in imask],
    #         # 'qc' : [mean(x[:,col]) for col in qmask]} #[iqmean(z) for z in iqy]}
    #         #, 'iqv' : var(iqy)} # [mean(x[:,col]) for col in mask]

def jsonAddFeatures(jdir=args.jdir, synthFeatures={}):
    print(' > generating synthetic features from %s' % jdir)
    for fname in os.listdir(jdir):
        mod = fname.split(':', 1)[0]
        fpath = os.path.join(jdir, fname)
        with open(fpath, 'r') as f:
            jfdict = json.load(f)
        # synthFeatures[mod] = {net : [COL_DICT[i] for i in jfdict[net]] for net in jfdict}
        for net in jfdict:
            synthFeatures[net] = {'mod' : mod, 'mask' : [COL_DICT[i] for i in jfdict[net]]}
            # if not key in synthFeatures:
            #     synthFeatures[key] = [val]
            # else:
            #     synthFeatures[key].append(val)
    return synthFeatures

def get_cs(synthFeatures):
    feats = synthFeatures.keys()
    f = partial(get_means, synthFeatures)
    # cv = list(map(f, tqdm(feats)))
    pool = Pool()
    cv = list(pool.map(f, tqdm(feats)))
    pool.close()
    pool.join()
    for i,key in enumerate(feats):
        synthFeatures[key]['class_centroid'] = cv[i]['class_c']
        synthFeatures[key]['centroid'] = cv[i]['c']
        # # synthFeatures[key]['var'] = cv[i]['v']
        # # synthFeatures[key]['iqcentroid'] = cv[i]['iqc']
        # synthFeatures[key]['icentroid'] = cv[i]['ic']
        # synthFeatures[key]['qcentroid'] = cv[i]['qc']
        # synthFeatures[key]['imask'] = cv[i]['imask']
        # synthFeatures[key]['qmask'] = cv[i]['qmask']
        # # synthFeatures[key]['iqvar'] = cv[i]['iqv']
    return synthFeatures

def addSynth(synthFeatures, fun=args.fun, metric='euclidean', fcs=None):
    print(' > adding synthetic features (%s) to %s' % (fun, args.file))
    if fun == 'projection':
        if fcs == None:
            jname = os.path.splitext(args.file)[0] + '_centroid.json'
            if os.path.exists(jname):
                print(' > loading centroids from %s' % jname)
                with open(jname,'r') as f:
                    cs = json.load(f)
            else:
                print(' > calculating centroids')
                cs = get_cs(synthFeatures)
                print(' |-> saving to %s' % jname)
                with open(jname, 'w') as f:
                    json.dump(cs, f)
        else:
            print(' > loading centroids from %s' % fcs)
            with open(fcs,'r') as f:
                cs = json.load(f)
        cols = cs.keys()
        if not metric in metrics:
            print(' ! metric %s not found. using euclidean.')
            metric = 'euclidean'
        for key in cols:
            cs[key]['centroid'] = np.array(cs[key]['centroid'])
            cs[key]['class_centroid'] = np.array(cs[key]['class_centroid'])
        f = partial(functions[fun], cols, cs, metrics[metric])
    else:
        if not fun in functions:
            print(' ! unknown function %s. using mean' % fun)
            fun = 'mean'
        cols = synthFeatures.keys()
        f = partial(addfeats, functions[fun], synthFeatures, cols)
    print(' > calculating synthetic features (%s)' % fun)
    # x = list(map(f, tqdm(range(df.shape[0]))))
    pool = Pool()
    x = list(pool.map(f, tqdm(range(df.shape[0]))))
    pool.close()
    pool.join()
    name,ext = os.path.splitext(args.file)
    if fun == 'projection':
        pre = '_'.join([fun,metric])
        f_out = '_'.join([name,fun,metric]) + '.csv'
    else:
        pre = fun
        f_out = '_'.join([name,fun]) + '.csv'
    # x = list(map(f, tqdm(range(df.shape[0]))))
    df0 = pd.DataFrame(np.vstack(x), columns=[pre + '_' + net for net in cols], index=df.index)
    dfs = pd.concat([dfgt, df0], axis=1)
    print(' > writing to %s' % f_out)
    with open(f_out,'w') as f:
        writer = csv.writer(f)
        writer.writerow(dfs.columns.tolist())
    with open(f_out,'a') as f:
        writer = csv.writer(f)
        for i in tqdm(range(dfs.shape[0])):
            writer.writerow(dfs.iloc[i].tolist())
    return f_out

def jsonSynth(fun=args.fun, fcs=None, metric='euclidean'):
    synthFeatures = jsonAddFeatures()
    return addSynth(synthFeatures, metric=metric, fun=fun, fcs=fcs)

if __name__ == '__main__':
    jsonSynth(args.fun, args.centroids, args.metric)
    # synthFeatures = jsonAddFeatures()
    # feats = synthFeatures.keys()
    # # f = partial(get_means, synthFeatures)
    # # %prun cv = list(map(f, tqdm(feats)))
    # cs = get_cs(synthFeatures)
    # for key in cs.keys():
    #     cs[key]['centroid'] = np.array(cs[key]['centroid'])
    #     cs[key]['class_centroid'] = np.array(cs[key]['class_centroid'])
    # f = partial(functions['projection'], cs.keys(), cs, metrics['euclidean'])
    # %prun x = np.vstack(map(f,tqdm(range(100))))
    # f = partial(addfeats, np.mean, synthFeatures)
    # # %prun activations(synthFeatures, cs, vs, euclidean, 0)
