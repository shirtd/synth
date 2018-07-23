from util.data import *
from util.afrl import *
from multiprocessing import Process, Pool, cpu_count
from scipy.spatial.distance import euclidean
from functools import partial
from numpy import mean, var, dot
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import json, csv
import os, sys

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

print(' > reading file %s' % args.file)
dfall = pd.read_csv(args.file)
dfgt = dfall[ICOLS]
df = dfall.drop(ICOLS, axis=1)
Xmods = {mod : df.loc[dfgt['MOD'] == mod].values for mod in MODS}
X = df.values
# X = df.apply(zscore).values

def activation(d, c, x, key):
    p = np.array(c[key]['centroid'])
    return d(x[c[key]['mask']], p) # / c[key]['var']

def activations(keys, cs, d, i):
    f = partial(activation, d, cs, X[i])
    return np.array(list(map(f, keys)))

# def crossover(a, b, full_row):
#     row = full_row[a:b]
def crossover(row):
    i,count = 0,0
    while row[i] == 0:
        i += 1
    sign = -1 if row[i] < 0 else 1
    while i < len(row):
        while i < len(row) and row[i]*sign > 0:
            i += 1
        sign *= -1
        count += 1
        i += 1
    return float(count) / 2048

def possum(row):
    return abs(sum(filter(lambda x: x > 0, row)))

def negsum(row):
    return abs(sum(filter(lambda x: x < 0, row)))

def addfeats(fun, feats, keys, i):
    x = X[i]
    return np.array(list(map(lambda k: fun(x[feats[k]['mask']]), keys)))

functions = {
    'projection' : activations, 'crossover' : crossover,
    'possum' : possum, 'negsum' : negsum,
    # 'icrossover' : partial(crossover, 0, 1024),
    # 'qcrossover' : partial(crossover, 1024, 2048),
    'median' : np.median, 'mean' : np.mean, 'var' : np.var,
    'min' : min, 'max' : max, 'sum' : sum
}

metrics = {
    'euclidean' : euclidean,
    'dot' : dot
}

def get_means(feats, key):
    mod,mask = feats[key]['mod'],feats[key]['mask']
    x = Xmods[mod] if mod != 'AllMod' else X
    y = [x[:,col] for col in mask]
    return {'c' : [mean(z) for z in y], 'v' : var(y)} # [mean(x[:,col]) for col in mask]

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
    pool = Pool()
    cv = list(pool.map(f, tqdm(feats)))
    pool.close()
    pool.join()
    # cv = list(map(f, tqdm(feats)))
    for i,key in enumerate(feats):
        synthFeatures[key]['centroid'] = cv[i]['c']
        synthFeatures[key]['var'] = cv[i]['v']
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
        f = partial(functions[fun], cols, cs, metrics[metric])
    else:
        if not fun in functions:
            print(' ! unknown function %s. using mean' % fun)
            fun = 'mean'
        cols = synthFeatures.keys()
        f = partial(addfeats, functions[fun], synthFeatures, cols)
    print(' > calculating synthetic features (%s)' % fun)
    pool = Pool()
    x = list(pool.map(f, tqdm(range(df.shape[0]))))
    pool.close()
    pool.join()

    if fun == 'projection':
        pre = '_'.join([fun,metric])
        f_out = '_'.join([name,fun,metric]) + '.csv'
    else:
        pre = fun
        f_out = '_'.join([name,fun]) + '.csv'
    # x = list(map(f, tqdm(range(df.shape[0]))))
    df0 = pd.DataFrame(np.vstack(x), columns=[pre + '_' + net for net in cols], index=df.index)
    dfs = pd.concat([dfgt, df0], axis=1)
    name,ext = os.path.splitext(args.file)
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
    # cs = get_cs(synthFeatures)
    # vs = get_vs(synthFeatures)
    # f = partial(activations, synthFeatures, cs, vs, euclidean)
    # # f = partial(addfeats, np.mean, synthFeatures)

# # %prun x = np.vstack(map(f,range(2)))
# # %prun activations(synthFeatures, cs, vs, euclidean, 0)
