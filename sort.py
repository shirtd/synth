from util.data import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys, os, csv
import argparse

parser = argparse.ArgumentParser(description='random forest predict.')
parser.add_argument('-d', '--dir', default='data', help='data directory.')
parser.add_argument('-c', '--chunk', type=int, default=0, help='chunk # (0-14). default: 0')
parser.add_argument('-s', '--snr', type=int, nargs='+', default=None, help='signal-to-noise ratios (+-10, +-6, +-2,). default: None (all)')

sortedReal = []
def sortReal(row):
    global sortedReal
    sortedReal.append(row.sort_values().tolist())

sortedImag = []
def sortImag(row):
    global sortedImag
    sortedImag.append(row.sort_values().tolist())

def colnames(k=1, n=1024):
    cola = ["sorted_a_%d" % (i) for i in range(0,n,k)]
    colb = ["sorted_b_%d" % (i) for i in range(0,n,k)]
    return cola + colb

if __name__ == '__main__':
    args = parser.parse_args()
    name = name_file(args.chunk, args.snr)
    fname = os.path.join(args.dir, name + '.csv')
    print('loading %s' % fname)
    source = pd.read_csv(fname)
    print(' > sorting %s rows' % source.shape[0])
    isort = np.sort(source.iloc[:,-2048:-1024].values)
    qsort = np.sort(source.iloc[:,-1024:].values)
    cat = np.concatenate([isort, qsort], axis=1)
    # sortedAll = map(np.concatenate, zip(sortedReal, sortedImag))
    sortedDf = pd.DataFrame(cat, columns=colnames(), index=source.index)
    f_out = os.path.join(args.dir, name + "_sorted.csv")
    out = pd.concat([source[ICOLS],sortedDf], axis=1)
    print('writing to %s' % f_out)
    with open(f_out,'w') as f:
        writer = csv.writer(f)
        writer.writerow(out.columns.tolist())
    with open(f_out,'a') as f:
        writer = csv.writer(f)
        for i in tqdm(range(out.shape[0])):
            writer.writerow(out.iloc[i].tolist())
