from util.data import *
from util.args import parser
from util.afrl import upload
from tqdm import tqdm
import pandas as pd
import numpy as np
import os,sys,csv

def join(args, chunk=None):
    chunk = chunk if chunk != None else args.chunk
    name = name_file(chunk, args.snr)
    
    base = os.path.join(args.dir,name + '.csv')
    files = [os.path.join(args.dir,name+'_%s.csv' % s) for s in args.suff]

    print(' > reading base %s' % base)
    df = pd.read_csv(base)

    print(' > reading %d files' % len(files))
    dfs = [pd.read_csv(fname).drop(ICOLS, axis=1) for fname in files]

    print(' > appending %d files to %s' % (len(files), base))
    dfc = pd.concat([df]+dfs,axis=1)

    file_out = os.path.join(args.dir,name+'_join.csv')
    print(' > writing to %s' % file_out)
    with open(file_out,'w') as f:
        writer = csv.writer(f)
        writer.writerow(dfc.columns.tolist())
    with open(file_out,'a') as f:
        writer = csv.writer(f)
        def write_row(i):
            writer.writerow(dfc.iloc[i,:].tolist())
        list(map(write_row, tqdm(range(dfc.shape[0]))))

    if args.upload:
        upload(file_out, args.force)

    return file_out

if __name__ == '__main__':
    join(parser['join'].parse_args())
