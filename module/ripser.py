from util.data import name_lcols, ICOLS
from subprocess import Popen, PIPE
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys, csv, os
import re

# def dtos(d):
#     return 'persistence intervals in dim %d:\n' % d
#
# def to_array(s):
#     return np.array(list(map(lambda x: list(map(float,x.split(','))),s.split('\n')[:-1])))

def getpair(s):
    return map(float, re.findall(r'[-+]?\d*\.\d+|\d+',s))

def getpairs(s):
    return filter(lambda x: len(x) == 2, map(getpair, s.split('\n')))

def getdgm(out):
    s0,s1 = re.split(r'persistence intervals in dim \d:\n', out.decode())[1:]
    return {0 : np.array(getpairs(s0)), 1 : np.array(getpairs(s1))}

def ripser(p,t=None):
    input = '\n'.join(map(lambda x: ','.join(map(str,x)), p)).encode()
    call = ['ripser', '--format', 'point-cloud']
    if t != None: call += ['--threshold', str(t)]
    cproc = Popen(call, stdin=PIPE, stdout=PIPE)
    out, err = cproc.communicate(input)
    return getdgm(out)
    # out = out.decode().replace(' [0, )\n','').replace(')','').replace(' [','')
    # i0,i1 = out.find(dtos(0)),out.find(dtos(1))
    # s0,s1 = out[i0+len(dtos(0)):i1],out[i1+len(dtos(1)):]
    # return {0 : to_array(s0), 1 : to_array(s1)}

def landscape(T, n, k, dgm):
    if len(dgm) == 0:
        return np.zeros(n*k)
    def topk(x):
        l = sorted(x,reverse=True)
        if len(l) < k:
            l = np.concatenate([l,np.zeros(k-len(l))])
        return l[:k]
    # T = np.linspace(0,max(dgm[:,1]),num=n)
    return np.array([topk([max([min([t-dgm[i,0],dgm[i,1]-t]),0]) for i in range(dgm.shape[0])]) for t in T]).T

def get_landscape(p, t=None, n=100, k=10):
    dgm = ripser(p, t)
    mx0,mx1 = max(dgm[0][:,1]),max(dgm[1][:,1])
    if t != None:
        t0,t1 = np.linspace(0, t, n),np.linspace(0, t, n)
    else:
        t0,t1 = np.linspace(0,mx0,n),np.linspace(0,mx1,n)
    L0,L1 = landscape(t0,n,k,dgm[0]),landscape(t1,n,k,dgm[1])
    return dgm, {0 : L0, 1 : L1}

# def dtol(n,k,d):
#     return np.vstack([landscape(dgm, n, k) for dgm in d])

def landscapes(fname, n, k, ROWS=None):
    print('reading file %s' % fname)
    df = pd.read_csv(fname)
    MODS = df.MOD.unique()

    def get_mod(mod,n=ROWS):
        X = df.loc[df['MOD'] == mod]
        n = X.shape[0] if n == None else n
        def get_modi(i):
            return np.vstack([X.iloc[i][-2048:-1024],X.iloc[i][-1024:]]).T
        pool = Pool()
        D = list(pool.map(ripser,list(map(get_modi,range(n)))))
        pool.close()
        pool.join()
        return D

    def get_landscapes(D):
        pool = Pool()
        T = np.linspace(0,max([max(dgm[:,1]) for dgm in D]), num=n)
        f = partial(landscape, T, n, k)
        L = np.vstack(list(pool.map(f,tqdm(D))))
        pool.close()
        pool.join()
        return L

    print('computing persistence diagrams')
    D = np.concatenate([get_mod(mod) for mod in tqdm(MODS)])
    D0,D1 = [d[0] for d in D],[d[1] for d in D]

    print('computing landscapes 0-1 with N=%d, K=%d' % (n,k))
    df0 = pd.DataFrame(get_landscapes(D0), columns=name_lcols(0, n, k))
    df1 = pd.DataFrame(get_landscapes(D1), columns=name_lcols(1, n, k))

    # dfpd = pd.concat([df, df0, df1], axis=1)
    dfpd = pd.concat([df[ICOLS], df0, df1], axis=1)

    folder,file = os.path.split(fname)
    name,ext = os.path.splitext(file)
    path_out = os.path.join(folder, name + '_ph' + ext)
    print('writing to %s' % path_out)
    with open(path_out,'w') as f:
        writer = csv.writer(f)
        writer.writerow(dfpd.columns.tolist())
    with open(path_out,'a') as f:
        writer = csv.writer(f)
        def write_row(i):
            writer.writerow(dfpd.iloc[i].tolist())
        list(map(write_row, tqdm(range(dfpd.shape[0]))))

    return path_out
