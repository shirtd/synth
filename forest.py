from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import log_loss
from util.data import ICOLS, name_file
import pandas as pd
import numpy as np
import os, sys, gc
import argparse
import json

parser = argparse.ArgumentParser(description='random forest predict.')
parser.add_argument('-T', '--train', type=int, default=4, help='training data chunk.')
parser.add_argument('-t', '--test', type=int, default=5, help='testing data chunk.')
parser.add_argument('-s','--snr', type=int, nargs='+', default=None, help='signal-to-noise ratios (+-10, +-6, +-2,). default: None (all)')
parser.add_argument('-d','--dir', default='data', help='data directory. default: data')
parser.add_argument('-b','--base', default='', help='base data.')
parser.add_argument('-f','--fun', default='forest', help='classification function. deafult: forest (random forest).')
parser.add_argument('--args', type=int, default=1000, help='classification function arguments. deafult: 1000 (estimators).')
parser.add_argument('--suff', nargs='+', help='suffixes of files to add. default: none.')
parser.add_argument('--save', action='store_true', help='save result.')
parser.add_argument('--stats', action='store_true', help='save stats result.')
parser.add_argument('-o','--out', default='RandForestModel', help='out file name.')

models = {
    'forest' : lambda x : RandomForestClassifier(n_estimators=x, criterion='entropy', n_jobs=-1),
    'knn' : lambda x : KNeighborsClassifier(n_neighbors=x, n_jobs = -1)
}

def get_df(chunk, snr=[10], suff='', dir='data'):
    name = name_file(chunk, snr, suff)
    fname = os.path.join(dir, name + '.csv')
    print('reading file %s' % fname)
    return pd.read_csv(fname)

def join_df(chunk, snr=[10], suffs=None, base='', dir='data', shuffle=False, subset=1):
    df_base = get_df(chunk, snr, base, dir)
    if suffs != None:
        dfs = [get_df(chunk, snr, suff, dir).drop(ICOLS, axis=1) for suff in suffs]
        dfout = pd.concat([df_base] + dfs, axis=1)
    else:
        dfout = df_base
    if subset < 1:
        dfmods = [dfout.loc[dfout['MOD'] == mod] for mod in MOD]
        dfsub = pd.concat([pd.concat([d.loc[d['SNR'] == s].sample(frac=subset) for s in snr]) for d in dfmods])
    else:
        dfsub = dfout
    return dfsub.sample(frac=1) if shuffle else dfsub

def format_data(df,gti=1):
    X = df.drop(ICOLS, axis=1).values
    y = df.iloc[:,gti].values
    return {'X' : X, 'y' : y}

def get_data(train=4, test=5, snr=[10], suffs=['synth','ph'], base='', dir='data'):
    df_train = join_df(train, snr, suffs, base, dir, shuffle=True)
    traindata = format_data(df_train)
    df_test = None
    gc.collect()

    df_test = join_df(test, snr, suffs, base, dir) # subset=0.3475)
    testdata = format_data(df_test)
    df_test = None
    gc.collect()

    return {'train' : traindata, 'test' : testdata}

def predict(data, fun='forest', arg=10):
    # Instantiate model with 1000 decision trees
    rf = models[fun](arg)

    # Train the model on training data
    print('fitting model "%s" with argument %s' % (fun,arg))
    rf.fit(data['train']['X'], data['train']['y']);

    #Predict into the model
    print(' > predicting')
    p = np.array(rf.predict_proba(data['test']['X']))

    #Print out the accuracy of the model
    ll = log_loss(data['test']['y'], p)
    score = 100 / (1 + ll)
    print('\tscore: %0.4f' % score)

    return {'data' : data, 'model' : rf}, {'p' : p.tolist(), 'loss' : ll, 'score' : score, 'features' : rf.feature_importances_.tolist()}

def write_out(res, fname='RandForestModel', dir='data'):
    #Save the model to a pickle file
    fout = os.path.join(dir,fname + '.pkl')
    print(' > saving output directory to %s' % fout)
    with open(fout, 'wb') as f:
        pickle.dump(res, f)
    return fout

if __name__ == '__main__':
    args = parser.parse_args()
    data = get_data(args.train, args.test, args.snr, args.suff, args.base, args.dir)
    res,stats = predict(data, args.fun, args.args)
    if args.stats:
        name = name_file(args.train, args.snr) + '_' + name_file(args.test, args.snr)
        path_out = os.path.join(args.dir, name + '_'.join(['']+args.suff) + '_stats.json')
        with open(path_out, 'w') as f:
            json.dump(stats, f)
    if args.save:
        write_out(res, dir=args.dir)
