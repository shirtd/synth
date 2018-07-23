from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from util.afrl import upload
from tqdm import tqdm
import pandas as pd
import argparse
import os, csv, gc

parser = argparse.ArgumentParser(description='select features and normalize.')
parser.add_argument('-T', '--train', default='data/chunk6_join.csv', help='training data file.')
parser.add_argument('-t', '--test', default='data/chunk7_join.csv', help='testing data file.')
parser.add_argument('-u', '--upload', action='store_true', help='upload training source')
parser.add_argument('-s', '--scale', action='store_true', help='scale data.')
# parser.add_argument('-n', '--norm', action='store_true', help='normalize data.')
args = parser.parse_args()

# PROCESS TRAIN
print('reading file %s' % args.train)
dftrain = pd.read_csv(args.train)
traingt = dftrain.iloc[:,:27]
trainidx = dftrain.index
cols = dftrain.iloc[:,27:].columns
X, y = dftrain.iloc[:,27:].values, dftrain['MODi'].values
dftrain = None
gc.collect()

print(' > training classifier')
clf = ExtraTreesClassifier(n_jobs=-1)
clf = clf.fit(X, y)

print(' > selecting features')
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X = None
gc.collect()

icols = model.get_support(indices=True)
cols_new = [cols[i] for i in icols]
print(' |-> %d features selected' % len(cols_new))

if args.scale:
    scaler = StandardScaler()
    print(' > training scaler')
    scaler.fit(X_new)
    dftrain_new = pd.DataFrame(scaler.transform(X_new), columns=cols_new, index=trainidx)
else:
    dftrain_new = pd.DataFrame(X_new, columns=cols_new, index=trainidx)
X_new = None
gc.collect()

train_out = os.path.splitext(args.train)[0] + '_select.csv'
_dftrain = pd.concat([traingt, dftrain_new], axis=1)
dftrain_new = None
gc.collect()

print('writing file %s' % train_out)
with open(train_out,'w') as f:
    writer = csv.writer(f)
    writer.writerow(_dftrain.columns.tolist())
with open(train_out,'a') as f:
    writer = csv.writer(f)
    for i in tqdm(range(_dftrain.shape[0])):
        writer.writerow(_dftrain.iloc[i].tolist())
_dftrain = None
gc.collect()

# PROCESS TEST
print('reading file %s' % args.test)
dftest = pd.read_csv(args.test)
testgt = dftest.iloc[:,:27]
testidx = dftest.index
_X, _y = dftest.iloc[:,27:].values, dftest['MODi'].values
dftest = None
gc.collect()

_X_new = model.transform(_X)
_X = None
gc.collect()

if args.scale:
    dftest_new = pd.DataFrame(scaler.transform(_X_new), columns=cols_new, index=testidx)
else:
    dftest_new = pd.DataFrame(_X_new, columns=cols_new, index=testidx)
_X_new = None
gc.collect()

test_out = os.path.splitext(args.test)[0] + '_select.csv'
_dftest = pd.concat([testgt, dftest_new], axis=1)
dftest_new = None
gc.collect()

print('writing file %s' % test_out)
with open(test_out,'w') as f:
    writer = csv.writer(f)
    writer.writerow(_dftest.columns.tolist())
with open(test_out,'a') as f:
    writer = csv.writer(f)
    for i in tqdm(range(_dftest.shape[0])):
        writer.writerow(_dftest.iloc[i].tolist())
_dftest = None
gc.collect()

if args.upload: upload(train_out)
