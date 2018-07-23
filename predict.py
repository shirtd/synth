from module.predictor import LocalPredictor
from util.args import parser
from util.data import *
from math import log as ln
import numpy as np

def predict(args):
    print('initializing predictor with %s and %s' % (args.train,args.test))
    pred = LocalPredictor(args.train, args.test, args.gt, args.col, args.fun, args.k, args.rows)

    print('running topological predict on %s' % (args.net))
    z = pred.topological_predict(args.net)
    a, c = pred.accuracy(z, l=[MODS[i] for i in pred.classes])
    s = pred.score(z)

if __name__ == '__main__':
    predict(parser['predict'].parse_args())
