# This file reproduces the functionality of the default topological_predict function in the Ayasdi SDK while running as much
# as possible solely on the local machine.
# Essentially this function produces KNN prediction using the metrics in Ayasdi platform and the corresponding output.

# Was tested on MNIST data, but should work for any dataset provided below requirements are met. If not, let me know.
# Prediction Prevalence should be covered by scipy.stats.mode returning count of each occurrence.

# REQUIREMENTS
# Set AYASDI_USERNAME and AYASDI_PASSWORD to your login information
# to_predict in all functions is a 2D numpy array of data to predict
# Need to initialize a LocalPredictor object before predictions can be made. This is done by calling LocalPredictor passing a path to a training file on the local machine

# USER = # ayasdi username
# PASS = # ayasdi password

from multiprocessing import Pool
from util.data import MODS
from util.afrl import USER,PASS
from ayasdi.core.api import Api
from functools import partial
from sklearn import neighbors
import math, scipy, sklearn
from sklearn import metrics
from math import log as ln
from tqdm import tqdm
import pandas as pd
import numpy as np
import os, sys, gc
import requests
import time

def color(net, i):
    nerrors = 0
    while True:
        if nerrors > 24:
            print(' ! %d errors' % nerrors)
        try:
            return np.array(net.get_coloring_values(name=MODS[i]))
        except requests.exceptions.SSLError as e:
            if nerrors > 24:
                print(e)
                time.sleep(60)
            else:
                time.sleep(5)
        except requests.exceptions.ConnectionError as e:
            if nerrors > 24:
                print(e)
                time.sleep(60)
            else:
                time.sleep(5)
        nerrors += 1

class LocalPredictor:
    def __init__(self, train, test, gti, col_set, metric='euclidean', K=10, rows=None, group=None):
        ''' Initializes a LocalPredictor object.
            trainFile is a path to the CSV file containing the training data on the local machine,
            the name of this file must match the source name on Ayasdi exactly and must contain a header,
            gti is a list of the indices of the ground truth columns to predict into,
            column_set_name is the name of the column set containing the features to use,
            K is the number of neighbors to return in all functions.'''

        dir,fname = os.path.split(train)
        print('connecting as %s' % USER)
        connection = Api(USER, PASS)
        print(' > loading source %s' % fname)
        self.source = connection.get_source(name=fname)
        self.metric = metric
        self.rows = rows
        self.gti = gti
        self.K = K

        # Dictionary to convert strings for metrics to function calls
        self.functions = {
                'angle':self.angle,
                'norm_angle' : self.norm_angle,
                'vne' : self.variance_normalized_euclidean,
                'iqr_euclidean' : self.iqr_euclidean,
                'absolute_correlation' : self.absolute_correlation,
                'norm_correlation' : self.norm_correlation,
                'cosine': self.cosine
            }

        self.data_init(train, test, col_set, group)
        self.net = None
        self.w = None

        for i in range(len(self.classes)):
            if not MODS[i] in [c['name'] for c in self.source.get_colorings()]:
                col = self.source.columns[self.source.colname_to_ids[MODS[i]]]
                self.source.create_coloring(name=MODS[i], column_id=col['index'])

    def data_init(self, train, test, col_set, group=None, gid=None):
        print(' > loading %s' % train)
        train_data = pd.read_csv(train)
        self.gtl = train_data.columns[self.gti]
        self.classes = train_data[self.gtl].unique().tolist()

        print(' > loading column set %s' % col_set)
        self.col_set = self.source.get_column_set(name=col_set)
        self.dati = self.col_set['column_indices']
        if gid == None and group == None:
            self.group = {'id' : None}
            self.rowi = range(train_data.shape[0])
        else:
            f = self.source.get_group
            self.group = f(id=gid) if gid != None else f(name=group)
            self.rowi = self.group['row_indices']

        df = train_data.iloc[self.rowi,:]
        # self.crows = {c : df.index[df[self.gtl] == c].tolist() for c in self.classes}
        self.columns = [df.columns[i] for i in self.dati]
        train_data = None

        # ground truth column
        self.truth = df.iloc[:,self.gti].values
        self.data = df.iloc[:,self.dati].values
        df = None
        gc.collect()

        print(' > loading %s' % test)
        test_raw = pd.read_csv(test)
        part = [test_raw.loc[test_raw[self.gtl] == c] for c in self.classes]
        if self.rows == None or max(map(lambda x: x.shape[0], part)) < self.rows:
            if self.rows != None: print(' ! desired rows > number of rows. using all.')
            # if test_raw.shape[0] > 100000:
            #     test_data = pd.concat([x.sample(n=695) for x in part])
            # else:
            test_data = test_raw
        # else:
        #     test_data = pd.concat([x.sample(n=self.rows) for x in part])
        test_raw = None
        self._snrs = test_data['SNR'].values
        self._truth = test_data.iloc[:,self.gti].values
        self._data = test_data.iloc[:,self.dati].values
        test_data = None
        gc.collect()

        self.init_res = self.knn_predict(self._data, self.metric)

    # Function to do general KNN prediction for an array of test points into an Ayasdi network.
    # User specifies the number of neighbors and metric as well as if to use mode or mean
    # Can also choose to return the closest row indices and the distances to each of the closest rows.
    # REQUIRES NUMPY ARRAYS
    def knn_predict(self, to_predict, metric, calc_method='mode', return_distances=True, return_predictions=False):
        ''' Performs simple KNN prediction for an array of testing points.
            to_predict is the array of testing points, metric is the metric used to calculate distances,
            network is the name of the network to perform prediction in,
            calc_method can be either 'mode' or 'mean' and determines how to decide the predicted value,
            return_closest_rows defaults to true and indicates whether or not to return the indices for the nearest neighbors,
            return_distances defaults to true and determines whether to return a list of distances to the K neighbors.'''

        # print('running knn_predict with %s' % (metric))
        index_dict = dict(enumerate(self.rowi))
        # data_points,to_predict = self.data,self._data
        print(' > fitting model with metric %s' % metric if isinstance(metric, basestring) else metric.__name__)
        nc, means, std = self.get_model(metric, self.data)
        self.data = None
        gc.collect()
        if metric is 'norm_correlation' or metric is 'norm_angle':
            self._data = np.divide((self._data - means), std, where=std!=0)
        print(' > predicting row indices with k=%d' % self.K)
        dist,ind = nc.kneighbors(self._data, n_neighbors=self.K, return_distance=return_distances)
        self._data = None
        gc.collect()
        # Convert indices in the smaller numpy array to the actual indices in the full size data set.
        indices = [[index_dict[x] for x in index.tolist()] for index in ind]
        # Set up return dictionary based on which flags are set.
        ret = {'closest_rows' : indices}
        if return_distances:
            ret['distances'] = dist
        if return_predictions:
            # Calculate predicted values for each test point
            print(' > predicting values with %s' % calc_method)
            f = scipy.stats.mode if calc_method == 'mode' else scipy.stats.mean
            predictions = [f(self.truth[i]) for i in indices]
            ret['knn_predictions'] = predictions
        return ret

    def net_init(self, network):
        print(' > initializing network %s' % network)
        self.net = self.source.get_network(name=network)
        n,m = self.net.json['node_count'],len(self.classes)
        f = partial(color, self.net)
        # w = list(map(f, tqdm(range(m))))
        pool = Pool()
        w = list(pool.map(f,range(m)))
        pool.close()
        pool.join()
        self.w = np.vstack(w)

    # Function that performs a majority of the rest of Ayasdi topological_predict.
    # Does not allow for interpolation_source possibilities, as containing_nodes is implemented elsewhere and containing_groups (deprecated).
    # Return formatting is different from Ayadi, but does contain the same data.
    def topological_predict(self, network, method='mode', return_containing_nodes=True, return_containing_groups=False):
        ''' to_predict is a numpy array consisting of points to predict.
            data_points is a numpy array with data to predict into, with the first column being ground truth.
            predict_columns is a dictionary containing 'calculation_method', 'source_name', and 'network_name'.
            K is the number of neighbors to look for, metric is the distance metric used to determine closest neighbors '''
        self.source.sync()
        if self.net == None or network != self.net.name:
            self.net_init(network)

        col_set = self.net.json['column_set_id']
        gid = self.net.json['row_group'] if 'row_group_id' in self.net.json else None
        if col_set != self.col_set['id'] or gid != self.group['id']:
            self.data_init(col_set, gid=gid)

        if return_containing_nodes:
            # Get a dictionary corresponding node IDs to row IDs
            points = self.net.get_points(range(self.net.json['node_count']))
            node_point_dict = {int(k) : v for k,v in points.iteritems()}
            # Get a dictionary corresponding row IDs to node IDs.
            row_node_dict = dict()
            for val in node_point_dict:
                for item in node_point_dict[val]:
                    if item not in row_node_dict:
                        row_node_dict[item] = [val]
                    else:
                        row_node_dict[item].append(val)
            keys = row_node_dict.keys()
            keys.sort()

        # Get a dictionary corresponding row IDs to group IDs.
        if return_containing_groups:
            group_point_dict = self.source.get_group_membership()

        ret = self.init_res.copy()

        # Go through the initial predictions and add their
        # corresponding closest_rows and containing_nodes to lists.
        if return_containing_nodes and return_containing_groups:
            rows = self.init_res['closest_rows']
            f = lambda i : zip(*[(row_node_dict[row],group_point_dict[row]) for row in i])
            x = map(lambda x: map(list, x),zip(*map(f, rows)))
            ret['containing_nodes'],ret['containing_groups'] = x
        elif return_containing_nodes:
            f = lambda i : [row_node_dict[row] for row in i]
            ret['containing_nodes'] = map(f, self.init_res['closest_rows'])
        elif return_containing_groups:
            f = lambda i : [group_point_dict[row] for row in i]
            ret['containing_groups'] = map(f, self.init_res['closest_rows'])

        p = self.prob_matrix(ret)
        ret['probability_matrix'] = p
        z,w = zip(*[max(enumerate(p[i,:]), key=lambda x: x[1]) for i in range(p.shape[0])])
        ret['predictions'],ret['confidence'],ret['weights'] = z,w,self.w
        # ret['gt'] = self._truth
        return ret

    def prob_matrix(self, z):
        cz = z['containing_nodes']
        p = np.zeros((len(cz),len(self.classes)), dtype=float)
        for i,x in enumerate(cz):
            us,cs = np.unique(np.hstack(x), return_counts=True)
            for j,l in enumerate(self.classes): # float(c)/self.K *
                p[i,j] = sum(self.w[j,u] for u,c in zip(us,cs))
        p = p / p.sum(axis=1, keepdims=1)
        p = np.vectorize(lambda x: max(min(x, 1-1E-15), 1E-15))(p)
        return p / p.sum(axis=1, keepdims=1)

    def score(self, z):
        p, n = self.prob_matrix(z), len(self._truth)
        l = -1*sum(ln(p[i,self._truth[i]]) for i in range(n))/n
        s = 100/(1 + l)
        print(' |-> score: %0.4f' % s)
        return s

    def accuracy(self, ret, l=None, verbose=True):
        ''' ret is a dictionary of prediction results returned from topological_predict.
            Calculates and prints accuracy and confidence of prediction.
            prints confusion matrix and classification report when verbose = True. '''

        a = metrics.accuracy_score(self._truth, ret['predictions'])
        c = metrics.confusion_matrix(self._truth, ret['predictions'])
        dfc = pd.DataFrame(c,index=['%d. %s' % (i,l[i]) for i in range(len(l))]) if l != None else pd.DataFrame(c)
        if verbose:
            print('\nconfusion matrix:\n')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                pd.set_option('display.expand_frame_repr', False)
                pd.set_option('display.width', None)
                print(dfc)
            print('\nclassifcation report:\n')
            print(metrics.classification_report(self._truth, ret['predictions'], target_names=l))
            print('accuracy: %0.4f' % a)
        return a, c

    # METRICS
    # Angle
    def angle(self,X,Y):
        top_sum = np.sum(np.multiply(X,Y))
        left_sum = math.sqrt(np.sum(np.power(X,2)))
        right_sum = math.sqrt(np.sum(np.power(Y,2)))
        return np.arccos(top_sum/(left_sum*right_sum))

    # Norm Angle
    def norm_angle(self,X,Y):
        return angle(X,Y)

    # Variance Normalized Euclidean
    def variance_normalized_euclidean(self,V,X,Y):
        a = np.divide(np.power(np.subtract(X,Y),2),V,where=V!=0)
        return math.sqrt(np.sum(a))

    # IQR Normalized Euclidean
    def iqr_euclidean(self,R,X,Y):
        a = np.divide(np.power(np.subtract(X,Y),2),R, where=R!=0)
        return math.sqrt(np.sum(a))

    # Absolute Correlation
    def absolute_correlation(self,n,X,Y):
        return 1 - abs(scipy.stats.pearsonr(X,Y))

    # Correlation
    def correlation(self,n,X,Y):
         return 1 - scipy.stats.pearsonr(X,Y)
         #return 1 - top/bottom

    # Norm Correlation
    def norm_correlation(self,X,Y):
        return correlation(X, Y)

    # Cosine
    def cosine(self,X,Y):
        top_sum = np.sum(np.multiply(X,Y))
        left_sqrt_sum = math.sqrt(np.sum(np.power(X,2)))
        right_sqrt_sum = math.sqrt(np.sum(np.power(Y,2)))
        return 1-(top_sum / (left_sqrt_sum * right_sqrt_sum))

    def get_function(self, metric, data):
        means,std = 0,0
        if hasattr(metric, '__call__'):
            fun = metric
        # elif metric == 'norm_angle':
        #     means = np.mean(data,axis = 0)
        #     std = np.std(data, axis = 0)
        #     data = np.divide((data - means), std, where =std!=0)
        #     fun = self.functions[metric]
        elif metric == 'norm_correlation' or metric == 'norm_angle':
            means = np.mean(data, axis = 0)
            std = np.std(data, axis = 0)
            data = np.divide((data - means), std, where=std!=0)
            fun = self.functions[metric]
        elif metric == 'absolute_correlation' or metric == 'correlation':
            return partial(self.functions[metric], data.shape[1])
            # def fun(X,Y):
            #     return self.functions[metric](X, Y, data.shape[1])
        elif metric == 'iqr_euclidean':
            R = scipy.stats.iqr(data, axis = 0)
            return partial(self.functions[metric], R)
            # def fun(X,Y):
            #     return self.functions[metric](X, Y, R)
        elif metric == 'vne':
            V = np.var(data, axis = 0)
            return partial(self.functions[metric], V)
            # def fun(X,Y):
            #     return self.functions[metric](X, Y, V)
        else:
            fun = self.functions[metric]
        return fun, data, means, std

    def get_model(self, metric, data):
        if not hasattr(metric, '__call__') and metric not in self.functions:
            means,std = np.array([0,0])
            n_neighbor = neighbors.NearestNeighbors(self.K, metric=metric, n_jobs = -1)
        else:
            fun, data,means,std = self.get_function(metric, data)
            n_neighbor = neighbors.NearestNeighbors(self.K, fun, n_jobs = -1)
        n_neighbor.fit(data)
        return n_neighbor, means, std

    #Categorical Cosine, Binary Jaccard, Affine, Distance Correlation
    #Not yet implemented
