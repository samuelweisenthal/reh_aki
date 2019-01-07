'''Version of fof that searches over HP'''

from sklearn import datasets
from sklearn.utils import shuffle
#
from sklearn.linear_model import LogisticRegression as lr
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.svm import SVC as svc

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler, Normalizer

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, average_precision_score, brier_score_loss, mean_squared_error, median_absolute_error, r2_score, make_scorer
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.calibration import calibration_curve as cal_curve
from sklearn.calibration import CalibratedClassifierCV

from scipy.stats import randint, uniform, lognorm, expon

#import matplotlib.pyplot as plt
import numpy as np
import pdb
import pandas
import collections
import pickle

from time import time, sleep

from n_cv import n_cv, desc_dict
from get_n_hosp import get_nhosp_per_id
from sampler import sampler

import sys
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-rs", "--random_state", type=int)
parser.add_argument("-skgf", "--use_sk_gkfold", type=int)
parser.add_argument("-ons", "--on_server", type=int)
parser.add_argument("-nj", "--n_jobs", type=int)
parser.add_argument("-cal3", "--cal_threeQ", type=int)
parser.add_argument("-x", "--X_path", type=str)
parser.add_argument("-y", "--y_path", type=str)
parser.add_argument("-g", "--group_path", type=str)
parser.add_argument("-o", "--op_dir", type=str)
parser.add_argument("-shuffle_y", "--shuffle_y", nargs='?', type=int)
parser.add_argument("-weight", "--weight", nargs='?', type=int)
parser.add_argument("-lasso_only", "--lasso_only", nargs='?', type=int)
parser.add_argument("-high_pen", "--high_pen", nargs='?', type=int)
parser.add_argument("-samp", "--samp", nargs='?', type=int)


args = parser.parse_args()
print "Arguments:", args._get_args

# print "random state", random_state
# print "use_k_gkf", use_sk_gkfold
# print "on server", on_server
# print "n_jobs", n_jobs
# print "cal_threeQ", cal_threeQ
# print "X_path", X_path
# print "y_path", y_path
# print "group_path", group_path
# print "op_dir", op_dir #try to make this one variable in sbatch script


# Difference between error estimated by nested CV and "true" error
# number of trials
N_HIDE_SEE_SPLITS = 2
# number of outer nested cv splits
N_OUTER_SPLITS = 5
# number of inner splits to optimize params and later optimize model on full (seen) data
N_INNER_SPLITS = 5
# number of splits to make when generating dataset for calibration. Only uses first. Eg, if make 3 splits, will have 1/3 and 2/3, one of which is for calibration
CAL_SPLITS = 4

#random_state = 13 # random state must be changed HERE for iterated trials

#
if args.on_server:
    print "Running on server."
    X = pandas.read_csv(args.X_path, index_col=0)
    y = pandas.read_csv(args.y_path, index_col=0, header=None)
    feat_names = list(X.columns)#if not list, can't pickle
    groups = pandas.read_csv(args.group_path, index_col=0, header=None)
    fname = args.op_dir + '/data_rs_' + str(args.random_state)+'.pickle'
    if not os.path.exists(args.op_dir):
        os.makedirs(args.op_dir)
    # print "Running on server."
    # X = pandas.read_csv('/homedirec/user/X_aki_full_10.csv', index_col=0)
    # y = pandas.read_csv('/homedirec/user/y_aki_full_10.csv', index_col=0, header=None)
    # feat_names = list(X.columns)#if not list, can't pickle
    # groups = pandas.read_csv('/homedirec/user/groups_aki_full_10.csv', index_col=0, header=None)
    # fname = '/homedirec/user/fof_full_10/data'+str(random_state)+'_'+str(use_sk_gkfold)+'.pickle'    
else:
    print "Running on mac."
    n_feat = 3
    X, y = datasets.make_classification(n_samples=900, n_features=n_feat, n_informative=2, n_redundant=0, weights=[0.5,0.5], random_state=7)
    
    #N_GROUPS = len(X)  # effectively no groups
    N_GROUPS = 350
    ## assign groups randomly (later, assign actual)
    np.random.seed(10) #for the groups
    groups = np.array([np.random.choice(range(N_GROUPS)) for el in y])
    groups = pandas.DataFrame(groups, columns=[1])
    #pandas.DataFrame(X).to_csv('X.csv')
    #pandas.DataFrame(y).to_csv('y.csv', header=False)
    #pandas.DataFrame(groups).to_csv('groups.csv', header=False)
    #X = pandas.read_csv('/homedirec/user/X.csv', index_col=0)
    #y = pandas.read_csv('/homedirec/user/y.csv', index_col=0, header=None)
    #feat_names = list(X.columns)#if not list, can't pickle
    #groups = pandas.read_csv('/homedirec/user/groups.csv', index_col=0, header=None)
    #groups = range(len(y))
    feat_names = []
    X = pandas.DataFrame(X)
    y = pandas.DataFrame(y)
    #X, y = datasets.make_classification(n_samples=100, n_features=10, weights=[0.93,0.07])
    fname = 'data'+str(args.random_state)+'_'+str(args.use_sk_gkfold)+'.pickle'
    
samp = args.samp
if samp:
    print "Sampling--generating iid sample!"
    X, y, groups = sampler(X, y, groups, random_state=args.random_state)
    groups.columns = [1]
#    pdb.set_trace()


sample_weights = [1 for el in groups[1]]
weight = args.weight
if weight==1:
    print "weighting samples"
    nhpid = get_nhosp_per_id(groups)
    sample_weights = [1/float(nhpid[el]) for el in groups[1]]
    print "sample weights head", sample_weights[0:10]

sample_weights = np.array(sample_weights)

X, y, groups = np.array(X), np.ravel(np.array(y)), np.array(groups)

shuffle_y = args.shuffle_y
if shuffle_y==1:
    print "shuffling y"
    print "y",y
    np.random.shuffle(y)
    print "after shuffle", y


#my_gs = GroupShuffler(random_state=random_state, verbose=10)

#pdb.set_trace()


#ratio examples/features in aki set is 30 or so


metrics = [log_loss, roc_auc_score, average_precision_score,
           brier_score_loss, roc_curve, pr_curve, cal_curve]
metric_names = ['ll', 'roc', 'pr', 'bri', 'roc_curve', 'pr_curve', 'cal_curve']

#['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

#grid_scoring='neg_log_loss'
def neg_log_loss(y, y_pred, labels):
    return -1*log_loss(y, y_pred, labels=labels) #so bigger is better

grid_scoring = make_scorer(neg_log_loss, labels=sorted(np.unique(y)))


#n_jobs = 1
mask = 0  # conceal outer fold results

lasso_only = args.lasso_only
highly_pen = args.high_pen

if lasso_only:
    "Print only running lasso"
    if highly_pen:
        print "highly penalizing lasso"
        pen = 2e-4
    else:
        print "running normal penalty"
        pen = 2e-3 
    ess = [
        Pipeline([("im", Imputer(missing_values=np.NAN, strategy="most_frequent", axis=0, verbose=10)),
                                     ("sc", StandardScaler()),
                       ("lr1", lr(n_jobs=args.n_jobs, penalty='l1', class_weight='balanced', random_state=args.random_state))]),

    ]

    es_names = [
        "lr1",

    ]

    paramss = [
        {

            "lr1__C": uniform(0,1)        },

        {

        "gbc__n_estimators": randint(10,150),
            "gbc__min_samples_split":randint(10,200), #larger -> more bias
            "gbc__min_samples_leaf": randint(50,1000),
        "gbc__max_depth": randint(2,10), # want weak (high bias) estimators, so cap max depth

        },
    ]

    # for RandSearch, must be fewer it than # combinations
    n_iters = [
        20
    ]
else:
    print "Searching over GBC and lasso"
    ess = [
        Pipeline([("im", Imputer(missing_values=np.NAN, strategy="most_frequent", axis=0, verbose=10)),
                                     ("sc", StandardScaler()),
                       ("lr1", lr(n_jobs=args.n_jobs, penalty='l1', class_weight='balanced', random_state=args.random_state))]),
     #   Pipeline([("im", Imputer(missing_values=np.NAN, strategy="most_frequent", axis=0, verbose=10)),
     #                                ("sc", StandardScaler()),
     #                  ("lr2", lr(n_jobs=n_jobs, class_weight='balanced', random_state=random_state))]),
     #   Pipeline([("im", Imputer(missing_values=np.NAN, strategy="most_frequent", axis=0, verbose=10)),
     #                  ("rf", rf(n_jobs=n_jobs, class_weight='balanced', random_state=random_state))]),

        Pipeline([("im", Imputer(missing_values=np.NAN, strategy="most_frequent", axis=0, verbose=10)),
                       ("gbc", gbc(random_state=args.random_state))]),
    ]

    es_names = [
        "lr1",
    #    "lr2",
    #    "rf",
        "gbc"
    ]

    paramss = [
        {
            #"lr1__C": expon(scale=0.1),
            "lr1__C": uniform(0,0.000000000001)#[2e-3],
            #One shot
        #"lr1__C": [2e-3]
        #highly pen
        #"lr1__C": [2e-4]
            #"lr1__C": uniform(0,1000),

        },
    #    {
    #     	#"lr2__C": expon(scale=1e-5, loc=1e-8),
    #        "lr2__C": uniform(0,11e-6),
    #        #"lr2__C": uniform(0,1000),
    #    },
    #    {
            #"rf__n_estimators": [100],       #"rf__min_samples_split": randint(10,200), #larger -> more bias
            #"rf__max_depth": [4] #randint(2, 5), #should technically grow each tree large as possible and increase no estimators to reduce ovefitting, but that's very expensive
        #"rf__criterion":['entropy'],
        #"rf__max_features":['sqrt']#randint(10,100) #can speed up runtime
            #"rf__min_samples_leaf": randint(50, 1000),#larger -> more bias
    #    },
        {
            "gbc__n_estimators": randint(1,200),#[100],
            "gbc__min_samples_split": randint(1,200),#[150],#randint(10,200), #larger -> more bias
#            "gbc__min_samples_leaf": randint(1,2000), #[100],#randint(50,1000)
            "gbc__max_depth": uniform(1,30),#[2], # want weak (high bias) estimators, so cap max depth
            "gbc__subsample": uniform(0,1),
            "gbc__max_features":["auto", "sqrt", "log2"],
            "gbc__max_leaf_nodes": randint(0,1000),
#            "gbc__learning_rate": uniform(0,10),
            "gbc__loss": ["deviance", "exponential"],
            "gbc__criterion": ["friedman_mse", "mse"]

        #One shot
#        "gbc__n_estimators": [100],
#            "gbc__min_samples_split":[150],#randint(10,200), #larger -> more bias
#            "gbc__min_samples_leaf": [100],#randint(50,1000)
#        "gbc__max_depth": [2], # want weak (high bias) estimators, so cap max depth

        },
    ]

    # for RandSearch, must be fewer it than # combinations
    n_iters = [
        1,#100
    #    100,
    #    1,#100
        100#100
    ]

#n_iters = [
#    60,
##    1,
##    1,
#    60
#]



#ms_est, ms_tru, param_res, y_trues_hats = nested_cv_check(X=X, y=y,
#                                                          N_HIDE_SEE_SPLITS=N_HIDE_SEE_SPLITS, N_OUTER_SPLITS=N_OUTER_SPLITS, N_INNER_SPLITS=N_INNER_SPLITS,
#                                                          N_GROUPS=N_GROUPS,groups,
#                                                          ess=ess, es_names=es_names, paramss=paramss, random_state=13,
#                                                          metric_names=metric_names, metrics=metrics,
#                                                          n_iters=n_iters, grid_scoring=grid_scoring,
#                                                          mask=mask)
#
#desc_dict(ms_est, ms_tru)

t0 = time()
fit_counter = 0
best_clf, m_d_outers, param_res, all_param_res, all_param_full_res, y_trues_hats, searches, best_searches, full_searches, best_full_search, all_co, best_co, all_full_co, best_full_co , fit_counter, out_splits = n_cv(X=X, y=y, groups=groups,
                                     sample_weights=sample_weights,
                                     N_OUTER_SPLITS=N_OUTER_SPLITS, N_INNER_SPLITS=N_INNER_SPLITS, CAL_SPLITS=CAL_SPLITS, use_sk_gkfold=args.use_sk_gkfold, cal_threeQ=args.cal_threeQ,
                                     es_names=es_names, ess=ess, paramss=paramss, random_state=args.random_state,
                                     metric_names=metric_names, metrics=metrics,
                                     n_iters=n_iters, n_jobs=args.n_jobs, scoring=grid_scoring,
                                     mask=mask, fit_counter=fit_counter)

data = [best_clf, m_d_outers, param_res, all_param_res, all_param_full_res, y_trues_hats, searches, best_searches, full_searches, best_full_search, all_co, best_co, all_full_co, best_full_co, feat_names, out_splits]


with open(fname,'wb') as f: pickle.dump(data,f)

print "runtime", (time()-t0)/float(60),"min"
print "fits", fit_counter
desc_dict(m_d_outers)

#pdb.set_trace()

# plt.plot(*roc_curve(y_trues_hats[0][0], y_trues_hats[0][1])[0:2]); plt.show()
# plt.scatter(X[:,0],X[:,1], c=y, alpha=0.5); plt.show()


# plt.scatter(searches['rf'][0]['param_rf__max_depth'].data,searches['rf'][0]['mean_train_score'],c='r'); plt.scatter(searches['rf'][0]['param_rf__max_depth'].data,searches['rf'][0]['mean_test_score'],c='b'); plt.show()

# plt.scatter(searches['gbc'][0]['param_gbc__min_samples_leaf'].data,searches['gbc'][0]['mean_train_score'],c='r'); plt.scatter(searches['gbc'][0]['param_gbc__min_samples_leaf'].data,searches['gbc'][0]['mean_test_score'],c='b'); plt.show()

# plt.scatter(searches['lr1'][0]['param_lr1__C'].data,searches['lr1'][0]['mean_train_score'],c='r'); plt.scatter(searches['lr1'][0]['param_lr1__C'].data,searches['lr1'][0]['mean_test_score'],c='b'); plt.show()

# plt.scatter(searches['lr2'][0]['param_lr2__C'].data,searches['lr2'][0]['mean_train_score'],c='r'); plt.scatter(searches['lr2'][0]['param_lr2__C'].data,searches['lr2'][0]['mean_test_score'],c='b'); plt.show()


