'''A version of n_cv designed to analyze errors
Should have just given the normal n_cv another param,
but was complicated because errors are continuous rvs'''

from sklearn import datasets

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

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, average_precision_score, brier_score_loss, mean_squared_error, median_absolute_error, r2_score
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

from time import time
from random_gkf import rand_gkf
#from profilehooks import profile


def desc(alist):
    return str(np.mean(alist)) + "+/-" + str(np.std(alist))


def desc_dict(est_d, tru_d=None):
    for n in [el for el in est_d.keys() if "curve" not in el]:
        print n, ":"
        print "\t Est", desc(est_d[n])
        if tru_d:
            print "\t True", desc(tru_d[n])


def init_dict(keys):
    return {m: [] for m in keys}


def init_param_dict(es_names, paramss):
    param_res = collections.defaultdict(dict)
    for es_name, params in zip(es_names, paramss):
        for param in params:
            param_res[es_name][param] = []

    return param_res


def update_mets(adict, y, y_hat, metrics, metric_names):
    for m, n in zip(metrics, metric_names):
        adict[n].append(m(y, y_hat))


def add_dict(adict, to_add):
    for n in adict.keys():
        adict[n].append(to_add[n])


def extend_dict(adict, to_add):
    for n in adict.keys():
        adict[n] = adict[n] + to_add[n]


def extend_param_dict(adict, to_add):
    for n in adict.keys():
        for m in adict[n].keys():
            adict[n][m] = adict[n][m] + to_add[n][m]


def av_dict(adict):
    for n, m in adict.items():
        adict[n] = np.mean(m)
    return adict


def get_splits(cv_obj):
    '''Gets list of splits from cv object'''
    splits = []
    for train,test in cv_obj:
        splits.append((train,test))
    
    return splits

def gkf(n_splits, X, y, groups):
    
    cv = GroupKFold(n_splits=n_splits)
    splits = get_splits(cv.split(X,y,groups))
    
    return splits


def split_cal(X, y, groups, n_splits, cal_threeQ):
        
    splits = gkf(n_splits=n_splits, X=X, y=y, groups=groups)
    if cal_threeQ:
        fit_ix, cal_ix = splits[0][1], splits[0][0] #devote 3/4 to calibration
	print "Devoting 3/4 to calibration"
    else:
    	fit_ix, cal_ix = splits[0][0], splits[0][1] #devote 3/4 to fitting (not calibration anymore)
	print "Devoting 3/4 to fitting"
    print "fit_ix", fit_ix
    print "cal_ix", cal_ix
    X_fit = X[fit_ix]
    y_fit = y[fit_ix]
    X_cal = X[cal_ix]
    y_cal = y[cal_ix] #y.iloc[cal_ix]
    groups_fit = groups[fit_ix]
    groups_cal = groups[cal_ix]
    
    return X, y, groups, 0, 0, 0 #don't calibrate since doing regression

#@profile
def get_best_clf(es_names, ess, paramss,
                 n_iters, n_jobs, cv, random_state,
                 X, y, groups,
                 scoring, fit_counter):

    est_res = []
    est_coef = init_dict(es_names)
    best_coef = init_dict(es_names)
    searches = init_dict(es_names)
    all_params = init_param_dict(es_names, paramss)
    
    def get_coef(est):
        if hasattr(est,'coef_'): #function
            return est.coef_
        if hasattr(est,'feature_importances_'): #forest
            return est.feature_importances_
        else:
            return 'no_coef?'

    
    for es_name, es, params, n_iter in zip(es_names, ess, paramss, n_iters):
        # need to make new cv each time; otherwise gets "used up".
        # Not sure of details.
        
        t0 = time()
        clf = RandomizedSearchCV(estimator=es, param_distributions=params,
                                 cv=cv, scoring=scoring, n_iter=n_iter, random_state=random_state, n_jobs=n_jobs, verbose=10)
        fit_counter += n_iter*5
        
        
        clf.fit(X, y)
        fit_counter += 1
        
        print "time to fit (including search)", es_name,":",time()-t0
        
        #pandas.DataFrame(clf.cv_results_)[[el for el in clf.cv_results_.keys() if ('std' in el or 'mean' in el)]]
        est_res.append((clf, clf.best_score_, es_name))
        est_coef[es_name].append(get_coef(clf.best_estimator_.named_steps[es_name]))
        searches[es_name].append(clf.cv_results_)
        for param, value in clf.best_params_.items():
            all_params[es_name][param].append(value)
    
    best_clf = sorted(est_res, key=lambda x: x[1], reverse=True)[0]  # want max of neg log loss
    print "name best clf", best_clf[2]
    best_clf_ob, _, best_clf_name = best_clf
    best_coef[best_clf_name].append(get_coef(best_clf_ob.best_estimator_.named_steps[best_clf_name]))
    return best_clf_ob, best_clf_name, est_coef, best_coef, searches, all_params, fit_counter

#@profile
def n_cv(
         X, y, groups,
         N_OUTER_SPLITS, N_INNER_SPLITS, CAL_SPLITS, use_sk_gkfold, cal_threeQ,
         es_names, ess, paramss, random_state,
         metric_names, metrics,
         n_iters, n_jobs, scoring,
         mask, fit_counter):
    
    
    if use_sk_gkfold:
        out_splits = gkf(n_splits=N_OUTER_SPLITS, X=X, y=y, groups=groups)
    else:
        out_splits = rand_gkf(groups=groups, n_splits=N_OUTER_SPLITS, random_state=random_state, shuffle_groups=True)
    
    m_d_outers = init_dict(keys=metric_names)
    all_coef_outers = init_dict(keys=es_names)
    best_coef_outers = init_dict(keys=es_names)

    # set a random state for random search.  Still thinking about this.
    # as such, the number of times the param shows up here is the # of times it won
    param_res = init_param_dict(es_names, paramss)
    all_param_res = init_param_dict(es_names, paramss)
    y_trues_hats = []
    
    #to store best clf obj
    searches = init_dict(keys=es_names)
    best_searches = []
    

    # outer loop of NCV
    for train_ix, test_ix in out_splits:
        print "size OUTER splits train,test", len(train_ix), len(test_ix)
        X_train, X_test, y_train, y_test = X[train_ix], X[test_ix], y[train_ix], y[test_ix]
        
        groups_train, groups_test = groups[train_ix], groups[test_ix]
        
        # hold out data for calib
        X_fit, y_fit, groups_fit, X_cal, y_cal, groups_cal = split_cal(X_train, y_train, groups_train, CAL_SPLITS, cal_threeQ)
        
        if use_sk_gkfold:
            inner_splits = gkf(n_splits=N_INNER_SPLITS, X=X_fit, y=y_fit, groups=groups_fit)
        else:
            inner_splits = rand_gkf(groups=groups_fit, n_splits=N_INNER_SPLITS, random_state=random_state, shuffle_groups=True)
        print "# inner splits (handled by sklearn Grid or Rand search)", N_INNER_SPLITS
        # inner loop of NCV handled by GridSearchCV

        best_clf, best_clf_name, est_coef, best_coef, est_searches, est_params, fit_counter = get_best_clf(es_names=es_names, ess=ess, paramss=paramss,
                                               n_iters=n_iters, n_jobs=n_jobs, cv=inner_splits, random_state=random_state,
                                               X=X_fit, y=y_fit, groups=groups_fit,
                                               scoring=scoring, fit_counter=fit_counter)
        extend_dict(searches,est_searches)
        extend_dict(all_coef_outers, est_coef)
        extend_dict(best_coef_outers, best_coef)
        extend_param_dict(all_param_res, est_params)

        for param, value in best_clf.best_params_.items():
            param_res[best_clf_name][param].append(value)
        best_searches.append((best_clf_name, best_clf.cv_results_))
        
        #calibrate best classifer
        #best_clf = CalibratedClassifierCV(best_clf.best_estimator_, cv='prefit').fit(X_cal, y_cal)        #does it return a clf or estimator?

        # get mets for each fold
        y_hat = best_clf.predict(X_test)
        #pdb.set_trace()
        if mask:
            y_hat = np.random.choice((0, 1), len(y_hat))
        y_trues_hats.append((y_test, y_hat))
        update_mets(m_d_outers, y=y_test, y_hat=y_hat,
                    metrics=metrics, metric_names=metric_names)
        # #### To not have to wait until the end. Not hardcoded path
        # contain = [searches, m_d_outers, all_param_res]
        # import pickle
        # fname = '/homedirec/user/fof_full_10/data_'+str(random_state)+'_intermed.pickle'
        # with open(fname, 'wb') as f: pickle.dump(contain, f)
        # #####

    # Now refit on whole (seen) dataset, using identical modeling method
    print "Fitting clf wrt to whole data"
    X_fit, y_fit, groups_fit, X_cal, y_cal, groups_cal = split_cal(X, y, groups, CAL_SPLITS, cal_threeQ)

    if use_sk_gkfold:
        all_splits = gkf(n_splits=N_INNER_SPLITS, X=X_fit, y=y_fit, groups=groups_fit)
    else:
        all_splits = rand_gkf(groups=groups_fit, n_splits=N_INNER_SPLITS, random_state=random_state, shuffle_groups=True)
    
    best_clf, best_clf_name, all_clf_coef, full_clf_coef, full_searches, all_param_full_res, fit_counter = get_best_clf(es_names=es_names, ess=ess, paramss=paramss,
                                           n_iters=n_iters, n_jobs=n_jobs, cv=all_splits, random_state=random_state,
                                           X=X_fit, y=y_fit, groups=groups_fit,
                                           scoring=scoring, fit_counter=fit_counter)
    best_full_search = best_clf.cv_results_
    best_est = best_clf.best_estimator_
    print "best clf name wrt whole data", best_clf_name
    print "mask", mask
    #best_clf = CalibratedClassifierCV(best_clf.best_estimator_, cv = 'prefit').fit(X_cal, y_cal)
    
    return best_clf, m_d_outers, param_res, all_param_res, all_param_full_res, y_trues_hats, searches, best_searches, full_searches, best_full_search, all_coef_outers, best_coef_outers, all_clf_coef, full_clf_coef, fit_counter, out_splits
