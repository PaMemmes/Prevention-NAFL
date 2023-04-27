import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from time import time
from collections import defaultdict
import json

from utils.utils import calc_metrics, calc_all

def train(x_train, x_test, y_train, y_test):
    params = {
        'num_rounds':        10,
        'max_depth':         8,
        'max_leaves':        2**8,
        'alpha':             0.9,
        'eta':               0.1,
        'gamma':             0.1,
        'learning_rate':     0.1,
        'subsample':         1,
        'reg_lambda':        1,
        'scale_pos_weight':  2,
        'objective':         'multi:softmax',
        'num_class':         3,
        'verbose':           True,
        'gpu_id':            0,
        'tree_method':       'gpu_hist'
    }

    hyperparameter_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.20],
        'max_leaves': [2**4, 2**6, 2**8]
    }

    bst = xgb.XGBClassifier(**params)
    clf = RandomizedSearchCV(bst, hyperparameter_grid, random_state=0, n_iter=50)
    
    start = time()
    model = clf.fit(x_train, y_train)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - start, len(clf.cv_results_["params"])) )  
    metrics, cm, cm_norm, preds = calc_all(model.best_estimator_, x_test, y_test)
    print(model.best_params_)
    print('METRICS ', metrics)
    with open('../results/results.json', 'w', encoding='utf-8') as f: 
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    return model.best_estimator_