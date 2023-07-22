import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from time import time
from collections import defaultdict
import json

from utils.utils import calc_all
from utils.plots import plot_confusion_matrix


def train(x_train, x_test, y_train, y_test) -> xgb.sklearn.XGBClassifier:
    params = {
        'num_rounds': 10,
        'max_depth': 8,
        'max_leaves': 2**8,
        'alpha': 0.9,
        'eta': 0.1,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'subsample': 1,
        'reg_lambda': 1,
        'scale_pos_weight': 2,
        'objective': 'multi:softmax',
        'num_class': 4,
        'verbose': True,
        'gpu_id': 0,
        'tree_method': 'exact'
    }

    hyperparameter_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.20],
        'max_leaves': [2**4, 2**6, 2**8],
        'eta': [x for x in np.linspace(0.1, 0.6, 6)],
        'gamma': [int(x) for x in np.linspace(0, 0.5, 6)]
    }

    bst = xgb.XGBClassifier(**params)
    clf = RandomizedSearchCV(bst, hyperparameter_grid, n_iter=80)

    start = time()
    model = clf.fit(x_train, y_train)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (
        time() - start, len(clf.cv_results_["params"])))
    cm, cm_norm, preds = calc_all(model.best_estimator_, x_test, y_test)
    plot_confusion_matrix(cm, name='cm_xg')

    print(model.best_params_)
    print('Model best estimate', type(model.best_estimator_))
    return model.best_estimator_
