import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from time import time
from collections import defaultdict
import json

from utils.utils import calc_all
from utils.plots import plot_confusion_matrix


def train(x_train, x_test, y_train, y_test, fib4_preds_test) -> xgb.sklearn.XGBClassifier:
    params = {
        'num_rounds': 10,
        'max_depth': 8,
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
        'eta': list(np.linspace(0.1, 0.6, 6)),
        'gamma': [int(x) for x in np.linspace(0, 10, 10)]
    }

    bst = xgb.XGBClassifier(**params)
    clf = GridSearchCV(bst, hyperparameter_grid)

    start = time()
    model = clf.fit(x_train, y_train)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (
        time() - start, len(clf.cv_results_["params"])))
    preds = model.best_estimator_.predict(x_test)
    print('Printing out results for xgboost model')
    cm, cm_norm = calc_all(preds, y_test, 'xg')
    plot_confusion_matrix(cm, name='cm_xg')

    print(model.best_params_)
    with open('../results/hps_tree.json', 'w', encoding='utf-8') as f:
        json.dump({'HPs': model.best_params_}, f, ensure_ascii=False, indent=4)

    print('Printing out results for FIB-4')
    cm, cm_norm = calc_all(fib4_preds_test, y_test, 'fib4')
    plot_confusion_matrix(cm, name='cm_fib4')

    return model.best_estimator_
