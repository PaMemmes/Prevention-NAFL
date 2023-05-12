from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report


def calc_all(model, x_test, y_test):

    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cm_norm = confusion_matrix(y_test, preds, normalize='all')
    cohen_kappa = cohen_kappa_score(y_test, preds)
    precision, recall, fscore, support = score(y_test, preds)
    report = classification_report(y_test, preds)
    print(report)
    print('Accuracy', accuracy)
    print('Precision', precision)
    print('Recall', recall)
    print('fscore:', fscore)
    print('Cohen kappa', cohen_kappa)
    return cm, cm_norm, preds


def remove_y_nans(x, y):
    indices_to_keep = ~y.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    len_y = len(y)
    x = x.loc[indices_to_keep]
    y = y.loc[indices_to_keep]
    print('Dropped rows (=Rows that have missing "Stage"): ', len_y - len(y))
    print('Perc dropped rows: ', 100 - (len(y) / len_y * 100))
    return x, y


def factorize(df, cols):
    for col in cols:
        df[col] = pd.factorize(df[col])[0] + 1
    return df


def one_hot_encoding(df, cols, cardinality):
    low_cardinality_cols = [
        col for col in cols if df[col].nunique() <= cardinality]
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    oh_df = pd.DataFrame(oh_encoder.fit_transform(df[low_cardinality_cols]))
    oh_df.index = df.index
    oh_df.columns = oh_encoder.get_feature_names_out()
    df = df.drop(low_cardinality_cols, axis=1)
    df = pd.concat([df, oh_df], axis=1)

    df.columns = df.columns.astype(str)
    return df


def get_categoricals(df):
    num_cols = df._get_numeric_data().columns

    cats = list(set(df.columns) - set(num_cols))
    return cats
