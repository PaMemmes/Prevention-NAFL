from collections import defaultdict
from functools import reduce
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer


def calc_all(model, x_test, y_test) -> list[np.array, np.array, np.array]:

    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cm_norm = confusion_matrix(y_test, preds, normalize='all')
    cohen_kappa = cohen_kappa_score(y_test, preds)
    precision, recall, fscore, support = score(y_test, preds)
    report = classification_report(y_test, preds)
    with open('../results/res.json', 'w', encoding='utf-8') as f:
        json.dump({'Precision': precision.tolist(), 
        'Recall': recall.tolist(), 'F1': fscore.tolist(), 'Support': support.tolist()
        , 'CohenCappa': cohen_kappa.tolist(), 'Accuracy':accuracy.tolist()}, f, ensure_ascii=False, indent=4)

    print(report)
    print('Accuracy', accuracy)
    print('Precision', precision)
    print('Recall', recall)
    print('fscore:', fscore)
    print('Cohen kappa', cohen_kappa)
    return cm, cm_norm, preds


def calc_all_nn(preds, y_test) -> list[np.array, np.array]:
    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cm_norm = confusion_matrix(y_test, preds, normalize='all')
    cohen_kappa = cohen_kappa_score(y_test, preds)
    precision, recall, fscore, support = score(y_test, preds)
    report = classification_report(y_test, preds)
    with open('../results/nn_res.json', 'w', encoding='utf-8') as f:
            json.dump({'Precision': precision.tolist(), 
            'Recall': recall.tolist(), 'F1': fscore.tolist(), 'Support': support.tolist()
            , 'CohenCappa': cohen_kappa.tolist(), 'Accuracy':accuracy.tolist()}, f, ensure_ascii=False, indent=4)
    print(report)
    print('Accuracy', accuracy)
    print('fscore:', fscore)
    print('Cohen kappa', cohen_kappa)
    return cm, cm_norm

def handle_nans_simple(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    x = pd.DataFrame(imp.fit_transform(x), columns=x.columns)
    x = one_hot_encoding(x, x.columns, cardinality=4)
    return x

def handle_nans_gracefully(x):
    num_cols = x._get_numeric_data().columns
    cat_cols = list(set(x.columns) - set(num_cols))
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    x_cat = pd.DataFrame(imp.fit_transform(x[cat_cols]), columns=cat_cols)
    x_num = mice(x[num_cols], 50)
    x = pd.concat([x_cat, x_num], axis=1)
    x = one_hot_encoding(x, x.columns, cardinality=4)
    return x

def mice(data, m) -> pd.DataFrame:
    data = data.replace([np.inf, -np.inf], np.nan)
    imp_dfs = []
    for i in range(m):
        imp = IterativeImputer(
            missing_values=np.nan,
            random_state=i,
            sample_posterior=True)
        imp_dfs.append(
            pd.DataFrame(
                imp.fit_transform(data),
                columns=data.columns))
    x = reduce(lambda x, y: x.add(y), imp_dfs) / len(imp_dfs)
    return x


def remove_y_nans(x, y) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices_to_keep = ~y.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    len_y = len(y)
    x = x.loc[indices_to_keep]
    y = y.loc[indices_to_keep]
    print('Dropped rows (=Rows that have missing "Stage"): ', len_y - len(y))
    print('Perc dropped rows: ', 100 - (len(y) / len_y * 100))
    return x, y


def one_hot_encoding(df, cols, cardinality) -> pd.DataFrame:
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
