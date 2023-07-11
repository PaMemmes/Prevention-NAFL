from functools import reduce
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.utils import remove_y_nans, one_hot_encoding, get_categoricals, mice


def preprocess(df, nn=False) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index]:
    df = df.drop('ID', axis=1)
    df['Age'] = df['Age'] / 365
    y = df['Stage'] - 1
    x = df.drop('Stage', axis=1)
    x = x.drop(['N_Days', 'Status', 'Drug'], axis=1)
    cat_cols = get_categoricals(x)

    for col in cat_cols:
        x[col].fillna(x[col].mode().values[0], inplace=True)

    x = one_hot_encoding(x, x.columns, cardinality=4)
    x = mice(x, 50)
    y = pd.DataFrame(y, columns=['Stage'])

    x, y = remove_y_nans(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=0.15, random_state=20)
    if nn == True:
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, stratify=y, test_size=0.15, random_state=20)
        # val is 0.85*0.15 = 0.1270
        return x_train, x_val, x_val, y_train.astype(int), y_val.astype(int), y_test.astype(int)

    return x_train, x_test, y_train.astype(int), y_test.astype(int), x.columns
