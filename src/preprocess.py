from functools import reduce
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.utils import remove_y_nans, handle_nans_gracefully, handle_nans_simple


def preprocess(df,
               val=False):
    df = df.drop('ID', axis=1)
    df['Age'] = df['Age'] / 365
    y = df['Stage'] - 1
    x = df.drop('Stage', axis=1)
    x = x.drop(['N_Days', 'Status', 'Drug'], axis=1)
    
    ##########################################
    x = handle_nans_simple(x)
    ##########################################

    y = pd.DataFrame(y, columns=['Stage'])
    x, y = remove_y_nans(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=0.15, random_state=20)
    if val:
        # val is 0.85*0.15 = 0.1275
        # train is 0.85*0.85 = 0.7225
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, stratify=y_train, test_size=0.15, random_state=20)
        

        scaler = MinMaxScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train))
        x_val = pd.DataFrame(scaler.transform(x_val))
        x_test = pd.DataFrame(scaler.transform(x_test))

        return x_train, x_val, x_test, y_train.astype(
            int), y_val.astype(int), y_test.astype(int), x.columns

    scaler = MinMaxScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train))
    x_test = pd.DataFrame(scaler.transform(x_test))
    return x_train, x_test, y_train.astype(int), y_test.astype(int), x.columns
