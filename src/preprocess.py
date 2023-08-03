from functools import reduce
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.utils import remove_y_nans, handle_nans_gracefully, handle_nans_simple
from random import shuffle

def preprocess(df,
               val=False):
    df = df.drop(['SampleID', 'FBID', 'group'], axis=1)
    df['Age'] = df['Age'] / 365
    df['Fibrosis'] = pd.to_numeric(df['Fibrosis'].map(lambda x: x.lstrip('S')))
    y = df['Fibrosis'].copy()
    
    fib4_preds = df.copy()
    fib4_preds['Fibrosis'].loc[fib4_preds['FIB4'] < 1.45] = 0
    fib4_preds['Fibrosis'].loc[fib4_preds['FIB4'] >= 1.45] = 1
    fib4_preds['Fibrosis'].loc[fib4_preds['FIB4'] > 3.25] = 2
    fib4_preds = fib4_preds['Fibrosis'].copy()
    
    df = df.drop('Fibrosis', axis=1)
    
    ##########################################
    x = handle_nans_simple(df)
    ##########################################
    y = pd.DataFrame(y, columns=['Fibrosis'])
    
    y['Fibrosis'] = y['Fibrosis'].replace(1, 0)
    y['Fibrosis'] = y['Fibrosis'].replace(2, 1)
    y['Fibrosis'] = y['Fibrosis'].replace(3, 1)
    y['Fibrosis'] = y['Fibrosis'].replace(4, 2)
    y['Fibrosis'] = y['Fibrosis'].replace(5, 2)

    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    fib4_preds = fib4_preds.reset_index(drop=True)

    ind_list=[i for i in range(len(y))]
    shuffle(ind_list)
    
    print(len(ind_list))
    print(len(x))
    print(len(y))
    print(len(fib4_preds))

    x = x.iloc[ind_list]
    y = y.iloc[ind_list]
    fib4_preds = fib4_preds.iloc[ind_list]

    x_train = x[:int(0.8*len(x))]
    y_train = y[:int(0.8*len(x))]
    fib4_preds_train = fib4_preds[:int(0.8*len(x))]

    x_test = x[:int(0.2*len(x))]
    y_test = y[:int(0.2*len(x))]
    fib4_preds_test = fib4_preds[:int(0.2*len(x))]

    assert len(x_train) == len(y_train)
    assert len(x_train) == len(fib4_preds_train)

    assert len(x_test) == len(y_test)
    assert len(x_test) == len(fib4_preds_test)
    
    return x_train, x_test, y_train.astype(int), y_test.astype(int), fib4_preds_test, x.columns
