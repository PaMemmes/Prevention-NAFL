import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess(df):
    df = df.drop('ID', axis=1)
    df = df.drop('Drug', axis=1)
    df = df.drop('Sex', axis=1)

    before = len(df)
    indices_to_keep = ~df.isin([np.nan,np.inf, -np.inf]).any(axis=1)
    df = df.loc[indices_to_keep]
    
    
    labels = df['Stage'] - 1
    labels = labels.loc[indices_to_keep]
    df = pd.get_dummies(df, drop_first=True)
    df = df.drop('Stage', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train.astype(int), y_test.astype(int), df.columns

