import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def remove_nan(x, y):
    indices_to_keep = ~y.isin([np.nan,np.inf, -np.inf]).any(axis=1)
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
    low_cardinality_cols = [col for col in cols if df[col].nunique() < cardinality]
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
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

def preprocess(df):
    df = df.drop('ID', axis=1)
    #data_profile = ProfileReport(df)
    #data_profile.to_file(f'../results/data.html')
    df['Age'] = df['Age']/365
    y = df['Stage'] - 1
    x = df.drop('Stage', axis=1)
    x = x.drop(['N_Days', 'Status', 'Drug'], axis=1)
    cat_cols = get_categoricals(x)
    
    for col in cat_cols:
        x[col].fillna(x[col].mode().values[0], inplace=True)

    x = one_hot_encoding(x, x.columns, cardinality=4)
    cols = x.columns
    si = SimpleImputer(missing_values=np.nan, strategy="median")
    x = pd.DataFrame(si.fit_transform(x[x._get_numeric_data().columns]), columns=x.columns)

    y = pd.DataFrame(y, columns=['Stage'])
    
    x, y = remove_nan(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.15, random_state=420)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train.astype(int), y_test.astype(int), x.columns

