import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from utils.utils import remove_nan, one_hot_encoding, get_categoricals

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

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.15, random_state=42)

    # scaler = MinMaxScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    return x_train, x_test, y_train.astype(int), y_test.astype(int), x.columns

