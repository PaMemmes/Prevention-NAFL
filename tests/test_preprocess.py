import pytest
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# from src.preprocess import preprocess

from src.utils.utils import calc_all, remove_y_nans, factorize, one_hot_encoding, get_categoricals

@pytest.fixture
def example_df():
    return pd.DataFrame({'Feature1': [1, 3, 5, float('Nan'), 100000, 4, -10, 4, 5, float('inf'), float('-inf'), float('nan')],
                        'Feature2': [3, 12, 5, 3, 3, 1, 4, 10, 51, 5, 10000, -1053445],
                        'Feature3': ['Yes', 'No','Yes', 'No','Yes', 'No','Yes', 'No','Yes', 'No','Yes', 'No'],
                        'Label': ['Benign', 'Benign', 'BOT', 'DDoS', 'Trojan', 'Worm', 'Scan', 'Benign', 'Benign', 'Trojan', 'Worm', 'Scan']})

@pytest.fixture
def example_xy(example_df):
    x = example_df
    y = example_df['Label']
    return x, y

def test_remove_infs(example_xy):
    x, y = example_xy
    y = pd.DataFrame(y, columns=['Label'])
    df, labels = remove_y_nans(x, y)
    assert y.isnull().sum().sum() == 0

def test_oh_encoding(example_df):
    df = example_df
    df = df.drop('Label', axis=1)    
    df_true = one_hot_encoding(df, df.columns, 3)
    df_false = one_hot_encoding(df, df.columns, 1)
    assert df_true.shape[1] == df_true.select_dtypes(include=np.number).shape[1]
    assert df_false.shape[1] != df_false.select_dtypes(include=np.number).shape[1]