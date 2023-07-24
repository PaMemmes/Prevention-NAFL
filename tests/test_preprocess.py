import pytest
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# from src.preprocess import preprocess

from src.utils.utils import calc_all, mice, remove_y_nans, one_hot_encoding


@pytest.fixture
def example_df():
    return pd.DataFrame(
        {
            'Feature1': [1,3,5,float('Nan'),100000,float(np.inf),-10,4,5,10,12,float('nan')],
            'Feature2': [3,12,5,3,3,1,4,10,float(-np.inf),5,10000,-1053445],
            'Feature3': [
                'Yes',
                'No',
                'Yes',
                'No',
                'Yes',
                'No',
                'Yes',
                'No',
                'Yes',
                'No',
                'Yes',
                'No'],
            'Label': [
                'Benign',
                'Benign',
                'BOT',
                'DDoS',
                'Trojan',
                'Worm',
                'Scan',
                'Benign',
                'Benign',
                'Trojan',
                'Worm',
                'Scan']})


@pytest.fixture
def example_xy(example_df):
    x = example_df
    y = example_df['Label']
    return x, y


def test_remove_y_nans(example_xy):
    x, y = example_xy
    y = pd.DataFrame(y, columns=['Label'])
    df, labels = remove_y_nans(x, y)
    assert labels.isnull().sum().sum() == 0
    assert len(df) == len(labels)

def test_oh_encoding(example_df):
    df = example_df
    df = df.drop('Label', axis=1)
    df_true = one_hot_encoding(df, df.columns, 3)
    df_false = one_hot_encoding(df, df.columns, 1)
    assert df_true.shape[1] == df_true.select_dtypes(
        include=np.number).shape[1]
    assert df_false.shape[1] != df_false.select_dtypes(
        include=np.number).shape[1]


def test_mice(example_xy):
    x, _ = example_xy
    x = x.drop('Label', axis=1)
    x = one_hot_encoding(x, x.columns, 3)
    assert x.isnull().sum().sum() != 0
    x = mice(x, 5)
    assert x.isnull().values.any() == 0
