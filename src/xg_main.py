import pandas as pd
import numpy as np

from preprocess import preprocess
from xg_model import train
from interpret import interpret_tree

if __name__ == '__main__':
    df = pd.read_csv('../data/kaggle_cirrhosis.csv')
    x_train, x_test, y_train, y_test, df_cols = preprocess(df, nn=False)
    model = train(x_train, x_test, y_train, y_test)
    interpret_tree(model, x_train.to_numpy(), x_test.to_numpy(), df_cols, nn=False)
