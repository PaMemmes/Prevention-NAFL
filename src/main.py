import pandas as pd
import numpy as np

from preprocess import preprocess
from model import train
from interpret import interpret_tree

if __name__ == '__main__':
    df = pd.read_csv('../data/kaggle_cirrhosis.csv')
    x_train, x_test, y_train, y_test, df_cols = preprocess(df)
    model = train(x_train, x_test, y_train, y_test)
    interpret_tree(model, x_train, x_test, df_cols)
