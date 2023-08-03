import pandas as pd
import numpy as np

from preprocess import preprocess
from xg_model import train
from interpret import interpret_tree

if __name__ == '__main__':
    df1 = pd.read_csv('../data/val1.csv')
    df2 = pd.read_csv('../data/val2.csv')
    df = pd.concat([df1, df2])

    x_train, x_test, y_train, y_test, fib4_preds_test, df_cols = preprocess(df, val=False)
    model = train(x_train, x_test, y_train, y_test, fib4_preds_test)
    interpret_tree(model, x_train.to_numpy(), x_test.to_numpy(), df_cols, nn=False)
