from typing import List
from typing import Optional

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from sklearn.utils import class_weight

from preprocess import preprocess

class KaggleDataSet(Dataset):
    def __init__(self, x, y) -> None:
        x, y = x.values, y.values
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        return self.x[idx], self.y[idx]

class TrainValTestDataModule(pl.LightningModule):
    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size
        df = pd.read_csv('../data/kaggle_cirrhosis.csv')
        x_train, x_val, x_test, y_train, y_val, y_test, df_cols = preprocess(
            df, val=True)
        class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                        classes = np.unique(y_train),
                                        y = y_train.values.reshape(-1))
        self.class_weights = dict(zip(np.unique(y_train), class_weights))

    def setup(self, stage: Optional[str] = None) -> None:
        df = pd.read_csv('../data/kaggle_cirrhosis.csv')
        x_train, x_val, x_test, y_train, y_val, y_test, df_cols = preprocess(
            df, val=True)
        self.df_cols = df_cols
        self.train_set = KaggleDataSet(x_train, y_train)
        self.val_set = KaggleDataSet(x_val, y_val)
        self.test_set = KaggleDataSet(x_test, y_test)

        print('Length of Train Set: ', len(self.train_set))
        print('Length of Val Set: ', len(self.val_set))
        print('Length of Test Set: ', len(self.test_set))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers= True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers= True)

class TrainTestDataModule(pl.LightningModule):
    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size
        df = pd.read_csv('../data/kaggle_cirrhosis.csv')
        x_train, x_val, x_test, y_train, y_val, y_test, df_cols = preprocess(
            df, val=True)
        class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                        classes = np.unique(y_train),
                                        y = y_train.values.reshape(-1))
        self.class_weights = dict(zip(np.unique(y_train), class_weights))

    def setup(self, stage: Optional[str] = None) -> None:
        df = pd.read_csv('../data/kaggle_cirrhosis.csv')
        x_train, x_test, y_train, y_test, df_cols = preprocess(df, val=False)
        
        self.df_cols = df_cols
        self.train_set = KaggleDataSet(x_train, y_train)
        self.test_set = KaggleDataSet(x_test, y_test)

        print('Length of Train Set: ', len(self.train_set))
        print('Length of Test Set: ', len(self.test_set))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers= True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers= True)

    def predict_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers= True)
