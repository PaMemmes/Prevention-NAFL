from torch import optim, nn, utils, Tensor
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import MulticlassAccuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

from preprocess import preprocess
from utils.utils import remove_y_nans, one_hot_encoding, get_categoricals

class KaggleDataSet(Dataset):
    def __init__(self, x, y):
        x, y = x.values, y.values
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(y,dtype=torch.long)
  
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NeuralNetwork(pl.LightningModule):
    def __init__(self, in_features):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_features, 8), nn.ReLU(), nn.Linear(8, 10))
        self.layer2 = nn.Sequential(nn.Linear(10, 15), nn.ReLU(), nn.Linear(15, 4))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)
        x = self.forward(x)
        train_loss = F.cross_entropy(x, y)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)
        x = self.forward(x)
        val_loss = F.cross_entropy(x, y)
        self.log('val_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)
        x = self.forward(x)
        test_loss = F.cross_entropy(x, y)
        accuracy = MulticlassAccuracy(num_classes=4)
        accuracy = accuracy(x, y)
        self.log_dict({'test_loss': test_loss, 'accuracy': accuracy}, prog_bar=True)

if __name__ == '__main__':
    df = pd.read_csv('../data/kaggle_cirrhosis.csv')
    x_train, x_val, x_test, y_train, y_val, y_test = preprocess(df, nn=True)

    train_set = KaggleDataSet(x_train, y_train)
    val_set = KaggleDataSet(x_val, y_val)
    test_set = KaggleDataSet(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=8)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=8)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

    neural_network = NeuralNetwork(x_train.shape[1])
    trainer = pl.Trainer(log_every_n_steps= 10, logger=tb_logger, callbacks=[EarlyStopping(monitor='val_loss', mode='min')])
    trainer.fit(model=neural_network, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=neural_network, dataloaders=test_loader)
