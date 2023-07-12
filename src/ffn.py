import os
from typing import List
from typing import Optional

from torch import optim, nn, utils, Tensor
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import MulticlassAccuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

from preprocess import preprocess
from utils.utils import remove_y_nans, one_hot_encoding, get_categoricals, calc_all_nn

MODEL_DIR = 'logs/'
BATCH_SIZE = 8
EPOCHS = 40
TRIALS = 50


def objective(trial) -> float:
    n_layers = trial.suggest_int('n_layers', 1, 6)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    output_dims = [
        trial.suggest_int(
            'n_units_l{}'.format(i),
            4,
            128,
            log=True) for i in range(n_layers)]

    model = NeuralNetwork(21, dropout, output_dims)
    datamodule = DataModule(batch_size=BATCH_SIZE)

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir='logs'),
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        accelerator='auto',
        log_every_n_steps=10,
        callbacks=PyTorchLightningPruningCallback(trial, monitor='train_loss')
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics['val_acc'].item()


def retrain_objective(trial) -> float:
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    output_dims = [
        trial.suggest_int(
            'n_units_l{}'.format(i),
            4,
            128,
            log=True) for i in range(n_layers)]

    model = NeuralNetwork(21, dropout, output_dims)
    datamodule = DataModule(batch_size=BATCH_SIZE)

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir='logs'),
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        accelerator='auto',
        log_every_n_steps=10,
        callbacks=PyTorchLightningPruningCallback(trial, monitor='train_loss')
    )
    trainer.fit(model, datamodule=datamodule)
    preds = trainer.predict(datamodule=datamodule)
    trainer.test(datamodule=datamodule)
    return trainer.callback_metrics['test_acc'].item()


class KaggleDataSet(Dataset):
    def __init__(self, x, y) -> None:
        x, y = x.values, y.values
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        return self.x[idx], self.y[idx]


class DataModule(pl.LightningModule):
    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        df = pd.read_csv('../data/kaggle_cirrhosis.csv')
        x_train, x_val, x_test, y_train, y_val, y_test = preprocess(
            df, nn=True)

        self.train_set = KaggleDataSet(x_train, y_train)
        self.val_set = KaggleDataSet(x_val, y_val)
        self.test_set = KaggleDataSet(x_test, y_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


class Net(nn.Module):
    def __init__(self, input_dim, dropout: float,
                 output_dims: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, 4))
        layers.append(nn.LogSoftmax(dim=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)

        return F.log_softmax(logits, dim=1)


class NeuralNetwork(pl.LightningModule):
    def __init__(self, input_dim, dropout, output_dims) -> None:
        super().__init__()
        self.model = Net(input_dim, dropout, output_dims)
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx) -> float:
        x, y = batch
        y = y.squeeze(1)
        x = self(x)
        train_loss = F.nll_loss(x, y)
        preds = torch.argmax(x, dim=1)
        train_acc = MulticlassAccuracy(num_classes=4)
        train_acc = train_acc(preds, y)
        self.log('train_loss', train_loss, prog_bar=True)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        y = y.squeeze(1)
        x = self(x)
        val_loss = F.nll_loss(x, y)
        x = torch.argmax(x, dim=1)
        val_acc = MulticlassAccuracy(num_classes=4)
        val_acc = val_acc(x, y)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc)
        self.log('hp_metric', val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx) -> None:
        x, y = batch
        y = y.squeeze(1)
        x = self.forward(x)
        test_loss = F.nll_loss(x, y)
        preds = torch.argmax(x, dim=1)
        accuracy = MulticlassAccuracy(num_classes=4)
        accuracy = accuracy(preds, y)
        self.log_dict(
            {'test_loss': test_loss, 'test_acc': accuracy}, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y = y.squeeze(1)
        y_hat = self.model(x)
        preds = torch.argmax(y_hat, dim=1)
        return preds


if __name__ == '__main__':

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=TRIALS)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    retrain_objective(study.best_trial)
