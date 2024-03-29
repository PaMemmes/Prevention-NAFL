import os
from collections import defaultdict
from typing import List
import json

from torch import optim, nn, utils, Tensor
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import MulticlassAccuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torchmetrics import Accuracy

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import calc_all_nn
from datasets import TrainValTestDataModule, TrainTestDataModule
from utils.plots import plot_confusion_matrix
from interpret import interpret_tree

MODEL_DIR = 'logs/'
BATCH_SIZE = 8
EPOCHS = 50
TRIALS = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def objective(trial) -> float:
    n_layers = trial.suggest_int('n_layers', 1, 7)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    output_dims = [
        trial.suggest_int(
            'n_units_l{}'.format(i),
            4,
            128,
            log=True) for i in range(n_layers)]

    datamodule = TrainValTestDataModule(batch_size=BATCH_SIZE)
    class_weights = list(datamodule.class_weights.values())
    class_weights = torch.Tensor(class_weights)
    model = NeuralNetwork(21, lr, dropout, output_dims, class_weights)
    hp_logger = TensorBoardLogger(save_dir='logs_hp', default_hp_metric=False)
    trainer = pl.Trainer(
        logger=hp_logger,
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        accelerator='gpu',
        log_every_n_steps=1,
        callbacks=PyTorchLightningPruningCallback(trial, monitor='val_loss')
    )
    trainer.fit(model, datamodule=datamodule)
    hp_logger.finalize('success')

    return trainer.callback_metrics['val_acc'].item()


def retrain_objective(trial) -> float:
    n_layers = trial.suggest_int('n_layers', 1, 9)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    output_dims = [
        trial.suggest_int(
            'n_units_l{}'.format(i),
            4,
            128,
            log=True) for i in range(n_layers)]
    
    datamodule = TrainTestDataModule(batch_size=BATCH_SIZE)
    class_weights = list(datamodule.class_weights.values())

    class_weights = torch.Tensor(class_weights)
    model = NeuralNetwork(21, lr, dropout, output_dims, class_weights)
    
    retrain_logger = TensorBoardLogger(save_dir='logs')
    trainer = pl.Trainer(
        logger=retrain_logger,
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        accelerator='gpu',
        log_every_n_steps=1
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path='best')
    y = datamodule.test_set.y.numpy()
    y = [item for sublist in y for item in sublist]
    cm, cm_norm = calc_all_nn(model.test_preds, y)
    plot_confusion_matrix(cm, name='cm_nn')
    plot_confusion_matrix(cm_norm, name='cm_nn_norm')

    retrain_logger.finalize('success')
    interpret_tree(model, datamodule.train_set.x, datamodule.test_set.x, datamodule.df_cols, nn=True)

    return trainer.callback_metrics['test_acc'].item()


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
    def __init__(self, input_dim, lr, dropout, output_dims, loss_weight) -> None:
        super().__init__()
        self.model = Net(input_dim, dropout, output_dims)
        self.lr = lr
        self.dropout = dropout
        self.output_dims = output_dims
        self.loss_weight = loss_weight.cuda()
        self.save_hyperparameters()

        self.test_preds = []

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx) -> float:
        x, y = batch
        y = y.squeeze(1)
        x = self(x)
        train_loss = F.nll_loss(x, y, weight=self.loss_weight)
        preds = torch.argmax(x, dim=1)
        train_acc = MulticlassAccuracy(num_classes=4).to(device)
        train_acc = train_acc(preds, y)
        self.log('train_loss', train_loss)
        self.log('train_acc', train_acc)
        self.log('step', self.current_epoch)
        self.logger.log_hyperparams(
            self.hparams, {
                'hp/LR': self.lr, 'hp/dropout': self.dropout})
        return train_loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        y = y.squeeze(1)
        x = self(x)
        val_loss = F.nll_loss(x, y, weight=self.loss_weight)
        x = torch.argmax(x, dim=1)
        val_acc = MulticlassAccuracy(num_classes=4).to(device)
        val_acc = val_acc(x, y)
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)
        self.log('step', self.current_epoch)
        self.logger.log_hyperparams(
            self.hparams, {
                'hp/LR': self.lr, 'hp/dropout': self.dropout})

    def test_step(self, batch, batch_idx) -> None:
        x, y = batch
        y = y.squeeze(1)
        x = self.forward(x)
        test_loss = F.nll_loss(x, y, weight=self.loss_weight)
        preds = torch.argmax(x, dim=1)
        self.test_preds.extend(preds.cpu().numpy())
        accuracy = Accuracy(task='multiclass', num_classes=4).to(device)
        accuracy = accuracy(preds, y)
        self.log_dict({'test_loss': test_loss, 'test_acc': accuracy})
        self.log('step', self.current_epoch)
        self.logger.log_hyperparams(
            self.hparams, {
                'hp/LR': self.lr, 'hp/dropout': self.dropout})

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
    print(study.best_trial)
    with open('../results/hps_nn.json', 'w', encoding='utf-8') as f:
        for key, value in trial.params.items():
            json.dump({key:value}, f, ensure_ascii=False, indent=4)
    retrain_objective(study.best_trial)
