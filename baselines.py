import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter


import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor, BasePredictionWriter, StochasticWeightAveraging
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner import Tuner

# commands
# python baselines.py --seq_len 12 --batch_size 8 --criterion MSELoss --max_epochs 1000 --n_features 7 --hidden_size 100 --num_layers 1 --dropout 1 --learning_rate 0.001 --num_workers 6 --scaler MinMaxScaler
# tensorboard --logdir=Models/lightning_logs/      

class ColoradoDataModule(L.LightningDataModule):
  def __init__(self, data_dir: str, scaler: int, seq_len: int, batch_size: int, num_workers: int, is_persistent: bool):
    super().__init__()
    self.data_dir = data_dir
    self.scaler = scaler
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.is_persistent = is_persistent
    self.X_train = None
    self.y_train = None
    self.X_val = None
    self.y_val = None
    self.X_test = None
    self.y_test = None

  def setup(self, stage: str):
    df = pd.read_csv(self.data_dir)
    df.index = df['Start_DateTime']
    df = df[['Start_DateTime', 'Energy_Consumption']].sort_index()
    df.dropna(inplace=True)
    df['Start_DateTime'] = pd.to_datetime(df['Start_DateTime'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('Start_DateTime', inplace=True)
    X = df.copy()
    y = X['Energy_Consumption'].shift(-1).ffill()
    X_tv, self.X_test, y_tv, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_tv, y_tv, test_size=0.25, shuffle=False)
    
    preprocessing = self.scaler
    preprocessing.fit(self.X_train) # should only fit to training data
        
    if stage == "fit" or stage is None:
      self.X_train = preprocessing.transform(self.X_train)
      self.y_train = self.y_train.values.reshape((-1, 1))
      self.X_val = preprocessing.transform(self.X_val)
      self.y_val = self.y_val.values.reshape((-1, 1))

    if stage == "test" or "predict" or stage is None:
      self.X_test = preprocessing.transform(self.X_test)
      self.y_test = self.y_test.values.reshape((-1, 1))

  def train_dataloader(self):
    train_dataset = TimeSeriesDataset(self.X_train, self.y_train, seq_len=self.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    return train_loader
  
  def val_dataloader(self):
    val_dataset = TimeSeriesDataset(self.X_val, self.y_val, seq_len=self.seq_len)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    return val_loader

  def test_dataloader(self):
    test_dataset = TimeSeriesDataset(self.X_test, self.y_test, seq_len=self.seq_len)
    test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    return test_loader

  def predict_dataloader(self):
    test_dataset = TimeSeriesDataset(self.X_test, self.y_test, seq_len=self.seq_len)
    test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    return test_loader

class TimeSeriesDataset(Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
    self.X = torch.tensor(X).float()
    self.y = torch.tensor(y).float()
    self.seq_len = seq_len

  def __len__(self):
    return self.X.__len__() - (self.seq_len-1)

  def __getitem__(self, index):
    return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])

class LSTM(L.LightningModule):
  def __init__(self, input_size, hidden_size, num_layers, criterion, dropout, learning_rate):
    super().__init__()
    self.save_hyperparameters(ignore=['criterion'])
    self.dropout = dropout
    self.criterion = criterion
    self.learning_rate = learning_rate
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    out, _ = self.lstm(x)
    out = self.fc(out[:, -1, :])  # Get the last time step
    return out

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    train_loss = self.criterion(y_hat, y) 
    self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return train_loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    val_loss = self.criterion(y_hat, y)
    self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return val_loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    test_loss = self.criterion(y_hat, y)
    self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return test_loss

  def predict_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    return y_hat

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.001)

class CustomWriter(BasePredictionWriter):
  def __init__(self, output_dir, write_interval):
    super().__init__(write_interval)
    self.output_dir = output_dir

  def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
    torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))
    # torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt")) # for batch indices if needed

def create_pred_actual_plot(predictions, colmod):
    actuals = []
    for batch in colmod.predict_dataloader():
      x, y = batch
      actuals.extend(y.numpy())

    predictions_flat = [item.item() for sublist in predictions for item in sublist]
    actuals_flat = [item for sublist in actuals for item in sublist]

    with SummaryWriter('Models/lightning_logs/predictions') as writer:
      for i, (pred, actual) in enumerate(zip(predictions_flat, actuals_flat)):
        writer.add_scalars('pred_graphs', {'Prediction': pred, 'Actual': actual}, i)

parser = ArgumentParser()
parser.add_argument("--seq_len", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--criterion", type=str, default="MSELoss")
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--n_features", type=int, default=7)
parser.add_argument("--hidden_size", type=int, default=100)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--dropout", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_workers", type=int, default=5)
parser.add_argument("--scaler", type=str, default="MinMaxScaler")

criterion_map = { "MSELoss": nn.MSELoss }
scaler_map = { "MinMaxScaler": MinMaxScaler }

args = parser.parse_args()

# Consider speed up trainer by reduce precision (e.g. Trainer(precision="16-mixed")) https://lightning.ai/docs/pytorch/stable/common/precision_basic.html

if __name__ == '__main__':
    model = LSTM(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, criterion=criterion_map.get(args.criterion)(), dropout=args.dropout, learning_rate=args.learning_rate)
    colmod = ColoradoDataModule(data_dir='ColoradoData_Boulder.csv', scaler=scaler_map.get(args.scaler)(), seq_len=args.seq_len, batch_size=args.batch_size, num_workers=args.num_workers, is_persistent=True if args.num_workers > 0 else False)
    trainer = L.Trainer(max_epochs=args.max_epochs, callbacks=[EarlyStopping(monitor="val_loss", mode="min"), StochasticWeightAveraging(swa_lrs=1e-2)], default_root_dir='Models')
    tuner = Tuner(trainer)
    trainer.fit(model, colmod)
    trainer.test(model, colmod)
    predictions = trainer.predict(model, colmod, return_predictions=True)
    create_pred_actual_plot(predictions, colmod)