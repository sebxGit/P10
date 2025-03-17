import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from argparse import ArgumentParser

df = pd.read_csv('ColoradoData_Boulder.csv')
df.index = df['Start_DateTime']
df = df[['Start_DateTime', 'Energy_Consumption']].sort_index()
df.dropna(inplace=True)
df['Start_DateTime'] = pd.to_datetime(df['Start_DateTime'], format='%Y-%m-%d %H:%M:%S')
df.set_index('Start_DateTime', inplace=True)

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

parser = ArgumentParser()
parser.add_argument("--devices", type=int, default=2)
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

criterion_map = {
    "MSELoss": nn.MSELoss,
}

scaler_map = {
    "MinMaxScaler": MinMaxScaler()
}

args = parser.parse_args()

X = df.copy()
y = X['Energy_Consumption'].shift(-1).ffill()
X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_cv, y_cv, test_size=0.25, shuffle=False)
preprocessing = scaler_map.get(args.scaler)
preprocessing.fit(X_train)

X_train = preprocessing.transform(X_train)
y_train = y_train.values.reshape((-1, 1))
X_val = preprocessing.transform(X_val)
y_val = y_val.values.reshape((-1, 1))
X_test = preprocessing.transform(X_test)
y_test = y_test.values.reshape((-1, 1))

train_dataset = TimeSeriesDataset(X_train, y_train, seq_len=args.seq_len)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)

val_dataset = TimeSeriesDataset(X_val, y_val, seq_len=args.seq_len)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)

test_dataset = TimeSeriesDataset(X_test, y_test, seq_len=args.seq_len)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)

if __name__ == '__main__':
    model = LSTM(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, criterion=criterion_map.get(args.criterion), dropout=args.dropout, learning_rate=args.learning_rate)
    trainer = L.Trainer(max_epochs=args.max_epochs, fast_dev_run=100)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)