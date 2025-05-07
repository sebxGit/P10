import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import holidays
import optuna
from models.D_PAD_adpGCN import DPAD_GCN
from models.LSTM import LSTM
from models.GRU import GRU
from models.MLP import MLP
from models.xPatch import xPatch
from models.PatchMixer import PatchMixer
from models.Fredformer import Fredformer
import argparse
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils import resample
from sklearn.multioutput import MultiOutputRegressor
import ast


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch import seed_everything

# Seed 
SEED = 42
seed_everything(SEED, workers=True)

def convert_to_hourly(data):

    # Remove unnecessary columns
    data = data.drop(columns=['Zip_Postal_Code'])

    # Convert date/time columns to datetime
    data['Start_DateTime'] = pd.to_datetime(data['Start_DateTime'])
    data['Charging_EndTime'] = pd.to_datetime(data['End_DateTime'])
    data['Charging_Time'] = pd.to_timedelta(data['Charging_Time'])

    ####################### CONVERT DATASET TO HOURLY  #######################

    hourly_rows = []

    # Iterate over each row in the dataframe to break charging sessions into hourly intervals
    for _, row in data.iterrows():
        start, end = row['Start_DateTime'], row['Charging_EndTime']
        energy = row['Energy_Consumption']

        hourly_intervals = pd.date_range(
            start=start.floor('1h'), end=end.ceil('1h'), freq='1h')
        total_duration = (end - start).total_seconds()

        for i in range(len(hourly_intervals) - 1):
            interval_start = max(start, hourly_intervals[i])
            interval_end = min(end, hourly_intervals[i+1])
            interval_duration = (interval_end - interval_start).total_seconds()

            # Calculate the energy consumption for the interval if interval is greater than 0 (Start and end time are different)
            if interval_duration > 0:
                energy_fraction = (interval_duration / total_duration) * energy

            hourly_rows.append({
                'Time': hourly_intervals[i],
                'Energy_Consumption': energy_fraction,
                "Session_Count": 1  # Count of sessions in the interval
            })

    hourly_df = pd.DataFrame(hourly_rows)

    hourly_df = hourly_df.groupby('Time').agg({
        'Energy_Consumption': 'sum',
        'Session_Count': 'sum'
    }).reset_index()

    # Convert the Time column to datetime
    hourly_df['Time'] = pd.to_datetime(
        hourly_df['Time'], format="%d-%m-%Y %H:%M:%S")
    hourly_df = hourly_df.set_index('Time')

    # Define time range for all 24 hours
    start_time = hourly_df.index.min().normalize()  # 00:00:00
    end_time = hourly_df.index.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(hours=1)  # 23:00:00

    # Change range to time_range_full, so from 00:00:00 to 23:00:00
    time_range_full = pd.date_range(start=start_time, end=end_time, freq='1h')
    hourly_df = hourly_df.reindex(time_range_full, fill_value=0)

    return hourly_df

def add_features(df):
  ####################### TIMED BASED FEATURES  #######################
  df['Day_of_Week'] = df.index.dayofweek
  df['Hour_of_Day'] = df.index.hour
  df['Month_of_Year'] = df.index.month
  df['Year'] = df.index.year
  df['Day/Night'] = (df['Hour_of_Day'] >= 6) & (df['Hour_of_Day'] <= 18)

  # Add holiday
  us_holidays = holidays.US(years=range(2018, 2023 + 1))
  df['IsHoliday'] = df.index.map(lambda x: 1 if x.date() in us_holidays else 0)

  # Add weekend
  df['Weekend'] = (df['Day_of_Week'] >= 5).astype(int)

  ####################### CYCLIC FEATURES  #######################

  df['HourSin'] = np.sin(2 * np.pi * df['Hour_of_Day'] / 24)
  df['HourCos'] = np.cos(2 * np.pi * df['Hour_of_Day'] / 24)
  df['DayOfWeekSin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
  df['DayOfWeekCos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
  df['MonthOfYearSin'] = np.sin(2 * np.pi * df['Month_of_Year'] / 12)
  df['MonthOfYearCos'] = np.cos(2 * np.pi * df['Month_of_Year'] / 12)

  ####################### SEASONAL FEATURES  #######################
  month_to_season = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2,
                     7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
  df['Season'] = df['Month_of_Year'].map(month_to_season)

  ####################### HISTORICAL CONSUMPTION FEATURES  #######################
  df['Energy_Consumption_1h'] = df['Energy_Consumption'].shift(1)
  df['Energy_Consumption_6h'] = df['Energy_Consumption'].shift(6)
  df['Energy_Consumption_12h'] = df['Energy_Consumption'].shift(12)
  df['Energy_Consumption_24h'] = df['Energy_Consumption'].shift(24)
  df['Energy_Consumption_1w'] = df['Energy_Consumption'].shift(24*7)
  df['Energy_Consumption_rolling'] = df['Energy_Consumption'].rolling(window=24).mean()

  return df

def filter_data(start_date, end_date, data):
    return data[(data.index >= start_date) & (data.index <= end_date)].copy()

class TimeSeriesDataset(Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1, pred_len: int = 24, stride: int = 24):
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.stride = stride

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    self.X = torch.tensor(X).float()
    self.y = torch.tensor(y).float()

  def __len__(self):
    return (len(self.X) - (self.seq_len + self.pred_len - 1)) // self.stride + 1

  def __getitem__(self, index):
    start_idx = index * self.stride
    x_window = self.X[start_idx: start_idx + self.seq_len]
    y_target = self.y[start_idx + self.seq_len: start_idx + self.seq_len + self.pred_len]
    return x_window, y_target
  
class BootstrapSampler:
    def __init__(self, dataset_size, random_state=None):
        self.dataset_size = dataset_size
        self.random_state = random_state

    def __iter__(self):
        indices = resample(range(self.dataset_size), replace=True, n_samples=self.dataset_size, random_state=self.random_state)
        return iter(indices)

    def __len__(self):
        return self.dataset_size

class ColoradoDataModule(L.LightningDataModule):
  def __init__(self, data_dir: str, scaler: int, seq_len: int, pred_len: int, stride: int, batch_size: int, num_workers: int, is_persistent: bool):
    super().__init__()
    self.data_dir = data_dir
    self.scaler = scaler
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.stride = stride
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.is_persistent = is_persistent
    self.X_train = None
    self.y_train = None
    self.X_val = None
    self.y_val = None
    self.X_test = None
    self.y_test = None
    self.X_train_val = None
    self.y_train_val = None

  def setup(self, stage: str):
    start_date = pd.to_datetime('2021-05-30')
    end_date = pd.to_datetime('2023-05-30')

    # Load and preprocess the data
    data = pd.read_csv(self.data_dir)
    data = convert_to_hourly(data)
    data = add_features(data)
    df = filter_data(start_date, end_date, data)

    df = df.dropna()

    X = df.copy()

    y = X.pop('Energy_Consumption')

    # 60/20/20 split
    self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size=0.25, shuffle=False)

    preprocessing = self.scaler
    preprocessing.fit(self.X_train)  # should only fit to training data
    
    if stage == "fit" or stage is None:
      self.X_train = preprocessing.transform(self.X_train)
      self.y_train = np.array(self.y_train)

      self.X_val = preprocessing.transform(self.X_val)
      self.y_val = np.array(self.y_val)

    if stage == "test" or "predict" or stage is None:
      self.X_test = preprocessing.transform(self.X_test)
      self.y_test = np.array(self.y_test)

  def train_dataloader(self):
    train_dataset = TimeSeriesDataset(self.X_train, self.y_train, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    bootstrap_sampler = BootstrapSampler(len(train_dataset), random_state=SEED)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=bootstrap_sampler, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return train_loader
  
  def val_dataloader(self):
    val_dataset = TimeSeriesDataset(self.X_val, self.y_val, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return val_loader

  def test_dataloader(self):
    test_dataset = TimeSeriesDataset(self.X_test, self.y_test, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return test_loader

  def predict_dataloader(self):
    val_dataset = TimeSeriesDataset(self.X_val, self.y_val, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return val_loader
  
  def sklearn_setup(self, set_name: str = "train"): 
    if set_name == "train": 
      X, y = resample(self.X_train, self.y_train, replace=True, n_samples=len(self.X_train), random_state=SEED)
    elif set_name == "val":
      X, y = self.X_val, self.y_val
    elif set_name == "test":
      X, y = self.X_test, self.y_test
    elif set_name == "train_val":
      X, y = resample(self.X_train_val, self.y_train_val, replace=True, n_samples=len(self.X_train_val), random_state=SEED)
    else:
      raise ValueError("Invalid set name. Choose from 'train', 'val', or 'test'.")

    seq_len, pred_len, stride = self.seq_len, self.pred_len, self.stride
    X_window, y_target = [], []

    max_start = len(X) - (seq_len + pred_len)+1

    for i in range(0, max_start, stride):
      X_win = X[i:i + seq_len]
      y_tar = y[i + seq_len:i + seq_len + pred_len]

      arr_x = np.asanyarray(X_win).reshape(-1)
      arr_y = np.asanyarray(y_tar).reshape(-1)

      X_window.append(arr_x)
      y_target.append(arr_y)

    return np.stack(X_window), np.stack(y_target)
  
class CustomWriter(BasePredictionWriter):
  def __init__(self, output_dir, write_interval, combined_name, model_name):
    super().__init__(write_interval)
    self.output_dir = output_dir
    self.combined_name = combined_name
    self.model_name = model_name

  def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
    filename = os.path.join(self.output_dir, f"{self.combined_name}/predictions_{self.model_name}.pt")
    os.makedirs(os.path.join(self.output_dir, self.combined_name), exist_ok=True)
    torch.save(predictions, filename)

class LightningModel(L.LightningModule):
  def __init__(self, model, criterion, optimizer, learning_rate):
    super().__init__()
    self.criterion = criterion
    self.learning_rate = learning_rate
    self.optimizer = optimizer
    self.model = model

  def forward(self, x):
    return self.model(x)

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
    return self.optimizer(self.parameters(), lr=self.learning_rate)

class Configs:
  def __init__(self, config_dict):
    for key, value in config_dict.items():
      setattr(self, key, value)

def create_and_save_ensemble(combined_name):
  all_predictions = []
  lengths = []
  folder_path = f'Tunings/{combined_name}'
  pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
  for i, pt_file in enumerate(pt_files):
    file_path = os.path.join(folder_path, pt_file)
    predictions = torch.load(file_path, weights_only=False)
    if type(predictions[0]) == torch.Tensor:
      predictions = [elem.item() for sublist in predictions for elem in sublist.flatten()]
    elif type(predictions[0]) == np.float64:
      predictions = predictions.tolist()
    lengths.append(len(predictions))
    all_predictions.append(predictions)

  most_freq_len = max(set(lengths), key=lengths.count)
  for i, pred in enumerate(all_predictions):
    if len(pred) < most_freq_len:
      all_predictions[i] += [0] * (most_freq_len - len(pred))
    elif len(pred) > most_freq_len:
      all_predictions[i] = pred[-most_freq_len:]

  ensemble_predictions = np.mean(all_predictions, axis=0)
  filename = f"{folder_path}/predictions_{combined_name}.pt"
  torch.save(ensemble_predictions, filename)

def objective(args, trial, all_subsets):
  # all_subsets_as_strings = [str(subset) for subset in all_subsets]
  # selected_subset_as_string = trial.suggest_categorical("model_subsets", all_subsets_as_strings)
  # selected_subset = ast.literal_eval(selected_subset_as_string)

  selected_subset = "['LSTM', 'GRU', 'S[xPatch, PatchMixer]', S['LSTM', 'GRU']]"
  bagging_models = [model_initializers[model]() for model in selected_subset if model in model_initializers]

  stacked_models = []
  for m in selected_models:
    if "S[" in m:
      model = m[2:-1].split(', ')
      stacked_models.append([model_initializers[model]() for model in model])

  # for bagging models
  for model in bagging_models:
    model_name = model.name if isinstance(model, torch.nn.Module) else model.__class__.__name__
    _hparams = ast.literal_eval(hparams[hparams['model'] == model_name]['parameters'].values[0])
    colmod = ColoradoDataModule(data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv', scaler=scaler_map.get(args.scaler)(), seq_len=args.seq_len, batch_size=_hparams['batch_size'], pred_len=args.pred_len, stride=args.stride, num_workers=_hparams['num_workers'], is_persistent=True if _hparams['num_workers'] > 0 else False)
    colmod.prepare_data()
    colmod.setup(stage=None)

    print(f"-----Training {model_name} model-----")
    if isinstance(model, torch.nn.Module):
      model = LightningModel(model=model, criterion=criterion_map.get(args.criterion)(), optimizer=optimizer_map.get(args.optimizer), learning_rate=_hparams['learning_rate'])
      trainer = L.Trainer(max_epochs=_hparams['max_epochs'], log_every_n_steps=50, precision='16-mixed', enable_checkpointing=False)
      trainer.fit(model, colmod)
      trainer.predict(model, colmod, return_predictions=False)

    elif isinstance(model, BaseEstimator):
      X_train, y_train = colmod.sklearn_setup("train") 
      X_val, y_val = colmod.sklearn_setup("val")

      model.fit(X_train, y_train)
      y_pred = model.predict(X_val).reshape(-1)
      if not os.path.exists(f"Tunings/{combined_name}"):
        os.makedirs(f"Tunings/{combined_name}")
      torch.save(y_pred, f"Tunings/{combined_name}/predictions_{model_name}.pt")

  # for stacking models
  for base_learners in stacked_models:
    meta_model = LinearRegression()
    predictions = []

    print(f"-----Starting Stack -----")
    for model in base_learners:
      model_name = model.name if isinstance(model, torch.nn.Module) else model.__class__.__name__
      _hparams = ast.literal_eval(hparams[hparams['model'] == model_name]['parameters'].values[0])
      colmod = ColoradoDataModule(data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv', scaler=scaler_map.get(args.scaler)(), seq_len=args.seq_len, batch_size=_hparams['batch_size'], pred_len=args.pred_len, stride=args.stride, num_workers=_hparams['num_workers'], is_persistent=True if _hparams['num_workers'] > 0 else False)
      colmod.prepare_data()
      colmod.setup(stage=None)

      print(f"-----Training {model_name} model-----")
      if isinstance(model, torch.nn.Module):
        model = LightningModel(model=model, criterion=criterion_map.get(args.criterion)(), optimizer=optimizer_map.get(args.optimizer), learning_rate=_hparams['learning_rate'])
        trainer = L.Trainer(max_epochs=_hparams['max_epochs'], log_every_n_steps=50, precision='16-mixed', enable_checkpointing=False)
        trainer.fit(model, colmod)
        y_pred = trainer.predict(model, colmod, return_predictions=True)
        print(y_pred)
        predictions.append(y_pred)
      elif isinstance(model, BaseEstimator):
        X_train, y_train = colmod.sklearn_setup("train") 
        X_val, y_val = colmod.sklearn_setup("val")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val).reshape(-1)
        predictions.append(y_pred)
        if not os.path.exists(f"Predictions/{combined_name}"):
          os.makedirs(f"Predictions/{combined_name}")
        torch.save(y_pred, f"Predictions/{combined_name}/predictions_{model_name}.pt")

    # Stack the predictions
    stacked_predictions = np.column_stack(predictions)
    # Train the meta-model
    print(stacked_predictions.shape, colmod.y_val.shape, y_val.shape)
    # meta_model.fit(stacked_predictions, y_val)
    # Make predictions on the validation set'

  exit()
  create_and_save_ensemble(combined_name)
  y_pred = torch.load(f"Tunings/{combined_name}/predictions_{combined_name}.pt")
  y_pred = y_pred.flatten()
  mae = nn.L1Loss(colmod.y_val, y_pred)
  return mae

parser = ArgumentParser()
parser.add_argument("--criterion", type=str, default="MAELoss")
parser.add_argument("--models", type=str, default="['LSTM', 'GRU', 'MLP', 'DPAD', 'S[xPatch, PatchMixer]']")
parser.add_argument("--input_size", type=int, default=21)
parser.add_argument("--pred_len", type=int, default=24)
parser.add_argument("--stride", type=int, default=24)
parser.add_argument("--seq_len", type=int, default=24*7)
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--scaler", type=str, default="MinMaxScaler")
criterion_map = { 
                  "MSELoss": nn.MSELoss, 
                  "MAELoss": nn.L1Loss,
                }

optimizer_map = { "Adam": torch.optim.Adam }

scaler_map = { "MinMaxScaler": MinMaxScaler }

args = parser.parse_args()

model_initializers = {
  "LSTM": lambda: LSTM(input_size=args.input_size, pred_len=args.pred_len, hidden_size=lstm_params['hidden_size'], num_layers=lstm_params['num_layers'], dropout=lstm_params['dropout'] ),
  "GRU": lambda: GRU(input_size=args.input_size, pred_len=args.pred_len, hidden_size=gru_params['hidden_size'], num_layers=gru_params['num_layers'], dropout=gru_params['dropout']),
  "MLP": lambda: MLP(num_features=args.seq_len*args.input_size, pred_len=args.pred_len, seq_len=mlp_params['batch_size'], hidden_size=mlp_params['hidden_size']),
  "xPatch": lambda: xPatch(xpatch_params),
  "PatchMixer": lambda: PatchMixer(patchmixer_params),
  "DPAD": lambda: DPAD_GCN(input_len=args.seq_len, output_len=args.pred_len, input_dim=args.input_size, enc_hidden=dpad_params['enc_hidden'], dec_hidden=dpad_params['dec_hidden'],
                                dropout=dpad_params['dropout'], num_levels=dpad_params['num_levels'], K_IMP=dpad_params['K_IMP'], RIN=dpad_params['RIN'])
}

if __name__ == "__main__":
  hparams = pd.read_csv('tuning.csv')
  lstm_params = ast.literal_eval(hparams[hparams['model'] == 'LSTM']['parameters'].values[0])
  gru_params = ast.literal_eval(hparams[hparams['model'] == 'GRU']['parameters'].values[0])
  mlp_params = ast.literal_eval(hparams[hparams['model'] == 'MLP']['parameters'].values[0])
  xpatch_params = Configs(ast.literal_eval(hparams[hparams['model'] == 'xPatch']['parameters'].values[0]))
  patchmixer_params = Configs(ast.literal_eval(hparams[hparams['model'] == 'PatchMixer']['parameters'].values[0]))
  dpad_params = ast.literal_eval(hparams[hparams['model'] == 'DPAD']['parameters'].values[0])
  rf_params = ast.literal_eval(hparams[hparams['model'] == 'RandomForest']['parameters'].values[0])
  ada_params = ast.literal_eval(hparams[hparams['model'] == 'AdaBoost']['parameters'].values[0])
  gb_params = ast.literal_eval(hparams[hparams['model'] == 'GradientBoosting']['parameters'].values[0])

  selected_models = ast.literal_eval(args.models)
  combined_name = "-".join([m for m in selected_models])

  all_subsets = []
  for r in range(1, len(selected_models) + 1):
    all_subsets.extend(combinations(selected_models, r))

  all_subsets = [list(subset) for subset in all_subsets if len(subset) >= 3]

  study = optuna.create_study(direction="minimize", study_name=f"Bagging-{combined_name}")
  study.optimize(lambda trial: objective(args, trial, all_subsets), n_trials=150, gc_after_trial=True, timeout=43000)
