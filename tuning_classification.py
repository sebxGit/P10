import argparse
import os
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import holidays
import optuna
from sklearn.discriminant_analysis import StandardScaler
from models.LSTM import LSTM
from models.GRU import GRU
from models.MLP import MLP
from models.D_PAD_adpGCN import DPAD_GCN
from models.xPatch import xPatch
from models.PatchMixer import PatchMixer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils import resample

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch import seed_everything
from joblib import Parallel, delayed

SEED = 42
seed_everything(SEED, workers=True)

def convert_Colorado_to_hourly(data):
    data = data.drop(columns=['Zip_Postal_Code'])
    data['Start_DateTime'] = pd.to_datetime(data['Start_DateTime'])
    data['Charging_EndTime'] = pd.to_datetime(data['End_DateTime'])
    data['Charging_Time'] = pd.to_timedelta(data['Charging_Time'])

    hourly_rows = []
    for _, row in data.iterrows():
        start, end = row['Start_DateTime'], row['Charging_EndTime']
        energy = row['Energy_Consumption']

        hourly_intervals = pd.date_range(
            start=start.floor('h'), end=end.ceil('h'), freq='h')
        total_duration = (end - start).total_seconds()

        for i in range(len(hourly_intervals) - 1):
            interval_start = max(start, hourly_intervals[i])
            interval_end = min(end, hourly_intervals[i+1])
            interval_duration = (interval_end - interval_start).total_seconds()

            if interval_duration > 0:
                energy_fraction = (interval_duration / total_duration) * energy

            hourly_rows.append({
                'Time': hourly_intervals[i],
                'Energy_Consumption': energy_fraction,
                "Session_Count": 1 
            })

    hourly_df = pd.DataFrame(hourly_rows)

    hourly_df = hourly_df.groupby('Time').agg({
        'Energy_Consumption': 'sum',
        'Session_Count': 'sum'
    }).reset_index()

    hourly_df['Time'] = pd.to_datetime(
        hourly_df['Time'], format="%d-%m-%Y %H:%M:%S")
    hourly_df = hourly_df.set_index('Time')

    start_time = hourly_df.index.min().normalize() 
    end_time = hourly_df.index.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(hours=1) 

    time_range_full = pd.date_range(start=start_time, end=end_time, freq='h')

    hourly_df = hourly_df.reindex(time_range_full, fill_value=0)

    return hourly_df

def add_features(hourly_df, dataset_name, historical_feature, weather_df=None):
  ####################### TIMED BASED FEATURES  #######################
  hourly_df['Day_of_Week'] = hourly_df.index.dayofweek
  hourly_df['Hour_of_Day'] = hourly_df.index.hour
  hourly_df['Month_of_Year'] = hourly_df.index.month
  hourly_df['Year'] = hourly_df.index.year
  hourly_df['Day/Night'] = (hourly_df['Hour_of_Day'] >= 6) & (hourly_df['Hour_of_Day'] <= 18)

  us_holidays = holidays.US(years=range(hourly_df.index.year.min(), hourly_df.index.year.max() + 1))
  hourly_df['IsHoliday'] = hourly_df.index.to_series().dt.date.isin(us_holidays).astype(int)

  hourly_df['Weekend'] = (hourly_df['Day_of_Week'] >= 5).astype(int)

  hourly_df['HourSin'] = np.sin(2 * np.pi * hourly_df['Hour_of_Day'] / 24)
  hourly_df['HourCos'] = np.cos(2 * np.pi * hourly_df['Hour_of_Day'] / 24)
  hourly_df['DayOfWeekSin'] = np.sin(2 * np.pi * hourly_df['Day_of_Week'] / 7)
  hourly_df['DayOfWeekCos'] = np.cos(2 * np.pi * hourly_df['Day_of_Week'] / 7)
  hourly_df['MonthOfYearSin'] = np.sin(2 * np.pi * hourly_df['Month_of_Year'] / 12)
  hourly_df['MonthOfYearCos'] = np.cos(2 * np.pi * hourly_df['Month_of_Year'] / 12)

  # 0 = Spring, 1 = Summer, 2 = Autumn, 3 = Winter
  month_to_season = {1: 4, 2: 4, 3: 0, 4: 0, 5: 0, 6: 1,
                     7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 3}
  hourly_df['Season'] = hourly_df['Month_of_Year'].map(month_to_season)

  if weather_df is not None:
    weather_df = pd.read_csv(weather_df, parse_dates=['time']).set_index(
        'time').rename(columns={'temperature': 'Temperature'})

    weather_df['Temperature'] = pd.to_numeric(
        weather_df['Temperature'], errors='coerce')

    hourly_df = hourly_df.join(weather_df, how='left')

  hourly_df['Energy_Consumption_1h'] = hourly_df[historical_feature].shift(1)
  hourly_df['Energy_Consumption_6h'] = hourly_df[historical_feature].shift(6)
  hourly_df['Energy_Consumption_12h'] = hourly_df[historical_feature].shift(
      12)
  hourly_df['Energy_Consumption_24h'] = hourly_df[historical_feature].shift(
      24)

  hourly_df['Energy_Consumption_1w'] = hourly_df[historical_feature].shift(
      24*7)

  hourly_df['Energy_Consumption_rolling'] = hourly_df[historical_feature].rolling(window=24).mean()

  return hourly_df

def filter_data(start_date, end_date, data):
    data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
    return data

class TimeSeriesDataset(Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1, pred_len: int = 24, stride: int = 24):
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.stride = stride

    if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

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
        indices = resample(range(self.dataset_size), replace=True,
                           n_samples=self.dataset_size, random_state=self.random_state)
        return iter(indices)

    def __len__(self):
        return self.dataset_size

def process_window(i, X, y, seq_len, pred_len):
  X_win = X[i:i + seq_len]
  y_tar = y[i + seq_len:i + seq_len + pred_len]
  arr_x = np.asanyarray(X_win).reshape(-1)
  arr_y = np.asanyarray(y_tar).reshape(-1)
  return arr_x, arr_y

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
    self.val_dates = []


  def setup(self, stage: str):
    start_date = pd.to_datetime('2021-05-30')
    end_date = pd.to_datetime('2023-05-30')

    # Load and preprocess the data
    data = pd.read_csv(self.data_dir)
    data = convert_Colorado_to_hourly(data)
    data = add_features(data, dataset_name='Colorado', historical_feature='Energy_Consumption', weather_df='Colorado/denver_weather.csv')
    df = filter_data(start_date, end_date, data)

    df = df.dropna()

    X = df.copy()

    y = X.pop('Energy_Consumption')

    # 60/20/20 split
    X_tv, self.X_test, y_tv, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_tv, y_tv, test_size=0.25, shuffle=False)

    self.val_dates = self.X_val.index.tolist()

    preprocessing = self.scaler
    preprocessing.fit(self.X_train) 
    
    if stage == "fit" or stage is None:
      self.X_train = preprocessing.transform(self.X_train)
      self.y_train = np.array(self.y_train)

    if stage == "test" or "predict" or stage is None:
      self.X_val = preprocessing.transform(self.X_val)
      self.y_val = np.array(self.y_val)

  def train_dataloader(self):
    train_dataset = TimeSeriesDataset(self.X_train, self.y_train, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    sampler = BootstrapSampler(len(train_dataset), random_state=SEED)
    if args.individual == 'False':
      train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    else:
      train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return train_loader
  
  def predict_dataloader(self):
    val_dataset = TimeSeriesDataset(self.X_val, self.y_val, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return val_loader
  
  def sklearn_setup(self, set_name: str = "train"):
    if set_name == "train":
      if args.individual == 'False':
        X, y = resample(self.X_train, self.y_train, replace=True, n_samples=len(self.X_train), random_state=SEED)
      else:
        X, y = self.X_train, self.y_train
    elif set_name == "val":
        X, y = self.X_val, self.y_val
    elif set_name == "test":
        X, y = self.X_test, self.y_test
    else:
        raise ValueError(
            "Invalid set name. Choose from 'train', 'val', or 'test'.")

    seq_len, pred_len, stride = self.seq_len, self.pred_len, self.stride
    max_start = len(X) - (seq_len + pred_len) + 1

    # Parallelize the loop
    results = Parallel(n_jobs=-1)(
        delayed(process_window)(i, X, y, seq_len, pred_len) for i in range(0, max_start, stride)
    )

    # Unpack results
    X_window, y_target = zip(*results)
    return np.array(X_window), np.array(y_target)

class SDUDataModule(L.LightningDataModule):
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
    self.train_dates = []
    self.val_dates = []
    self.test_dates = []


  def setup(self, stage: str):
    start_date = pd.to_datetime('2029-12-31')
    end_date = pd.to_datetime('2030-12-31')

    df = pd.read_csv(self.data_dir, skipinitialspace=True)

    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], format="%b %d, %Y, %I:%M:%S %p")

    # Keep only relevant columns
    df = df[['Timestamp', 'Aggregated charging load',
            'Total number of EVs', 'Number of charging EVs',
             'Number of driving EVs', 
             'Year', 'Month', 'Day', 'Hour']]

    df['Timestamp'] = df['Timestamp'].dt.floor('h')

    df.set_index('Timestamp', inplace=True)

    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    df['dayofweek'] = df.index.dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    df['lag1h'] = df['Aggregated charging load'].shift(1)
    df['lag24h'] = df['Aggregated charging load'].shift(24)  # 1 day

    df = df.dropna()


    df = filter_data(start_date, end_date, df)

    df = df.dropna()

    X = df.copy()

    y = X.pop('Aggregated charging load')

    # 60/20/20 split
    self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split( X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size=0.25, shuffle=False)

    self.train_dates = self.X_train.index.tolist()
    self.val_dates = self.X_val.index.tolist()
    self.test_dates = self.X_test.index.tolist()

    preprocessing = self.scaler
    preprocessing.fit(self.X_train)  #

    if stage == "fit" or stage is None:
      self.X_train = preprocessing.transform(self.X_train)
      self.y_train = np.array(self.y_train)


    if stage == "test" or "predict" or stage is None:
      self.X_val = preprocessing.transform(self.X_val)
      self.y_val = np.array(self.y_val)


  def train_dataloader(self):
    train_dataset = TimeSeriesDataset(
        self.X_train, self.y_train, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    sampler = BootstrapSampler(len(train_dataset), random_state=SEED)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler,
                              shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return train_loader

  def predict_dataloader(self):
    val_dataset = TimeSeriesDataset(
        self.X_val, self.y_val, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return val_loader

  def sklearn_setup(self, set_name: str = "train"):
    if set_name == "train":
      if args.individual == 'False':
        X, y = resample(self.X_train, self.y_train, replace=True,
                        n_samples=len(self.X_train), random_state=SEED)
      else:
        X, y = self.X_train, self.y_train
    elif set_name == "val":
        X, y = self.X_val, self.y_val
    elif set_name == "test":
        X, y = self.X_test, self.y_test
    else:
        raise ValueError(
            "Invalid set name. Choose from 'train', 'val', or 'test'.")

    seq_len, pred_len, stride = self.seq_len, self.pred_len, self.stride
    max_start = len(X) - (seq_len + pred_len) + 1

    results = Parallel(n_jobs=-1)(
        delayed(process_window)(i, X, y, seq_len, pred_len) for i in range(0, max_start, stride)
    )

    X_window, y_target = zip(*results)
    return np.array(X_window), np.array(y_target)

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

def recall_score(TP, FN):
  return TP / (TP + FN)

def get_baseloads_and_parts(colmod, y_pred, actuals):
  if args.dataset == "Colorado":
    y_pred = [pred * args.multiplier for pred in y_pred]
    actuals_flat = [item * args.multiplier for sublist in actuals for item in sublist]
    baseload1 = pd.read_csv('Colorado/ElectricityDemandColorado/ColoradoDemand_val.csv')
    baseload1['Timestamp (Hour Ending)'] = pd.to_datetime(baseload1['Timestamp (Hour Ending)'])

    range1_start = pd.Timestamp('2022-08-11 00:00')
    range1_end = pd.Timestamp('2023-01-03 23:00')

    range1_start = pd.to_datetime(range1_start)
    range1_end = pd.to_datetime(range1_end)

    df_pred_act = pd.DataFrame({'y_pred': y_pred[-len(actuals_flat):], 'actuals_flat': actuals_flat})
    df_pred_act.index = colmod.val_dates[-len(actuals_flat):]

    df_part1 = df_pred_act[(df_pred_act.index >= range1_start) & (df_pred_act.index <= range1_end)]
    baseload1 = baseload1[(baseload1['Timestamp (Hour Ending)'] >= range1_start) & (baseload1['Timestamp (Hour Ending)'] <= range1_end)]
    baseload1 = baseload1[-len(actuals_flat):]

    baseloads = [baseload1]
    dfs = [df_part1]

  elif args.dataset == "SDU":
    y_pred = [pred for pred in y_pred]
    actuals_flat = [item for sublist in actuals for item in sublist]

    val_start_date = pd.to_datetime('2030-08-07 01:00:00')
    val_end_date = pd.to_datetime('2030-10-19 00:00:00')

    df = pd.read_csv('SDU Dataset/DumbCharging_2020_to_2032/Measurements.csv', skipinitialspace=True)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%b %d, %Y, %I:%M:%S %p")
    df.set_index('Timestamp', inplace=True)

    df = df[(df.index >= val_start_date) & (df.index <= val_end_date)]
    
    df = df.iloc[:len(actuals_flat)]
    df = df.iloc[1100:1400] 

    df = df[['Aggregated base load']]

    df_pred_act = pd.DataFrame({'y_pred': y_pred, 'actuals_flat': actuals_flat})
    df_pred_act.index = colmod.val_dates[:len(actuals_flat)]

    df_pred_act = df_pred_act.iloc[1100:1400]

    dates.clear()
    dates.extend(df_pred_act.index)

    baseloads = [df]
    dfs = [df_pred_act]

  return baseloads, dfs

def objective(args, trial, study):
    params = {
        'input_size': 22 if args.dataset == "Colorado" else 16,
        'pred_len': args.pred_len,
        'seq_len': 24*7,
        'stride': args.pred_len,
        'batch_size': trial.suggest_int('batch_size', 32, 256, step=16) if args.model != "DPAD" else trial.suggest_int('batch_size', 16, 48, step=16),
        'criterion': torch.nn.L1Loss(),
        # 'criterion': torch.nn.HuberLoss(delta=0.25),
        'optimizer': torch.optim.Adam,
        # 'scaler': MaxAbsScaler(),
        'scaler': MinMaxScaler(),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'seed': 42,
        'max_epochs': trial.suggest_int('max_epochs', 1000, 2000, step=100),
        # 'num_workers': trial.suggest_int('num_workers', 6, 14) if args.model != "DPAD" else 2,
        'num_workers': 10,
        'is_persistent': True
    }

    if args.dataset == "Colorado":
      colmod = ColoradoDataModule(data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv', scaler=params['scaler'], seq_len=params['seq_len'], pred_len=params['pred_len'], stride=params['stride'], batch_size=params['batch_size'], num_workers=params['num_workers'], is_persistent=params['is_persistent'])
    elif args.dataset == "SDU":
      colmod = SDUDataModule(data_dir='SDU Dataset/DumbCharging_2020_to_2032/Measurements.csv', scaler=params['scaler'], seq_len=params['seq_len'], pred_len=params['pred_len'], stride=params['stride'], batch_size=params['batch_size'], num_workers=params['num_workers'], is_persistent=params['is_persistent'])
    
    colmod.prepare_data()
    colmod.setup(stage="None")
    seed_everything(params['seed'], workers=True)

    model = None

    if args.model == "LSTM":
      _params = {
        'hidden_size': trial.suggest_int('hidden_size', 50, 200),
        'num_layers': trial.suggest_int('num_layers', 1, 10),
        'dropout': trial.suggest_float('dropout', 0.0, 1),
      }
      model = LSTM(input_size=params['input_size'], pred_len=params['pred_len'], hidden_size=_params['hidden_size'], num_layers=_params['num_layers'], dropout=_params['dropout'])
    elif args.model == "GRU":
      _params = {
          'hidden_size': trial.suggest_int('hidden_size', 50, 200),
          'num_layers': trial.suggest_int('num_layers', 1, 10),
          'dropout': trial.suggest_float('dropout', 0.0, 1),
      }
      model = GRU(input_size=params['input_size'], pred_len=params['pred_len'], hidden_size=_params['hidden_size'], num_layers=_params['num_layers'], dropout=_params['dropout'])
    elif args.model == "MLP":
      model = MLP(num_features=params['seq_len']*params['input_size'], seq_len=params['batch_size'], pred_len=params['pred_len'], hidden_size=trial.suggest_int('hidden_size', 25, 250, step=25))
    elif args.model == "AdaBoost":
      _params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate_model': trial.suggest_float('learning_rate_model', 0.01, 1.0),
      }
      model = MultiOutputRegressor(AdaBoostRegressor(n_estimators=_params['n_estimators'], learning_rate=_params['learning_rate_model'], random_state=params['seed']), n_jobs=-1)
    elif args.model == "RandomForest":
      _params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
      }
      model =  MultiOutputRegressor(RandomForestRegressor(n_estimators=_params['n_estimators'], max_depth=_params['max_depth'], min_samples_split=_params['min_samples_split'], min_samples_leaf=_params['min_samples_leaf'], max_features=_params['max_features'], random_state=params['seed']), n_jobs=-1)
    elif args.model == "GradientBoosting":
      _params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'learning_rate_model': trial.suggest_float('learning_rate_model', 0.01, 1.0),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
      }
      model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=_params['n_estimators'], max_depth=_params['max_depth'], min_samples_split=_params['min_samples_split'], subsample=_params['subsample'], min_samples_leaf=_params['min_samples_leaf'], learning_rate=_params['learning_rate_model'], random_state=params['seed']), n_jobs=-1)
    elif args.model == "DPAD":
        _params = {
          'enc_hidden': trial.suggest_int('enc_hidden', 108, 324, step=24),
          'dec_hidden': trial.suggest_int('dec_hidden', 108, 324, step=24),
          'num_levels': trial.suggest_int('num_levels', 1, 3),
          'dropout': 0.5,
          'K_IMP': trial.suggest_int('K_IMP', 1, 10),
          'RIN': trial.suggest_int('RIN', 0, 1)
        }
        model = DPAD_GCN(input_len=params['seq_len'], output_len=params['pred_len'], input_dim=params['input_size'], enc_hidden=_params['enc_hidden'], dec_hidden=_params['dec_hidden'], dropout=_params['dropout'], num_levels=_params['num_levels'], K_IMP=_params['K_IMP'], RIN=_params['RIN'])
    elif args.model == "xPatch":
      params_xpatch = Configs(
        dict(
        seq_len = params['seq_len'],
        pred_len = params['pred_len'],
        enc_in = params['input_size'],
        patch_len = trial.suggest_int('patch_len', 2, 16, step=2),
        stride=trial.suggest_int('stride', 1, 7, step=2),
        padding_patch = trial.suggest_categorical('padding_patch', ['end', 'None']),
        revin = trial.suggest_int('revin', 0, 1),
        ma_type = trial.suggest_categorical('ma_type', ['reg', 'ema']),
        alpha = trial.suggest_float('alpha', 0.0, 1.0),
        beta = trial.suggest_float('beta', 0.0, 1.0),
        )
      )
      model = xPatch(params_xpatch)
    elif args.model == "PatchMixer":
      _params = Configs({
        "enc_in": params['input_size'],                # Number of input channels
        "seq_len": params['seq_len'],               # Context window (lookback length)
        "pred_len": params['pred_len'],
        "batch_size": params['batch_size'],
        "patch_len": trial.suggest_int("patch_len", 4, 32, step=4),  # Patch size
        "stride": trial.suggest_int("stride", 2, 16, step=2),  # Stride for patching
        "mixer_kernel_size": trial.suggest_int("mixer_kernel_size", 2, 16, step=2),  # Kernel size for the PatchMixer layer
        "d_model": trial.suggest_int("d_model", 128, 1024, step=64),  # Dimension of the model
        "dropout": trial.suggest_float("dropout", 0.0, 0.8, step=0.1),  # Dropout rate for the model
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.8, step=0.1),  # Dropout rate for the head layers
        "e_layers": trial.suggest_int("e_layers", 1, 10),  # Number of PatchMixer layers (depth)
      })
      model = PatchMixer(_params)
    else:
      raise ValueError("Model not found")
      
    if isinstance(model, torch.nn.Module):
      print(f"-----Tuning {model.name} model-----")
      tuned_model = LightningModel(model=model, criterion=params['criterion'], optimizer=params['optimizer'], learning_rate=params['learning_rate'])

      # Trainer for fitting using DDP - Multi GPU
      trainer = L.Trainer(max_epochs=params['max_epochs'], log_every_n_steps=0, precision='16-mixed' if args.mixed == 'True' else None, enable_checkpointing=False, strategy='ddp_find_unused_parameters_true') 
      # trainer = L.Trainer(max_epochs=params['max_epochs'], log_every_n_steps=0, precision='16-mixed' if args.mixed == 'True' else None, enable_checkpointing=False)

      trainer.fit(tuned_model, colmod)

      # New Trainer for inference on one GPU
      trainer = L.Trainer(max_epochs=params['max_epochs'], log_every_n_steps=0, precision='16-mixed' if args.mixed == 'True' else None, enable_checkpointing=False, devices=1)

      y_pred = trainer.predict(tuned_model, colmod, return_predictions=True)
      y_pred = [value.item() for tensor in y_pred for value in tensor.flatten()]

    elif isinstance(model, BaseEstimator):
      name = model.__class__.__name__
      print(f"-----Training {type(model.estimator).__name__ if name == 'MultiOutputRegressor' else name} model-----")
      X_train, y_train = colmod.sklearn_setup("train")
      X_val, y_val = colmod.sklearn_setup("val")
      
      model.fit(X_train, y_train)
      y_pred = model.predict(X_val).reshape(-1)

    act = []
    for batch in colmod.predict_dataloader():
      x, y = batch
      act.extend(y.numpy())

    baseloads, dfs = get_baseloads_and_parts(colmod, y_pred, act)

    recall_scores = []
    mae_loss_scores = []

    for i, (baseload, df) in enumerate(zip(baseloads, dfs)):
      if args.dataset == "Colorado":
        y_pred = df['y_pred'].values
        actuals_flat = df['actuals_flat'].values
        baseload = baseload['Demand (MWh)'].values / args.downscaling

      elif args.dataset == "SDU":
        y_pred = df['y_pred'].values
        actuals_flat = df['actuals_flat'].values
        baseload = baseload['Aggregated base load'].values

      actuals = np.array(actuals_flat) + baseload
      predictions = np.array(y_pred) + baseload
      
      actual_class = np.where(actuals > args.threshold, 1, 0)
      pred_class = np.where(predictions > args.threshold, 1, 0)

      TP = np.sum((pred_class == 1) & (actual_class == 1))
      TN = np.sum((pred_class == 0) & (actual_class == 0))
      FP = np.sum((pred_class == 1) & (actual_class == 0))
      FN = np.sum((pred_class == 0) & (actual_class == 1))
      
      recall_scores.append(recall_score(TP, FN))
      mae_loss_scores.append(params['criterion'](torch.tensor(predictions), torch.tensor(actuals)))
      
    total_recall_score = np.mean(recall_scores) if len(recall_scores) > 0 else 0
    total_mae_loss_score = np.mean(mae_loss_scores) if len(mae_loss_scores) > 0 else float('inf')

    dates = colmod.val_dates[-len(predictions):]
    plt.figure(figsize=(15, 8))
    plt.plot(dates, actuals, label='Actuals')
    plt.plot(dates, predictions, label=f'predictions')
    plt.xlabel('Dates')
    plt.ylabel('Electricity Consumption (kWh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Predictions/{args.dataset}_{args.model}_{total_recall_score}_{total_mae_loss_score}_plot.png')
    plt.show()
    plt.clf()
    plt.close()

    if len(study.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) and study.best_trials:
      for best_trial in study.best_trials:
        if total_recall_score >= best_trial.values[0]:
          best_list.clear()
          best_list.append({'baseload': baseload, 'predictions': predictions, 'actuals': actuals, 'recall': total_recall_score})


    return total_recall_score, total_mae_loss_score

def safe_objective(args, trial, study):
  try:
    return objective(args, trial, study)
  except Exception as e:
    print(f"Failed trial: {e}. Skipped this trial.")
    return float('inf')
  finally:
    gc.collect()
    torch.cuda.empty_cache()
  
def tune_model_with_optuna(args, n_trials):
  if args.individual == "True":
    path_pkl = f'Tunings/{args.dataset}_{args.pred_len}h_{args.model}_classification_individual_tuning.pkl'
    path_csv = f'Tunings/{args.dataset}_{args.pred_len}h_classification_individual_tuning.csv'
  else:
    path_pkl = f'Tunings/{args.dataset}_{args.pred_len}h_{args.model}_classification_tuning.pkl'
    path_csv = f'Tunings/{args.dataset}_{args.pred_len}h_classification_tuning.csv'
  study_name = f'{args.dataset}_{args.pred_len}h_{args.model}_{"Individual" if args.individual == "True" else "BootstrapSampling"}_classification_tuning'

  if args.load == 'True':
    try:
      print("Loaded an old study:")
      study = joblib.load(path_pkl)
    except Exception as e:
      print("No previous tuning found. Starting a new tuning.", e) 
      study = optuna.create_study(directions=["maximize", "minimize"], study_name=study_name)
  else:
    print("Starting a new tuning.")
    study = optuna.create_study(directions=["maximize", "minimize"], study_name=study_name)

  study.optimize(lambda trial: safe_objective(args, trial, study), n_trials=n_trials, gc_after_trial=True, timeout=37800) #change

  if not os.path.exists(f'Tunings'):
    os.makedirs(f'Tunings', exist_ok=True)

  joblib.dump(study, path_pkl)
  baseload, predictions, actuals, recall = best_list[0]['baseload'], best_list[0]['predictions'], best_list[0]['actuals'], best_list[0]['recall']
  try:
    df_tuning = pd.read_csv(path_csv, delimiter=',')
  except Exception:
    df_tuning = pd.DataFrame(columns=['model', 'trials', 'rec', 'mae', 'parameters'])

  best_trial = max(study.best_trials, key=lambda t: t.values[0]) 

  new_row = {'model': args.model, 'trials': len(study.trials), 'rec': best_trial.values[0], 'mae': best_trial.values[1], 'parameters': best_trial.params}
  new_row_df = pd.DataFrame([new_row]).dropna(axis=1, how='all')
  df_tuning = pd.concat([df_tuning, new_row_df], ignore_index=True)
  df_tuning = df_tuning.sort_values(by=['model', 'rec'], ascending=True).reset_index(drop=True)

  df_tuning.to_csv(path_csv, index=False)

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="Colorado")
  parser.add_argument("--pred_len", type=int, default=24)
  parser.add_argument("--model", type=str, default="RandomForest")
  parser.add_argument("--load", type=str, default='False') 
  parser.add_argument("--mixed", type=str, default='True')
  parser.add_argument("--individual", type=str, default="True")
  parser.add_argument("--threshold", type=float, default=500)
  parser.add_argument("--downscaling", type=int, default=13)
  parser.add_argument("--multiplier", type=int, default=2)
  parser.add_argument("--trials", type=int, default=150) 
  args = parser.parse_args()
  
  best_list = []
  dates = []

  best_params = tune_model_with_optuna(args, n_trials=args.trials)