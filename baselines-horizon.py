import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
from models.D_PAD_adpGCN import DPAD_GCN
from models.LSTM import LSTM
from models.GRU import GRU
from models.MLP import MLP
from models.xPatch import xPatch
from models.PatchMixer import PatchMixer
from models.Fredformer import Fredformer
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils import resample
from sklearn.multioutput import MultiOutputRegressor


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
    end_time = hourly_df.index.max().normalize() + pd.Timedelta(days=1) - \
        pd.Timedelta(hours=1)  # 23:00:00

    # Change range to time_range_full, so from 00:00:00 to 23:00:00
    time_range_full = pd.date_range(start=start_time, end=end_time, freq='1h')
    hourly_df = hourly_df.reindex(time_range_full, fill_value=0)

    return hourly_df


def add_features(hourly_df, weather_df=None):

  ####################### TIMED BASED FEATURES  #######################
  hourly_df['Day_of_Week'] = hourly_df.index.dayofweek

  # Add hour of the day
  hourly_df['Hour_of_Day'] = hourly_df.index.hour

  # Add month of the year
  hourly_df['Month_of_Year'] = hourly_df.index.month

  # Add year
  hourly_df['Year'] = hourly_df.index.year

  # Add day/night
  hourly_df['Day/Night'] = (hourly_df['Hour_of_Day']
                            >= 6) & (hourly_df['Hour_of_Day'] <= 18)

  # Add holiday
  us_holidays = holidays.US(years=range(2018, 2023 + 1))
  hourly_df['IsHoliday'] = hourly_df.index.map(
      lambda x: 1 if x.date() in us_holidays else 0)

  # Add weekend
  hourly_df['Weekend'] = (hourly_df['Day_of_Week'] >= 5).astype(int)

  ####################### CYCLIC FEATURES  #######################
  # Cos and sin transformations for cyclic features (hour of the day, day of the week, month of the year)

  hourly_df['HourSin'] = np.sin(2 * np.pi * hourly_df['Hour_of_Day'] / 24)
  hourly_df['HourCos'] = np.cos(2 * np.pi * hourly_df['Hour_of_Day'] / 24)
  hourly_df['DayOfWeekSin'] = np.sin(2 * np.pi * hourly_df['Day_of_Week'] / 7)
  hourly_df['DayOfWeekCos'] = np.cos(2 * np.pi * hourly_df['Day_of_Week'] / 7)
  hourly_df['MonthOfYearSin'] = np.sin(
      2 * np.pi * hourly_df['Month_of_Year'] / 12)
  hourly_df['MonthOfYearCos'] = np.cos(
      2 * np.pi * hourly_df['Month_of_Year'] / 12)

  ####################### SEASONAL FEATURES  #######################
  # 0 = Spring, 1 = Summer, 2 = Autumn, 3 = Winter
  month_to_season = {1: 4, 2: 4, 3: 0, 4: 0, 5: 0, 6: 1,
                     7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 3}
  hourly_df['Season'] = hourly_df['Month_of_Year'].map(month_to_season)

  ####################### WEATHER FEATURES  #######################
  if weather_df is not None:
    weather_df = pd.read_csv(weather_df, parse_dates=['time']).set_index(
        'time').rename(columns={'temperature': 'Temperature'})

    # make sure tempture is a number
    weather_df['Temperature'] = pd.to_numeric(
        weather_df['Temperature'], errors='coerce')

    hourly_df = hourly_df.join(weather_df, how='left')

  ####################### HISTORICAL CONSUMPTION FEATURES  #######################
  # Lag features
  # 1h
  hourly_df['Energy_Consumption_1h'] = hourly_df['Energy_Consumption'].shift(1)

  # 6h
  hourly_df['Energy_Consumption_6h'] = hourly_df['Energy_Consumption'].shift(6)

  # 12h
  hourly_df['Energy_Consumption_12h'] = hourly_df['Energy_Consumption'].shift(
      12)

  # 24h
  hourly_df['Energy_Consumption_24h'] = hourly_df['Energy_Consumption'].shift(
      24)

  # 1 week
  hourly_df['Energy_Consumption_1w'] = hourly_df['Energy_Consumption'].shift(
      24*7)

  # Rolling average
  # 24h
  hourly_df['Energy_Consumption_rolling'] = hourly_df['Energy_Consumption'].rolling(
      window=24).mean()

  return hourly_df


def filter_data(start_date, end_date, data):
    return data[(data.index >= start_date) & (data.index <= end_date)].copy()


def add_featuresSDU(df):
    # Ensure Timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Use the Timestamp column instead of index
    df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
    df['Hour_of_Day'] = df['Timestamp'].dt.hour
    df['Month_of_Year'] = df['Timestamp'].dt.month
    df['Year'] = df['Timestamp'].dt.year
    df['Day/Night'] = (df['Hour_of_Day'] >= 6) & (df['Hour_of_Day'] <= 18)

    # Add holiday
    dk_hols = holidays.DK(years=range(
        df['Timestamp'].dt.year.min(), df['Timestamp'].dt.year.max() + 1))
    df['IsHoliday'] = df['Timestamp'].dt.date.isin(dk_hols).astype(int)

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
    # df['Aggregated_charging_load_1h'] = df['Aggregated charging load'].shift(1)
    # df['Aggregated_charging_load_6h'] = df['Aggregated charging load'].shift(6)
    # df['Aggregated_charging_load_12h'] = df['Aggregated charging load'].shift(
    #     12)
    # df['Aggregated_charging_load_24h'] = df['Aggregated charging load'].shift(
    #     24)
    # df['Aggregated_charging_load_1w'] = df['Aggregated charging load'].shift(
    #     24*7)
    # df['Aggregated_charging_rolling'] = df['Aggregated charging load'].rolling(
    #     window=24).mean()

    return df


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
    y_target = self.y[start_idx +
                      self.seq_len: start_idx + self.seq_len + self.pred_len]
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
    self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
        self.X_train_val, self.y_train_val, test_size=0.25, shuffle=False)

    preprocessing = self.scaler
    preprocessing.fit(self.X_train)  # should only fit to training data

    if stage == "fit" or stage is None:
      self.X_train = preprocessing.transform(self.X_train)
      self.y_train = np.array(self.y_train)

      # self.X_val = preprocessing.transform(self.X_val)
      # self.y_val = np.array(self.y_val)

    if stage == "test" or "predict" or stage is None:
      self.X_val = preprocessing.transform(self.X_val)
      self.y_val = np.array(self.y_val)

      # self.X_test = preprocessing.transform(self.X_test)
      # self.y_test = np.array(self.y_test)

  def train_dataloader(self):
    train_dataset = TimeSeriesDataset(
        self.X_train, self.y_train, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    sampler = BootstrapSampler(len(train_dataset), random_state=SEED)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler,
                              shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return train_loader

  # def predict_dataloader(self):
  #   test_dataset = TimeSeriesDataset(self.X_test, self.y_test, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
  #   test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
  #   return test_loader

  def predict_dataloader(self):
    val_dataset = TimeSeriesDataset(
        self.X_val, self.y_val, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return val_loader

  def sklearn_setup(self, set_name: str = "train"):
    if set_name == "train":
      X, y = resample(self.X_train, self.y_train, replace=True,
                      n_samples=len(self.X_train), random_state=SEED)
    elif set_name == "val":
      X, y = self.X_val, self.y_val
    elif set_name == "test":
      X, y = self.X_test, self.y_test
    else:
      raise ValueError(
          "Invalid set name. Choose from 'train', 'val', or 'test'.")

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

  def setup(self, stage: str):
    # Define the start and end dates
    start_date = pd.to_datetime('2024-12-31')
    end_date = pd.to_datetime('2032-12-31')

    # df = pd.read_csv(self.data_dir, skipinitialspace=True)
    # df.columns = df.columns.str.strip()

    # # Fix timestamp format and floor
    # df['Timestamp'] = df['Timestamp'].astype(str).str.strip()
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%b %d, %Y, %I:%M:%S %p")
    # df['Timestamp'] = df['Timestamp'].dt.floor('h')

    # df = convert_SDU_to_hourly(df)

    # Load the CSV
    df = pd.read_csv(self.data_dir, skipinitialspace=True)

    # Convert 'Timestamp' to datetime with exact format
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], format="%b %d, %Y, %I:%M:%S %p")

    # Keep only relevant columns
    df = df[['Timestamp', 'Aggregated charging load',
            'Total number of EVs', 'Number of charging EVs',
             'Number of driving EVs', 'Overload duration [min]']]

    # Ensure numeric columns are correctly parsed
    numeric_cols = [
        'Aggregated charging load',
        'Total number of EVs',
        'Number of charging EVs',
        'Overload duration [min]'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Use lowercase 'h' to avoid deprecation warning
    df['Timestamp'] = df['Timestamp'].dt.floor('h')

    # Optional: Aggregate if multiple entries exist for the same hour
    df = df.groupby('Timestamp')[numeric_cols].sum().reset_index()

    df = add_featuresSDU(df)

    df = df.set_index('Timestamp')

    df = filter_data(start_date, end_date, df)

    df_clean = df.dropna()

    print(df.columns)

    # save df to csv
    df_clean.to_csv("SDU_hourly.csv", index=True)

    X = df_clean.copy()

    y = X.pop('Aggregated charging load')

    # 60/20/20 split
    self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
        self.X_train_val, self.y_train_val, test_size=0.25, shuffle=False)

    preprocessing = self.scaler
    preprocessing.fit(self.X_train)  # should only fit to training data

    if stage == "fit" or stage is None:
      self.X_train = preprocessing.transform(self.X_train)
      self.y_train = np.array(self.y_train)

      # self.X_val = preprocessing.transform(self.X_val)
      # self.y_val = np.array(self.y_val)

    if stage == "test" or "predict" or stage is None:
      self.X_val = preprocessing.transform(self.X_val)
      self.y_val = np.array(self.y_val)

      # self.X_test = preprocessing.transform(self.X_test)
      # self.y_test = np.array(self.y_test)

  def train_dataloader(self):
    train_dataset = TimeSeriesDataset(
        self.X_train, self.y_train, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    sampler = BootstrapSampler(len(train_dataset), random_state=SEED)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler,
                              shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return train_loader

  # def predict_dataloader(self):
  #   test_dataset = TimeSeriesDataset(self.X_test, self.y_test, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
  #   test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
  #   return test_loader

  def predict_dataloader(self):
    val_dataset = TimeSeriesDataset(
        self.X_val, self.y_val, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return val_loader

  def sklearn_setup(self, set_name: str = "train"):
    if set_name == "train":
      X, y = resample(self.X_train, self.y_train, replace=True,
                      n_samples=len(self.X_train), random_state=SEED)
    elif set_name == "val":
      X, y = self.X_val, self.y_val
    elif set_name == "test":
      X, y = self.X_test, self.y_test
    else:
      raise ValueError(
          "Invalid set name. Choose from 'train', 'val', or 'test'.")

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
    filename = os.path.join(
        self.output_dir, f"{self.combined_name}/predictions_{self.model_name}.pt")
    os.makedirs(os.path.join(self.output_dir,
                self.combined_name), exist_ok=True)
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
    self.log("train_loss", train_loss, on_step=True,
             on_epoch=True, prog_bar=True, logger=True)
    return train_loss

  def predict_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    return y_hat

  def configure_optimizers(self):
    return self.optimizer(self.parameters(), lr=self.learning_rate)


params = dict(
    input_size=17,
    seq_len=(24*7)*3,
    pred_len=24,
    stride=24,
    batch_size=24,
    criterion=nn.MSELoss(),  # nn.L1Loss() (MAE), nn.MSELoss() (MSE)
    optimizer=torch.optim.Adam,
    max_epochs=500,
    n_features=7,
    hidden_size=100,
    num_layers=1,
    dropout=0,
    learning_rate=0.001,
    num_workers=0,
    is_persistent=False,
    scaler=MinMaxScaler(),
    seed=SEED,
)


class Configs:
  def __init__(self, config_dict):
    for key, value in config_dict.items():
      setattr(self, key, value)


params_xpatch = Configs(
    dict(
        seq_len=(24*7)*3,
        pred_len=24,
        enc_in=18,
        patch_len=12,
        stride=8,
        padding_patch='end',
        revin=1,
        ma_type='ema',
        alpha=0.3,
        beta=0.3,
    )
)

params_PatchMixer = Configs(
    dict(
        enc_in=21,                # Number of input channels (nvals)
        seq_len=24*7,               # Lookback window length
        pred_len=24,              # Forecasting length
        batch_size=32,             # Batch size
        patch_len=16,             # Patch size
        stride=8,                 # Stride for patching
        mixer_kernel_size=8,      # Kernel size for the PatchMixer layer
        d_model=512,              # Dimension of the model
        dropout=0.05,              # Dropout rate for the model
        head_dropout=0.0,         # Dropout rate for the head layers
        e_layers=2,               # Number of PatchMixer layers (depth)
    )
)

params_Fredformer = Configs(
    dict(
        enc_in=21,                # Number of input channels
        seq_len=24*7,               # Context window (lookback length)
        pred_len=24,              # Target window (forecasting length)
        output=0,                 # Output dimension (default 0)

        # Model architecture
        e_layers=2,               # Number of layers
        n_heads=8,                # Number of attention heads
        d_model=512,              # Dimension of the model
        d_ff=2048,                 # Dimension of feed-forward network
        dropout=0.05,              # Dropout rate
        fc_dropout=0.05,           # Fully connected dropout
        head_dropout=0.0,         # Dropout rate for the head layers
        individual=1,             # Whether to use individual heads

        # Patching
        patch_len=12,             # Patch size
        stride=8,                 # Stride for patching
        padding_patch="end",      # Padding type for patches

        # RevIN
        revin=1,               # Whether to use RevIN
        affine=0,             # Affine transformation in RevIN
        subtract_last=0,      # Subtract last value in RevIN

        # Ablation and Nystrom
        use_nys=0,            # Whether to use Nystrom approximation
        ablation=0,            # Ablation study configuration

        # Crossformer-specific parameters
        cf_dim=48,               # Crossformer dimension
        cf_depth=2,               # Crossformer depth
        cf_heads=6,               # Crossformer number of heads
        cf_mlp=128,               # Crossformer MLP dimension
        cf_head_dim=32,           # Crossformer head dimension
        cf_drop=0.2,              # Crossformer dropout rate

        # MLP-specific parameters
        mlp_hidden=64,           # Hidden layer size for MLP
        mlp_drop=0.3              # Dropout rate for MLP
    )
)

# colmod = ColoradoDataModule(data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv', scaler=params['scaler'], seq_len=params['seq_len'], pred_len=params['pred_len'], stride=params['stride'], batch_size=params['batch_size'], num_workers=params['num_workers'], is_persistent=params['is_persistent'])
colmod = SDUDataModule(data_dir='SDU Dataset/DumbCharging_2020_to_2032/Measurements.csv', scaler=params['scaler'], seq_len=params['seq_len'], pred_len=params['pred_len'], stride=params['stride'], batch_size=params['batch_size'], num_workers=params['num_workers'], is_persistent=params['is_persistent'])
colmod.prepare_data()
colmod.setup(stage=None)

ensemble_models = [
    # MLP(num_features=params['seq_len']*params['input_size'], seq_len=params['batch_size'], pred_len=params['pred_len']),
    # GRU(input_size=params['input_size'], pred_len=params['pred_len'] ,hidden_size=params['hidden_size'], num_layers=params['num_layers'], dropout=params['dropout']),
    LSTM(input_size=params['input_size'], pred_len=params['pred_len'], hidden_size=params['hidden_size'], num_layers=params['num_layers'], dropout=params['dropout']),
    # MultiOutputRegressor(AdaBoostRegressor(n_estimators=100, random_state=42)),
    # MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5, min_samples_leaf=1, min_samples_split=2)),
    # MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=10, learning_rate=params['learning_rate'], random_state=params['seed'])),
    # xPatch(params_xpatch),
    # PatchMixer(params_PatchMixer),
    # DPAD_GCN(input_len=params['seq_len'], output_len=params['pred_len'], input_dim=params['input_size'], enc_hidden=168, dec_hidden=168, dropout=0.5, num_levels=2, K_IMP=6, RIN=1),
    # Fredformer(params_Fredformer),
]

model_names = [
    m.name if isinstance(m, torch.nn.Module)
    else type(m.estimator).__name__ if m.__class__.__name__ == "MultiOutputRegressor"
    else m.__class__.__name__
    for m in ensemble_models
]
combined_name = "-".join(model_names)

return_pred = []

for _model in ensemble_models:
  if isinstance(_model, torch.nn.Module):
    print(f"-----Training {_model.name} model-----")
    model = LightningModel(
        model=_model, criterion=params['criterion'], optimizer=params['optimizer'], learning_rate=params['learning_rate'])
    pred_writer = CustomWriter(output_dir="Predictions", write_interval="epoch",
                               combined_name=combined_name, model_name=_model.name)
    # trainer = L.Trainer(max_epochs=params['max_epochs'], log_every_n_steps=params['batch_size']//2, precision='16-mixed', accelerator="auto", devices=1, enable_progress_bar=True, enable_model_summary=False, profiler="simple")
    trainer = L.Trainer(max_epochs=params['max_epochs'], log_every_n_steps=params['batch_size']//2, accelerator="auto", precision='16-mixed',
                        devices=1, enable_progress_bar=True, callbacks=[EarlyStopping(monitor="train_loss", mode="min"), pred_writer], min_epochs=1500)
    trainer.fit(model, colmod)
    # trainer.test(model, colmod)
    y_pred_nn = trainer.predict(model, colmod, return_predictions=True)
    print(f"shape of y_pred_nn: {y_pred_nn[0].shape}")
    print(f"length of y_pred_nn: {len(y_pred_nn)}")
  elif isinstance(_model, BaseEstimator):
    print(f"-----Training {type(_model.estimator).__name__} model-----")
    X_train, y_train = colmod.sklearn_setup("train")
    X_test, y_test = colmod.sklearn_setup("test")
    _model.fit(X_train, y_train)
    y_pred = _model.predict(X_test)
    y_pred_flat = y_pred.reshape(-1)
    if not os.path.exists(f"Predictions/{combined_name}"):
      os.makedirs(f"Predictions/{combined_name}")
    torch.save(
        y_pred_flat, f"Predictions/{combined_name}/predictions_{type(_model.estimator).__name__}.pt")


# ensemble predictions
all_predictions = []
lengths = []
folder_path = f'Predictions/{combined_name}'
pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
for pt_file in pt_files:
  file_path = os.path.join(folder_path, pt_file)
  predictions = torch.load(file_path, weights_only=False)

  # Extract the model name from the file name
  model_name = pt_file.split('_')[1].split(
      '.')[0] if '_' in pt_file else pt_file.split('.')[0]

  if type(predictions[0]) == torch.Tensor:
    predictions = [elem.item()
                   for sublist in predictions for elem in sublist.flatten()]
    print(f"lengths of predictions: {len(predictions)} for model {model_name}")
  elif type(predictions[0]) == np.float64:
    predictions = predictions.tolist()
    print(f"lengths of predictions: {len(predictions)} for model {model_name}")
  lengths.append(len(predictions))
  all_predictions.append(predictions)

most_freq_len = max(set(lengths), key=lengths.count)
for i, pred in enumerate(all_predictions):
  if len(pred) < most_freq_len:
    all_predictions[i] += [0] * (most_freq_len - len(pred))
  elif len(pred) > most_freq_len:
    all_predictions[i] = pred[-most_freq_len:]

ensemble_predictions = np.mean(all_predictions, axis=0)
print(f"Length of ensemble predictions: {len(ensemble_predictions)}")
filename = f"{folder_path}/predictions_combined.pt"
torch.save(ensemble_predictions, filename)

# plot
# colmod = ColoradoDataModule(data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv', scaler=params['scaler'], seq_len=params['seq_len'], pred_len=params['pred_len'], stride=params['stride'], batch_size=params['batch_size'], num_workers=params['num_workers'], is_persistent=params['is_persistent'])
colmod = SDUDataModule(data_dir='SDU Dataset/DumbCharging_2020_to_2032/Measurements.csv', scaler=params['scaler'], seq_len=params['seq_len'], pred_len=params[
                       'pred_len'], stride=params['stride'], batch_size=params['batch_size'], num_workers=params['num_workers'], is_persistent=params['is_persistent'])
colmod.prepare_data()
colmod.setup(stage=None)

# combined_name = "combined"
combined_name = "-".join(model_names)

actuals = []
for batch in colmod.predict_dataloader():
  x, y = batch
  actuals.extend(y.numpy())

actuals_flat = [item for sublist in actuals for item in sublist]

folder_path = f'Predictions/{combined_name}'
pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

metrics = []
plt.figure(figsize=(15, 5))
plt.plot(actuals_flat, label='Actuals')
for i, pt_file in enumerate(pt_files):
  file_path = os.path.join(folder_path, pt_file)
  predictions = torch.load(file_path, weights_only=False)
  model_name = pt_file.split('_')[1].split('.')[0]
  # model_name = pt_file.split('.')[0].split('_')[-1] #use this with loss function names

  if type(predictions[0]) == torch.Tensor:
    predictions = [value.item()
                   for tensor in predictions for value in tensor.flatten()]
  elif type(predictions[0]) == np.float64:
    predictions = predictions.tolist()

  # predictions = predictions[:len(actuals_flat)] # reduce length of predictions to match actuals
  print(predictions[:10])

  metrics.append({
      'model': model_name,
      'mse': mean_squared_error(predictions, actuals_flat),
      'mae': mean_absolute_error(predictions, actuals_flat),
      'mape': mean_absolute_percentage_error(predictions, actuals_flat)})
  plt.plot(predictions, label=model_name)

loss_func_df = pd.concat([pd.DataFrame([m]) for m in metrics])
loss_func_df.set_index('model', inplace=True)
loss_func_df.to_csv(f'{folder_path}/loss_func_metrics.csv')

plt.xlabel('Samples')
plt.ylabel('Energy Consumption')
plt.title(f'Predictions vs Actuals ({combined_name})')
plt.legend()

# Ensure the directory exists before saving the file
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# plt.savefig(f'{folder_path}/predictions_vs_actuals_{combined_name}.png')
plt.show()

actuals = []
for batch in colmod.predict_dataloader():
  x, y = batch
  actuals.extend(y.numpy())

actuals_flat = [item for sublist in actuals for item in sublist]

folder_path = f'Predictions/{combined_name}'
pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

predictions = []
metrics = []

actuals_flat = actuals_flat[-500:]
belows = {}

plt.figure(figsize=(8, 5))
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Data points')
plt.ylabel('Pred-Act')
for i, pt_file in enumerate(pt_files):
    file_path = os.path.join(folder_path, pt_file)
    predictions = torch.load(file_path, weights_only=False)
    model_name = pt_file.split('.')[0].split('_')[-1]

    if len(predictions) > len(actuals_flat):
        predictions = predictions[-len(actuals_flat):]
    if type(predictions[0]) == torch.Tensor:
        predictions = [item.item()
                       for sublist in predictions for item in sublist.flatten()]
    elif type(predictions[0]) == np.float64:
        predictions = predictions.tolist()

    predictions = predictions[-500:]

    diff = np.array(predictions) - np.array(actuals_flat)
    plt.plot(diff, label=model_name)

    total_below_actuals = np.sum(
        np.array(predictions) < np.array(actuals_flat))
    percentage_total_below_actuals = total_below_actuals / \
        len(predictions) * 100
    belows.update({model_name: percentage_total_below_actuals})

# save belows to csv
belows_df = pd.DataFrame(list(belows.items()), columns=[
                         'Model', 'Percentage Below Actuals'])
belows_df.to_csv(f'Predictions/belows.csv', index=False)
plt.legend()
plt.savefig(f'{folder_path}/delta_{combined_name}.png')
# plt.show()
