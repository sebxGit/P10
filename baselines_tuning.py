import argparse
import calendar
import os
import gc
from typing import Any
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
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler
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

# tensorboard --logdir=Predictions/MLP-GRU-LSTM
SEED = 42
seed_everything(SEED, workers=True)


def convert_Colorado_to_hourly(data):

    # Remove unnecessary columns
    data = data.drop(columns=['Zip_Postal_Code'])

    # Convert date/time columns to datetime
    data['Start_DateTime'] = pd.to_datetime(data['Start_DateTime'])
    data['Charging_EndTime'] = pd.to_datetime(data['End_DateTime'])
    data['Charging_Time'] = pd.to_timedelta(data['Charging_Time'])

    ####################### CONVERT DATASET TO HOURLY  #######################

    # Split the session into hourly intervals
    hourly_rows = []

    # Iterate over each row in the dataframe to break charging sessions into hourly intervals
    for _, row in data.iterrows():
        start, end = row['Start_DateTime'], row['Charging_EndTime']
        energy = row['Energy_Consumption']

        # Generate hourly intervals
        hourly_intervals = pd.date_range(
            start=start.floor('h'), end=end.ceil('h'), freq='h')
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

    # Create a new dataframe from the hourly intervals
    hourly_df = pd.DataFrame(hourly_rows)

    # Aggregate the hourly intervals
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
    time_range_full = pd.date_range(start=start_time, end=end_time, freq='h')

    # Reindex the hourly data to include all hours in the range
    hourly_df = hourly_df.reindex(time_range_full, fill_value=0)

    # Return the hourly data
    return hourly_df

def add_features(hourly_df, dataset_name, historical_feature, weather_df=None):
  ####################### TIMED BASED FEATURES  #######################
  hourly_df['Day_of_Week'] = hourly_df.index.dayofweek

  # Add hour of the day
  hourly_df['Hour_of_Day'] = hourly_df.index.hour

  # Add month of the year
  hourly_df['Month_of_Year'] = hourly_df.index.month

  # Add year
  hourly_df['Year'] = hourly_df.index.year

  # Add day/night
  hourly_df['Day/Night'] = (hourly_df['Hour_of_Day'] >= 6) & (hourly_df['Hour_of_Day'] <= 18)

  # Add holiday
  if dataset_name == 'Colorado':
    us_holidays = holidays.US(years=range(hourly_df.index.year.min(), hourly_df.index.year.max() + 1))
    hourly_df['IsHoliday'] = hourly_df.index.to_series().dt.date.isin(us_holidays).astype(int)
  elif dataset_name == 'SDU':
    dk_holidays = holidays.DK(years=range(
        hourly_df.index.year.min(), hourly_df.index.year.max() + 1))
    hourly_df['IsHoliday'] = hourly_df.index.to_series().dt.date.isin(dk_holidays).astype(int)

  # Add weekend
  hourly_df['Weekend'] = (hourly_df['Day_of_Week'] >= 5).astype(int)

  ####################### CYCLIC FEATURES  #######################
  # Cos and sin transformations for cyclic features (hour of the day, day of the week, month of the year)

  hourly_df['HourSin'] = np.sin(2 * np.pi * hourly_df['Hour_of_Day'] / 24)
  hourly_df['HourCos'] = np.cos(2 * np.pi * hourly_df['Hour_of_Day'] / 24)
  hourly_df['DayOfWeekSin'] = np.sin(2 * np.pi * hourly_df['Day_of_Week'] / 7)
  hourly_df['DayOfWeekCos'] = np.cos(2 * np.pi * hourly_df['Day_of_Week'] / 7)
  hourly_df['MonthOfYearSin'] = np.sin(2 * np.pi * hourly_df['Month_of_Year'] / 12)
  hourly_df['MonthOfYearCos'] = np.cos(2 * np.pi * hourly_df['Month_of_Year'] / 12)

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
  hourly_df['Energy_Consumption_1h'] = hourly_df[historical_feature].shift(1)

  # 6h
  hourly_df['Energy_Consumption_6h'] = hourly_df[historical_feature].shift(6)

  # 12h
  hourly_df['Energy_Consumption_12h'] = hourly_df[historical_feature].shift(
      12)

  # 24h
  hourly_df['Energy_Consumption_24h'] = hourly_df[historical_feature].shift(
      24)

  # 1 week
  hourly_df['Energy_Consumption_1w'] = hourly_df[historical_feature].shift(
      24*7)

  # Rolling average
  # 24h
  hourly_df['Energy_Consumption_rolling'] = hourly_df[historical_feature].rolling(window=24).mean()

  return hourly_df

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
    df['Aggregated_charging_load_1h'] = df['Aggregated charging load'].shift(1)
    df['Aggregated_charging_load_6h'] = df['Aggregated charging load'].shift(6)
    df['Aggregated_charging_load_12h'] = df['Aggregated charging load'].shift(
        12)
    df['Aggregated_charging_load_24h'] = df['Aggregated charging load'].shift(
        24)
    df['Aggregated_charging_load_1w'] = df['Aggregated charging load'].shift(24*7)
    df['Aggregated_charging_rolling'] = df['Aggregated charging load'].rolling(window=24).mean()

    return df

def filter_data(start_date, end_date, data):
    ####################### FILTER DATASET  #######################
    data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
    # print(data.head())

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

    # Ensure data is numeric and handle non-numeric values
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


# class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1, pred_len: int = 1, stride = 1):
        self.seq_len = seq_len
        self.pred_len = pred_len  # will be 1 for classical
        self.stride = stride          # always 1 for classical

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, index):
        x_window = self.X[index: index + self.seq_len]
        y_target = self.y[index + self.seq_len]  # single step ahead
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

  def setup(self, stage: str):
    start_date = pd.to_datetime('2021-05-30')
    end_date = pd.to_datetime('2023-05-30')

    # Load and preprocess the data
    data = pd.read_csv(self.data_dir)
    data = convert_Colorado_to_hourly(data)
    data = add_features(data, dataset_name='Colorado', historical_feature='Energy_Consumption', weather_df='Colorado/denver_weather.csv')
    df = filter_data(start_date, end_date, data)

    df = df.dropna()

    df.to_csv('colorado_test.csv')

    X = df.copy()

    y = X.pop('Energy_Consumption')



    # 60/20/20 split
    X_tv, self.X_test, y_tv, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_tv, y_tv, test_size=0.25, shuffle=False)

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
  def __init__(self, data_dir: str, scaler: Any, seq_len: int, pred_len: int, stride: int, batch_size: int, num_workers: int, is_persistent: bool):
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
    start_date = pd.to_datetime('2029-01-01')
    end_date = pd.to_datetime('2032-01-01')
    df = pd.read_csv(self.data_dir, skipinitialspace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%b %d, %Y, %I:%M:%S %p")
    df['Timestamp'] = df['Timestamp'].dt.floor('h')
    df = df[['Timestamp', 'Aggregated charging load', 'Day', 'Month', 'Year', 'Hour']]

    # Mask zero away the zero values
    # df['Aggregated charging load'] = df['Aggregated charging load'].mask(df['Aggregated charging load'] == 0, np.nan)
    # print(f"Number of zero values: {df['Aggregated charging load'].isna().sum()} out of {len(df)}")
    # df = df[df["Aggregated charging load"] != 0.0]

    # print(f"Number of Masked values: {df['Aggregated charging load'].isna().sum()} out of {len(df)}")
    df.set_index('Timestamp', inplace=True) 

    # df['hour'] = df.index.hour
    # df['dayofweek'] = df.index.dayofweek
    # df['month'] = df.index.month

    # print("CSV Columns:", df.columns.tolist())


    #print(df.head())

    # df['Aggregated charging load'] = df['Aggregated charging load'].interpolate(method='time')
    # df['Aggregated charging load'] = df['Aggregated charging load'].interpolate(method='linear')
    # df['Aggregated charging load'] = df['Aggregated charging load'].interpolate(method='pchip')

    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # df['days_in_month'] = df.apply(lambda row: calendar.monthrange(row['Year'], row['Month'])[1], axis=1)
    # df['day_sin'] = np.sin(2 * np.pi * df['Day'] / df['days_in_month'])
    # df['day_cos'] = np.cos(2 * np.pi * df['Day'] / df['days_in_month'])

    df['dayofweek'] = df.index.dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    #remove  
    # df = df.drop(columns=['month', 'Year', 'hour', 'Month', 'Hour', 'Day'])
    
    # Add Logp1 transformation to the target variable 
    # df['Aggregated charging load'] = np.log1p(df['Aggregated charging load'])

    ### features 
    df['lag1h'] = df['Aggregated charging load'].shift(1)
    # df['lag3h'] = df['Aggregated charging load'].shift(3)
    # df['lag6h'] = df['Aggregated charging load'].shift(6)

    # df['lag12h'] = df['Aggregated charging load'].shift(12)
    df['lag24h'] = df['Aggregated charging load'].shift(24)  # 1 day
    # df['lag1w'] = df['Aggregated charging load'].shift(24*7)  # 1 week

    # df['roll_std_24h'] = df['Aggregated charging load'].rolling(window=24).std()
    # df['roll_min_24h'] = df['Aggregated charging load'].rolling(window=24).min()
 

    # df['rolling1h'] = df['Aggregated charging load'].rolling(window=2).mean()  # 1 hour rolling mean
    # df['rolling3h'] = df['Aggregated charging load'].rolling(window=3).mean()  # 3 hour rolling mean
    # df['rolling6h'] = df['Aggregated charging load'].rolling(window=6).mean()  # 6 hour rolling mean
    # df['rolling12h'] = df['Aggregated charging load'].rolling(window=12).mean()  # 12 hour rolling mean
    # df['roll_max_24h'] = df['Aggregated charging load'].rolling(window=24).max()

    df = df.dropna()

    print("final", df.columns.tolist())

    df = filter_data(start_date, end_date, df)

    # df.to_csv('sdu_test.csv')

    # exit()

    X = df.copy()
    y = X.pop('Aggregated charging load')

    self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size=0.25, shuffle=False)

    # Start and end dates for the dataset
    # print(self.X_train.index[1], self.X_train.index[-1])
    # print(self.X_val.index[1], self.X_val.index[-1])
    # print(self.X_test.index[1], self.X_test.index[-1])
    
    # exit()

    preprocessing = self.scaler
    preprocessing.fit(self.X_train)


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
    if args.individual == 'False':
      train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    else:
      train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
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

    # Parallelize the loop
    results = Parallel(n_jobs=-1)(
        delayed(process_window)(i, X, y, seq_len, pred_len) for i in range(0, max_start, stride)
    )

    # Unpack results
    X_window, y_target = zip(*results)
    return np.array(X_window), np.array(y_target)


class SDUDataModule2(L.LightningDataModule):
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
    start_date = pd.to_datetime('2027-01-01')
    end_date = pd.to_datetime('2028-07-01')

    # Load the CSV
    df = pd.read_csv(self.data_dir, skipinitialspace=True)

    # Convert 'Timestamp' to datetime with exact format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%b %d, %Y, %I:%M:%S %p")

    # df = df[['Timestamp', 'Aggregated charging load',
    #          'Total number of EVs', 'Number of charging EVs',
    #          'Number of driving EVs', 'Overload duration [min]']]

    # Keep only relevant columns
    df = df[['Timestamp', 'Aggregated charging load',
            'Total number of EVs', 'Number of charging EVs',
             'Number of driving EVs', 'Overload duration [min]']]

    # Ensure numeric columns are correctly parsed
    numeric_cols = [
        'Aggregated charging load', 'Total number of EVs', 'Number of charging EVs',
        'Number of driving EVs', 'Overload duration [min]'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')


    # Use lowercase 'h' to avoid deprecation warning
    df['Timestamp'] = df['Timestamp'].dt.floor('h')

    # Optional: Aggregate if multiple entries exist for the same hour
    df = df.groupby('Timestamp')[numeric_cols].sum().reset_index()

    df = df[['Timestamp', 'Aggregated charging load',
        'Total number of EVs', 'Number of charging EVs',
          'Number of driving EVs']]
    
    #df = add_featuresSDU(df)

    df = df.set_index('Timestamp')

    df = filter_data(start_date, end_date, df)

    df = df.dropna()

    X = df.copy()

    y = X.pop('Aggregated charging load')

    # 60/20/20 split
    self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split( X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size=0.25, shuffle=False)

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
    train_dataset = TimeSeriesDataset(self.X_train, self.y_train, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    # sampler = BootstrapSampler(len(train_dataset), random_state=SEED)
    # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
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

    # Parallelize the loop
    results = Parallel(n_jobs=-1)(
        delayed(process_window)(i, X, y, seq_len, pred_len) for i in range(0, max_start, stride)
    )

    # Unpack results
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

def get_actuals_and_prediction_flattened(colmod, prediction):
  actuals = []
  for batch in colmod.predict_dataloader():
    x, y = batch
    actuals.extend(y.numpy())

  actuals_flattened = [item for sublist in actuals for item in sublist]
  predictions_flattened = [value.item() for tensor in prediction for value in tensor.flatten()]
  predictions_flattened = predictions_flattened[-len(actuals_flattened):]

  return predictions_flattened, actuals_flattened

def objective(args, trial):
    params = {  
        'input_size': 22 if args.dataset == "Colorado" else 13,
        'pred_len': args.pred_len,
        'seq_len': 24*7,
        'stride': args.pred_len,
        # 'batch_size': 24, 
        'batch_size': trial.suggest_int('batch_size', 16, 128, step=16) if args.model != "DPAD" else trial.suggest_int('batch_size', 16, 48, step=16),
        'criterion': torch.nn.HuberLoss(), ###CHANGE
        'optimizer': torch.optim.Adam,
        'scaler': MinMaxScaler(), ###CHANGE
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'seed': 42,
        'max_epochs': trial.suggest_int('max_epochs', 1000, 5000, step=100), ###CHANGE
        # 'max_epochs': 1000,
        'num_workers': trial.suggest_int('num_workers', 5, 20) if args.model != "DPAD" else 2, ###CHANGE
        # 'num_workers': 12,
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
        # 'num_layers': trial.suggest_int('num_layers', 1, 10), ### CHANGE 
        'num_layers': 5,
        #'dropout': trial.suggest_float('dropout', 0.0, 1),
        'dropout': 0.001,
      }
      model = LSTM(input_size=params['input_size'], pred_len=params['pred_len'], hidden_size=_params['hidden_size'], num_layers=_params['num_layers'], dropout=_params['dropout'])
    elif args.model == "GRU":
      _params = {
        'hidden_size': trial.suggest_int('hidden_size', 50, 200),
        # 'num_layers': trial.suggest_int('num_layers', 1, 10), ### CHANGE
        'num_layers': 5,
        # 'dropout': trial.suggest_float('dropout', 0.0, 1),
        'dropout': 0.001,
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
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'learning_rate_model': trial.suggest_float('learning_rate_model', 0.01, 1.0),
      }
      model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=_params['n_estimators'], max_depth=_params['max_depth'], min_samples_split=_params['min_samples_split'], min_samples_leaf=_params['min_samples_leaf'], learning_rate=_params['learning_rate_model'], random_state=params['seed']), n_jobs=-1)
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
        patch_len = trial.suggest_int('patch_len', 12, 48, step=6),
        stride=trial.suggest_int('stride', 12, 48, step=6),
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
        "stride": trial.suggest_int("stride", 1, 16, step=1),  # Stride for patching 
        "mixer_kernel_size": trial.suggest_int("mixer_kernel_size", 2, 16, step=2),  # Kernel size for the PatchMixer layer
        "d_model": trial.suggest_int("d_model", 128, 1024, step=64),  # Dimension of the model
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.05),  # Dropout rate for the model
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.5, step=0.05),  # Dropout rate for the head layers
        "e_layers": trial.suggest_int("e_layers", 1, 4),  # Number of PatchMixer layers (depth)
      })
      model = PatchMixer(_params)
    else:
      raise ValueError("Model not found")
      
    if isinstance(model, torch.nn.Module):
      print(f"-----Tuning {model.name} model-----")
      tuned_model = LightningModel(model=model, criterion=params['criterion'], optimizer=params['optimizer'], learning_rate=params['learning_rate'])

      # Trainer for fitting using DDP - Multi GPU
      trainer = L.Trainer(max_epochs=params['max_epochs'], log_every_n_steps=0, precision='16-mixed', enable_checkpointing=False, strategy='ddp_find_unused_parameters_true')  ###CHANGE
      # trainer = L.Trainer(max_epochs=params['max_epochs'], log_every_n_steps=1, precision='16-mixed' if args.mixed == 'True' else None, enable_checkpointing=False)

      trainer.fit(tuned_model, colmod)

      # New Trainer for inference on one GPU
      trainer = L.Trainer(max_epochs=params['max_epochs'], log_every_n_steps=0, precision='16-mixed' if args.mixed == 'True' else None, enable_checkpointing=False, devices=1)  ###CHANGE

      predictions = trainer.predict(tuned_model, colmod, return_predictions=True)

      pred, act = get_actuals_and_prediction_flattened(colmod, predictions)

      # pred = np.expm1(pred)  # Inverse log1p transformation
      # act = np.expm1(act)  # Inverse log1p transformation

      # print(f"Predictions: {pred[:30]}")
      # print(f"Actuals: {act[:30]}")

      # print(f"Predictions: min={pred.min()}, max={pred.max()}, mean={pred.mean()}")
      # print(f"Actuals: min={act.min()}, max={act.max()}, mean={act.mean()}")

      act_tensor = torch.tensor(act, dtype=torch.float32)
      pred_tensor = torch.tensor(pred, dtype=torch.float32)

      # train_loss = torch.nn.HuberLoss(act_tensor, pred_tensor) ## Changed
      criterion = torch.nn.HuberLoss()
      train_loss = criterion(pred_tensor, act_tensor)

    elif isinstance(model, BaseEstimator):
      name = model.__class__.__name__
      print(f"-----Training {type(model.estimator).__name__ if name == 'MultiOutputRegressor' else name} model-----")
      X_train, y_train = colmod.sklearn_setup("train")
      X_val, y_val = colmod.sklearn_setup("val")
      
      model.fit(X_train, y_train)

      y_pred = model.predict(X_val)
      
      act = y_val.reshape(-1)
      pred = y_pred.reshape(-1)

      # pred = np.expm1(pred)  # Inverse log1p transformation
      # act = np.expm1(act)  # Inverse log1p transformation

      act_tensor = torch.tensor(act, dtype=torch.float32)
      pred_tensor = torch.tensor(pred, dtype=torch.float32)

      # train_loss = torch.nn.HuberLoss(act_tensor, pred_tensor) ## Changed 
      criterion = torch.nn.HuberLoss()
      train_loss = criterion(pred_tensor, act_tensor)

    plt.figure(figsize=(10, 5))
    plt.plot(act, label='Actuals', color='blue')
    plt.plot(pred, label='Predictions', color='orange')
    plt.title(f'Actuals vs Predictions for {args.model} model')
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.savefig(f"sdu_testplot_{args.model}.png")
    plt.show()
    plt.close()
    return train_loss

def safe_objective(args, trial):
  try:
    return objective(args, trial)
  except Exception as e:
    print(f"Failed trial: {e}. Skipped this trial.")
    return float('inf')
  finally:
    gc.collect()
    torch.cuda.empty_cache()
  
def tune_model_with_optuna(args, n_trials):
  if args.load == 'True':
    try:
      if args.individual:
        study = joblib.load(f'Tunings/{args.dataset}_{args.pred_len}h_{args.model}_individual_tuning.pkl')
      else:
        study = joblib.load(f'Tunings/{args.dataset}_{args.pred_len}h_{args.model}_tuning.pkl')
      print("Loaded an old study:")
    except Exception as e:
      print("No previous tuning found. Starting a new tuning.", e) 
      study = optuna.create_study(direction="minimize")
  else:
    print("Starting a new tuning.")
    study = optuna.create_study(direction="minimize")

  study.optimize(lambda trial: objective(args, trial), n_trials=n_trials, gc_after_trial=True, timeout=37800)

  print("Len trials:", len(study.trials))
  print("Best params:", study.best_params)
  print("Best validation loss:", study.best_value)

  if study.best_value != float('inf'):
    try:
      if args.individual:
        df_tuning = pd.read_csv(f'Tunings/{args.dataset}_{args.pred_len}h_individual_tuning.csv', delimiter=',')
      else:
        df_tuning = pd.read_csv(f'Tunings/{args.dataset}_{args.pred_len}h_tuning.csv', delimiter=',')
    except Exception:
      df_tuning = pd.DataFrame(columns=['model', 'trials', 'val_loss', 'parameters'])

    new_row = {'model': args.model, 'trials': len(study.trials), 'val_loss': study.best_value, 'parameters': study.best_params}
    new_row_df = pd.DataFrame([new_row]).dropna(axis=1, how='all')
    df_tuning = pd.concat([df_tuning, new_row_df], ignore_index=True)
    df_tuning = df_tuning.sort_values(by=['model', 'val_loss'], ascending=True).reset_index(drop=True)

    if not os.path.exists(f'Tunings'):
      os.makedirs(f'Tunings', exist_ok=True)
    if args.individual:
      df_tuning.to_csv(f'Tunings/{args.dataset}_{args.pred_len}h_individual_tuning.csv', index=False)
      joblib.dump(study, f'Tunings/{args.dataset}_{args.pred_len}h_{args.model}_individual_tuning.pkl')
    else:
      df_tuning.to_csv(f'Tunings/{args.dataset}_{args.pred_len}h_tuning.csv', index=False)
      joblib.dump(study, f'Tunings/{args.dataset}_{args.pred_len}h_{args.model}_tuning.pkl')
    return study.best_params

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="SDU")
  parser.add_argument("--pred_len", type=int, default=24)
  parser.add_argument("--model", type=str, default="xPatch")
  parser.add_argument("--load", type=str, default='False')
  parser.add_argument("--mixed", type=str, default='True')
  parser.add_argument("--individual", type=str, default="False")
  args = parser.parse_args()

  best_params = tune_model_with_optuna(args, n_trials=20)