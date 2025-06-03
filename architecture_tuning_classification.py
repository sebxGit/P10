import gc
import sys
import os
import joblib
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
import shutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils import resample
from sklearn.multioutput import MultiOutputRegressor
import ast

from joblib import Parallel, delayed


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
    df['Aggregated_charging_load_1w'] = df['Aggregated charging load'].shift(
        24*7)
    df['Aggregated_charging_rolling'] = df['Aggregated charging load'].rolling(
        window=24).mean()

    return df

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
    preprocessing.fit(self.X_train)  # should only fit to training data
    
    if stage == "fit" or stage is None:
      self.X_train = preprocessing.transform(self.X_train)
      self.y_train = np.array(self.y_train)

    if stage == "test" or "predict" or stage is None:
      self.X_test = preprocessing.transform(self.X_test)
      self.y_test = np.array(self.y_test)

  def train_dataloader(self):
    train_dataset = TimeSeriesDataset(self.X_train, self.y_train, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    sampler = BootstrapSampler(len(train_dataset), random_state=SEED)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    #train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return train_loader
  
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
    else:
        raise ValueError("Invalid set name. Choose from 'train', 'val', or 'test'.")

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
    self.val_dates = []

  def setup(self, stage: str):
    # Define the start and end dates
    # start_date = pd.to_datetime('2024-12-31')
    # end_date = pd.to_datetime('2032-12-31')
    start_date = pd.to_datetime('2029-12-31')
    end_date = pd.to_datetime('2030-12-31')

    # Load the CSV
    df = pd.read_csv(self.data_dir, skipinitialspace=True)

    # Convert 'Timestamp' to datetime with exact format
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], format="%b %d, %Y, %I:%M:%S %p")

    # Keep only relevant columns
    df = df[['Timestamp', 'Aggregated charging load',
            'Total number of EVs', 'Number of charging EVs',
             'Number of driving EVs', 
             'Year', 'Month', 'Day', 'Hour']]

    # Ensure numeric columns are correctly parsed
    # numeric_cols = [
    #     'Aggregated charging load',
    #     'Total number of EVs',
    #     'Number of driving EVs',
    #     'Number of charging EVs',
    #     'Overload duration [min]'
    # ]
    # df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

  
    # Use lowercase 'h' to avoid deprecation warning
    df['Timestamp'] = df['Timestamp'].dt.floor('h')

    # Optional: Aggregate if multiple entries exist for the same hour
    # df = df.groupby('Timestamp')[numeric_cols].sum().reset_index()

    #df = add_featuresSDU(df)

    # df['hour'] = df.index.hour
    # df['dayofweek'] = df.index.dayofweek
    # df['month'] = df.index.month

    # print("CSV Columns:", df.columns.tolist())


    #print(df.head())

    # df['Aggregated charging load'] = df['Aggregated charging load'].interpolate(method='time')
    # df['Aggregated charging load'] = df['Aggregated charging load'].interpolate(method='linear')
    # df['Aggregated charging load'] = df['Aggregated charging load'].interpolate(method='pchip')

    df.set_index('Timestamp', inplace=True)

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
    # df = df.drop(columns=['montdh', 'Year', 'hour', 'Month', 'Hour', 'Day'])
    
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

    # print("final", df.columns.tolist())

    df = filter_data(start_date, end_date, df)

    df = df.dropna()

    X = df.copy()

    y = X.pop('Aggregated charging load')

    # 60/20/20 split
    self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split( X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size=0.25, shuffle=False)

    self.val_dates = self.X_val.index.tolist()

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

  shortest_len = min(lengths)
  for i, pred in enumerate(all_predictions):
    if len(pred) > shortest_len:
      all_predictions[i] = pred[-shortest_len:]

  ensemble_predictions = np.mean(all_predictions, axis=0)
  filename = f"{folder_path}/predictions_{combined_name}.pt"
  torch.save(ensemble_predictions, filename)

def get_baseloads_and_parts(colmod, y_pred, actuals):
  if args.dataset == "Colorado":
    y_pred = [pred * args.multiplier for pred in y_pred]
    actuals_flat = [item * args.multiplier for sublist in actuals for item in sublist]
    baseload1 = pd.read_csv('Colorado/ElectricityDemandColorado/ColoradoDemand_val.csv')
    baseload1['Timestamp (Hour Ending)'] = pd.to_datetime(baseload1['Timestamp (Hour Ending)'])

    min_length = min(len(y_pred), len(actuals_flat))

    range1_start = pd.Timestamp('2022-08-11 00:00')
    range1_end = pd.Timestamp('2023-01-03 23:00')

    range1_start = pd.to_datetime(range1_start)
    range1_end = pd.to_datetime(range1_end)

    df_pred_act = pd.DataFrame({'y_pred': y_pred[-min_length:], 'actuals_flat': actuals_flat[-min_length:]})
    df_pred_act.index = colmod.val_dates[-min_length:]

    df_part1 = df_pred_act[(df_pred_act.index >= range1_start) & (df_pred_act.index <= range1_end)]
    baseload1 = baseload1[(baseload1['Timestamp (Hour Ending)'] >= range1_start) & (baseload1['Timestamp (Hour Ending)'] <= range1_end)]
    baseload1 = baseload1[-min_length:]

    dates.clear()
    dates.extend(df_pred_act.index)

    baseloads = [baseload1]
    dfs = [df_part1]

  elif args.dataset == "SDU":
    y_pred = [pred for pred in y_pred]
    actuals_flat = [item for sublist in actuals for item in sublist]

    test_start_date = pd.to_datetime('2029-10-19 05:00:00')
    test_end_date = pd.to_datetime('2031-05-26 14:00:00')

    df = pd.read_csv('SDU Dataset/DumbCharging_2020_to_2032/Measurements.csv', skipinitialspace=True)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%b %d, %Y, %I:%M:%S %p")
    df.set_index('Timestamp', inplace=True)

    df = df[(df.index >= test_start_date) & (df.index <= test_end_date)]
    
    df = df.iloc[:len(actuals_flat)]

    df = df[['Aggregated base load']]

    df_pred_act = pd.DataFrame({'y_pred': y_pred, 'actuals_flat': actuals_flat})
    df_pred_act.index = colmod.val_dates[:len(actuals_flat)]

    baseloads = [df]
    dfs = [df_pred_act]

  return baseloads, dfs

def recall_score(TP, FN):
  return TP / (TP + FN)

def objective(args, trial, all_subsets, study):
  all_subsets_as_strings = [str(subset) for subset in all_subsets]
  selected_subset_as_string = trial.suggest_categorical("model_subsets", all_subsets_as_strings)
  selected_subset = ast.literal_eval(selected_subset_as_string)

  # bagging_models = [model_initializers[model]() for model in selected_subset if model in model_initializers]

  for model_name in selected_subset:
    print(f"-----Training {model_name} model-----")
    if model_name in model_initializers:
      model = model_initializers[model_name]()
    
    model_name = model.name if isinstance(model, torch.nn.Module) else type(model.estimator).__name__ if model.__class__.__name__ == 'MultiOutputRegressor' else model.__class__.__name__
    _hparams = ast.literal_eval(hparams[hparams['model'] == model_name]['parameters'].values[0])

    if model_name == 'DPAD':
      _hparams['dropout'] = 0.5
      _hparams['num_workers'] = 2

    if args.dataset == "Colorado":
      colmod = ColoradoDataModule(data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv', scaler=scaler_map.get(args.scaler)(), seq_len=args.seq_len, batch_size=_hparams['batch_size'], pred_len=args.pred_len, stride=args.stride, num_workers=_hparams['num_workers'], is_persistent=True if _hparams['num_workers'] > 0 else False)
    else: 
      colmod = SDUDataModule(data_dir='SDU Dataset/DumbCharging_2020_to_2032/Measurements.csv', scaler=scaler_map.get(args.scaler)(), seq_len=args.seq_len, batch_size=_hparams['batch_size'], pred_len=args.pred_len, stride=args.stride, num_workers=_hparams['num_workers'], is_persistent=True if _hparams['num_workers'] > 0 else False)

    colmod.prepare_data()
    colmod.setup(stage=None)

    print(f"-----Training {model_name} model-----")
    if isinstance(model, torch.nn.Module):
      model = LightningModel(model=model, criterion=criterion_map.get(args.criterion)(), optimizer=optimizer_map.get(args.optimizer), learning_rate=_hparams['learning_rate'])
      pred_writer = CustomWriter(output_dir="Tunings", write_interval="epoch", combined_name=combined_name, model_name=model_name)
      trainer = L.Trainer(max_epochs=_hparams['max_epochs'], log_every_n_steps=100, precision='16-mixed', callbacks=[pred_writer], enable_checkpointing=False, strategy='ddp_find_unused_parameters_true')
      trainer.fit(model, colmod)

      trainer = L.Trainer(max_epochs=_hparams['max_epochs'], log_every_n_steps=100, precision='16-mixed', callbacks=[pred_writer], enable_checkpointing=False, devices=1)
      trainer.predict(model, colmod)

    elif isinstance(model, BaseEstimator):
      X_train, y_train = colmod.sklearn_setup("train") 
      X_val, y_val = colmod.sklearn_setup("val")

      model.fit(X_train, y_train)
      y_pred = model.predict(X_val).reshape(-1)
      if not os.path.exists(f"Tunings/{combined_name}"):
        os.makedirs(f"Tunings/{combined_name}")
      torch.save(y_pred, f"Tunings/{combined_name}/predictions_{model_name}.pt")

  create_and_save_ensemble(combined_name)
  y_pred = torch.load(f"Tunings/{combined_name}/predictions_{combined_name}.pt", weights_only=False)
  y_pred = y_pred.flatten()

  act = []
  for batch in colmod.predict_dataloader():
    x, y = batch
    act.extend(y.numpy())

  baseloads, dfs = get_baseloads_and_parts(colmod, y_pred, act)

  recall_scores = []
  mae_scores = []

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
    mae_scores.append(mean_absolute_error(actuals, predictions))

  total_recall_score = np.mean(recall_scores) if len(recall_scores) > 0 else 0
  total_mae_score = np.mean(mae_scores) if len(mae_scores) > 0 else float('inf')

  tuning_results.append({'combined_name': combined_name, 'rec': total_recall_score, 'mae': total_mae_score, 'parameters': trial.params})

  if len(study.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) and study.best_trials:
    for best_trial in study.best_trials:
      if total_recall_score >= best_trial.values[0]:
        best_list.clear()
        best_list.append({'baseload': baseload, 'predictions': predictions, 'actuals': actuals})

  if os.path.exists(f"Tunings/{combined_name}"):
    shutil.rmtree(f"Tunings/{combined_name}")
  return total_recall_score, total_mae_score

parser = ArgumentParser()
parser.add_argument("--criterion", type=str, default="MAELoss")
parser.add_argument("--models", type=str, default="['LSTM', 'GRU', 'AdaBoostRegressor' , 'RandomForestRegressor', 'GradientBoostingRegressor']") #'RandomForestRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'PatchMixer', 
parser.add_argument("--dataset", type=str, default="Colorado")
parser.add_argument("--input_size", type=int, default=22)
parser.add_argument("--pred_len", type=int, default=24)
parser.add_argument("--stride", type=int, default=24)
parser.add_argument("--seq_len", type=int, default=24*7)
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--scaler", type=str, default="MinMaxScaler")
parser.add_argument("--load", type=str, default='False')
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--threshold", type=float, default=500)
parser.add_argument("--downscaling", type=int, default=13)
parser.add_argument("--multiplier", type=int, default=2)

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
  "xPatch": lambda: xPatch(xpatch_params),
  "PatchMixer": lambda: PatchMixer(patchmixer_params),
  "RandomForestRegressor": lambda: MultiOutputRegressor(RandomForestRegressor(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], min_samples_split=rf_params['min_samples_split'], min_samples_leaf=rf_params['min_samples_leaf'], max_features=rf_params['max_features'], random_state=SEED), n_jobs=-1),
  "GradientBoostingRegressor": lambda: MultiOutputRegressor(GradientBoostingRegressor(n_estimators=gb_params['n_estimators'], max_depth=gb_params['max_depth'], min_samples_split=gb_params['min_samples_split'], min_samples_leaf=gb_params['min_samples_leaf'], learning_rate=gb_params['learning_rate_model'], random_state=SEED), n_jobs=-1),
  "AdaBoostRegressor": lambda: MultiOutputRegressor(AdaBoostRegressor(n_estimators=ada_params['n_estimators'], learning_rate=ada_params['learning_rate'], random_state=SEED), n_jobs=-1),
  # "DPAD": lambda: DPAD_GCN(input_len=args.seq_len, output_len=args.pred_len, input_dim=args.input_size, enc_hidden=dpad_params['enc_hidden'], dec_hidden=dpad_params['dec_hidden'], dropout=dpad_params['dropout'], num_levels=dpad_params['num_levels'], K_IMP=dpad_params['K_IMP'], RIN=dpad_params['RIN'])
}

def safe_objective(args, trial, all_subsets, study):
  try:
    return objective(args, trial, all_subsets, study)
  except Exception as e:
    print(f"Failed trial: {e}. Skipped this trial.")
    return float('inf')
  finally:
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
  hparams = pd.read_csv(f'./Tunings/{args.dataset}_{args.pred_len}h_classification_tuning.csv')
  lstm_params = ast.literal_eval(hparams[hparams['model'] == 'LSTM']['parameters'].values[0])
  gru_params = ast.literal_eval(hparams[hparams['model'] == 'GRU']['parameters'].values[0])
  xpatch_params = Configs({**ast.literal_eval(hparams[hparams['model'] == 'xPatch']['parameters'].values[0]), "enc_in": args.input_size, "pred_len": args.pred_len, 'seq_len': args.seq_len })
  patchmixer_params = Configs({**ast.literal_eval(hparams[hparams['model'] == 'PatchMixer']['parameters'].values[0]), "enc_in": args.input_size, "pred_len": args.pred_len, "seq_len": args.seq_len })
  # dpad_params = ast.literal_eval(hparams[hparams['model'] == 'DPAD']['parameters'].values[0])
  rf_params = ast.literal_eval(hparams[hparams['model'] == 'RandomForestRegressor']['parameters'].values[0])
  ada_params = ast.literal_eval(hparams[hparams['model'] == 'AdaBoostRegressor']['parameters'].values[0])
  gb_params = ast.literal_eval(hparams[hparams['model'] == 'GradientBoostingRegressor']['parameters'].values[0])

  selected_models = ast.literal_eval(args.models)
  combined_name = "-".join([m for m in selected_models])

  all_subsets = []
  for r in range(1, len(selected_models) + 1):
    all_subsets.extend(combinations(selected_models, r))

  all_subsets = [list(subset) for subset in all_subsets if len(subset) >= 3]

  if args.load == "True":
    try:
      study = joblib.load(f'Tunings/{args.dataset}_{args.pred_len}h_{args.models}_architecture_tuning_classification.pkl')
      print("Loaded an old study:")
    except Exception as e:
      print("No previous tuning found. Starting a new tuning.", e)
      study = optuna.create_study(directions=["maximize", "minimize"], study_name=f"Bagging-{combined_name}")
  else:
    print("Starting a new tuning.")
    study = optuna.create_study(directions=["maximize", "minimize"], study_name=f"Bagging-{combined_name}")

  tuning_results = []
  best_list = []
  dates = []

  study.optimize(lambda trial: objective(args, trial, all_subsets, study), n_trials=args.trials, gc_after_trial=True, timeout=37800)

  joblib.dump(study, f'Tunings/{args.dataset}_{args.pred_len}h_{args.models}_architecture_tuning_classification.pkl')

  unique_results = []
  for d in tuning_results:
    if d not in unique_results and d['mae'] <= 100:
      unique_results.append(d)
  sorted_trials = sorted(unique_results, key=lambda x: x.get('rec', float('inf')), reverse=True)
  top_10_tunings = sorted_trials[:10]
  df_top_10 = pd.DataFrame(top_10_tunings)
  df_top_10.to_csv(f'Tunings/{args.dataset}_{args.pred_len}h_architecture_tuning_classification.csv', index=False)
  
  baseload, predictions, actuals = best_list[0]['baseload'], best_list[0]['predictions'], best_list[0]['actuals']

  #baseload plot
  plt.figure(figsize=(15, 4))
  plt.plot(dates, baseload, label='Baseload')
  plt.axhline(y=args.threshold, color='red', linestyle='--', label='Threshold')
  plt.xlabel('Dates')
  plt.ylabel('Electricity Consumption (kWh)')
  plt.legend()
  plt.tight_layout()
  plt.savefig(f'Tunings/{args.dataset}_{args.pred_len}h_{args.models}_classification_baseload_plot.png')
  plt.show()
  plt.clf()

  # pred and act plot
  plt.figure(figsize=(15, 4))
  plt.plot(dates, actuals, label='Actuals')
  plt.plot(dates, predictions, label=f'predictions')
  plt.axhline(y=args.threshold, color='red', linestyle='--', label='Threshold')
  plt.xlabel('Dates')
  plt.ylabel('Electricity Consumption (kWh)')
  plt.legend()
  plt.tight_layout()
  plt.savefig(f'Tunings/{args.dataset}_{args.pred_len}h_{args.models}_classification_predact_plot.png')
  plt.show()
  plt.clf()
