import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import holidays
import optuna
from models.LSTM import LSTM
from models.GRU import GRU
from models.MLP import MLP
from models.D_PAD_adpGCN import DPAD_GCN
from models.xPatch import xPatch
from models.Fredformer import Fredformer
from models.PatchMixer import PatchMixer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
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

# tensorboard --logdir=Predictions/MLP-GRU-LSTM

def convert_to_hourly(data):

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

def add_features(hourly_df):
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
    ####################### FILTER DATASET  #######################
    data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
    # print(data.head())

    return data

class TimeSeriesDataset(Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1, pred_len: int = 24, stride: int = 24):
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.stride = stride

    self.X = torch.tensor(X).float()
    self.y = torch.tensor(y).float()

  def __len__(self):
    return (len(self.X) - (self.seq_len + self.pred_len - 1)) // self.stride + 1

  def __getitem__(self, index):
    start_idx = index * self.stride
    x_window = self.X[start_idx: start_idx + self.seq_len]
    y_target = self.y[start_idx + self.seq_len: start_idx + self.seq_len + self.pred_len]
    return x_window, y_target

class BootstrapSampler(Sampler):
  def __init__(self, data_source, window_size, num_samples=None):
    self.data_source = data_source
    self.window_size = window_size
    self.num_samples = num_samples if num_samples is not None else len(data_source) // 2

  def __iter__(self):
    indices = []
    for _ in range(self.num_samples):
      start_idx = np.random.randint(0, len(self.data_source) - self.window_size + 1)
      indices.extend(range(start_idx, start_idx + self.window_size))
    return iter(indices)

  def __len__(self):
    return self.num_samples

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
    data = convert_to_hourly(data)
    data = add_features(data)
    df = filter_data(start_date, end_date, data)

    df = df.dropna()

    #df.to_csv('final_df_hourly.csv', index=True)  

    X = df.copy()

    y = X.pop('Energy_Consumption')

    #y = create_multi_step_targets(X['Energy_Consumption'], horizon=24)
    #X=X.iloc[:-24]
    #y=y[:-24] 

    # 60/20/20 split
    X_tv, self.X_test, y_tv, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_tv, y_tv, test_size=0.25, shuffle=False)

    preprocessing = self.scaler
    preprocessing.fit(self.X_train)  # should only fit to training data
    

    if stage == "fit" or stage is None:
      self.X_train = preprocessing.transform(self.X_train)
      self.y_train = np.array(self.y_train)

      y_train_df = pd.DataFrame(self.y_train, columns=["target"])
      combined = pd.concat([y_train_df, pd.DataFrame(self.X_train),], axis=1)
      combined.to_csv('train.csv', index=True)

      self.X_val = preprocessing.transform(self.X_val)
      self.y_val = np.array(self.y_val)

    if stage == "test" or "predict" or stage is None:
      self.X_test = preprocessing.transform(self.X_test)
      self.y_test = np.array(self.y_test)

  def train_dataloader(self):
    train_dataset = TimeSeriesDataset(self.X_train, self.y_train, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    # window_size = round(len(self.X_train)*0.97)
    # bootstrap_sampler = BootstrapSampler(train_dataset, window_size=window_size)
    # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=bootstrap_sampler, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
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
    test_dataset = TimeSeriesDataset(self.X_test, self.y_test, seq_len=self.seq_len, pred_len=self.pred_len, stride=self.stride)
    test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent, drop_last=False)
    return test_loader
  
  def sklearn_setup(self, set_name: str = "train"): 
    if set_name == "train": 
      X = self.X_train 
      y = self.y_train
    elif set_name == "val":
      X = self.X_val 
      y = self.y_val
    elif set_name == "test":
      X = self.X_test 
      y = self.y_test
    else:
      raise ValueError("Invalid set name. Choose from 'train', 'val', or 'test'.")


    seq_len = self.seq_len
    pred_len = 24

    X_window, y_target = [], []
  
    for i in range(len(X) - seq_len - pred_len + 1):
        X_window.append(X[i:i + seq_len].flatten())
        y_target.append(y[i + seq_len:i + seq_len + pred_len])

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
      
def objective(args, trial):
    params = {
        'input_size': 21,
        'pred_len': 24,
        'seq_len': trial.suggest_int('seq_len', 24*7, 24*7*3, step=24),
        'stride': 24,
        'batch_size': trial.suggest_int('batch_size', 1, 64),
        'criterion': torch.nn.L1Loss(),
        'optimizer': torch.optim.Adam,
        'scaler': MinMaxScaler(),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'max_epochs': trial.suggest_int('max_epochs', 50, 1000, step=50),
        'num_workers': trial.suggest_int('num_workers', 0, 20),
        'seed': 42,
        'is_persistent': False,
    }

    colmod = ColoradoDataModule(data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv', scaler=params['scaler'], seq_len=params['seq_len'], pred_len=params['pred_len'], stride=params['stride'], batch_size=params['batch_size'], num_workers=params['num_workers'], is_persistent=params['is_persistent'])
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
      model = AdaBoostRegressor(n_estimators=_params['n_estimators'], learning_rate=_params['learning_rate_model'], random_state=params['seed'])
    elif args.model == "RandomForest":
      _params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
      }
      model = RandomForestRegressor(n_estimators=_params['n_estimators'], max_depth=_params['max_depth'], min_samples_split=_params['min_samples_split'], min_samples_leaf=_params['min_samples_leaf'], max_features=_params['max_features'], random_state=params['seed'])
    elif args.model == "GradientBoosting":
      _params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'learning_rate_model': trial.suggest_float('learning_rate_model', 0.01, 1.0),
      }
      model = GradientBoostingRegressor(n_estimators=_params['n_estimators'], max_depth=_params['max_depth'], min_samples_split=_params['min_samples_split'], min_samples_leaf=_params['min_samples_leaf'], learning_rate=_params['learning_rate_model'], random_state=params['seed'])
    elif args.model == "DPAD":
       _params = {
        'enc_hidden': trial.suggest_int('enc_hidden', 1, 400),
        'dec_hidden': trial.suggest_int('dec_hidden', 1, 400),
        'num_levels': trial.suggest_int('num_levels', 1, 10),
        'dropout': trial.suggest_float('dropout', 0.0, 1),
        'K_IMP': trial.suggest_int('K_IMP', 1, 10),
        'RIN': trial.suggest_int('RIN', 0, 1)
       }
       DPAD_GCN(input_len=params['seq_len'], output_len=params['pred_len'], input_dim=params['input_size'], enc_hidden=_params['enc_hidden'], dec_hidden=_params['dec_hidden'], dropout=_params['dropout'], num_levels=_params['num_levels'], K_IMP=_params['K_IMP'], RIN=_params['RIN'])
    elif args.model == "xPatch":
      params_xpatch = Configs(
        dict(
        seq_len = params['seq_len'],
        pred_len = params['pred_len'],
        enc_in = params['input_size'],
        patch_len = trial.suggest_int('patch_len', 1, 24),
        stride = trial.suggest_int('stride', 1, 24),
        padding_patch = trial.suggest_categorical('padding_patch', ['end', 'None']),
        revin = trial.suggest_int('revin', 0, 1),
        ma_type = trial.suggest_categorical('ma_type', ['reg', 'ema', 'dema']),
        alpha = trial.suggest_float('alpha', 0.0, 1.0),
        beta = trial.suggest_float('beta', 0.0, 1.0),
        )
      )
      model = xPatch(params_xpatch)
    elif args.model == "Fredformer":
      _params = Configs(
        dict(
          enc_in=params['input_size'],                # Number of input channels
          seq_len=params['seq_len'],               # Context window (lookback length)
          pred_len=params['pred_len'],              # Target window (forecasting length)
          output=0,                 # Output dimension (default 0)

          # Model architecture
          e_layers = trial.suggest_int("e_layers", 1, 4),  # Number of layers
          n_heads = trial.suggest_int("n_heads", 1, 16),  # Number of attention heads
          d_model = trial.suggest_int("d_model", 128, 1024, step=64),  # Dimension of the model
          d_ff = trial.suggest_int("d_ff", 512, 4096, step=128),  # Dimension of feed-forward network
          dropout = trial.suggest_float("dropout", 0.0, 1, step=0.05),  # Dropout rate
          fc_dropout = trial.suggest_float("fc_dropout", 0.0, 1, step=0.05),  # Fully connected dropout
          head_dropout = trial.suggest_float("head_dropout", 0.0, 1, step=0.05),  # Dropout rate for the head layers
          individual = trial.suggest_categorical("individual", [0, 1]),  # Whether to use individual heads

          # Patching
          patch_len = trial.suggest_int("patch_len", 4, 16, step=1),  # Patch size
          stride = trial.suggest_int("stride", 1, 16, step=1),  # Stride for patching
          padding_patch = trial.suggest_categorical("padding_patch", ["end", "start", "none"]),  # Padding type for patches

          # RevIN
          revin = trial.suggest_categorical("revin", [0, 1]),  # Whether to use RevIN
          affine = trial.suggest_categorical("affine", [0, 1]),  # Affine transformation in RevIN
          subtract_last = trial.suggest_categorical("subtract_last", [0, 1]),  # Subtract last value in RevIN

          # Ablation and Nystrom
          use_nys = trial.suggest_categorical("use_nys", [0, 1]),  # Whether to use Nystrom approximation
          ablation = trial.suggest_categorical("ablation", [0, 1]),  # Ablation study configuration

          # Crossformer-specific parameters
          cf_dim = trial.suggest_int("cf_dim", 16, 128, step=8),  # Crossformer dimension
          cf_depth = trial.suggest_int("cf_depth", 1, 4),  # Crossformer depth
          cf_heads = trial.suggest_int("cf_heads", 1, 8),  # Crossformer number of heads
          cf_mlp = trial.suggest_int("cf_mlp", 64, 256, step=16),  # Crossformer MLP dimension
          cf_head_dim = trial.suggest_int("cf_head_dim", 16, 64, step=8),  # Crossformer head dimension
          cf_drop = trial.suggest_float("cf_drop", 0.0, 0.5, step=0.05),  # Crossformer dropout rate

          # MLP-specific parameters
          mlp_hidden = trial.suggest_int("mlp_hidden", 32, 128, step=16),  # Hidden layer size for MLP
          mlp_drop = trial.suggest_float("mlp_drop", 0.0, 0.5, step=0.05),  # Dropout rate for MLP
        )
      )
      model = Fredformer(_params)
    elif args.model == "PatchMixer":
      # _params = Configs({
      #   "enc_in": params['input_size'],                # Number of input channels
      #   "seq_len": params['seq_len'],               # Context window (lookback length)
      #   "pred_len": params['pred_len'],
      #   "batch_size": params['batch_size'],
      #   "patch_len": trial.suggest_int("patch_len", 4, 32, step=4),  # Patch size
      #   "stride": trial.suggest_int("stride", 1, 16, step=1),  # Stride for patching
      #   "mixer_kernel_size": trial.suggest_int("mixer_kernel_size", 2, 16, step=2),  # Kernel size for the PatchMixer layer
      #   "d_model": trial.suggest_int("d_model", 128, 1024, step=64),  # Dimension of the model
      #   "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.05),  # Dropout rate for the model
      #   "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.5, step=0.05),  # Dropout rate for the head layers
      #   "e_layers": trial.suggest_int("e_layers", 1, 4),  # Number of PatchMixer layers (depth)
      # })
      _params = Configs(
        dict(
          enc_in = 21,                # Number of input channels (nvals)
          seq_len = 12,               # Lookback window length
          pred_len = 24,              # Forecasting length
          batch_size = 24,             # Batch size
          patch_len = 16,             # Patch size
          stride = 8,                 # Stride for patching
          mixer_kernel_size = 8,      # Kernel size for the PatchMixer layer
          d_model = 512,              # Dimension of the model
          dropout = 0.05,              # Dropout rate for the model
          head_dropout = 0.0,         # Dropout rate for the head layers
          e_layers = 2,               # Number of PatchMixer layers (depth)
        )
      )
      model = PatchMixer(_params)
    else:
      raise ValueError("Model not found")
      
    if isinstance(model, torch.nn.Module):
      print(f"-----Tuning {model.name} model-----")
      tuned_model = LightningModel(model=model, criterion=params['criterion'], optimizer=params['optimizer'], learning_rate=params['learning_rate'])
      trainer = L.Trainer(max_epochs=params['max_epochs'], log_every_n_steps=0, enable_checkpointing=False)
      trainer.fit(tuned_model, colmod)
      train_loss = trainer.callback_metrics["train_loss"].item()

    elif isinstance(model, BaseEstimator):
      name = model.__class__.__name__
      print(f"-----Training {type(model.estimator).__name__ if name == "MultiOutputRegressor" else name} model-----")
      X_train = colmod.X_train
      y_train = colmod.y_train.ravel()
      model.fit(X_train, y_train)
      train_loss = mean_absolute_error(y_train, model.predict(X_train))
    return train_loss

def tune_model_with_optuna(args, n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(args, trial), n_trials=n_trials)

    print("Best params:", study.best_params)
    print("Best validation loss:", study.best_value)
    return study.best_params

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="LSTM")
    args = parser.parse_args()

    best_params = tune_model_with_optuna(args, n_trials=100)
