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
  def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
    self.X = torch.tensor(X).float()
    self.y = torch.tensor(y).float()
    self.seq_len = seq_len

  def __len__(self):
    return self.X.__len__() - (self.seq_len-1)

  def __getitem__(self, index):
    return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])

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
    data = pd.read_csv(self.data_dir)
    data = convert_to_hourly(data)
    data = add_features(data)

    start_date = pd.to_datetime('2021-11-30')
    end_date = pd.to_datetime('2023-11-30')

    data = pd.get_dummies(data, columns=['Season'])

    df = filter_data(start_date, end_date, data)
    df = df.dropna()
    X = df.copy()
    y = X['Energy_Consumption'].shift(-1).ffill()
    X_tv, self.X_test, y_tv, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_tv, y_tv, test_size=0.25, shuffle=False)

    preprocessing = self.scaler
    preprocessing.fit(self.X_train)  # should only fit to training data

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
    window_size = round(len(self.X_train)*0.80)
    # bootstrap_sampler = BootstrapSampler(train_dataset, window_size=window_size)
    # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=bootstrap_sampler, shuffle=False, num_workers=self.num_workers, persistent_workers=self.is_persistent)
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

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.mean((predictions - targets) ** 2) #MSE
        weight = torch.where(predictions < targets, 2.0, 1.0)
        weighted_loss = torch.mean(loss * weight)
        return weighted_loss

class AsymmetricMAEandMSELoss(nn.Module):
    def __init__(self):
        super(AsymmetricMAEandMSELoss, self).__init__()

    def forward(self, predictions, targets):
        mae_loss = torch.mean(torch.abs(predictions - targets))
        mse_loss = torch.mean((predictions - targets) ** 2)
        loss = torch.where(predictions > targets, mae_loss, mse_loss)

        mean_loss = torch.mean(loss)
        return mean_loss

class WeightedAsymmetricMAEandMSELoss(nn.Module): 
    def __init__(self):
        super(WeightedAsymmetricMAEandMSELoss, self).__init__()

    def forward(self, predictions, targets):
        mae_loss = torch.mean(torch.abs(predictions - targets))
        mse_loss = torch.mean((predictions - targets) ** 2)
        loss = torch.where(predictions > targets, mae_loss, mse_loss*2)

        mean_loss = torch.mean(loss)
        return mean_loss

class CustomLogCoshLoss(nn.Module):
    def __init__(self):
        super(CustomLogCoshLoss, self).__init__()

    def forward(self, predictions, targets):
        error = predictions - targets
        lc_loss = torch.log(torch.cosh(error))
        mse_loss = torch.mean(error ** 2)
        loss = torch.where(predictions > targets, lc_loss, mse_loss*2)
        return torch.mean(loss)

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

# class MLP(torch.nn.Module):
#   def __init__(self, num_features, seq_len, num_classes):
#     super().__init__()
#     self.name = "MLP"

#     self.all_layers = torch.nn.Sequential(
#       torch.nn.Linear(num_features, seq_len),
#       torch.nn.ReLU(),
#       torch.nn.Linear(seq_len, 25),
#       torch.nn.ReLU(),
#       torch.nn.Linear(25, num_classes),
#     )

#   def forward(self, x):
#     x = torch.flatten(x, start_dim=1)
#     logits = self.all_layers(x)
#     return logits

# class LSTM(torch.nn.Module):
#   def __init__(self, input_size, hidden_size, num_layers, dropout):
#     super().__init__()
#     self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
#     self.fc = nn.Linear(hidden_size, 1)
#     self.name = "LSTM"

#   def forward(self, x):
#     out, _ = self.lstm(x)
#     out = self.fc(out[:, -1, :])  # Get the last time step
#     return out

# class GRU(torch.nn.Module):
#   def __init__(self, input_size, hidden_size, num_layers, dropout):
#     super().__init__()
#     self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
#     self.fc = nn.Linear(hidden_size, 1)
#     self.name = "GRU"

#   def forward(self, x):
#     out, _ = self.gru(x)
#     out = self.fc(out[:, -1, :])  # Get the last time step
#     return out

def objective(args, trial):
    params = {
        'seq_len': trial.suggest_int('seq_len', 1, 24),
        'batch_size': trial.suggest_int('batch_size', 1, 64),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'max_epochs': trial.suggest_int('max_epochs', 50, 1000),
        'criterion': torch.nn.L1Loss(),
        'optimizer': torch.optim.Adam,
        'scaler': MinMaxScaler(),
        'num_workers': trial.suggest_int('num_workers', 0, 20),
        'input_size': 22,
        'forecasting_horizon': 1,
    }

    colmod = ColoradoDataModule(
    data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv',
    scaler=params['scaler'],
    seq_len=params['seq_len'],
    batch_size=params['batch_size'],
    num_workers=params['num_workers'],
    is_persistent=True if params['num_workers'] > 0 else False
    )

    colmod.prepare_data()
    colmod.setup(stage="fit")

    model = None

    if args.model == "LSTM":
      _params = {
        'hidden_size': trial.suggest_int('hidden_size', 50, 200),
        'num_layers': trial.suggest_int('num_layers', 1, 10),
        'dropout': trial.suggest_float('dropout', 0.0, 1),
      }
      model = LSTM(input_size=params['input_size'], hidden_size=_params['hidden_size'], num_layers=_params['num_layers'], dropout=_params['dropout'])
    elif args.model == "GRU":
      _params = {
        'hidden_size': trial.suggest_int('hidden_size', 50, 200),
        'num_layers': trial.suggest_int('num_layers', 1, 10),
        'dropout': trial.suggest_float('dropout', 0.0, 1),
      }
      model = GRU(input_size=params['input_size'], hidden_size=_params['hidden_size'], num_layers=_params['num_layers'], dropout=_params['dropout'])
    elif args.model == "MLP":
      model = MLP(num_features=params['input_size'], seq_len=params['seq_len'], num_classes=1)
    elif args.model == "AdaBoost":
      _params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate_model': trial.suggest_float('learning_rate_model', 0.01, 1.0),
      }
      model = AdaBoostRegressor(n_estimators=_params['n_estimators'], learning_rate=_params['learning_rate_model'], random_state=42)
    elif args.model == "RandomForest":
      _params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
      }
      model = RandomForestRegressor(n_estimators=_params['n_estimators'], max_depth=_params['max_depth'], min_samples_split=_params['min_samples_split'], min_samples_leaf=_params['min_samples_leaf'], max_features=_params['max_features'], random_state=42)
    elif args.model == "GradientBoosting":
      _params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'learning_rate_model': trial.suggest_float('learning_rate_model', 0.01, 1.0),
      }
      model = GradientBoostingRegressor(n_estimators=_params['n_estimators'], max_depth=_params['max_depth'], min_samples_split=_params['min_samples_split'], min_samples_leaf=_params['min_samples_leaf'], learning_rate=_params['learning_rate_model'], random_state=42)
    elif args.model == "DPAD":
       _params = {
        'enc_hidden': trial.suggest_int('enc_hidden', 1, 400),
        'dec_hidden': trial.suggest_int('dec_hidden', 1, 400),
        'num_levels': trial.suggest_int('num_levels', 1, 10),
        'dropout': trial.suggest_float('dropout', 0.0, 1),
        'K_IMP': trial.suggest_int('K_IMP', 1, 10),
        'RIN': trial.suggest_int('RIN', 0, 1)
       }
       DPAD_GCN(input_len=params['seq_len'], output_len=1, input_dim=params['input_size'], enc_hidden=168, dec_hidden=168, dropout=0.5, num_levels=2, K_IMP=6, RIN=1)
    elif args.model == "xPatch":
      class Configs:
        def __init__(self, config_dict):
          for key, value in config_dict.items():
            setattr(self, key, value)

      params_xpatch = Configs(
        dict(
        seq_len = params['seq_len'],
        pred_len = params['forecasting_horizon'],
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

    if isinstance(model, torch.nn.Module):
      print(f"-----Tuning {model.name} model-----")
      tuned_model = LightningModel(
          model=model,
          criterion=params['criterion'],
          optimizer=params['optimizer'],
          learning_rate=params['learning_rate']
      )
      trainer = L.Trainer(
          max_epochs=params['max_epochs'],
          callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
          log_every_n_steps=0,
          enable_checkpointing=False
      )
      trainer.fit(tuned_model, colmod)
      val_loss = trainer.callback_metrics["val_loss"].item()

    elif isinstance(model, BaseEstimator):
      print(f"-----Tuning {model.__class__.__name__} model-----")
      X_train = colmod.X_train
      y_train = colmod.y_train.ravel()
      model.fit(X_train, y_train)
      val_loss = mean_absolute_error(y_train, model.predict(X_train))
    return val_loss

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


