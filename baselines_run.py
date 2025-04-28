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
from lossfunctions.AsymmetricMAEandMSELoss import AsymmetricMAEandMSELoss
from lossfunctions.WeightedAsymmetricMAEandMSELoss import WeightedAsymmetricMAEandMSELoss
from lossfunctions.CustomLogCoshLoss import CustomLogCoshLoss
from lossfunctions.WeightedMSELoss import WeightedMSELoss
import ast

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

seed_everything(42, workers=True)
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
    elif set_name == "train_val":
      X = self.X_train_val 
      y = self.y_train_val
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

def create_and_save_ensemble(combined_name):
  all_predictions = []
  lengths = []
  folder_path = f'Predictions/{combined_name}'
  pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
  for i, pt_file in enumerate(pt_files):
    file_path = os.path.join(folder_path, pt_file)
    predictions = torch.load(file_path)
    if type(predictions[0]) == torch.Tensor:
      predictions = torch.cat(predictions).tolist()
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

def plot_and_save_with_metrics(combined_name, colmod):
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
  for pt_file in pt_files:
    file_path = os.path.join(folder_path, pt_file)
    predictions = torch.load(file_path)
    model_name = pt_file.split('_')[1].split('.')[0]
    # model_name = pt_file.split('.')[0].split('_')[-1] #use this with loss function names

    if type(predictions[0]) == torch.Tensor: 
      predictions = [value.item() for tensor in predictions for value in tensor.flatten()]
    elif type(predictions[0]) == np.float64:
      predictions = predictions.tolist()

    predictions = predictions[-len(actuals_flat):] # reduce length of predictions to match actuals

    if len(predictions) == len(actuals_flat):
      metrics.append({
        'model': model_name,
        'mse': mean_squared_error(predictions, actuals_flat),
        'mae': mean_absolute_error(predictions, actuals_flat),
        'mape': mean_absolute_percentage_error(predictions, actuals_flat)})
      plt.plot(predictions, label=model_name)

  if metrics:
    loss_func_df = pd.concat([pd.DataFrame([m]) for m in metrics], ignore_index=True)
  else:
    loss_func_df = pd.DataFrame(columns=['model', 'mse', 'mae', 'mape'])
  loss_func_df.set_index('model', inplace=True)
  loss_func_df.to_csv(f'{folder_path}/loss_func_metrics.csv')

  plt.xlabel('Samples')
  plt.ylabel('Energy Consumption')
  plt.title(f'Predictions vs Actuals ({combined_name})')
  plt.legend()

  plt.savefig(f'{folder_path}/predictions_vs_actuals_{combined_name}.png')
  plt.show()

parser = ArgumentParser()
parser.add_argument("--input_size", type=int, default=21)
parser.add_argument("--pred_len", type=int, default=24)
parser.add_argument("--stride", type=int, default=24)
parser.add_argument("--seq_len", type=int, default=24*7)
parser.add_argument("--criterion", type=str, default="MAELoss")
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--scaler", type=str, default="MinMaxScaler")

criterion_map = { 
                  "MSELoss": nn.MSELoss, 
                  "MAELoss": nn.L1Loss,
                  "WeightedMSELoss": WeightedMSELoss,
                  "AsymmetricMAEandMSELoss": AsymmetricMAEandMSELoss,
                  "WeightedAsymmetricMAEandMSELoss": WeightedAsymmetricMAEandMSELoss,
                  "CustomLogCoshLoss": CustomLogCoshLoss 
                }

optimizer_map = { "Adam": torch.optim.Adam }

scaler_map = { "MinMaxScaler": MinMaxScaler }

args = parser.parse_args()

if __name__ == "__main__":
  hparams = pd.read_csv('tuning.csv')
  mlp_params = ast.literal_eval(hparams[hparams['model'] == 'MLP']['parameters'].values[0])
  gru_params = ast.literal_eval(hparams[hparams['model'] == 'GRU']['parameters'].values[0])
  lstm_params = ast.literal_eval(hparams[hparams['model'] == 'LSTM']['parameters'].values[0])
  patchmixer_params = Configs(ast.literal_eval(hparams[hparams['model'] == 'PatchMixer']['parameters'].values[0]))
  xpatch_params = Configs(ast.literal_eval(hparams[hparams['model'] == 'xPatch']['parameters'].values[0]))
  fredformer_params = Configs(ast.literal_eval(hparams[hparams['model'] == 'Fredformer']['parameters'].values[0]))
  dpad_params = ast.literal_eval(hparams[hparams['model'] == 'DPAD']['parameters'].values[0])

  ensemble_models = [
    MLP(num_features=args.seq_len*args.input_size, pred_len=args.pred_len, seq_len=mlp_params['batch_size'], hidden_size=mlp_params['hidden_size']),
    GRU(input_size=args.input_size, pred_len=args.pred_len, hidden_size=gru_params['hidden_size'], num_layers=gru_params['num_layers'], dropout=gru_params['dropout']),
    LSTM(input_size=args.input_size, pred_len=args.pred_len, hidden_size=lstm_params['hidden_size'], num_layers=lstm_params['num_layers'], dropout=lstm_params['dropout']),
    PatchMixer(patchmixer_params),
    xPatch(xpatch_params),
    Fredformer(fredformer_params),
    DPAD_GCN(input_len=args.seq_len, output_len=args.pred_len, input_dim=args.input_size, enc_hidden=dpad_params['enc_hidden'], dec_hidden=dpad_params['dec_hidden'], dropout=dpad_params['dropout'], num_levels=dpad_params['num_levels'], K_IMP=dpad_params['K_IMP'], RIN=dpad_params['RIN']),
  ]

  model_names = [m.name if isinstance(m, torch.nn.Module) else m.__class__.__name__ for m in ensemble_models]
  combined_name = "-".join(model_names)

  for model in ensemble_models:
    model_name = model.name if isinstance(model, torch.nn.Module) else model.__class__.__name__
    _hparams = ast.literal_eval(hparams[hparams['model'] == model_name]['parameters'].values[0])
    colmod = ColoradoDataModule(data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv', scaler=scaler_map.get(args.scaler)(), seq_len=args.seq_len, batch_size=_hparams['batch_size'], pred_len=args.pred_len, stride=args.stride, num_workers=_hparams['num_workers'], is_persistent=True if _hparams['num_workers'] > 0 else False)
    colmod.prepare_data()
    colmod.setup(stage=None)

    if isinstance(model, torch.nn.Module):
      print(f"-----Training {model_name} model-----")
      model = LightningModel(model=model, criterion=criterion_map.get(args.criterion)(), optimizer=optimizer_map.get(args.optimizer), learning_rate=_hparams['learning_rate'])
      pred_writer = CustomWriter(output_dir="Predictions", write_interval="epoch", combined_name=combined_name, model_name=model_name)
      trainer = L.Trainer(max_epochs=_hparams['max_epochs'], callbacks=[EarlyStopping(monitor="val_loss", mode="min"), pred_writer], log_every_n_steps=_hparams['batch_size']//2, precision='16-mixed', enable_checkpointing=False, strategy='ddp_find_unused_parameters_true')
      trainer.fit(model, colmod)
      trainer.predict(model, colmod, return_predictions=False)
    
    if isinstance(model, BaseEstimator):
      print(f"-----Training {model_name} model-----")
      # X_train_sample, y_train_sample = resample(colmod.X_train, colmod.y_train, replace=True, n_samples=len(colmod.X_train), random_state=42)
      X_train_val, y_train_val = colmod.sklearn_setup("train_val") 
      X_test, y_test = colmod.sklearn_setup("test")
      model.fit(X_train_val, y_train_val)
      y_pred = model.predict(X_test)
      if not os.path.exists(f"Predictions/{combined_name}"):
        os.makedirs(f"Predictions/{combined_name}")
      torch.save(y_pred, f"Predictions/{combined_name}/predictions_{model_name}.pt")

  combined_name = "LSTM-PatchMixer-xPatch"
  # create_and_save_ensemble(combined_name)
  colmod = ColoradoDataModule(data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv', scaler=scaler_map.get(args.scaler)(), seq_len=args.seq_len, batch_size=96, pred_len=args.pred_len, stride=args.stride, num_workers=5, is_persistent=True if 5 > 0 else False)
  colmod.prepare_data()
  colmod.setup(stage=None)
  plot_and_save_with_metrics(combined_name, colmod)