import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import holidays

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils import resample

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import BasePredictionWriter

# commands
# python baselines.py --seq_len 12 --batch_size 8 --criterion MSELoss --max_epochs 1000 --n_features 7 --hidden_size 100 --num_layers 1 --dropout 1 --learning_rate 0.001 --num_workers 6 --scaler MinMaxScaler
# tensorboard --logdir=Models/lightning_logs/      

def convert_to_quarterly(data):

    # Remove unnecessary columns
    data = data.drop(columns=['Zip_Postal_Code'])

    # Convert date/time columns to datetime
    data['Start_DateTime'] = pd.to_datetime(data['Start_DateTime'])
    data['Charging_EndTime'] = pd.to_datetime(data['End_DateTime'])
    data['Charging_Time'] = pd.to_timedelta(data['Charging_Time'])

    ####################### CONVERT DATASET TO QUARTERLY  #######################

    # Split the session into quarterly intervals
    quarterly_rows = []

    # Iterate over each row in the dataframe to break charging sessions into quarterly intervals
    for _, row in data.iterrows():
        start, end = row['Start_DateTime'], row['Charging_EndTime']
        energy = row['Energy_Consumption']

        # Generate quarterly intervals
        quarterly_intervals = pd.date_range(start=start.floor(
            '15min'), end=end.ceil('15min'), freq='15min')
        total_duration = (end - start).total_seconds()

        for i in range(len(quarterly_intervals) - 1):
            interval_start = max(start, quarterly_intervals[i])
            interval_end = min(end, quarterly_intervals[i+1])
            interval_duration = (interval_end - interval_start).total_seconds()

            # Calculate the energy consumption for the interval if interval is greater than 0 (Start and end time are different)
            if interval_duration > 0:
                energy_fraction = (interval_duration / total_duration) * energy

            quarterly_rows.append({
                'Time': quarterly_intervals[i],
                'Energy_Consumption': energy_fraction,
                "Session_Count": 1  # Count of sessions in the interval
            })

    quarterly_df = pd.DataFrame(quarterly_rows)

    quarterly_df = quarterly_df.groupby('Time').agg({
        'Energy_Consumption': 'sum',
        'Session_Count': 'sum'
    }).reset_index()

    # Convert the Time column to datetime
    quarterly_df['Time'] = pd.to_datetime(
        quarterly_df['Time'], format="%d-%m-%Y %H:%M:%S")
    quarterly_df = quarterly_df.set_index('Time')

    # Define time range for all 24 hours
    start_time = quarterly_df.index.min().normalize()  # 00:00:00
    end_time = quarterly_df.index.max().normalize() + pd.Timedelta(days=1) - \
        pd.Timedelta(hours=1)  # 23:00:00

    # Change range to time_range_full, so from 00:00:00 to 23:00:00
    time_range_full = pd.date_range(
        start=start_time, end=end_time, freq='15min')

    quarterly_df = quarterly_df.reindex(time_range_full, fill_value=0)

    return quarterly_df

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
  df['Energy_Consumption_rolling'] = df['Energy_Consumption'].rolling(
      window=24).mean()

  return df

def filter_data(start_date, end_date, data):
    return data[(data.index >= start_date) & (data.index <= end_date)].copy()

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
    start_date = pd.to_datetime('2021-05-30')
    end_date = pd.to_datetime('2023-05-30')

    # Load and preprocess the data
    data = pd.read_csv(self.data_dir)
    data = convert_to_quarterly(data)
    data = add_features(data)
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
    # window_size = round(len(self.X_train)*0.97)
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

class MLP(torch.nn.Module):
  def __init__(self, num_features, seq_len, num_classes):
    super().__init__()
    self.name = "MLP"

    self.all_layers = torch.nn.Sequential(
      torch.nn.Linear(num_features, seq_len),
      torch.nn.ReLU(),
      torch.nn.Linear(seq_len, 25),
      torch.nn.ReLU(),
      torch.nn.Linear(25, num_classes),
    )

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)
    logits = self.all_layers(x)
    return logits

class LSTM(torch.nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout):
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    self.fc = nn.Linear(hidden_size, 1)
    self.name = "LSTM"

  def forward(self, x):
    out, _ = self.lstm(x)
    out = self.fc(out[:, -1, :])  # Get the last time step
    return out

class GRU(torch.nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout):
    super().__init__()
    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    self.fc = nn.Linear(hidden_size, 1)
    self.name = "GRU"

  def forward(self, x):
    out, _ = self.gru(x)
    out = self.fc(out[:, -1, :])  # Get the last time step
    return out

def run(ensemble_models, colmod, combined_name):
    for _model in ensemble_models:
        if isinstance(_model, torch.nn.Module):
            print(f"-----Training {_model.name} model-----")
            model = LightningModel(model=_model, criterion=criterion_map.get(args.criterion)(), optimizer=optimizer_map.get(args.optimizer), learning_rate=args.learning_rate)
            pred_writer = CustomWriter(output_dir="Predictions", write_interval="epoch", combined_name=combined_name, model_name=_model.name)
            trainer = L.Trainer(max_epochs=args.max_epochs, callbacks=[EarlyStopping(monitor="val_loss", mode="min"), pred_writer], log_every_n_steps=args.batch_size)
            trainer.fit(model, colmod)
            trainer.test(model, colmod)
            trainer.predict(model, colmod, return_predictions=False)

        if isinstance(_model, BaseEstimator):
            print(f"-----Training {_model.__class__.__name__} model-----")
            # X_train_sample, y_train_sample = resample(colmod.X_train, colmod.y_train, replace=True, n_samples=len(colmod.X_train), random_state=42)
            # _model.fit(X_train_sample, y_train_sample.ravel()) # ravel() converts a 2D to a 1D array
            _model.fit(colmod.X_train, colmod.y_train.ravel()) # ravel() converts a 2D to a 1D array
            y_pred = _model.predict(colmod.X_test)
            if not os.path.exists(f"Predictions/{combined_name}"):
              os.makedirs(f"Predictions/{combined_name}")
            torch.save(y_pred, f"Predictions/{combined_name}/predictions_{_model.__class__.__name__}.pt")

def bagging(combined_name):
    all_predictions = []
    lengths = []
    folder_path = f'Predictions/{combined_name}'
    pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    for i, pt_file in enumerate(pt_files):
      file_path = os.path.join(folder_path, pt_file)
      predictions = torch.load(file_path)
      if type(predictions[0]) == torch.Tensor:
        predictions = [item.item() for sublist in predictions for item in sublist]
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

def get_metrics_and_ensemble_plot(combined_name, colmod):
    actuals = []
    for batch in colmod.predict_dataloader():
      x, y = batch
      actuals.extend(y.numpy())

    actuals_flat = [item for sublist in actuals for item in sublist]

    folder_path = f'Predictions/{combined_name}'
    pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

    metrics = []
    plt.figure(figsize=(10, 5))
    plt.plot(actuals_flat, label='Actuals')

    for pt_file in pt_files:
      file_path = os.path.join(folder_path, pt_file)
      predictions = torch.load(file_path)
      model_name = pt_file.split('.')[0].split('_')[-1]

      if len(predictions) > len(actuals_flat):
        predictions = predictions[-len(actuals_flat):]
      if type(predictions[0]) == torch.Tensor: 
        predictions = [item.item() for sublist in predictions for item in sublist]
      elif type(predictions[0]) == np.float64:
        predictions = predictions.tolist()

      metrics.append({
        'model': model_name,
        'mse': mean_squared_error(predictions, actuals_flat),
        'mae': mean_absolute_error(predictions, actuals_flat),
        'mape': mean_absolute_percentage_error(predictions, actuals_flat)})
      
      _combined_name = combined_name
      if '_' in combined_name:
          _combined_name = combined_name.split('_')[0]

      if model_name == _combined_name:
        plt.plot(predictions, label=model_name)

    loss_func_df = pd.concat([pd.DataFrame([m]) for m in metrics])
    loss_func_df.set_index('model', inplace=True)
    # print(loss_func_df)
    loss_func_df.to_csv('loss_func_metrics.csv')

    plt.xlabel('Samples')
    plt.ylabel('Energy Consumption')
    plt.title(f'Predictions vs Actuals ({combined_name})')
    plt.legend()

    plt.savefig(f'{folder_path}/predictions_vs_actuals_{combined_name}.png')
    # plt.show()

def get_delta_and_below_actuals(combined_name, colmod):
    actuals = []
    for batch in colmod.predict_dataloader():
      x, y = batch
      actuals.extend(y.numpy())

    actuals_flat = [item for sublist in actuals for item in sublist]

    folder_path = f'Predictions/{combined_name}'
    pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

    predictions = []

    actuals_flat = actuals_flat[-500:]
    belows = {}

    plt.figure(figsize=(8, 5))
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Data points')
    plt.ylabel('Pred-Act')
    for i, pt_file in enumerate(pt_files):
        file_path = os.path.join(folder_path, pt_file)
        predictions = torch.load(file_path)
        model_name = pt_file.split('.')[0].split('_')[-1]

        if len(predictions) > len(actuals_flat):
            predictions = predictions[-len(actuals_flat):]
        if type(predictions[0]) == torch.Tensor: 
            predictions = [item.item() for sublist in predictions for item in sublist]
        elif type(predictions[0]) == np.float64:
            predictions = predictions.tolist()
        
        predictions = predictions[-500:]

        diff = np.array(predictions) - np.array(actuals_flat)
        plt.plot(diff, label=model_name)

        total_below_actuals = np.sum(np.array(predictions) < np.array(actuals_flat))
        percentage_total_below_actuals = total_below_actuals / len(predictions) * 100
        belows.update({model_name: percentage_total_below_actuals})

    # save belows to csv
    belows_df = pd.DataFrame(list(belows.items()), columns=['Model', 'Percentage Below Actuals'])
    belows_df.to_csv(f'{folder_path}/belows.csv', index=False)
    plt.legend()
    plt.savefig(f'{folder_path}/delta_{combined_name}.png')
    # plt.show()

parser = ArgumentParser()
parser.add_argument("--input_size", type=int, default=22)
parser.add_argument("--seq_len", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--criterion", type=str, default="MSELoss")
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--max_epochs", type=int, default=1000)
parser.add_argument("--n_features", type=int, default=7)
parser.add_argument("--hidden_size", type=int, default=100)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--dropout", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_workers", type=int, default=5)
parser.add_argument("--scaler", type=str, default="MinMaxScaler")

criterion_map = { 
                  "MSELoss": nn.MSELoss, 
                  "WeightedMSELoss": WeightedMSELoss,
                  "AsymmetricMAEandMSELoss": AsymmetricMAEandMSELoss,
                  "WeightedAsymmetricMAEandMSELoss": WeightedAsymmetricMAEandMSELoss,
                  "CustomLogCoshLoss": CustomLogCoshLoss 
                }

optimizer_map = {
                  "Adam": torch.optim.Adam,
                }

scaler_map = { "MinMaxScaler": MinMaxScaler }

args = parser.parse_args()

# Consider speed up trainer by reduce precision (e.g. Trainer(precision="16-mixed")) https://lightning.ai/docs/pytorch/stable/common/precision_basic.html

if __name__ == '__main__':
    colmod = ColoradoDataModule(data_dir='Colorado/Preprocessing/TestDataset/CleanedColoradoData.csv', scaler=scaler_map.get(args.scaler)(), seq_len=args.seq_len, batch_size=args.batch_size, num_workers=args.num_workers, is_persistent=True if args.num_workers > 0 else False)
    colmod.prepare_data()
    colmod.setup(stage=None)

    ensemble_models = [
      MLP(num_features=args.seq_len*args.input_size, seq_len=args.batch_size, num_classes=1),
      GRU(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout),
      LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout),
      AdaBoostRegressor(n_estimators=100, random_state=42),
      RandomForestRegressor(n_estimators=100, random_state=42),
    ]

    model_names = [m.name if isinstance(m, torch.nn.Module) else m.__class__.__name__ for m in ensemble_models]
    combined_name = "-".join(model_names)
    combined_name = f"{combined_name}_{args.criterion}"

    run(ensemble_models, colmod, combined_name)
    bagging(combined_name)
    get_metrics_and_ensemble_plot(combined_name, colmod)
    get_delta_and_below_actuals(combined_name, colmod)
