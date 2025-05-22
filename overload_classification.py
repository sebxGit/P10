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

class Configs:
  def __init__(self, config_dict):
    for key, value in config_dict.items():
      setattr(self, key, value)


parser = ArgumentParser()
parser.add_argument("--models", type=str, default="['xPatch', 'LSTM', 'GRU', 'PatchMixer']") 
parser.add_argument("--pred_len", type=int, default=24)
parser.add_argument("--dataset", type=str, default="Colorado")

args = parser.parse_args()

if __name__ == "__main__":
  hparams = pd.read_csv(f'./Tunings/{args.dataset}_{args.pred_len}h_tuning.csv')
  lstm_params = ast.literal_eval(hparams[hparams['model'] == 'LSTM']['parameters'].values[0])
  gru_params = ast.literal_eval(hparams[hparams['model'] == 'GRU']['parameters'].values[0])
  xpatch_params = Configs({**ast.literal_eval(hparams[hparams['model'] == 'xPatch']['parameters'].values[0]), "enc_in": args.input_size, "pred_len": args.pred_len, 'seq_len': args.seq_len })
  patchmixer_params = Configs({**ast.literal_eval(hparams[hparams['model'] == 'PatchMixer']['parameters'].values[0]), "enc_in": args.input_size, "pred_len": args.pred_len, "seq_len": args.seq_len })
  # dpad_params = ast.literal_eval(hparams[hparams['model'] == 'DPAD']['parameters'].values[0])
  rf_params = ast.literal_eval(hparams[hparams['model'] == 'RandomForest']['parameters'].values[0])
  ada_params = ast.literal_eval(hparams[hparams['model'] == 'AdaBoost']['parameters'].values[0])
  gb_params = ast.literal_eval(hparams[hparams['model'] == 'GradientBoosting']['parameters'].values[0])

  selected_models = ast.literal_eval(args.models)
  combined_name = "-".join([m for m in selected_models])
  print(combined_name)