import os
import math
import tempfile
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# TSFM libraries
from tsfm_public.toolkit.time_series_forecasting_pipeline import TimeSeriesForecastingPipeline
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.callbacks import TrackingCallback

from tsfm_public import (
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
    get_datasets,
)

bitcoin_data = pd.read_csv('sample_btcusd.csv')
# print(len(bitcoin_data))
# print(bitcoin_data.isna().sum())

bitcoin_data['Timestamp'] = pd.to_datetime(bitcoin_data['Timestamp'], unit='s')
# print(bitcoin_data.tail())

# On va moyenner les données minutes par des données horaires
bt_data_resampled = bitcoin_data.resample('H', on = 'Timestamp').mean().dropna().reset_index()
# print(len(bt_data_resampled)) ~ 22 000, on a enlevé quasiment 100% des données

print(f"Checking for no values after resampling:\n{bt_data_resampled.isna().sum()}\n")
print(f"Number of entries after resampling: {len(bt_data_resampled)}")

### Data Prep
timestamp_column = 'TimeStamp'
target_column = ['Close']
observable_columns = ["Open", "High", "Low"]

SEED = 42
set_seed(SEED)

### Paramètres de prévision (Forecasting Parameters = FP)
context_length = 512
forecast_length = 96 # prédire le prix dans 96 minutes en utilisant les 512 minutes précédentes!

### Splitting the data
data_lenght = len(bt_data_resampled)

train_start_index = 0
train_end_index = round(data_lenght * 0.8)

# Validation set
eval_start_index = round(data_lenght * 0.8) - context_length
eval_end_index = round(data_lenght * 0.9) 

#Test
test_start_index = round(data_lenght * 0.9) - context_length
test_end_index = data_lenght

split_config = {
    "train": [train_start_index, train_end_index],
    "valid": [eval_start_index, eval_end_index],
    "test": [test_start_index, test_end_index],
}

print(f"train_start_index: {train_start_index}, train_end_index: {train_end_index}")
print(f"eval_start_index: {eval_start_index}, eval_end_index: {eval_end_index}")
print(f"test_start_index: {test_start_index}, test_start_index: {test_end_index}")