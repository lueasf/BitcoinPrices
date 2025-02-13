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
bt_data_resampled = bitcoin_data.resample('h', on = 'Timestamp').mean().dropna().reset_index()
# print(len(bt_data_resampled)) ~ 22 000, on a enlevé quasiment 100% des données

# print(f"Checking for no values after resampling:\n{bt_data_resampled.isna().sum()}\n")
# print(f"Number of entries after resampling: {len(bt_data_resampled)}")

### Data Prep
timestamp_column = 'Timestamp'
target_columns = ['Close']
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

# print(f"train_start_index: {train_start_index}, train_end_index: {train_end_index}")
# print(f"eval_start_index: {eval_start_index}, eval_end_index: {eval_end_index}")
# print(f"test_start_index: {test_start_index}, test_start_index: {test_end_index}")

# On utilise la Data Prep pour config le TimeSeriesPreprocessor
column_specifiers = {
    "timestamp_column": timestamp_column,  # Time reference column
    "target_columns": target_columns,  # Target variable (Close price)
    "observable_columns": observable_columns  # Observable variables (Open, High, Low prices)
}

tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=context_length,
    prediction_length=forecast_length,
    scaling=True, # applique une normalisation sur les données
    encode_categorical=False,
    scaler_type="standard"
)

# On recup les datasets avec en params (le tsp, les données et la config)
train_dataset, valid_dataset, test_dataset = get_datasets(
    tsp, bt_data_resampled, split_config
)

### Zero-shot Model
# On charge le modèle Tiny Time Mixer
zeroshot_model = TinyTimeMixerForPrediction.from_pretrained("models/zero_shot_model", prediction_filter_length=24)
zeroshot_trainer = Trainer(model=zeroshot_model,)
res = zeroshot_trainer.evaluate(test_dataset)
# print(res)
'''
Le résultat obtenu est :
eval_loss: 0.004135717637836933  (perte mesurée sur le test)
eval_runtime: 5.2537  (temps d'évaluation)
eval_samples_per_second: 504.408  (nombre d'échantillons évalués par seconde)
eval_steps_per_second: 63.194   (nb de paquets évalués par seconde)
'''

### Prediction avec le modèle Zero-shot
# On configure le pipeline de prévision
zs_forecast_pipeline = TimeSeriesForecastingPipeline(
    model=zeroshot_model,
    device="cpu",
    timestamp_column=timestamp_column,
    id_colums=[],
    target_columns=target_columns,
    freq='h'
)

zs_forecast = pd.read_pickle("models/zs_forecast.pkl")
# print(zs_forecast)
'''
Rappel : zs.pkl contient des prédictions enregistrés (en pickle) par le modèle sur mes données test.

On obtient des valeurs et des NaN dans la colone close car les vraies valeurs ne sont pas 
encore connues, et donc on ne peut pas encore comparer les valeurs prédites avec les valeurs réelles.
'''

# Comparaison 1
fcast_df = pd.DataFrame({
    "pred": zs_forecast.loc[11]['Close_prediction'],  # Predicted values for the next 24 time steps
    "actual": zs_forecast.loc[11]['Close'][:24]       # Actual values for the same time period
})

ax = fcast_df.plot()

ax.set_xlabel("Time Steps")  
ax.set_ylabel("Close Price") 
ax.set_title("Predicted vs Actual Close Price for Row 11")
plt.show()

# Comparaison 2
def compare_forecast(forecast, date_col, prediction_col, actual_col, hours_out):
    comp = pd.DataFrame()
    comp[date_col] = forecast[date_col]

    actual = []
    pred = []