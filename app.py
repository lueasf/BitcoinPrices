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
from sklearn.metrics import root_mean_squared_error

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

### Comparaison 1
fcast_df = pd.DataFrame({
    "pred": zs_forecast.loc[11]['Close_prediction'],  # Predicted values for the next 24 time steps
    "actual": zs_forecast.loc[11]['Close'][:24]       # Actual values for the same time period
})

ax = fcast_df.plot()

ax.set_xlabel("Time Steps")  
ax.set_ylabel("Close Price") 
ax.set_title("Predicted vs Actual Close Price for Row 11")
plt.show()

### Comparaison 2
def compare_forecast(forecast, date_col, prediction_col, actual_col, hours_out):
    comp = pd.DataFrame()
    comp[date_col] = forecast[date_col]

    actual = []
    pred = []

    for i in range(len(forecast)):
        pred.append(forecast[prediction_col].values[i][hours_out - 1])
        actual.append(forecast[actual_col].values[i][hours_out -1])
    
    comp['actual'] = actual
    comp['pred'] = pred

    return comp

# Valeurs sur un jour sans les NaN
one_day_pred = compare_forecast(zs_forecast, 'Timestamp', 'Close_prediction', 'Close', 24)
out = one_day_pred.dropna(subset=['actual', 'pred'])

# Calcul du RMSE - Root Mean Squared Error
rmse = '{:.10f}'.format(root_mean_squared_error(out['actual'], out['pred']))
print(f"RMSE: {rmse}")

# Graphique qui représente les valeurs prédites et les valeurs réelles sur le dataset.
out.plot(x="Timestamp", y=["pred", "actual"], figsize=(20, 5), title=f"RMSE for zero-shot model: {rmse}")
plt.show()




### Fine-tuning
OUT_DIR = ""

learning_rate = 0.0001 
num_epochs = 10  
batch_size = 32
 
finetune_forecast_args = TrainingArguments(
    output_dir=os.path.join(OUT_DIR, "output"),  
    overwrite_output_dir=True,  
    learning_rate=learning_rate,  
    num_train_epochs=num_epochs,  
    do_eval=True,  
    evaluation_strategy="epoch",  
    per_device_train_batch_size=batch_size,  
    per_device_eval_batch_size=batch_size,  
    dataloader_num_workers=8,  
    save_strategy="epoch",  
    logging_strategy="epoch",  
    save_total_limit=1,  
    logging_dir=os.path.join(OUT_DIR, "logs"),  
    load_best_model_at_end=True,  
    metric_for_best_model="eval_loss",  
    greater_is_better=False,  
)

# Avant de faire le fine-tuning, on doit charger un modèle pré-entrainé, qui a déjà été entrainé sur des données similaires.
finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained("models/finetuned_forecast_model")

# pour empecher le modèle de se sur-entrainer, on utilise un early stopping callback.
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2, # s'arrete après 2 époques sans amélioration
    early_stopping_threshold=0.001, # minimum requis d'amélioration pour continuer
)

tracking_callback = TrackingCallback()

optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
scheduler = OneCycleLR(
    optimizer,
    learning_rate,
    epochs=num_epochs,
    steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
)

finetune_forecast_trainer = Trainer(
    model=finetune_forecast_model,
    args=finetune_forecast_args,  
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,  
    callbacks=[early_stopping_callback, tracking_callback],  
    optimizers=(optimizer, scheduler),  
)

# finetune_forecast_trainer.train()

### Pipeline de prévision des séries temporelles après le fine-tuning 
forecast_pipeline = TimeSeriesForecastingPipeline(
    model=finetune_forecast_model,
    device="cpu",
    timestamp_column=timestamp_column,
    id_columns=[],
    target_columns=target_columns,
    observable_columns=observable_columns,
    freq='h'
)

# finetune_forecast_trainer.evaluate(tsp.preprocess(bt_data_resampled[test_start_index:test_end_index]))

### Évaluation du modèle fine-tuné
# On recharge le modèle pré-entrainé, car l'exéc du fine-tuning est longue.
forecast_finetuned = pd.read_pickle("models/forecast_finetuned.pkl")

forecast_predictions = compare_forecast(forecast_finetuned, "Timestamp", "Close_prediction", "Close", 12)
forecast_out = forecast_predictions.dropna(subset=["actual", "pred"])

rmse2 = '{:.10f}'.format(root_mean_squared_error(forecast_out['actual'], forecast_out['pred']))

forecast_out.plot(x="Timestamp", y=["pred", "actual"], figsize=(20, 5), title=f"RMSE for fine-tuned model: {rmse2}")
plt.show()