# BitcoinPrices
Predicting Bitcoin Close Prices using Foundational Models (Time Series.)

# Présentation
Ce projet permet d'apprendre la prévision de **séries temporelles** en 
utilisant le modèle **Tiny Time Mixer (TTM) d’IBM**. En travaillant sur les prix 
du Bitcoin, il couvre la préparation des données, l'entraînement du modèle et 
l’évaluation des prédictions à l’aide de métriques comme le RMSE. 

Le Tiny Time Mixer (TTM) est un modèle pré-entraîné développé par IBM pour 
la prévision de séries temporelles multivariées. Introduit en 2024, il est 
performant, léger et efficace.

# Objectifs du projet
- Installer l'envirronement de travail pour le modèle TTM.
- Préparer le dataset (prix du Bitcoin).
- Comprendre l'architecture du modèle TTM.
- Entrainer le modèle.
- Analyser les performances du modèle.
- Explorer les différentes options de personnalisation du modèle.

# Installation 
```bash
git clone
cd BitcoinPrices
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

