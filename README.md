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

# Dataset

## Fichiers
sample_btcusd.csv est un fichier d'1 million de ligne qui sert de dataset, qui est tronqué d'un fichier de 6 millions de lignes qui contient des données sur 
le bitcoin, à partir de 2012 et jusqu'à 2014 à CHAQUE SECONDES.

À la base, la colone TimeStamp est de type UNIX time (nb de secondes, depuis le 1 Janvier 1970), on peut la convertir avec Pandas :
- pd.to_datetime(df['TimeStamp'], unit='s')

Pour réduire la taille, on utilise :
- resample() pour moyenner les minutes en heures.
- dropna() pour supprimer les valeurs manquantes.
- reset_index() pour réinitialiser l'index.

## Valeurs manquantes
On peut utiliser:
- isna(), qui est une méthode Pandas qui renvoie un masque booléen de la même forme que le DataFrame, indiquant les valeurs manquantes.
Si on trouve une valeur manquante, on peut utiliser :
- ffill() qui permet de remplir les valeurs manquantes avec la dernière valeur non manquante.

## Data Prep
On va performer un **Zero-shot learning**. On va faire une prédiction sans entrainer le modèle. 
D'abord on va couper le dataset en :
- Training set (80%)
- Validation set (10%) permet d'ajuster le modèle.
- Test set (10%)

Puis on fera un **Fine-tuning** pour améliorer les performances.

