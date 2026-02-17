# ML — Prédiction de crues (bassin Vilaine Amont)

Pipeline de Machine Learning pour prédire les hauteurs d'eau aux 11 stations hydrométriques du bassin Vilaine Amont, en utilisant les données historiques H/Q et les précipitations Open-Meteo.

## Infrastructure

**NVIDIA DGX Spark** — Grace Blackwell GB10, CUDA 13.0, 128 Go mémoire unifiée.

## Installation

```bash
cd ml
python3 -m venv venv
source venv/bin/activate

# Tout installer (torch CUDA 13.0 + dépendances PyPI)
pip install -r requirements.txt
```

Vérifier que le GPU est bien détecté :

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GB10
```

## Étapes

### 1. Collecte des données

```bash
# Données hydrologiques (H + Q) — 11 stations, depuis 2000
python collect_hydro.py

# Précipitations Open-Meteo — 11 points géographiques
python collect_meteo.py
```

Les CSV sont stockés dans `data/raw/` (gitignored).

Les scripts sont **incrémentaux** : relancer ne récupère que les nouvelles données depuis le dernier timestamp collecté. La date de fin est automatiquement aujourd'hui.

Options utiles :
- `--start 2010-01-01` pour ajuster la date de début
- `--station J706062001` pour ne collecter qu'une station (hydro uniquement)
- `--full` pour forcer une collecte complète (ignore les CSV existants)

### 2. Validation des données

```bash
python validate_data.py
```

Vérifie pour chaque station : fichiers présents, plage temporelle, taux de couverture horaire, trous > 24h, statistiques min/max/moyenne.

### 3. Préparation du dataset

```bash
python prepare_dataset.py
```

Fusionne les 11 stations, interpole les trous, crée les features (dH/dt, sin/cos temporel → 48 features), normalise et découpe en fenêtres glissantes.

Split chronologique :
- **Train** : jusqu'à fin 2023
- **Validation** : 2024
- **Test** : janvier 2025 (inclut la crue du 27/01/2025)

Options :
- `--window 168` pour une fenêtre d'entrée de 7 jours (défaut : 72h)
- `--target J709063002` pour changer la station cible (défaut : Châteaubourg)

Les fichiers `.npy` et métadonnées sont dans `data/processed/`.

### 4. Entraînement — Baseline XGBoost

```bash
python train_xgboost.py
```

Rapide (quelques minutes), sert de référence. Un modèle par horizon de prédiction.

### 5. Entraînement — LSTM

```bash
python train_lstm.py
```

LSTM 2 couches (128→64) avec early stopping. Utilise automatiquement le GPU.

Options : `--epochs 200 --hidden1 256 --hidden2 128 --lr 0.0005`

### 6. Entraînement — TFT (LSTM + Attention)

```bash
python train_tft.py
```

LSTM + Multi-Head Self-Attention. Plus lent mais capture mieux les dépendances inter-stations.

Options : `--epochs 100 --hidden 64 --attention-heads 4`

### 7. Évaluation comparative

```bash
python evaluate.py
```

Compare les 3 modèles (NSE, RMSE, MAE) et génère un graphique `models/checkpoints/comparison.png`.

### 8. Export ONNX

```bash
python export_onnx.py
```

Exporte le meilleur modèle en ONNX dans `models/onnx/`, prêt pour l'inférence dans le backend Node.js via `onnxruntime-node`.

Options : `--model lstm` ou `--model tft` pour forcer un modèle spécifique.

## Structure

```
ml/
├── config.py              # stations, coordonnées, paramètres
├── collect_hydro.py       # collecte H+Q (Hydro EauFrance)
├── collect_meteo.py       # collecte précipitations (Open-Meteo)
├── validate_data.py       # validation des données brutes
├── prepare_dataset.py     # nettoyage, features, fenêtrage
├── train_xgboost.py       # baseline XGBoost
├── train_lstm.py          # modèle LSTM
├── train_tft.py           # modèle TFT (LSTM + Attention)
├── evaluate.py            # comparaison des modèles
├── export_onnx.py         # export ONNX
├── requirements.txt       # dépendances Python
├── data/                  # gitignored
│   ├── raw/               # CSV bruts par station
│   └── processed/         # dataset unifié (.npy)
└── models/                # gitignored
    ├── checkpoints/       # poids PyTorch + résultats JSON
    └── onnx/              # modèle exporté + métadonnées
```

## Métriques cibles

- **NSE > 0.8** sur le jeu de test (standard en hydrologie)
- Focus sur la **crue du 27/01/2025** : le modèle doit anticiper la montée
- Horizons de prédiction : **t+1h, t+3h, t+6h, t+12h, t+24h**
