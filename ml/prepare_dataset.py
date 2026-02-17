"""Préparation du dataset unifié pour l'entraînement.

Étapes :
1. Charger les CSV bruts (H, Q, précipitations) pour les 11 stations
2. Aligner sur un index horaire commun
3. Interpoler les trous (linéaire, max 6h)
4. Créer les features dérivées (dH/dt, dQ/dt, encodage temporel)
5. Normaliser (min-max)
6. Créer les fenêtres glissantes (input → multi-horizon output)
7. Split chronologique train/val/test

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --window 168 --target J706062001
"""

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FORECAST_HORIZONS,
    INPUT_WINDOW_HOURS,
    PROCESSED_DIR,
    RAW_DIR,
    STATION_CODES,
    STATIONS,
    TARGET_STATION,
    TRAIN_END,
    VAL_END,
)


def load_raw_data() -> pd.DataFrame:
    """Charge et fusionne toutes les données brutes en un DataFrame unifié."""
    # Index horaire commun
    all_timestamps = set()

    # Charger d'abord pour trouver la plage
    station_data = {}
    for station in STATIONS:
        code = station["code"]

        for var in ["h", "q"]:
            path = RAW_DIR / f"{code}_{var}.csv"
            if path.exists():
                df = pd.read_csv(path, parse_dates=["timestamp"])
                station_data[(code, var)] = df
                all_timestamps.update(df["timestamp"].dt.floor("h"))

        precip_path = RAW_DIR / f"{code}_precip.csv"
        if precip_path.exists():
            df = pd.read_csv(precip_path, parse_dates=["timestamp"])
            # Open-Meteo retourne des timestamps tz-aware, on les convertit en tz-naive
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            station_data[(code, "precip")] = df
            all_timestamps.update(df["timestamp"].dt.floor("h"))

    if not all_timestamps:
        raise ValueError("Aucune donnée brute trouvée dans data/raw/")

    # Créer l'index horaire
    ts_min = min(all_timestamps)
    ts_max = max(all_timestamps)
    print(f"Plage temporelle : {ts_min} → {ts_max}")

    hourly_index = pd.date_range(ts_min, ts_max, freq="h")
    result = pd.DataFrame({"timestamp": hourly_index})

    # Joindre chaque série
    for station in tqdm(STATIONS, desc="Fusion des stations"):
        code = station["code"]

        for var in ["h", "q"]:
            col_name = f"{code}_{var}"
            key = (code, var)
            if key in station_data:
                df = station_data[key].copy()
                df["timestamp"] = df["timestamp"].dt.floor("h")
                # Moyenne si plusieurs valeurs dans la même heure
                df = df.groupby("timestamp")[var].mean().reset_index()
                df.columns = ["timestamp", col_name]
                result = result.merge(df, on="timestamp", how="left")
            else:
                result[col_name] = np.nan

        precip_key = (code, "precip")
        col_name = f"{code}_precip"
        if precip_key in station_data:
            df = station_data[precip_key].copy()
            df["timestamp"] = df["timestamp"].dt.floor("h")
            df = df.groupby("timestamp")["precipitation"].mean().reset_index()
            df.columns = ["timestamp", col_name]
            result = result.merge(df, on="timestamp", how="left")
        else:
            result[col_name] = np.nan

    return result


def interpolate_gaps(df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
    """Interpole les trous linéairement (max max_gap_hours consécutifs)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].interpolate(method="linear", limit=max_gap_hours)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute dH/dt, dQ/dt pour chaque station + encodage temporel."""
    for code in STATION_CODES:
        h_col = f"{code}_h"
        q_col = f"{code}_q"

        if h_col in df.columns:
            df[f"{code}_dh"] = df[h_col].diff()
        if q_col in df.columns:
            df[f"{code}_dq"] = df[q_col].diff()

    # Encodage temporel cyclique
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    day_of_year = df["timestamp"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return df


def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalisation min-max. Retourne le df normalisé et les paramètres."""
    norm_params = {}
    numeric_cols = [c for c in df.columns if c != "timestamp"]

    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 1e-8:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.0
        norm_params[col] = {"min": float(col_min), "max": float(col_max)}

    return df, norm_params


def create_windows(
    df: pd.DataFrame,
    target_station: str,
    input_window: int,
    horizons: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crée les fenêtres glissantes input→output.

    Returns:
        X: (N, input_window, n_features) — séquences d'entrée
        y: (N, len(horizons)) — hauteurs cibles aux horizons
        timestamps: (N,) — timestamp du dernier pas de la fenêtre d'entrée
    """
    feature_cols = [c for c in df.columns if c != "timestamp"]
    target_col = f"{target_station}_h"

    if target_col not in feature_cols:
        raise ValueError(f"Colonne cible {target_col} introuvable")

    data = df[feature_cols].values
    target_idx = feature_cols.index(target_col)
    timestamps = df["timestamp"].values

    max_horizon = max(horizons)
    n_samples = len(df) - input_window - max_horizon

    if n_samples <= 0:
        raise ValueError(f"Pas assez de données : {len(df)} lignes pour window={input_window}, max_horizon={max_horizon}")

    X = np.empty((n_samples, input_window, len(feature_cols)), dtype=np.float32)
    y = np.empty((n_samples, len(horizons)), dtype=np.float32)
    ts = np.empty(n_samples, dtype="datetime64[ns]")

    for i in tqdm(range(n_samples), desc="Fenêtrage", leave=False):
        X[i] = data[i : i + input_window]
        for j, h in enumerate(horizons):
            y[i, j] = data[i + input_window + h - 1, target_idx]
        ts[i] = timestamps[i + input_window - 1]

    # Remplacer les NaN restants par 0
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    return X, y, ts


def split_by_date(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    train_end: str,
    val_end: str,
) -> dict:
    """Split chronologique train/val/test."""
    train_mask = timestamps <= np.datetime64(train_end)
    val_mask = (timestamps > np.datetime64(train_end)) & (timestamps <= np.datetime64(val_end))
    test_mask = timestamps > np.datetime64(val_end)

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "ts_train": timestamps[train_mask],
        "ts_val": timestamps[val_mask],
        "ts_test": timestamps[test_mask],
    }


def main():
    parser = argparse.ArgumentParser(description="Préparation du dataset")
    parser.add_argument("--window", type=int, default=INPUT_WINDOW_HOURS, help="Fenêtre d'entrée (heures)")
    parser.add_argument("--target", default=TARGET_STATION, help="Station cible")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Chargement
    print("1/6 — Chargement des données brutes...")
    df = load_raw_data()
    print(f"    {len(df)} lignes, {len(df.columns)} colonnes")

    # 2. Interpolation
    print("2/6 — Interpolation des trous...")
    df = interpolate_gaps(df)

    # 3. Features dérivées
    print("3/6 — Création des features dérivées...")
    df = add_derived_features(df)
    # Supprimer la première ligne (NaN du diff)
    df = df.iloc[1:].reset_index(drop=True)
    print(f"    {len(df.columns) - 1} features")

    # 4. Normalisation
    print("4/6 — Normalisation min-max...")
    df, norm_params = normalize_features(df)

    # Sauvegarder les paramètres de normalisation
    norm_path = PROCESSED_DIR / "norm_params.json"
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"    Paramètres → {norm_path.name}")

    # Sauvegarder les noms de features
    feature_cols = [c for c in df.columns if c != "timestamp"]
    features_path = PROCESSED_DIR / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # 5. Fenêtrage
    print(f"5/6 — Fenêtrage (window={args.window}h, horizons={FORECAST_HORIZONS}h)...")
    X, y, timestamps = create_windows(df, args.target, args.window, FORECAST_HORIZONS)
    print(f"    X: {X.shape}, y: {y.shape}")

    # 6. Split
    print(f"6/6 — Split chronologique (train≤{TRAIN_END}, val≤{VAL_END}, test=reste)...")
    splits = split_by_date(X, y, timestamps, TRAIN_END, VAL_END)

    for name in ["train", "val", "test"]:
        xk, yk = f"X_{name}", f"y_{name}"
        print(f"    {name}: X={splits[xk].shape}, y={splits[yk].shape}")
        np.save(PROCESSED_DIR / f"X_{name}.npy", splits[xk])
        np.save(PROCESSED_DIR / f"y_{name}.npy", splits[yk])
        np.save(PROCESSED_DIR / f"ts_{name}.npy", splits[f"ts_{name}"])

    # Sauvegarder les métadonnées
    meta = {
        "target_station": args.target,
        "input_window": args.window,
        "forecast_horizons": FORECAST_HORIZONS,
        "n_features": X.shape[2],
        "feature_names": feature_cols,
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "shapes": {
            "train": list(splits["X_train"].shape),
            "val": list(splits["X_val"].shape),
            "test": list(splits["X_test"].shape),
        },
    }
    meta_path = PROCESSED_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Dataset prêt dans {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
