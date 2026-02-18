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
    DH_CLIP,
    DQ_CLIP,
    FORECAST_HORIZONS,
    INPUT_WINDOW_HOURS,
    PROCESSED_DIR,
    PROPAGATION_HOURS,
    RAW_DIR,
    RIVER_BRANCHES,
    RIVER_DISTANCES_KM,
    STATION_BRANCH,
    STATION_CODES,
    STATIONS,
    STATIONS_NO_Q,
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
                if df["timestamp"].dt.tz is not None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
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


def drop_no_q_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes Q des stations sans données de débit."""
    cols_to_drop = [f"{code}_q" for code in STATIONS_NO_Q if f"{code}_q" in df.columns]
    if cols_to_drop:
        print(f"    Suppression colonnes Q sans données : {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
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


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip les outliers des features dérivées (dH/dt, dQ/dt)."""
    dh_cols = [c for c in df.columns if c.endswith("_dh")]
    dq_cols = [c for c in df.columns if c.endswith("_dq")]
    for col in dh_cols:
        df[col] = df[col].clip(-DH_CLIP, DH_CLIP)
    for col in dq_cols:
        df[col] = df[col].clip(-DQ_CLIP, DQ_CLIP)
    clipped = len(dh_cols) + len(dq_cols)
    print(f"    Clipping appliqué sur {clipped} colonnes (dH: ±{DH_CLIP}, dQ: ±{DQ_CLIP})")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features de lag pour les stations amont (H décalé du temps de propagation)."""
    for code, hours in PROPAGATION_HOURS.items():
        if hours is None:
            continue
        h_col = f"{code}_h"
        if h_col in df.columns:
            lag_col = f"{code}_h_lag{hours}h"
            df[lag_col] = df[h_col].shift(hours)
    return df


def add_static_spatial_features(df: pd.DataFrame, norm_params: dict) -> tuple[pd.DataFrame, dict]:
    """Ajoute les features statiques de topologie (après normalisation).

    Pour chaque station : distance normalisée, is_upstream, is_barrage, one-hot branche.
    """
    max_dist = max(abs(v) for v in RIVER_DISTANCES_KM.values())
    barrage_codes = {s["code"] for s in STATIONS if s.get("barrage")}

    n = len(df)
    new_cols = {}
    for code in STATION_CODES:
        dist_norm = RIVER_DISTANCES_KM[code] / max_dist  # [-1, 1]
        is_upstream = 1.0 if RIVER_DISTANCES_KM[code] > 0 else 0.0
        is_barrage = 1.0 if code in barrage_codes else 0.0
        branch = STATION_BRANCH[code]

        new_cols[f"{code}_dist_to_target"] = np.full(n, dist_norm)
        norm_params[f"{code}_dist_to_target"] = {"min": -1.0, "max": 1.0}

        new_cols[f"{code}_is_upstream"] = np.full(n, is_upstream)
        norm_params[f"{code}_is_upstream"] = {"min": 0.0, "max": 1.0}

        new_cols[f"{code}_is_barrage"] = np.full(n, is_barrage)
        norm_params[f"{code}_is_barrage"] = {"min": 0.0, "max": 1.0}

        for b in RIVER_BRANCHES:
            col = f"{code}_branch_{b}"
            new_cols[col] = np.full(n, 1.0 if branch == b else 0.0)
            norm_params[col] = {"min": 0.0, "max": 1.0}

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df, norm_params


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
        h_current = data[i + input_window - 1, target_idx]  # dernier pas de la fenêtre
        for j, h in enumerate(horizons):
            # Delta target : y = H_futur_norm - H_actuel_norm
            y[i, j] = data[i + input_window + h - 1, target_idx] - h_current
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
    print("1/8 — Chargement des données brutes...")
    df = load_raw_data()
    print(f"    {len(df)} lignes, {len(df.columns)} colonnes")

    # 1b. Suppression des colonnes Q inutiles
    df = drop_no_q_columns(df)

    # 2. Interpolation
    print("2/8 — Interpolation des trous...")
    df = interpolate_gaps(df)

    # 3. Features dérivées
    print("3/8 — Création des features dérivées...")
    df = add_derived_features(df)

    # 3b. Clipping des outliers sur dH/dt et dQ/dt
    df = clip_outliers(df)

    # 4. Features de lag (stations amont)
    print("4/8 — Ajout des features de lag...")
    df = add_lag_features(df)

    # Supprimer la première ligne (NaN du diff)
    df = df.iloc[1:].reset_index(drop=True)
    print(f"    {len(df.columns) - 1} features (avant static)")

    # 5. Normalisation
    print("5/8 — Normalisation min-max...")
    df, norm_params = normalize_features(df)

    # 6. Features statiques spatiales (après normalisation, déjà en [0,1] ou [-1,1])
    print("6/8 — Ajout des features statiques spatiales...")
    df, norm_params = add_static_spatial_features(df, norm_params)

    # Sauvegarder les paramètres de normalisation (NaN → null pour JSON valide)
    norm_path = PROCESSED_DIR / "norm_params.json"
    clean_params = {}
    for k, v in norm_params.items():
        clean_params[k] = {
            "min": None if np.isnan(v["min"]) else v["min"],
            "max": None if np.isnan(v["max"]) else v["max"],
        }
    with open(norm_path, "w") as f:
        json.dump(clean_params, f, indent=2)
    print(f"    Paramètres → {norm_path.name}")

    # Sauvegarder les noms de features
    feature_cols = [c for c in df.columns if c != "timestamp"]
    features_path = PROCESSED_DIR / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # 7. Fenêtrage
    print(f"7/8 — Fenêtrage (window={args.window}h, horizons={FORECAST_HORIZONS}h)...")
    X, y, timestamps = create_windows(df, args.target, args.window, FORECAST_HORIZONS)
    print(f"    X: {X.shape}, y: {y.shape}")

    # 8. Split
    print(f"8/8 — Split chronologique (train≤{TRAIN_END}, val≤{VAL_END}, test=reste)...")
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
        "target_mode": "delta",
        "log_transform_cols": [],
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
