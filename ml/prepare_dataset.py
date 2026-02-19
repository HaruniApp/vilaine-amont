"""Préparation du dataset unifié pour l'entraînement.

Étapes :
1. Charger les CSV bruts (H, Q, précipitations) pour les 11 stations
2. Aligner sur un index horaire commun
3. Interpoler les trous (linéaire, max 6h)
4. Créer les features dérivées (dH/dt, dQ/dt via différence centrale)
5. Ajouter la feature release (lâché de barrage)
6. Normaliser (P1/P99 robuste)
7. Créer les fenêtres glissantes (input → multi-horizon multi-station output)
8. Split chronologique train/val/test

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --window 72
"""

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    BARRAGE_CODES,
    DH_CLIP,
    DQ_CLIP,
    FORECAST_HORIZONS,
    FUTURE_PRECIP_HOURS,
    INPUT_WINDOW_HOURS,
    PROCESSED_DIR,
    RAW_DIR,
    STATION_CODES,
    STATIONS,
    STATIONS_NO_Q,
    TRAIN_END,
    VAL_END,
)


def load_raw_data() -> pd.DataFrame:
    """Charge et fusionne toutes les données brutes en un DataFrame unifié."""
    all_timestamps = set()

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
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            station_data[(code, "precip")] = df
            all_timestamps.update(df["timestamp"].dt.floor("h"))

        for soil_var in ["soil_moisture_0_to_7cm", "soil_moisture_0_to_28cm"]:
            soil_path = RAW_DIR / f"{code}_{soil_var}.csv"
            if soil_path.exists():
                df = pd.read_csv(soil_path, parse_dates=["timestamp"])
                if df["timestamp"].dt.tz is not None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
                station_data[(code, soil_var)] = df
                all_timestamps.update(df["timestamp"].dt.floor("h"))

    if not all_timestamps:
        raise ValueError("Aucune donnée brute trouvée dans data/raw/")

    ts_min = min(all_timestamps)
    ts_max = max(all_timestamps)
    print(f"Plage temporelle : {ts_min} → {ts_max}")

    hourly_index = pd.date_range(ts_min, ts_max, freq="h")
    result = pd.DataFrame({"timestamp": hourly_index})

    for station in tqdm(STATIONS, desc="Fusion des stations"):
        code = station["code"]

        for var in ["h", "q"]:
            col_name = f"{code}_{var}"
            key = (code, var)
            if key in station_data:
                df = station_data[key].copy()
                df["timestamp"] = df["timestamp"].dt.floor("h")
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

        for soil_var in ["soil_moisture_0_to_7cm", "soil_moisture_0_to_28cm"]:
            soil_key = (code, soil_var)
            soil_col = f"{code}_{soil_var}"
            if soil_key in station_data:
                df = station_data[soil_key].copy()
                df["timestamp"] = df["timestamp"].dt.floor("h")
                df = df.groupby("timestamp")[soil_var].mean().reset_index()
                df.columns = ["timestamp", soil_col]
                result = result.merge(df, on="timestamp", how="left")
            else:
                result[soil_col] = np.nan

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
    """Ajoute dH/dt, dQ/dt (différence centrale) pour chaque station."""
    for code in STATION_CODES:
        h_col = f"{code}_h"
        q_col = f"{code}_q"

        if h_col in df.columns:
            # Différence centrale : (H[t+1] - H[t-1]) / 2
            df[f"{code}_dh"] = (df[h_col].shift(-1) - df[h_col].shift(1)) / 2.0
        if q_col in df.columns:
            df[f"{code}_dq"] = (df[q_col].shift(-1) - df[q_col].shift(1)) / 2.0

    return df


def add_release_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la feature de détection de lâché de barrage pour les 3 stations barrage."""
    for code in BARRAGE_CODES:
        dh_col = f"{code}_dh"
        precip_col = f"{code}_precip"
        if dh_col in df.columns and precip_col in df.columns:
            dh = df[dh_col].fillna(0)
            precip = df[precip_col].fillna(0)
            # release = chute de H sans pluie → signal de lâché
            # dH < 0 et precip ≈ 0 → release > 0
            # dH < 0 et precip > 0 → release ≈ 0 (décrue naturelle)
            # dH ≥ 0 → release = 0
            df[f"{code}_release"] = (-dh).clip(lower=0) * (1 - precip.clip(upper=1))
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


def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalisation robuste P1/P99. Retourne le df normalisé et les paramètres."""
    norm_params = {}
    numeric_cols = [c for c in df.columns if c != "timestamp"]

    for col in numeric_cols:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        if p99 - p1 > 1e-8:
            df[col] = (df[col] - p1) / (p99 - p1)
        else:
            df[col] = 0.0
        norm_params[col] = {"min": float(p1), "max": float(p99)}

    return df, norm_params


def build_feature_order(df: pd.DataFrame) -> list[str]:
    """Construit l'ordre des features : groupées par station.

    Pour chaque station :
    - Stations normales (avec Q) : h, q, precip, dh, dq
    - Stations no_q sans barrage : h, precip, dh
    - Stations barrage : h, precip, dh, release
    """
    feature_cols = []
    for code in STATION_CODES:
        feature_cols.append(f"{code}_h")
        if code not in STATIONS_NO_Q:
            feature_cols.append(f"{code}_q")
        feature_cols.append(f"{code}_precip")
        feature_cols.append(f"{code}_dh")
        if code not in STATIONS_NO_Q:
            feature_cols.append(f"{code}_dq")
        if code in BARRAGE_CODES:
            feature_cols.append(f"{code}_release")
        feature_cols.append(f"{code}_soil_moisture_0_to_7cm")
        feature_cols.append(f"{code}_soil_moisture_0_to_28cm")

    # Vérifier que toutes les colonnes existent
    available = set(df.columns) - {"timestamp"}
    missing = [c for c in feature_cols if c not in available]
    if missing:
        print(f"    WARNING: colonnes manquantes: {missing}")
    extra = available - set(feature_cols)
    if extra:
        print(f"    Colonnes ignorées: {sorted(extra)}")

    return feature_cols


def build_station_feature_map(feature_cols: list[str]) -> dict:
    """Construit le mapping station → indices de features dans le vecteur."""
    station_map = {}
    for code in STATION_CODES:
        indices = []
        vars_list = []
        for i, col in enumerate(feature_cols):
            if col.startswith(f"{code}_"):
                indices.append(i)
                var = col[len(code) + 1:]  # strip "{code}_"
                vars_list.append(var)
        station_map[code] = {"indices": indices, "vars": vars_list}
    return station_map


def build_output_map() -> tuple[dict, int]:
    """Construit le mapping station → indices dans le vecteur de sortie.

    Ordre : pour chaque station, H×5 horizons, puis Q×5 si la station a Q.
    Returns: (output_map, n_outputs)
    """
    output_map = {}
    offset = 0
    for code in STATION_CODES:
        entry = {"h_start": offset, "h_end": offset + len(FORECAST_HORIZONS)}
        offset += len(FORECAST_HORIZONS)
        if code not in STATIONS_NO_Q:
            entry["q_start"] = offset
            entry["q_end"] = offset + len(FORECAST_HORIZONS)
            offset += len(FORECAST_HORIZONS)
        output_map[code] = entry
    return output_map, offset


def create_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    input_window: int,
    horizons: list[int],
    output_map: dict,
    n_outputs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crée les fenêtres glissantes input→output.

    Returns:
        X: (N, input_window, n_features) — séquences d'entrée passées
        future_precip: (N, n_stations * future_hours) — précip futures
        y: (N, n_outputs) — deltas H et Q pour toutes les stations
        timestamps: (N,) — timestamp du dernier pas de la fenêtre d'entrée
    """
    data = df[feature_cols].values
    timestamps = df["timestamp"].values
    n_stations = len(STATION_CODES)

    # Trouver les indices de précip dans feature_cols
    precip_indices = []
    for code in STATION_CODES:
        pcol = f"{code}_precip"
        precip_indices.append(feature_cols.index(pcol))

    # Trouver les indices de H et Q pour le target
    h_indices = {}
    q_indices = {}
    for code in STATION_CODES:
        h_indices[code] = feature_cols.index(f"{code}_h")
        if code not in STATIONS_NO_Q:
            q_indices[code] = feature_cols.index(f"{code}_q")

    max_horizon = max(horizons)
    n_samples = len(df) - input_window - max_horizon

    if n_samples <= 0:
        raise ValueError(f"Pas assez de données : {len(df)} lignes pour window={input_window}, max_horizon={max_horizon}")

    X = np.empty((n_samples, input_window, len(feature_cols)), dtype=np.float32)
    future_precip = np.empty((n_samples, n_stations * FUTURE_PRECIP_HOURS), dtype=np.float32)
    y = np.empty((n_samples, n_outputs), dtype=np.float32)
    ts = np.empty(n_samples, dtype="datetime64[ns]")

    for i in tqdm(range(n_samples), desc="Fenêtrage", leave=False):
        window_end = i + input_window  # index du premier pas après la fenêtre

        # Séquence passée
        X[i] = data[i:window_end]

        # Précipitations futures : T+1 à T+FUTURE_PRECIP_HOURS
        for s_idx, code in enumerate(STATION_CODES):
            p_idx = precip_indices[s_idx]
            for h in range(FUTURE_PRECIP_HOURS):
                future_idx = window_end + h
                if future_idx < len(data):
                    future_precip[i, s_idx * FUTURE_PRECIP_HOURS + h] = data[future_idx, p_idx]
                else:
                    future_precip[i, s_idx * FUTURE_PRECIP_HOURS + h] = 0.0

        # Multi-target : delta H et Q pour toutes les stations
        for code in STATION_CODES:
            h_idx = h_indices[code]
            h_current = data[window_end - 1, h_idx]
            om = output_map[code]

            # Delta H
            for j, horizon in enumerate(horizons):
                target_idx = window_end + horizon - 1
                if target_idx < len(data):
                    y[i, om["h_start"] + j] = data[target_idx, h_idx] - h_current
                else:
                    y[i, om["h_start"] + j] = 0.0

            # Delta Q (si applicable)
            if code not in STATIONS_NO_Q:
                q_idx = q_indices[code]
                q_current = data[window_end - 1, q_idx]
                for j, horizon in enumerate(horizons):
                    target_idx = window_end + horizon - 1
                    if target_idx < len(data):
                        y[i, om["q_start"] + j] = data[target_idx, q_idx] - q_current
                    else:
                        y[i, om["q_start"] + j] = 0.0

        ts[i] = timestamps[window_end - 1]

    # Remplacer les NaN restants par 0
    X = np.nan_to_num(X, nan=0.0)
    future_precip = np.nan_to_num(future_precip, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    return X, future_precip, y, ts


def split_by_date(
    X: np.ndarray,
    future_precip: np.ndarray,
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
        "X_val": X[val_mask],
        "X_test": X[test_mask],
        "fp_train": future_precip[train_mask],
        "fp_val": future_precip[val_mask],
        "fp_test": future_precip[test_mask],
        "y_train": y[train_mask],
        "y_val": y[val_mask],
        "y_test": y[test_mask],
        "ts_train": timestamps[train_mask],
        "ts_val": timestamps[val_mask],
        "ts_test": timestamps[test_mask],
    }


def main():
    parser = argparse.ArgumentParser(description="Préparation du dataset")
    parser.add_argument("--window", type=int, default=INPUT_WINDOW_HOURS, help="Fenêtre d'entrée (heures)")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Chargement
    print("1/7 — Chargement des données brutes...")
    df = load_raw_data()
    print(f"    {len(df)} lignes, {len(df.columns)} colonnes")

    # 1b. Suppression des colonnes Q inutiles
    df = drop_no_q_columns(df)

    # 2. Interpolation
    print("2/7 — Interpolation des trous...")
    df = interpolate_gaps(df)

    # 3. Features dérivées (différence centrale)
    print("3/7 — Création des features dérivées (différence centrale)...")
    df = add_derived_features(df)

    # 3b. Clipping des outliers sur dH/dt et dQ/dt
    df = clip_outliers(df)

    # 3c. Feature release (lâché de barrage)
    print("    Ajout des features release (barrages)...")
    df = add_release_features(df)

    # Supprimer les bords (NaN de la différence centrale)
    df = df.iloc[1:-1].reset_index(drop=True)

    # 4. Construire l'ordre des features
    feature_cols = build_feature_order(df)
    print(f"    {len(feature_cols)} features par pas de temps")

    # 5. Normalisation P1/P99
    print("4/7 — Normalisation P1/P99...")
    df, norm_params = normalize_features(df)

    # Sauvegarder les paramètres de normalisation
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
    features_path = PROCESSED_DIR / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # 6. Construire output map
    output_map, n_outputs = build_output_map()
    print(f"    Output: {n_outputs} valeurs ({len(STATION_CODES)} stations × H + Q)")

    # 7. Fenêtrage
    print(f"5/7 — Fenêtrage (window={args.window}h, horizons={FORECAST_HORIZONS}h)...")
    X, future_precip, y, timestamps = create_windows(
        df, feature_cols, args.window, FORECAST_HORIZONS, output_map, n_outputs
    )
    print(f"    X: {X.shape}, future_precip: {future_precip.shape}, y: {y.shape}")

    # 8. Split
    print(f"6/7 — Split chronologique (train≤{TRAIN_END}, val≤{VAL_END}, test=reste)...")
    splits = split_by_date(X, future_precip, y, timestamps, TRAIN_END, VAL_END)

    for name in ["train", "val", "test"]:
        xk, fpk, yk = f"X_{name}", f"fp_{name}", f"y_{name}"
        print(f"    {name}: X={splits[xk].shape}, fp={splits[fpk].shape}, y={splits[yk].shape}")
        np.save(PROCESSED_DIR / f"X_{name}.npy", splits[xk])
        np.save(PROCESSED_DIR / f"fp_{name}.npy", splits[fpk])
        np.save(PROCESSED_DIR / f"y_{name}.npy", splits[yk])
        np.save(PROCESSED_DIR / f"ts_{name}.npy", splits[f"ts_{name}"])

    # Sauvegarder les métadonnées
    station_feature_map = build_station_feature_map(feature_cols)

    # Vars per station pour le reshape dans le modèle
    vars_per_station = {}
    for code in STATION_CODES:
        vars_per_station[code] = station_feature_map[code]["vars"]

    meta = {
        "input_window": args.window,
        "forecast_horizons": FORECAST_HORIZONS,
        "n_features": len(feature_cols),
        "n_stations": len(STATION_CODES),
        "n_outputs": n_outputs,
        "future_precip_size": len(STATION_CODES) * FUTURE_PRECIP_HOURS,
        "future_precip_hours": FUTURE_PRECIP_HOURS,
        "feature_names": feature_cols,
        "station_codes": STATION_CODES,
        "station_feature_map": station_feature_map,
        "vars_per_station": vars_per_station,
        "output_map": output_map,
        "target_mode": "delta",
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

    print(f"\n7/7 — Métadonnées sauvegardées")
    print(f"\n✓ Dataset prêt dans {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
