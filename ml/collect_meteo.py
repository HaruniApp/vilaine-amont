"""Collecte des données météo horaires depuis Open-Meteo Historical Weather API.

Variables collectées :
- precipitation (mm/h)
- soil_moisture_0_to_7cm (m³/m³)
- soil_moisture_0_to_28cm (m³/m³)

Open-Meteo accepte des plages longues (plusieurs années) en une seule requête,
mais on découpe en segments annuels pour fiabilité.

Mode incrémental : si un CSV existe déjà, ne récupère que les données
postérieures au dernier timestamp présent.

Usage:
    python collect_meteo.py
    python collect_meteo.py --start 2000-01-01
    python collect_meteo.py --full   # ignore les CSV existants
"""

import argparse
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    COLLECT_END_DATE,
    COLLECT_START_DATE,
    METEO_BASE_URL,
    RAW_DIR,
    STATIONS,
)

MAX_RETRIES = 3
RETRY_DELAY = 5
REQUEST_DELAY = 0.5  # Open-Meteo est généreux mais restons polis

METEO_VARIABLES = ["precipitation", "soil_moisture_0_to_7cm", "soil_moisture_0_to_28cm"]

# Mapping variable → suffixe de fichier CSV
CSV_SUFFIXES = {
    "precipitation": "precip",
    "soil_moisture_0_to_7cm": "soil_moisture_0_to_7cm",
    "soil_moisture_0_to_28cm": "soil_moisture_0_to_28cm",
}


def fetch_meteo(lat: float, lon: float, start: str, end: str) -> dict[str, pd.DataFrame] | None:
    """Récupère les variables météo horaires pour un point géographique.

    Returns:
        Dict mapping variable name → DataFrame with (timestamp, value) columns.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(METEO_VARIABLES),
        "timezone": "Europe/Paris",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(METEO_BASE_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            if not times:
                return None

            result = {}
            timestamps = pd.to_datetime(times)
            for var in METEO_VARIABLES:
                values = hourly.get(var, [])
                if values:
                    result[var] = pd.DataFrame({"timestamp": timestamps, var: values})

            return result if result else None

        except (requests.RequestException, ValueError) as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} ({e})")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  ÉCHEC: {e}")
                return None


def generate_yearly_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Découpe en segments annuels."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    ranges = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=365), end)
        ranges.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end
    return ranges


def get_last_timestamp(csv_path) -> str | None:
    """Retourne le dernier timestamp d'un CSV existant, ou None."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        if df.empty:
            return None
        last = df["timestamp"].max()
        return last.strftime("%Y-%m-%d")
    except Exception:
        return None


def collect_station_meteo(station: dict, start_date: str, end_date: str, full: bool = False) -> None:
    """Collecte les variables météo pour une station."""
    code = station["code"]
    label = station["label"]
    lat, lon = station["lat"], station["lon"]

    # Determine effective start date from existing CSVs (incremental mode)
    csv_paths = {var: RAW_DIR / f"{code}_{CSV_SUFFIXES[var]}.csv" for var in METEO_VARIABLES}
    existing_dfs = {}
    effective_start = start_date

    if not full:
        # Find the earliest "last timestamp" across all variable CSVs
        last_timestamps = []
        for var in METEO_VARIABLES:
            last_ts = get_last_timestamp(csv_paths[var])
            if last_ts is not None:
                last_timestamps.append(last_ts)
                existing_dfs[var] = pd.read_csv(csv_paths[var], parse_dates=["timestamp"])

        if last_timestamps:
            # Use the earliest last_ts to ensure all variables are up to date
            earliest_last = min(last_timestamps)
            if earliest_last >= end_date:
                print(f"\n  {label} ({code}) — déjà à jour, skip")
                return
            effective_start = earliest_last
            print(f"\n  {label} ({code}) — incrémental depuis {earliest_last}")
        else:
            print(f"\n  {label} ({code}) — lat={lat}, lon={lon}")
    else:
        print(f"\n  {label} ({code}) — lat={lat}, lon={lon}")

    ranges = generate_yearly_ranges(effective_start, end_date)
    all_dfs = {var: [] for var in METEO_VARIABLES}

    for seg_start, seg_end in tqdm(ranges, desc=f"  {code}", leave=False):
        result = fetch_meteo(lat, lon, seg_start, seg_end)
        if result is not None:
            for var in METEO_VARIABLES:
                if var in result:
                    all_dfs[var].append(result[var])
        time.sleep(REQUEST_DELAY)

    for var in METEO_VARIABLES:
        new_df = pd.concat(all_dfs[var], ignore_index=True) if all_dfs[var] else pd.DataFrame()
        existing_df = existing_dfs.get(var)

        if existing_df is not None and not new_df.empty:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        elif existing_df is not None:
            df = existing_df
        elif not new_df.empty:
            df = new_df
        else:
            print(f"  ⚠ Aucune donnée {var} pour {code}")
            continue

        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        df.to_csv(csv_paths[var], index=False)
        print(f"  ✓ {var}: {len(df)} points → {csv_paths[var].name}")


def main():
    parser = argparse.ArgumentParser(description="Collecte des données météo Open-Meteo")
    parser.add_argument("--start", default=COLLECT_START_DATE, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", default=COLLECT_END_DATE, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--full", action="store_true", help="Collecte complète (ignore les CSV existants)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    mode = "complète" if args.full else "incrémentale"
    print(f"Collecte Open-Meteo ({mode}) : {args.start} → {args.end}")
    print(f"Variables : {', '.join(METEO_VARIABLES)}")
    print(f"Stations : {len(STATIONS)}")

    for station in STATIONS:
        collect_station_meteo(station, args.start, args.end, full=args.full)

    print("\n✓ Collecte météo terminée.")


if __name__ == "__main__":
    main()
