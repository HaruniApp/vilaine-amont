"""Collecte des données H (hauteur) et Q (débit) depuis l'API Hydro EauFrance.

L'API retourne au maximum ~1 an de données horaires par requête.
On découpe donc la plage en segments annuels et on concatène.

Mode incrémental : si un CSV existe déjà, ne récupère que les données
postérieures au dernier timestamp présent.

Usage:
    python collect_hydro.py
    python collect_hydro.py --start 2020-01-01
    python collect_hydro.py --station J706062001
    python collect_hydro.py --full   # ignore les CSV existants
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
    HYDRO_BASE_URL,
    RAW_DIR,
    STATIONS,
    STATIONS_NO_Q,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

MAX_RETRIES = 3
RETRY_DELAY = 5  # secondes
REQUEST_DELAY = 2  # délai entre requêtes pour ne pas surcharger l'API


def fetch_series(station_id: str, start: str, end: str, variable: str) -> list[dict]:
    """Récupère une série H ou Q pour une station sur une plage de dates."""
    url = HYDRO_BASE_URL.format(station_id=station_id)
    params = {
        "hydro_series[startAt]": start,
        "hydro_series[endAt]": end,
        "hydro_series[variableType]": "simple_and_interpolated_and_hourly_variable",
        "hydro_series[simpleAndInterpolatedAndHourlyVariable]": variable,
        "hydro_series[statusData]": "most_valid",
        "hydro_series[threshold]": "1",
    }
    headers = {
        **HEADERS,
        "Referer": f"https://www.hydro.eaufrance.fr/stationhydro/{station_id}/series",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()

            series = data.get("series", {})
            if not series:
                return []
            raw_data = series.get("data", [])
            return [{"t": pt["t"], "v": pt["v"]} for pt in raw_data if pt.get("v") is not None]

        except (requests.RequestException, ValueError) as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} pour {station_id} {variable} ({e})")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  ÉCHEC {station_id} {variable} {start}→{end}: {e}")
                return []


def to_hydro_date(d: datetime) -> str:
    """Convertit en format dd/MM/yyyy attendu par l'API Hydro EauFrance."""
    return d.strftime("%d/%m/%Y")


def generate_yearly_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Découpe une plage de dates en segments d'environ 1 an (format dd/MM/yyyy)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    ranges = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=365), end)
        ranges.append((to_hydro_date(current), to_hydro_date(chunk_end)))
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


def collect_station(station: dict, start_date: str, end_date: str, full: bool = False) -> None:
    """Collecte H et Q pour une station et sauvegarde en CSV."""
    code = station["code"]
    label = station["label"]
    print(f"\n{'='*60}")
    print(f"Station: {label} ({code})")
    print(f"{'='*60}")

    variables = ["H"] if code in STATIONS_NO_Q else ["H", "Q"]

    for variable in variables:
        out_path = RAW_DIR / f"{code}_{variable.lower()}.csv"
        existing_df = None
        effective_start = start_date

        # Mode incrémental : reprendre après le dernier timestamp
        if not full:
            last_ts = get_last_timestamp(out_path)
            if last_ts is not None:
                if last_ts >= end_date:
                    print(f"\n  {variable}: déjà à jour (jusqu'au {last_ts}), skip")
                    continue
                effective_start = last_ts
                existing_df = pd.read_csv(out_path, parse_dates=["timestamp"])
                print(f"\n  {variable}: incrémental depuis {last_ts}")

        ranges = generate_yearly_ranges(effective_start, end_date)
        all_records = []
        print(f"  Variable {variable} — {len(ranges)} segments")

        for seg_start, seg_end in tqdm(ranges, desc=f"  {variable}", leave=False):
            records = fetch_series(code, seg_start, seg_end, variable)
            all_records.extend(records)
            time.sleep(REQUEST_DELAY)

        if not all_records and existing_df is None:
            print(f"  ⚠ Aucune donnée {variable} pour {code}")
            continue

        new_df = pd.DataFrame(all_records)
        if not new_df.empty:
            new_df["t"] = pd.to_datetime(new_df["t"])
            new_df.columns = ["timestamp", variable.lower()]

        # Fusionner avec les données existantes
        if existing_df is not None and not new_df.empty:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        elif existing_df is not None:
            df = existing_df
        else:
            df = new_df

        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

        df.to_csv(out_path, index=False)
        print(f"  ✓ {variable}: {len(df)} points → {out_path.name}")
        print(f"    Plage: {df['timestamp'].min()} → {df['timestamp'].max()}")


def main():
    parser = argparse.ArgumentParser(description="Collecte des données hydrologiques")
    parser.add_argument("--start", default=COLLECT_START_DATE, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", default=COLLECT_END_DATE, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--station", default=None, help="Code station unique (sinon toutes)")
    parser.add_argument("--full", action="store_true", help="Collecte complète (ignore les CSV existants)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    stations = STATIONS
    if args.station:
        stations = [s for s in STATIONS if s["code"] == args.station]
        if not stations:
            print(f"Station {args.station} inconnue")
            return

    mode = "complète" if args.full else "incrémentale"
    print(f"Collecte Hydro EauFrance ({mode}) : {args.start} → {args.end}")
    print(f"Stations : {len(stations)}")

    for station in stations:
        collect_station(station, args.start, args.end, full=args.full)

    print("\n✓ Collecte terminée.")


if __name__ == "__main__":
    main()
