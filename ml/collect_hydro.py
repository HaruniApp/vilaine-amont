"""Collecte des données H (hauteur) et Q (débit) depuis l'API Hydro EauFrance.

L'API retourne au maximum ~1 an de données horaires par requête.
On découpe donc la plage en segments annuels et on concatène.

Usage:
    python collect_hydro.py
    python collect_hydro.py --start 2020-01-01 --end 2025-02-01
    python collect_hydro.py --station J706062001
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
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

MAX_RETRIES = 3
RETRY_DELAY = 5  # secondes
REQUEST_DELAY = 1  # délai entre requêtes pour ne pas surcharger l'API


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


def collect_station(station: dict, start_date: str, end_date: str) -> None:
    """Collecte H et Q pour une station et sauvegarde en CSV."""
    code = station["code"]
    label = station["label"]
    print(f"\n{'='*60}")
    print(f"Station: {label} ({code})")
    print(f"{'='*60}")

    ranges = generate_yearly_ranges(start_date, end_date)

    for variable in ["H", "Q"]:
        all_records = []
        print(f"\n  Variable {variable} — {len(ranges)} segments")

        for seg_start, seg_end in tqdm(ranges, desc=f"  {variable}", leave=False):
            records = fetch_series(code, seg_start, seg_end, variable)
            all_records.extend(records)
            time.sleep(REQUEST_DELAY)

        if not all_records:
            print(f"  ⚠ Aucune donnée {variable} pour {code}")
            continue

        df = pd.DataFrame(all_records)
        df["t"] = pd.to_datetime(df["t"])
        df = df.drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)
        df.columns = ["timestamp", variable.lower()]

        out_path = RAW_DIR / f"{code}_{variable.lower()}.csv"
        df.to_csv(out_path, index=False)
        print(f"  ✓ {variable}: {len(df)} points → {out_path.name}")
        print(f"    Plage: {df['timestamp'].min()} → {df['timestamp'].max()}")


def main():
    parser = argparse.ArgumentParser(description="Collecte des données hydrologiques")
    parser.add_argument("--start", default=COLLECT_START_DATE, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", default=COLLECT_END_DATE, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--station", default=None, help="Code station unique (sinon toutes)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    stations = STATIONS
    if args.station:
        stations = [s for s in STATIONS if s["code"] == args.station]
        if not stations:
            print(f"Station {args.station} inconnue")
            return

    print(f"Collecte Hydro EauFrance : {args.start} → {args.end}")
    print(f"Stations : {len(stations)}")

    for station in stations:
        collect_station(station, args.start, args.end)

    print("\n✓ Collecte terminée.")


if __name__ == "__main__":
    main()
