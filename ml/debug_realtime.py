"""Reproduit le pipeline de forecast.js en Python pour comparer.

Fetch les mêmes données temps réel, applique le même preprocessing que
forecast.js, puis compare avec le preprocessing de prepare_dataset.py.

Usage:
    python debug_realtime.py
"""

import json
from datetime import datetime, timedelta

import numpy as np
import requests
import onnxruntime as ort

from config import DH_CLIP, DQ_CLIP, ONNX_DIR, PROCESSED_DIR, STATION_CODES, STATIONS, FORECAST_HORIZONS

STATIONS_NO_Q = {s["code"] for s in STATIONS if s.get("barrage") or s.get("no_q")}
STATION_COORDS = {s["code"]: (s["lat"], s["lon"]) for s in STATIONS}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}


def fetch_hydro(station_id, start_str, end_str, variable):
    """Fetch comme forecast.js."""
    url = f"https://www.hydro.eaufrance.fr/stationhydro/ajax/{station_id}/series"
    params = {
        "hydro_series[startAt]": start_str,
        "hydro_series[endAt]": end_str,
        "hydro_series[variableType]": "simple_and_interpolated_and_hourly_variable",
        "hydro_series[simpleAndInterpolatedAndHourlyVariable]": variable,
        "hydro_series[statusData]": "most_valid",
        "hydro_series[threshold]": "1",
    }
    headers = {**HEADERS, "Referer": f"https://www.hydro.eaufrance.fr/stationhydro/{station_id}/series"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if not resp.ok:
            print(f"  {station_id} {variable}: HTTP {resp.status_code}")
            return []
        data = resp.json()
        points = data.get("series", {}).get("data", [])
        print(f"  {station_id} {variable}: {len(points)} pts")
        return points
    except Exception as e:
        print(f"  {station_id} {variable}: {e}")
        return []


def fetch_precip(lat, lon, past_hours):
    """Fetch comme forecast.js."""
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&past_hours={past_hours}&forecast_hours=0&hourly=precipitation&timezone=Europe%2FParis")
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        times = data.get("hourly", {}).get("time", [])
        precip = data.get("hourly", {}).get("precipitation", [])
        return [{"t": t, "v": precip[i] if i < len(precip) else 0} for i, t in enumerate(times)]
    except Exception as e:
        print(f"  Precip ({lat},{lon}): {e}")
        return []


def round_to_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)


def build_hourly_index(count, last_hour):
    return [last_hour - timedelta(hours=count - 1 - i) for i in range(count)]


def align_hydro_js_style(points, hourly_ts):
    """Reproduit alignToHourlyGrid de forecast.js : floor to hour, last value wins."""
    ts_map = {}
    for p in points:
        t = datetime.fromisoformat(p["t"].replace("Z", "+00:00")).replace(tzinfo=None)
        t_hour = t.replace(minute=0, second=0, microsecond=0)
        ts_map[t_hour] = p["v"]

    result = [None] * len(hourly_ts)
    last_val = None
    for i, ts in enumerate(hourly_ts):
        val = ts_map.get(ts)
        if val is not None:
            result[i] = val
            last_val = val
        else:
            result[i] = last_val

    # Fill leading nulls
    if result[0] is None:
        first_known = next((v for v in result if v is not None), 0)
        for i in range(len(result)):
            if result[i] is None:
                result[i] = first_known
            else:
                break

    return result


def align_hydro_train_style(points, hourly_ts):
    """Reproduit prepare_dataset.py : floor to hour, MEAN of values in same hour."""
    from collections import defaultdict
    ts_sums = defaultdict(list)
    for p in points:
        t = datetime.fromisoformat(p["t"].replace("Z", "+00:00")).replace(tzinfo=None)
        t_hour = t.replace(minute=0, second=0, microsecond=0)
        ts_sums[t_hour].append(p["v"])

    ts_map = {k: sum(v) / len(v) for k, v in ts_sums.items()}

    result = [None] * len(hourly_ts)
    for i, ts in enumerate(hourly_ts):
        result[i] = ts_map.get(ts)

    # Linear interpolation (max 6h gaps) like prepare_dataset
    import pandas as pd
    s = pd.Series(result)
    s = s.interpolate(method="linear", limit=6)

    # Remaining NaN → forward fill then 0
    s = s.fillna(method="ffill").fillna(0)
    return s.tolist()


def align_precip(points, hourly_ts):
    """Reproduit alignPrecipToGrid de forecast.js."""
    ts_map = {}
    for p in points:
        # Open-Meteo: "2024-01-01T00:00" format (Europe/Paris, no tz suffix)
        t = datetime.fromisoformat(p["t"])
        ts_map[t] = p.get("v", 0) or 0

    return [ts_map.get(ts, 0) for ts in hourly_ts]


def compute_derivative(values):
    d = [0.0]
    for i in range(1, len(values)):
        d.append((values[i] or 0) - (values[i-1] or 0))
    return d


def normalize(value, fmin, fmax):
    if fmin is None or fmax is None or fmax == fmin:
        return 0.0
    return (value - fmin) / (fmax - fmin)


def main():
    with open(ONNX_DIR / "tft_meta.json") as f:
        meta = json.load(f)
    with open(ONNX_DIR / "norm_params.json") as f:
        norm_params = json.load(f)

    input_window = meta["input_window"]  # 72
    now = datetime.utcnow()
    last_hour = round_to_hour(now)
    start_date = last_hour - timedelta(hours=input_window + 1)
    hourly_ts = build_hourly_index(input_window, last_hour)

    start_str = start_date.strftime("%d/%m/%Y")
    end_str = now.strftime("%d/%m/%Y")

    print(f"Grid: {hourly_ts[0].isoformat()} → {hourly_ts[-1].isoformat()} (UTC)")
    print(f"Fetch range: {start_str} → {end_str}")
    print()

    # Fetch hydro data (sequential like fixed forecast.js)
    print("Fetching hydro data...")
    hydro_raw = {}
    for code in STATION_CODES:
        hydro_raw[(code, "H")] = fetch_hydro(code, start_str, end_str, "H")
        if code not in STATIONS_NO_Q:
            hydro_raw[(code, "Q")] = fetch_hydro(code, start_str, end_str, "Q")
        else:
            hydro_raw[(code, "Q")] = []

    # Fetch precip
    print("\nFetching precipitation...")
    precip_raw = {}
    for code in STATION_CODES:
        lat, lon = STATION_COORDS[code]
        precip_raw[code] = fetch_precip(lat, lon, input_window + 2)

    # Build features with BOTH alignment strategies
    target_col = f"{meta['target_station']}_h"

    for style_name, align_fn in [("JS (forward-fill)", align_hydro_js_style),
                                   ("Train (mean+interp)", align_hydro_train_style)]:
        print(f"\n{'='*60}")
        print(f"Preprocessing style: {style_name}")
        print(f"{'='*60}")

        station_data = {}
        for code in STATION_CODES:
            station_data[code] = {
                "h": align_fn(hydro_raw[(code, "H")], hourly_ts),
                "q": align_fn(hydro_raw[(code, "Q")], hourly_ts),
                "precip": align_precip(precip_raw[code], hourly_ts),
            }
            station_data[code]["dh"] = compute_derivative(station_data[code]["h"])
            station_data[code]["dq"] = compute_derivative(station_data[code]["q"])
            # Clip outliers pour cohérence avec l'entraînement
            station_data[code]["dh"] = [max(-DH_CLIP, min(DH_CLIP, v)) if v is not None else None for v in station_data[code]["dh"]]
            station_data[code]["dq"] = [max(-DQ_CLIP, min(DQ_CLIP, v)) if v is not None else None for v in station_data[code]["dq"]]

        # Temporal features (UTC, like fixed forecast.js)
        hour_sin, hour_cos, doy_sin, doy_cos = [], [], [], []
        for dt in hourly_ts:
            hour = dt.hour
            doy = dt.timetuple().tm_yday
            import math
            hour_sin.append(math.sin(2 * math.pi * hour / 24))
            hour_cos.append(math.cos(2 * math.pi * hour / 24))
            doy_sin.append(math.sin(2 * math.pi * doy / 365.25))
            doy_cos.append(math.cos(2 * math.pi * doy / 365.25))

        feature_map = {}
        for code in STATION_CODES:
            feature_map[f"{code}_h"] = station_data[code]["h"]
            feature_map[f"{code}_q"] = station_data[code]["q"]
            feature_map[f"{code}_precip"] = station_data[code]["precip"]
            feature_map[f"{code}_dh"] = station_data[code]["dh"]
            feature_map[f"{code}_dq"] = station_data[code]["dq"]
        feature_map["hour_sin"] = hour_sin
        feature_map["hour_cos"] = hour_cos
        feature_map["doy_sin"] = doy_sin
        feature_map["doy_cos"] = doy_cos

        # Build tensor
        tensor = np.zeros((1, input_window, meta["n_features"]), dtype=np.float32)
        for t in range(input_window):
            for f_idx, fname in enumerate(meta["feature_names"]):
                raw = (feature_map.get(fname) or [0] * input_window)[t] or 0
                np_f = norm_params.get(fname, {})
                tensor[0, t, f_idx] = normalize(raw, np_f.get("min"), np_f.get("max"))

        # Show target H at last timestep
        target_idx = meta["feature_names"].index(target_col)
        h_norm = tensor[0, -1, target_idx]
        np_t = norm_params[target_col]
        h_raw = h_norm * (np_t["max"] - np_t["min"]) + np_t["min"]
        print(f"\n{target_col} at t=71: norm={h_norm:.6f} → raw={h_raw:.1f} mm → {h_raw/1000:.3f} m")

        # Show a few key features at last timestep
        print(f"Key features at t=71:")
        for fname in [f"{meta['target_station']}_q", f"{meta['target_station']}_dh",
                       f"{meta['target_station']}_dq", "hour_sin", "hour_cos"]:
            f_idx = meta["feature_names"].index(fname)
            print(f"  {fname}: norm={tensor[0, -1, f_idx]:.6f}")

        # ONNX inference
        session = ort.InferenceSession(str(ONNX_DIR / "tft.onnx"))
        outputs = session.run(None, {"input": tensor})
        predictions = outputs[0][0]

        print(f"\nPredictions:")
        for i, h in enumerate(FORECAST_HORIZONS):
            raw_mm = predictions[i] * (np_t["max"] - np_t["min"]) + np_t["min"]
            print(f"  t+{h:>2d}h: norm={predictions[i]:.6f} → {raw_mm:.1f} mm → {raw_mm/1000:.3f} m")


if __name__ == "__main__":
    main()
