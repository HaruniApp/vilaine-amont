"""Reproduit le pipeline de forecast.js en Python pour comparer.

Fetch les mêmes données temps réel, applique le même preprocessing que
forecast.js, puis lance l'inférence ONNX et compare.

Usage:
    python debug_realtime.py
"""

import json
import math
from datetime import datetime, timedelta

import numpy as np
import requests
import onnxruntime as ort

from config import (
    BARRAGE_CODES,
    DH_CLIP, DQ_CLIP,
    FUTURE_PRECIP_HOURS,
    ONNX_DIR,
    STATION_CODES, STATIONS,
)

STATIONS_NO_Q = {s["code"] for s in STATIONS if s.get("barrage") or s.get("no_q")}
STATION_COORDS = {s["code"]: (s["lat"], s["lon"]) for s in STATIONS}

VARS_PER_STATION = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}


def fetch_hydro(station_id, start_str, end_str, variable):
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


def fetch_precip(lat, lon, past_hours, forecast_hours):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&past_hours={past_hours}&forecast_hours={forecast_hours}"
           f"&hourly=precipitation&timezone=Europe%2FParis")
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


def align_hydro(points, hourly_ts):
    """Forward-fill alignment (matches forecast.js)."""
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

    if result[0] is None:
        first_known = next((v for v in result if v is not None), 0)
        for i in range(len(result)):
            if result[i] is None:
                result[i] = first_known
            else:
                break

    return result


def align_precip(points, hourly_ts):
    ts_map = {}
    for p in points:
        t = datetime.fromisoformat(p["t"])
        ts_map[t] = p.get("v", 0) or 0
    return [ts_map.get(ts, 0) for ts in hourly_ts]


def compute_central_derivative(values):
    d = [0.0]
    for i in range(1, len(values) - 1):
        prev = values[i - 1] or 0
        nxt = values[i + 1] or 0
        d.append((nxt - prev) / 2.0)
    d.append(0.0)
    return d


def normalize(value, fmin, fmax):
    if fmin is None or fmax is None or fmax == fmin:
        return 0.0
    return (value - fmin) / (fmax - fmin)


def main():
    with open(ONNX_DIR / "station_attn_meta.json") as f:
        meta = json.load(f)
    with open(ONNX_DIR / "norm_params.json") as f:
        norm_params = json.load(f)

    input_window = meta["input_window"]  # 72
    future_hours = meta["future_precip_hours"]  # 6
    n_stations = meta["n_stations"]
    now = datetime.utcnow()
    last_hour = round_to_hour(now)
    hourly_ts = build_hourly_index(input_window, last_hour)

    # Future timestamps
    future_ts = [last_hour + timedelta(hours=h) for h in range(1, future_hours + 1)]
    all_precip_ts = hourly_ts + future_ts

    start_date = last_hour - timedelta(hours=input_window + 1)
    start_str = start_date.strftime("%d/%m/%Y")
    end_str = now.strftime("%d/%m/%Y")

    print(f"Grid: {hourly_ts[0].isoformat()} → {hourly_ts[-1].isoformat()} (UTC)")
    print(f"Future precip: {future_ts[0].isoformat()} → {future_ts[-1].isoformat()}")
    print()

    # Fetch hydro data
    print("Fetching hydro data...")
    hydro_raw = {}
    for code in STATION_CODES:
        hydro_raw[(code, "H")] = fetch_hydro(code, start_str, end_str, "H")
        if code not in STATIONS_NO_Q:
            hydro_raw[(code, "Q")] = fetch_hydro(code, start_str, end_str, "Q")
        else:
            hydro_raw[(code, "Q")] = []

    # Fetch precip (past + future)
    print("\nFetching precipitation (past + forecast)...")
    precip_raw = {}
    for code in STATION_CODES:
        lat, lon = STATION_COORDS[code]
        precip_raw[code] = fetch_precip(lat, lon, input_window + 2, future_hours + 1)

    # Organize data
    station_data = {}
    for code in STATION_CODES:
        h_arr = align_hydro(hydro_raw[(code, "H")], hourly_ts)
        q_arr = align_hydro(hydro_raw[(code, "Q")], hourly_ts)
        all_precip = align_precip(precip_raw[code], all_precip_ts)
        precip_past = all_precip[:input_window]
        precip_future = all_precip[input_window:input_window + future_hours]
        station_data[code] = {"h": h_arr, "q": q_arr, "precip": precip_past, "precip_future": precip_future}

    # Derivatives (central) + clip
    for code in STATION_CODES:
        station_data[code]["dh"] = [max(-DH_CLIP, min(DH_CLIP, v)) for v in compute_central_derivative(station_data[code]["h"])]
        station_data[code]["dq"] = [max(-DQ_CLIP, min(DQ_CLIP, v)) for v in compute_central_derivative(station_data[code]["q"])]

    # Release feature for barrages
    for code in BARRAGE_CODES:
        dh = station_data[code]["dh"]
        precip = station_data[code]["precip"]
        station_data[code]["release"] = [
            max(0, -(dh[i] or 0)) * (1 - min(1, precip[i] or 0))
            for i in range(input_window)
        ]

    # --- Build past tensor (padded) ---
    n_features_padded = n_stations * VARS_PER_STATION
    past_tensor = np.zeros((1, input_window, n_features_padded), dtype=np.float32)

    for t in range(input_window):
        for s, code in enumerate(STATION_CODES):
            base = s * VARS_PER_STATION
            sd = station_data[code]
            np_h = norm_params.get(f"{code}_h", {})
            np_precip = norm_params.get(f"{code}_precip", {})
            np_dh = norm_params.get(f"{code}_dh", {})

            past_tensor[0, t, base + 0] = normalize(sd["h"][t] or 0, np_h.get("min"), np_h.get("max"))
            past_tensor[0, t, base + 2] = normalize(sd["precip"][t] or 0, np_precip.get("min"), np_precip.get("max"))
            past_tensor[0, t, base + 3] = normalize(sd["dh"][t] or 0, np_dh.get("min"), np_dh.get("max"))

            if code not in STATIONS_NO_Q:
                np_q = norm_params.get(f"{code}_q", {})
                np_dq = norm_params.get(f"{code}_dq", {})
                past_tensor[0, t, base + 1] = normalize(sd["q"][t] or 0, np_q.get("min"), np_q.get("max"))
                past_tensor[0, t, base + 4] = normalize(sd["dq"][t] or 0, np_dq.get("min"), np_dq.get("max"))
            elif code in BARRAGE_CODES:
                np_release = norm_params.get(f"{code}_release", {})
                past_tensor[0, t, base + 4] = normalize(sd.get("release", [0]*input_window)[t], np_release.get("min"), np_release.get("max"))

    # --- Build future precip tensor ---
    future_precip_tensor = np.zeros((1, n_stations * future_hours), dtype=np.float32)
    for s, code in enumerate(STATION_CODES):
        np_precip = norm_params.get(f"{code}_precip", {})
        pf = station_data[code]["precip_future"]
        for h in range(future_hours):
            val = pf[h] if h < len(pf) else 0
            future_precip_tensor[0, s * future_hours + h] = normalize(val, np_precip.get("min"), np_precip.get("max"))

    # --- Run ONNX inference ---
    session = ort.InferenceSession(str(ONNX_DIR / "station_attn.onnx"))
    outputs = session.run(None, {
        "past_input": past_tensor,
        "future_precip": future_precip_tensor,
    })
    predictions = outputs[0][0]

    print(f"\nPredictions ({len(predictions)} outputs):")

    output_map = meta["output_map"]
    forecast_horizons = meta["forecast_horizons"]

    for code in STATION_CODES:
        om = output_map[code]
        np_h = norm_params.get(f"{code}_h", {})
        h_range = (np_h.get("max", 0) or 0) - (np_h.get("min", 0) or 0)
        last_h_norm = normalize(station_data[code]["h"][-1] or 0, np_h.get("min"), np_h.get("max"))

        print(f"\n  {code} H:")
        for j, h in enumerate(forecast_horizons):
            delta = predictions[om["h_start"] + j]
            raw_mm = (last_h_norm + delta) * h_range + (np_h.get("min", 0) or 0)
            print(f"    t+{h:>2d}h: delta={delta:.6f} → {raw_mm:.1f} mm → {raw_mm/1000:.3f} m")

        if "q_start" in om:
            np_q = norm_params.get(f"{code}_q", {})
            q_range = (np_q.get("max", 0) or 0) - (np_q.get("min", 0) or 0)
            last_q_norm = normalize(station_data[code]["q"][-1] or 0, np_q.get("min"), np_q.get("max"))

            print(f"  {code} Q:")
            for j, h in enumerate(forecast_horizons):
                delta = predictions[om["q_start"] + j]
                raw_ls = (last_q_norm + delta) * q_range + (np_q.get("min", 0) or 0)
                print(f"    t+{h:>2d}h: delta={delta:.6f} → {raw_ls:.0f} L/s → {raw_ls/1000:.3f} m³/s")


if __name__ == "__main__":
    main()
