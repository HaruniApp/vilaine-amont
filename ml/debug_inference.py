"""Compare l'inférence ONNX Python vs les valeurs attendues du test set.

Prend un échantillon du test set, fait l'inférence ONNX, et affiche
les valeurs pour comparaison avec forecast.js.

Usage:
    python debug_inference.py
    python debug_inference.py --sample 0      # premier échantillon du test
    python debug_inference.py --sample -1     # dernier échantillon du test
"""

import argparse
import json

import numpy as np
import onnxruntime as ort

from config import PROCESSED_DIR, ONNX_DIR, FORECAST_HORIZONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Index de l'échantillon test")
    args = parser.parse_args()

    # Charger les données test
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    ts_test = np.load(PROCESSED_DIR / "ts_test.npy")

    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)

    with open(PROCESSED_DIR / "norm_params.json") as f:
        norm_params = json.load(f)

    feature_names = meta["feature_names"]
    target_col = f"{meta['target_station']}_h"
    target_idx = feature_names.index(target_col)
    np_target = norm_params[target_col]

    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_test.shape[2]} ({len(feature_names)} names)")
    print(f"Sample index: {args.sample}")
    print(f"Timestamp: {ts_test[args.sample]}")
    print()

    # Extraire l'échantillon
    sample = X_test[args.sample]  # shape (72, n_features)
    target = y_test[args.sample]  # shape (5,)

    # Valeurs du dernier pas de temps
    print("=" * 60)
    print(f"Features au dernier pas de temps (t=71) — normalisées:")
    print("=" * 60)
    for i, fname in enumerate(feature_names):
        val = sample[71, i]
        # Dénormaliser
        np_f = norm_params.get(fname, {})
        fmin = np_f.get("min")
        fmax = np_f.get("max")
        if fmin is not None and fmax is not None and fmax != fmin:
            raw = val * (fmax - fmin) + fmin
            print(f"  {fname:30s}: norm={val:.6f}  raw={raw:.1f}")
        else:
            print(f"  {fname:30s}: norm={val:.6f}  (no denorm)")

    # Valeur H cible au dernier pas
    h_norm = sample[71, target_idx]
    h_raw = h_norm * (np_target["max"] - np_target["min"]) + np_target["min"]
    print(f"\n{target_col} au t=71: norm={h_norm:.6f} → raw={h_raw:.1f} mm → {h_raw/1000:.3f} m")

    # Targets attendus
    print(f"\nTargets attendus (normalisés, from y_test):")
    for i, h in enumerate(FORECAST_HORIZONS):
        raw_mm = target[i] * (np_target["max"] - np_target["min"]) + np_target["min"]
        print(f"  t+{h}h: norm={target[i]:.6f} → {raw_mm:.1f} mm → {raw_mm/1000:.3f} m")

    # Inférence ONNX
    print(f"\n{'=' * 60}")
    print("Inférence ONNX")
    print("=" * 60)

    onnx_path = str(ONNX_DIR / "tft.onnx")
    session = ort.InferenceSession(onnx_path)

    input_tensor = sample[np.newaxis, :, :].astype(np.float32)  # (1, 72, 51)
    outputs = session.run(None, {"input": input_tensor})
    predictions = outputs[0][0]  # shape (5,)

    print(f"Raw model output: [{', '.join(f'{v:.6f}' for v in predictions)}]")
    print()
    for i, h in enumerate(FORECAST_HORIZONS):
        raw_mm = predictions[i] * (np_target["max"] - np_target["min"]) + np_target["min"]
        err_mm = raw_mm - (target[i] * (np_target["max"] - np_target["min"]) + np_target["min"])
        print(f"  t+{h:>2d}h: norm={predictions[i]:.6f} → {raw_mm:.1f} mm → {raw_mm/1000:.3f} m  (err={err_mm:+.1f} mm)")

    # Stats globales sur tout le test set
    print(f"\n{'=' * 60}")
    print("Stats sur tout le test set (ONNX)")
    print("=" * 60)

    all_preds = []
    for i in range(len(X_test)):
        single = X_test[i:i+1].astype(np.float32)
        out = session.run(None, {"input": single})
        all_preds.append(out[0])
    all_preds = np.concatenate(all_preds)

    for i, h in enumerate(FORECAST_HORIZONS):
        pred = all_preds[:, i]
        true = y_test[:, i]
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        mae = np.mean(np.abs(pred - true))
        bias = np.mean(pred - true)
        bias_mm = bias * (np_target["max"] - np_target["min"])
        rmse_mm = rmse * (np_target["max"] - np_target["min"])
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        nse = 1 - ss_res / ss_tot
        print(f"  t+{h:>2d}h: NSE={nse:.4f}  RMSE={rmse_mm:.1f}mm  MAE(norm)={mae:.6f}  bias={bias_mm:+.1f}mm")


if __name__ == "__main__":
    main()
