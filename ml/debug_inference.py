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


def denorm(val, np_f):
    """Dénormalise une valeur avec les paramètres min/max."""
    return val * (np_f["max"] - np_f["min"]) + np_f["min"]


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
    t_range = np_target["max"] - np_target["min"]
    target_mode = meta.get("target_mode", "absolute")

    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_test.shape[2]} ({len(feature_names)} names)")
    print(f"Target mode: {target_mode}")
    print(f"Sample index: {args.sample}")
    print(f"Timestamp: {ts_test[args.sample]}")
    print()

    # Extraire l'échantillon
    sample = X_test[args.sample]  # shape (72, n_features)
    target = y_test[args.sample]  # shape (5,)

    # Valeurs du dernier pas de temps
    print("=" * 60)
    print("Features au dernier pas de temps (t=71) — normalisées:")
    print("=" * 60)
    for i, fname in enumerate(feature_names):
        val = sample[71, i]
        np_f = norm_params.get(fname, {})
        fmin = np_f.get("min")
        fmax = np_f.get("max")
        if fmin is not None and fmax is not None and fmax != fmin:
            raw = denorm(val, np_f)
            print(f"  {fname:30s}: norm={val:.6f}  raw={raw:.1f}")
        else:
            print(f"  {fname:30s}: norm={val:.6f}  (no denorm)")

    # Valeur H cible au dernier pas
    h_norm = sample[71, target_idx]
    h_mm = denorm(h_norm, np_target)
    print(f"\n{target_col} au t=71: norm={h_norm:.6f} → {h_mm:.1f} mm → {h_mm/1000:.3f} m")

    # Targets attendus
    print(f"\nTargets attendus (y_test, mode={target_mode}):")
    for i, h in enumerate(FORECAST_HORIZONS):
        if target_mode == "delta":
            delta_mm = target[i] * t_range
            abs_mm = h_mm + delta_mm
            print(f"  t+{h}h: delta_norm={target[i]:.6f} → delta={delta_mm:+.1f} mm → H={abs_mm:.1f} mm ({abs_mm/1000:.3f} m)")
        else:
            raw_mm = denorm(target[i], np_target)
            print(f"  t+{h}h: norm={target[i]:.6f} → {raw_mm:.1f} mm → {raw_mm/1000:.3f} m")

    # Inférence ONNX
    print(f"\n{'=' * 60}")
    print("Inférence ONNX")
    print("=" * 60)

    onnx_path = str(ONNX_DIR / "tft.onnx")
    session = ort.InferenceSession(onnx_path)

    input_tensor = sample[np.newaxis, :, :].astype(np.float32)
    outputs = session.run(None, {"input": input_tensor})
    predictions = outputs[0][0]  # shape (5,)

    print(f"Raw model output: [{', '.join(f'{v:.6f}' for v in predictions)}]")
    print()
    for i, h in enumerate(FORECAST_HORIZONS):
        if target_mode == "delta":
            pred_delta_mm = predictions[i] * t_range
            true_delta_mm = target[i] * t_range
            pred_abs_mm = h_mm + pred_delta_mm
            true_abs_mm = h_mm + true_delta_mm
            err_mm = pred_abs_mm - true_abs_mm
            print(f"  t+{h:>2d}h: delta={pred_delta_mm:+.1f} mm → H={pred_abs_mm:.1f} mm ({pred_abs_mm/1000:.3f} m)  (err={err_mm:+.1f} mm)")
        else:
            pred_mm = denorm(predictions[i], np_target)
            true_mm = denorm(target[i], np_target)
            err_mm = pred_mm - true_mm
            print(f"  t+{h:>2d}h: norm={predictions[i]:.6f} → {pred_mm:.1f} mm → {pred_mm/1000:.3f} m  (err={err_mm:+.1f} mm)")

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

    # H actuel normalisé pour chaque sample (dernier timestep)
    h_last_norm = X_test[:, -1, target_idx]

    for i, h in enumerate(FORECAST_HORIZONS):
        if target_mode == "delta":
            # Convertir en absolu mm pour des métriques significatives
            pred_abs_mm = (h_last_norm + all_preds[:, i]) * t_range + np_target["min"]
            true_abs_mm = (h_last_norm + y_test[:, i]) * t_range + np_target["min"]
        else:
            pred_abs_mm = all_preds[:, i] * t_range + np_target["min"]
            true_abs_mm = y_test[:, i] * t_range + np_target["min"]

        rmse = float(np.sqrt(np.mean((pred_abs_mm - true_abs_mm) ** 2)))
        mae = float(np.mean(np.abs(pred_abs_mm - true_abs_mm)))
        bias = float(np.mean(pred_abs_mm - true_abs_mm))
        ss_res = np.sum((true_abs_mm - pred_abs_mm) ** 2)
        ss_tot = np.sum((true_abs_mm - np.mean(true_abs_mm)) ** 2)
        nse = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        print(f"  t+{h:>2d}h: NSE={nse:.4f}  RMSE={rmse:.1f}mm  MAE={mae:.1f}mm  bias={bias:+.1f}mm")


if __name__ == "__main__":
    main()
