"""Analyse des erreurs du modèle TFT par régime hydrologique.

Cherche si le modèle sous-estime systématiquement en crue.

Usage:
    python debug_errors.py
"""

import json

import numpy as np
import onnxruntime as ort

from config import PROCESSED_DIR, ONNX_DIR, FORECAST_HORIZONS


def main():
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
    np_t = norm_params[target_col]
    t_min, t_max = np_t["min"], np_t["max"]
    t_range = t_max - t_min
    log_cols = set(meta.get("log_transform_cols", []))
    target_is_log = target_col in log_cols
    target_mode = meta.get("target_mode", "absolute")

    def denorm_to_mm(normalized):
        """Dénormalise: min-max inverse, puis expm1 si log-transform."""
        val = normalized * t_range + t_min
        if target_is_log:
            val = np.expm1(val)
        return val

    # Run ONNX on all test samples
    session = ort.InferenceSession(str(ONNX_DIR / "tft.onnx"))
    all_preds = []
    for i in range(len(X_test)):
        out = session.run(None, {"input": X_test[i:i+1].astype(np.float32)})
        all_preds.append(out[0])
    all_preds = np.concatenate(all_preds)

    # Current H at last timestep (normalized)
    h_last_norm = X_test[:, -1, target_idx]
    h_last_mm = denorm_to_mm(h_last_norm)

    # Errors in mm for each horizon
    print("=" * 70)
    print(f"Distribution des erreurs par horizon (mm) [target_mode={target_mode}]")
    print("=" * 70)
    for i, h in enumerate(FORECAST_HORIZONS):
        if target_mode == "delta":
            pred_mm = denorm_to_mm(all_preds[:, i] + h_last_norm)
            true_mm = denorm_to_mm(y_test[:, i] + h_last_norm)
        else:
            pred_mm = denorm_to_mm(all_preds[:, i])
            true_mm = denorm_to_mm(y_test[:, i])
        err_mm = pred_mm - true_mm
        abs_err = np.abs(err_mm)
        print(f"\n  t+{h}h:")
        print(f"    Biais moyen : {np.mean(err_mm):+.1f} mm")
        print(f"    MAE         : {np.mean(abs_err):.1f} mm")
        print(f"    Médiane |e| : {np.median(abs_err):.1f} mm")
        print(f"    P90 |e|     : {np.percentile(abs_err, 90):.1f} mm")
        print(f"    P95 |e|     : {np.percentile(abs_err, 95):.1f} mm")
        print(f"    P99 |e|     : {np.percentile(abs_err, 99):.1f} mm")
        print(f"    Max |e|     : {np.max(abs_err):.1f} mm")

    # --- Analyse par régime hydrologique ---
    print(f"\n{'=' * 70}")
    print("Erreur t+1h par niveau d'eau actuel")
    print("=" * 70)

    if target_mode == "delta":
        pred_t1_mm = denorm_to_mm(all_preds[:, 0] + h_last_norm)
        true_t1_mm = denorm_to_mm(y_test[:, 0] + h_last_norm)
    else:
        pred_t1_mm = denorm_to_mm(all_preds[:, 0])
        true_t1_mm = denorm_to_mm(y_test[:, 0])
    err_t1_mm = pred_t1_mm - true_t1_mm

    # Bins par niveau H actuel
    bins = [0, 200, 400, 600, 800, 1000, 1200, 1500, 2000, 3000]
    print(f"\n  {'H actuel (mm)':>20s} | {'N':>6s} | {'Biais':>8s} | {'MAE':>8s} | {'P95|e|':>8s} | {'H prédit moy':>12s} | {'H vrai moy':>12s}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}")
    for j in range(len(bins) - 1):
        mask = (h_last_mm >= bins[j]) & (h_last_mm < bins[j+1])
        n = mask.sum()
        if n == 0:
            continue
        errs = err_t1_mm[mask]
        preds = pred_t1_mm[mask]
        trues = true_t1_mm[mask]
        label = f"{bins[j]:>4d}–{bins[j+1]:>4d}"
        print(f"  {label:>20s} | {n:>6d} | {np.mean(errs):>+8.1f} | {np.mean(np.abs(errs)):>8.1f} | {np.percentile(np.abs(errs), 95):>8.1f} | {np.mean(preds):>12.1f} | {np.mean(trues):>12.1f}")

    # --- Analyse spécifique : H > 1500 mm (comme maintenant) ---
    print(f"\n{'=' * 70}")
    print("Focus : échantillons avec H actuel > 1500 mm (crue)")
    print("=" * 70)

    mask_crue = h_last_mm > 1500
    n_crue = mask_crue.sum()
    print(f"\n  Nombre d'échantillons : {n_crue} / {len(X_test)} ({100*n_crue/len(X_test):.1f}%)")

    if n_crue > 0:
        for i, h in enumerate(FORECAST_HORIZONS):
            if target_mode == "delta":
                pred = denorm_to_mm(all_preds[mask_crue, i] + h_last_norm[mask_crue])
                true = denorm_to_mm(y_test[mask_crue, i] + h_last_norm[mask_crue])
            else:
                pred = denorm_to_mm(all_preds[mask_crue, i])
                true = denorm_to_mm(y_test[mask_crue, i])
            err = pred - true
            print(f"\n  t+{h}h (N={n_crue}):")
            print(f"    Biais moyen : {np.mean(err):+.1f} mm")
            print(f"    MAE         : {np.mean(np.abs(err)):.1f} mm")
            print(f"    P95 |e|     : {np.percentile(np.abs(err), 95):.1f} mm")
            print(f"    H prédit moy: {np.mean(pred):.1f} mm ({np.mean(pred)/1000:.3f} m)")
            print(f"    H vrai moy  : {np.mean(true):.1f} mm ({np.mean(true)/1000:.3f} m)")

    # --- Tendance (crue vs décrue) ---
    print(f"\n{'=' * 70}")
    print("Erreur t+1h par tendance (crue vs décrue)")
    print("=" * 70)

    # dH basé sur H dénormalisé (en mm) entre les 2 derniers timesteps
    h_prev_mm = denorm_to_mm(X_test[:, -2, target_idx])
    dh_mm = h_last_mm - h_prev_mm

    for label, mask in [("Décrue (dH < -10 mm/h)", dh_mm < -10),
                         ("Stable (|dH| ≤ 10 mm/h)", np.abs(dh_mm) <= 10),
                         ("Crue (dH > 10 mm/h)", dh_mm > 10)]:
        n = mask.sum()
        if n == 0:
            continue
        errs = err_t1_mm[mask]
        print(f"\n  {label} (N={n}):")
        print(f"    Biais : {np.mean(errs):+.1f} mm")
        print(f"    MAE   : {np.mean(np.abs(errs)):.1f} mm")
        print(f"    P95   : {np.percentile(np.abs(errs), 95):.1f} mm")

    # --- Les pires prédictions ---
    print(f"\n{'=' * 70}")
    print("Top 10 pires erreurs t+1h")
    print("=" * 70)

    abs_err_t1 = np.abs(err_t1_mm)
    worst = np.argsort(abs_err_t1)[-10:][::-1]
    print(f"\n  {'Timestamp':>25s} | {'H actuel':>10s} | {'H prédit':>10s} | {'H vrai':>10s} | {'Erreur':>10s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for idx in worst:
        ts = str(ts_test[idx])[:19]
        h_act = h_last_mm[idx]
        h_pred = pred_t1_mm[idx]
        h_true = true_t1_mm[idx]
        err = err_t1_mm[idx]
        print(f"  {ts:>25s} | {h_act:>10.1f} | {h_pred:>10.1f} | {h_true:>10.1f} | {err:>+10.1f}")


if __name__ == "__main__":
    main()
