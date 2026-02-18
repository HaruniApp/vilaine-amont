"""Évaluation comparative de tous les modèles entraînés.

Compare XGBoost, LSTM et TFT sur les métriques RMSE, MAE, NSE,
avec un focus particulier sur les épisodes de crue.

Usage:
    python evaluate.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CHECKPOINTS_DIR, FORECAST_HORIZONS, PROCESSED_DIR, TARGET_STATION


def nash_sutcliffe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _normalize_keys(metrics: dict) -> dict:
    """Normalise les clés rmse_mm/mae_mm → rmse/mae."""
    return {
        "rmse": metrics.get("rmse") or metrics.get("rmse_mm", 0),
        "mae": metrics.get("mae") or metrics.get("mae_mm", 0),
        "nse": metrics["nse"],
    }


def _norm_to_mm(metrics: dict, t_range: float) -> dict:
    """Convertit les RMSE/MAE normalisées en mm."""
    return {
        "rmse": metrics["rmse"] * t_range,
        "mae": metrics["mae"] * t_range,
        "nse": metrics["nse"],
    }


def load_results() -> dict:
    """Charge les résultats de tous les modèles disponibles, tout en mm."""
    # Charger la plage de normalisation de la cible pour convertir en mm
    norm_path = PROCESSED_DIR / "norm_params.json"
    t_range = 1.0
    if norm_path.exists():
        with open(norm_path) as f:
            norm_params = json.load(f)
        target_col = f"{TARGET_STATION}_h"
        if target_col in norm_params:
            np_t = norm_params[target_col]
            if np_t["min"] is not None and np_t["max"] is not None:
                t_range = np_t["max"] - np_t["min"]

    results = {}

    xgb_path = CHECKPOINTS_DIR / "xgboost_results.json"
    if xgb_path.exists():
        with open(xgb_path) as f:
            data = json.load(f)
        results["XGBoost"] = {}
        for k, v in data.items():
            results["XGBoost"][k] = {
                "val": _norm_to_mm({"rmse": v["val_rmse"], "mae": v["val_mae"], "nse": v["val_nse"]}, t_range),
                "test": _norm_to_mm({"rmse": v["test_rmse"], "mae": v["test_mae"], "nse": v["test_nse"]}, t_range),
            }

    lstm_path = CHECKPOINTS_DIR / "lstm_results.json"
    if lstm_path.exists():
        with open(lstm_path) as f:
            data = json.load(f)
        results["LSTM"] = {}
        for k in data.get("test", {}):
            results["LSTM"][k] = {
                "val": _norm_to_mm(data["val"][k], t_range),
                "test": _norm_to_mm(data["test"][k], t_range),
            }

    tft_path = CHECKPOINTS_DIR / "tft_results.json"
    if tft_path.exists():
        with open(tft_path) as f:
            data = json.load(f)
        results["TFT"] = {}
        for k in data.get("test", {}):
            results["TFT"][k] = {
                "val": _normalize_keys(data["val"][k]),
                "test": _normalize_keys(data["test"][k]),
            }

    return results


def print_comparison(results: dict) -> None:
    """Affiche un tableau comparatif."""
    models = list(results.keys())
    if not models:
        print("Aucun résultat trouvé.")
        return

    horizons = list(results[models[0]].keys())

    # Tableau par horizon
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"  {horizon}")
        print(f"{'='*60}")
        print(f"  {'Modèle':<12} {'NSE (test)':>12} {'RMSE (mm)':>12} {'MAE (mm)':>11}")
        print(f"  {'-'*47}")

        best_nse = -999
        best_model = ""
        for model in models:
            if horizon in results[model]:
                m = results[model][horizon]["test"]
                if m["nse"] > best_nse:
                    best_nse = m["nse"]
                    best_model = model
                marker = ""
                print(f"  {model:<12} {m['nse']:>12.4f} {m['rmse']:>12.1f} {m['mae']:>11.1f}{marker}")

        if best_model:
            print(f"  → Meilleur : {best_model}")


def plot_comparison(results: dict) -> None:
    """Génère un graphique comparatif NSE par horizon."""
    models = list(results.keys())
    if len(models) < 2:
        return

    horizons = sorted(results[models[0]].keys(), key=lambda x: int(x.split("+")[1].rstrip("h")))
    horizon_labels = [h.replace("t+", "") for h in horizons]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"XGBoost": "#2563eb", "LSTM": "#0d9488", "TFT": "#d97706"}

    for ax, metric, title in zip(axes, ["nse", "rmse", "mae"], ["NSE ↑", "RMSE ↓ (mm)", "MAE ↓ (mm)"]):
        for model in models:
            values = [results[model][h]["test"][metric] for h in horizons]
            ax.plot(horizon_labels, values, "o-", label=model, color=colors.get(model, "#666"))
        ax.set_xlabel("Horizon")
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = CHECKPOINTS_DIR / "comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nGraphique → {out_path}")


def identify_best_model(results: dict) -> str | None:
    """Identifie le meilleur modèle global (NSE moyen sur test)."""
    models = list(results.keys())
    if not models:
        return None

    avg_nse = {}
    for model in models:
        nses = [results[model][h]["test"]["nse"] for h in results[model]]
        avg_nse[model] = np.mean(nses)

    best = max(avg_nse, key=avg_nse.get)
    print(f"\n{'='*60}")
    print(f"  MEILLEUR MODÈLE : {best} (NSE moyen = {avg_nse[best]:.4f})")
    print(f"{'='*60}")

    for model, nse in sorted(avg_nse.items(), key=lambda x: -x[1]):
        print(f"  {model:<12} NSE moyen = {nse:.4f}")

    # Sauvegarder
    with open(CHECKPOINTS_DIR / "best_model.json", "w") as f:
        json.dump({"best_model": best, "avg_nse": avg_nse}, f, indent=2)

    return best


def main():
    print("Évaluation comparative des modèles")
    print("=" * 60)

    results = load_results()
    print(f"Modèles trouvés : {list(results.keys())}")

    print_comparison(results)
    plot_comparison(results)
    best = identify_best_model(results)

    if best:
        print(f"\n→ Utiliser export_onnx.py --model {best.lower()} pour exporter le meilleur modèle")


if __name__ == "__main__":
    main()
