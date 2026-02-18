"""Évaluation du modèle Station-Attention.

Affiche les métriques NSE, RMSE, MAE par station et horizon,
avec un focus sur la station cible Châteaubourg.

Usage:
    python evaluate.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CHECKPOINTS_DIR, FORECAST_HORIZONS, STATION_CODES, STATIONS_NO_Q


def main():
    print("Évaluation Station-Attention")
    print("=" * 60)

    results_path = CHECKPOINTS_DIR / "station_attn_results.json"
    if not results_path.exists():
        print("Aucun résultat trouvé. Lancer train_station_attention.py d'abord.")
        return

    with open(results_path) as f:
        results = json.load(f)

    for split in ["val", "test"]:
        if split not in results:
            continue

        print(f"\n{'='*60}")
        print(f"  {split.upper()}")
        print(f"{'='*60}")

        # Per-station H summary
        print(f"\n  {'Station':<20} ", end="")
        for h in FORECAST_HORIZONS:
            print(f"{'t+' + str(h) + 'h':>10}", end="")
        print()
        print(f"  {'-'*20} " + "-" * 10 * len(FORECAST_HORIZONS))

        avg_nse_per_horizon = {h: [] for h in FORECAST_HORIZONS}

        for code in STATION_CODES:
            label = code[:10]
            print(f"  {label:<20} ", end="")
            for h in FORECAST_HORIZONS:
                key = f"{code}_H_t+{h}h"
                if key in results[split]:
                    nse = results[split][key]["nse"]
                    avg_nse_per_horizon[h].append(nse)
                    print(f"{nse:>10.4f}", end="")
                else:
                    print(f"{'—':>10}", end="")
            print()

        # Average
        print(f"  {'AVG H':<20} ", end="")
        for h in FORECAST_HORIZONS:
            vals = avg_nse_per_horizon[h]
            if vals:
                print(f"{np.mean(vals):>10.4f}", end="")
        print()

        # Q summary
        q_stations = [c for c in STATION_CODES if c not in STATIONS_NO_Q]
        if q_stations:
            print(f"\n  {'Station Q':<20} ", end="")
            for h in FORECAST_HORIZONS:
                print(f"{'t+' + str(h) + 'h':>10}", end="")
            print()
            print(f"  {'-'*20} " + "-" * 10 * len(FORECAST_HORIZONS))

            avg_q_per_horizon = {h: [] for h in FORECAST_HORIZONS}
            for code in q_stations:
                label = code[:10]
                print(f"  {label:<20} ", end="")
                for h in FORECAST_HORIZONS:
                    key = f"{code}_Q_t+{h}h"
                    if key in results[split]:
                        nse = results[split][key]["nse"]
                        avg_q_per_horizon[h].append(nse)
                        print(f"{nse:>10.4f}", end="")
                    else:
                        print(f"{'—':>10}", end="")
                print()

            print(f"  {'AVG Q':<20} ", end="")
            for h in FORECAST_HORIZONS:
                vals = avg_q_per_horizon[h]
                if vals:
                    print(f"{np.mean(vals):>10.4f}", end="")
            print()

    # Plot NSE H for Châteaubourg
    target = "J706062001"
    if "test" in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        horizons = FORECAST_HORIZONS
        nses = [results["test"].get(f"{target}_H_t+{h}h", {}).get("nse", 0) for h in horizons]
        ax.plot([str(h) for h in horizons], nses, "o-", color="#0d9488", label="H NSE")

        if any(f"{target}_Q_t+{h}h" in results["test"] for h in horizons):
            nses_q = [results["test"].get(f"{target}_Q_t+{h}h", {}).get("nse", 0) for h in horizons]
            ax.plot([str(h) for h in horizons], nses_q, "s-", color="#d97706", label="Q NSE")

        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("NSE")
        ax.set_title(f"Station-Attention — {target}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_path = CHECKPOINTS_DIR / "station_attn_eval.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"\nGraphique → {out_path}")


if __name__ == "__main__":
    main()
