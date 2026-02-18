"""Export du meilleur modèle PyTorch en format ONNX.

Le modèle ONNX sera utilisé par le backend Node.js via onnxruntime-node
pour servir les prédictions en temps réel.

Usage:
    python export_onnx.py                           # auto-détecte
    python export_onnx.py --model station_attn      # force station-attention
"""

import argparse
import json
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch

from config import CHECKPOINTS_DIR, ONNX_DIR, PROCESSED_DIR


def load_station_attn():
    """Charge le modèle Station-Attention."""
    from train_station_attention import StationAttentionModel

    with open(CHECKPOINTS_DIR / "station_attn_config.json") as f:
        config = json.load(f)

    model = StationAttentionModel(
        n_stations=config["n_stations"],
        vars_per_station=config["vars_per_station"],
        future_precip_hours=config["future_precip_hours"],
        n_horizons=config["n_horizons"],
        hidden_size=config["hidden_size"],
        n_heads=config["n_heads"],
        n_attn_layers=config["n_attn_layers"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
        stations_with_q_indices=config["stations_with_q_indices"],
    )
    model.load_state_dict(
        torch.load(CHECKPOINTS_DIR / "station_attn_best.pt", weights_only=True, map_location="cpu")
    )
    model.eval()
    return model, config


def export_to_onnx(model: torch.nn.Module, config: dict) -> str:
    """Exporte le modèle Station-Attention en ONNX (2 inputs)."""
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)

    input_window = meta["input_window"]
    n_stations = config["n_stations"]
    vars_per_station = config["vars_per_station"]
    future_precip_size = meta["future_precip_size"]

    # Dummy inputs
    dummy_past = torch.randn(1, input_window, n_stations * vars_per_station)
    dummy_future = torch.randn(1, future_precip_size)

    onnx_path = ONNX_DIR / "station_attn.onnx"

    torch.onnx.export(
        model,
        (dummy_past, dummy_future),
        str(onnx_path),
        input_names=["past_input", "future_precip"],
        output_names=["predictions"],
        dynamic_axes={
            "past_input": {0: "batch_size"},
            "future_precip": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
        opset_version=17,
    )

    # Repack: intégrer les poids externes dans un seul fichier .onnx
    onnx_model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(onnx_model, str(onnx_path))
    data_path = onnx_path.with_suffix(".onnx.data")
    if data_path.exists():
        data_path.unlink()
        print(f"  Poids intégrés dans {onnx_path.name}")
    onnx.checker.check_model(onnx_model)
    print(f"✓ Modèle ONNX valide : {onnx_path}")

    # Test d'inférence avec onnxruntime
    session = ort.InferenceSession(str(onnx_path))
    test_past = np.random.randn(1, input_window, n_stations * vars_per_station).astype(np.float32)
    test_future = np.random.randn(1, future_precip_size).astype(np.float32)
    outputs = session.run(None, {"past_input": test_past, "future_precip": test_future})
    print(f"  Shape sortie : {outputs[0].shape}")
    print(f"  Valeurs test : {outputs[0][0][:10]}...")

    # Comparer avec PyTorch
    with torch.no_grad():
        pt_output = model(torch.from_numpy(test_past), torch.from_numpy(test_future)).numpy()
    diff = np.abs(pt_output - outputs[0]).max()
    print(f"  Écart max PyTorch vs ONNX : {diff:.8f}")

    # Sauvegarder les métadonnées pour le backend
    export_meta = {
        "model_name": "station_attn",
        "input_window": input_window,
        "n_features_padded": n_stations * vars_per_station,
        "n_features_raw": meta["n_features"],
        "n_stations": n_stations,
        "vars_per_station": vars_per_station,
        "future_precip_size": future_precip_size,
        "future_precip_hours": meta["future_precip_hours"],
        "n_outputs": config["n_outputs"],
        "forecast_horizons": meta["forecast_horizons"],
        "station_codes": meta["station_codes"],
        "station_feature_map": meta["station_feature_map"],
        "output_map": meta["output_map"],
        "feature_names": meta["feature_names"],
        "target_mode": meta.get("target_mode", "delta"),
    }
    meta_path = ONNX_DIR / "station_attn_meta.json"
    with open(meta_path, "w") as f:
        json.dump(export_meta, f, indent=2)
    print(f"  Métadonnées → {meta_path.name}")

    # Copier les paramètres de normalisation
    shutil.copy(PROCESSED_DIR / "norm_params.json", ONNX_DIR / "norm_params.json")
    print(f"  norm_params.json → {ONNX_DIR.name}/")

    return str(onnx_path)


def main():
    parser = argparse.ArgumentParser(description="Export ONNX")
    parser.add_argument("--model", choices=["station_attn", "auto"], default="auto")
    args = parser.parse_args()

    model_name = args.model
    if model_name == "auto":
        if (CHECKPOINTS_DIR / "station_attn_best.pt").exists():
            model_name = "station_attn"
        else:
            print("Aucun modèle entraîné trouvé. Lancer train_station_attention.py d'abord.")
            return

    print(f"Export du modèle : {model_name}")

    if model_name == "station_attn":
        model, config = load_station_attn()
    else:
        print(f"Modèle inconnu : {model_name}")
        return

    onnx_path = export_to_onnx(model, config)
    print(f"\n✓ Export terminé : {onnx_path}")
    print(f"  Copier {ONNX_DIR}/ dans le backend pour l'inférence.")


if __name__ == "__main__":
    main()
