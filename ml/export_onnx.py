"""Export du meilleur modèle PyTorch en format ONNX.

Le modèle ONNX sera utilisé par le backend Node.js via onnxruntime-node
pour servir les prédictions en temps réel.

Usage:
    python export_onnx.py                    # auto-détecte le meilleur modèle
    python export_onnx.py --model lstm       # force l'export du LSTM
    python export_onnx.py --model tft        # force l'export du TFT
"""

import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from config import CHECKPOINTS_DIR, ONNX_DIR, PROCESSED_DIR, PROPAGATION_HOURS, RIVER_BRANCHES


def load_lstm():
    """Charge le modèle LSTM."""
    from train_lstm import FloodLSTM

    with open(CHECKPOINTS_DIR / "lstm_config.json") as f:
        config = json.load(f)

    model = FloodLSTM(
        n_features=config["n_features"],
        n_horizons=config["n_horizons"],
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "lstm_best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    return model, config


def load_tft():
    """Charge le modèle TFT."""
    from train_tft import SimplifiedTFT

    with open(CHECKPOINTS_DIR / "tft_config.json") as f:
        config = json.load(f)

    model = SimplifiedTFT(
        n_features=config["n_features"],
        n_horizons=config["n_horizons"],
        hidden_size=config["hidden_size"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "tft_best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    return model, config


def export_to_onnx(model: torch.nn.Module, config: dict, model_name: str) -> str:
    """Exporte un modèle PyTorch en ONNX."""
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)

    input_window = meta["input_window"]
    n_features = config["n_features"]

    # Dummy input
    dummy = torch.randn(1, input_window, n_features)

    onnx_path = ONNX_DIR / f"{model_name}.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["predictions"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
        opset_version=17,
    )

    # Vérifier le modèle ONNX
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"✓ Modèle ONNX valide : {onnx_path}")

    # Test d'inférence avec onnxruntime
    session = ort.InferenceSession(str(onnx_path))
    test_input = np.random.randn(1, input_window, n_features).astype(np.float32)
    outputs = session.run(None, {"input": test_input})
    print(f"  Shape sortie : {outputs[0].shape}")
    print(f"  Valeurs test : {outputs[0][0]}")

    # Comparer avec PyTorch
    with torch.no_grad():
        pt_output = model(torch.from_numpy(test_input)).numpy()
    diff = np.abs(pt_output - outputs[0]).max()
    print(f"  Écart max PyTorch vs ONNX : {diff:.8f}")

    # Sauvegarder les métadonnées pour le backend
    export_meta = {
        "model_name": model_name,
        "input_window": input_window,
        "n_features": n_features,
        "forecast_horizons": meta["forecast_horizons"],
        "target_station": meta["target_station"],
        "feature_names": meta["feature_names"],
        "target_mode": meta.get("target_mode", "absolute"),
        "log_transform_cols": meta.get("log_transform_cols", []),
        "propagation_hours": {k: v for k, v in PROPAGATION_HOURS.items()},
        "river_branches": RIVER_BRANCHES,
    }
    meta_path = ONNX_DIR / f"{model_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(export_meta, f, indent=2)
    print(f"  Métadonnées → {meta_path.name}")

    # Copier aussi les paramètres de normalisation
    import shutil
    shutil.copy(PROCESSED_DIR / "norm_params.json", ONNX_DIR / "norm_params.json")
    print(f"  norm_params.json → {ONNX_DIR.name}/")

    return str(onnx_path)


def main():
    parser = argparse.ArgumentParser(description="Export ONNX")
    parser.add_argument("--model", choices=["lstm", "tft", "auto"], default="auto",
                        help="Modèle à exporter (auto = meilleur selon evaluate.py)")
    args = parser.parse_args()

    model_name = args.model

    if model_name == "auto":
        best_path = CHECKPOINTS_DIR / "best_model.json"
        if best_path.exists():
            with open(best_path) as f:
                best = json.load(f)
            model_name = best["best_model"].lower()
            print(f"Auto-détecté : {model_name} (NSE={best['avg_nse'].get(best['best_model'], '?')})")
        else:
            # Fallback : prendre le LSTM s'il existe
            if (CHECKPOINTS_DIR / "lstm_best.pt").exists():
                model_name = "lstm"
            elif (CHECKPOINTS_DIR / "tft_best.pt").exists():
                model_name = "tft"
            else:
                print("Aucun modèle entraîné trouvé. Lancer train_lstm.py ou train_tft.py d'abord.")
                return

    print(f"Export du modèle : {model_name}")

    if model_name == "lstm":
        model, config = load_lstm()
    elif model_name == "tft":
        model, config = load_tft()
    else:
        print(f"Modèle inconnu : {model_name}")
        return

    onnx_path = export_to_onnx(model, config, model_name)
    print(f"\n✓ Export terminé : {onnx_path}")
    print(f"  Copier {ONNX_DIR}/ dans le backend pour l'inférence.")


if __name__ == "__main__":
    main()
