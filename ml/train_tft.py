"""Temporal Fusion Transformer (TFT) simplifié pour la prédiction de crues.

LSTM encoder + Multi-Head Self-Attention + dense decoder.
Suréchantillonnage des crues pour réduire le biais en crue.

Usage:
    python train_tft.py
    python train_tft.py --epochs 100 --hidden 64 --lr 0.001
"""

import argparse
import json
import warnings

import numpy as np
import torch

from config import (
    CHECKPOINTS_DIR,
    EPOCHS,
    FORECAST_HORIZONS,
    LEARNING_RATE,
    PATIENCE,
    PROCESSED_DIR,
)

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description="Entraînement TFT")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--hidden", type=int, default=256, help="Hidden size du TFT")
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Le TFT via pytorch-forecasting nécessite un format DataFrame spécifique.
    # On l'entraîne ici avec une approche simplifiée utilisant les données
    # déjà fenêtrées, en wrappant dans un modèle PyTorch classique.

    print("Chargement des données...")
    X_train_np = np.load(PROCESSED_DIR / "X_train.npy")
    y_train_np = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = torch.from_numpy(np.load(PROCESSED_DIR / "X_val.npy"))
    y_val = torch.from_numpy(np.load(PROCESSED_DIR / "y_val.npy"))
    X_test = torch.from_numpy(np.load(PROCESSED_DIR / "X_test.npy"))
    y_test = torch.from_numpy(np.load(PROCESSED_DIR / "y_test.npy"))

    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)

    with open(PROCESSED_DIR / "norm_params.json") as f:
        norm_params = json.load(f)

    n_features = meta["n_features"]
    n_horizons = len(meta["forecast_horizons"])
    input_window = meta["input_window"]
    target_col = f"{meta['target_station']}_h"
    target_idx = meta["feature_names"].index(target_col)
    np_t = norm_params[target_col]
    t_min, t_max = np_t["min"], np_t["max"]
    t_range = t_max - t_min

    print(f"Features: {n_features}, Window: {input_window}, Horizons: {n_horizons}")

    # --- Suréchantillonnage des crues ---
    # H normalisé au dernier timestep de chaque fenêtre
    h_last_norm = X_train_np[:, -1, target_idx]
    h_last_mm = h_last_norm * t_range + t_min

    # Seuils de suréchantillonnage (en mm)
    FLOOD_THRESHOLD_1 = 800   # crue modérée → x2
    FLOOD_THRESHOLD_2 = 1200  # crue forte → x4
    FLOOD_THRESHOLD_3 = 1500  # crue majeure → x8

    mask_1 = (h_last_mm >= FLOOD_THRESHOLD_1) & (h_last_mm < FLOOD_THRESHOLD_2)
    mask_2 = (h_last_mm >= FLOOD_THRESHOLD_2) & (h_last_mm < FLOOD_THRESHOLD_3)
    mask_3 = h_last_mm >= FLOOD_THRESHOLD_3

    n1, n2, n3 = mask_1.sum(), mask_2.sum(), mask_3.sum()
    print(f"\nSuréchantillonnage crues:")
    print(f"  H >= {FLOOD_THRESHOLD_1}mm: {n1} samples → x2 = +{n1}")
    print(f"  H >= {FLOOD_THRESHOLD_2}mm: {n2} samples → x4 = +{n2 * 3}")
    print(f"  H >= {FLOOD_THRESHOLD_3}mm: {n3} samples → x8 = +{n3 * 7}")

    # Dupliquer les échantillons de crue
    extra_X = []
    extra_y = []
    if n1 > 0:
        extra_X.append(X_train_np[mask_1])          # x1 extra → total x2
        extra_y.append(y_train_np[mask_1])
    if n2 > 0:
        for _ in range(3):                           # x3 extra → total x4
            extra_X.append(X_train_np[mask_2])
            extra_y.append(y_train_np[mask_2])
    if n3 > 0:
        for _ in range(7):                           # x7 extra → total x8
            extra_X.append(X_train_np[mask_3])
            extra_y.append(y_train_np[mask_3])

    if extra_X:
        X_train_np = np.concatenate([X_train_np] + extra_X)
        y_train_np = np.concatenate([y_train_np] + extra_y)
        print(f"  Dataset train: {len(X_train_np)} samples (après suréchantillonnage)")

    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Modèle TFT simplifié (encoder-only avec attention)
    model = SimplifiedTFT(
        n_features=n_features,
        n_horizons=n_horizons,
        hidden_size=args.hidden,
        n_heads=args.attention_heads,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres: {total_params:,}")

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    test_ds = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warmup linéaire sur 5 epochs puis ReduceLROnPlateau
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nEntraînement — {args.epochs} epochs max")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'LR':>10}")
    print("-" * 44)

    criterion = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n += 1
        train_loss /= n

        # Val
        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item()
                nv += 1
        val_loss /= nv

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "tft_best.pt")
            marker = " ★"
        else:
            patience_counter += 1

        print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>12.6f} {lr:>10.2e}{marker}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping à l'epoch {epoch}")
            break

    # Évaluation
    print(f"\n{'='*50}")
    print("Évaluation TFT")
    print(f"{'='*50}")

    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "tft_best.pt", weights_only=True))
    model.eval()

    results = {"model": "TFT", "val": {}, "test": {}}

    target_mode = meta.get("target_mode", "absolute")

    for split_name, loader, X_split in [("Val", val_loader, X_val), ("Test", test_loader, X_test)]:
        all_preds, all_targets, all_h_last = [], [], []
        offset = 0
        with torch.no_grad():
            for X_b, y_b in loader:
                all_preds.append(model(X_b.to(device)).cpu().numpy())
                all_targets.append(y_b.numpy())
                bs = X_b.shape[0]
                all_h_last.append(X_split[offset:offset + bs, -1, target_idx].numpy())
                offset += bs

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        h_last = np.concatenate(all_h_last)

        split_key = "val" if split_name == "Val" else "test"
        print(f"\n{split_name}:")

        if target_mode == "delta":
            # Convert delta predictions to absolute mm for meaningful metrics
            for i, h in enumerate(FORECAST_HORIZONS):
                pred_abs_mm = (h_last + y_pred[:, i]) * t_range + t_min
                true_abs_mm = (h_last + y_true[:, i]) * t_range + t_min
                rmse = float(np.sqrt(np.mean((true_abs_mm - pred_abs_mm) ** 2)))
                mae = float(np.mean(np.abs(true_abs_mm - pred_abs_mm)))
                ss_res = np.sum((true_abs_mm - pred_abs_mm) ** 2)
                ss_tot = np.sum((true_abs_mm - np.mean(true_abs_mm)) ** 2)
                nse = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                results[split_key][f"t+{h}h"] = {"rmse_mm": rmse, "mae_mm": mae, "nse": nse}
                print(f"  t+{h}h: NSE={nse:.4f}, RMSE={rmse:.1f}mm, MAE={mae:.1f}mm")
        else:
            for i, h in enumerate(FORECAST_HORIZONS):
                rmse = float(np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2)))
                mae = float(np.mean(np.abs(y_true[:, i] - y_pred[:, i])))
                ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
                ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
                nse = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                results[split_key][f"t+{h}h"] = {"rmse": rmse, "mae": mae, "nse": nse}
                print(f"  t+{h}h: NSE={nse:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f}")

    # Sauvegarder les résultats
    with open(CHECKPOINTS_DIR / "tft_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Sauvegarder config pour export
    model_config = {
        "n_features": n_features,
        "n_horizons": n_horizons,
        "hidden_size": args.hidden,
        "n_heads": args.attention_heads,
        "dropout": 0.1,
    }
    with open(CHECKPOINTS_DIR / "tft_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\n✓ Modèle TFT dans {CHECKPOINTS_DIR}")


class SimplifiedTFT(torch.nn.Module):
    """TFT simplifié : LSTM encoder + Multi-Head Self-Attention + dense decoder.

    Capture les dépendances temporelles (LSTM) et les relations inter-variables
    (attention), ce qui est l'essence du TFT pour notre cas d'usage.
    """

    def __init__(self, n_features: int, n_horizons: int, hidden_size: int = 32,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(n_features, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers=2,
                                  batch_first=True, dropout=dropout)
        self.attention = torch.nn.MultiheadAttention(hidden_size, n_heads,
                                                     dropout=dropout, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size * 4, hidden_size),
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)

        # Self-attention sur la sortie LSTM
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.norm1(lstm_out + attn_out)

        # FFN avec résiduelle
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Dernier pas de temps
        return self.output(x[:, -1, :])


if __name__ == "__main__":
    main()
