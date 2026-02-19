"""Station-Attention model pour la prédiction multi-station de crues.

Architecture :
  Per-station LSTM (partagé) → Cross-station Attention → Shared Decoder (H + Q)

Input passé:  (batch, 72, n_features)  → reshape → 11 stations × (batch, 72, 7)
Input futur:  (batch, n_stations * future_hours)

Output: (batch, n_outputs)  → 11 stations × 24 horizons H + 7 stations × 24 horizons Q

Usage:
    python train_station_attention.py
    python train_station_attention.py --epochs 100 --hidden 64
"""

import argparse
import json
import warnings

import numpy as np
import torch
import torch.nn as nn

from config import (
    BARRAGE_CODES,
    CHECKPOINTS_DIR,
    EPOCHS,
    FORECAST_HORIZONS,
    LEARNING_RATE,
    PATIENCE,
    PROCESSED_DIR,
    STATION_CODES,
    STATIONS_NO_Q,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Nombre de variables d'entrée par station (uniforme après padding)
VARS_PER_STATION = 7


class StationAttentionModel(nn.Module):
    """Per-station LSTM → Cross-station Attention → Shared Decoder.

    Chaque station a 7 variables d'entrée :
    - Stations normales : H, Q, precip, dH, dQ, soil_moisture_0_to_7cm, soil_moisture_0_to_28cm
    - Stations no_q (Taillis) : H, 0, precip, dH, 0, soil_moisture_0_to_7cm, soil_moisture_0_to_28cm
    - Stations barrage : H, 0, precip, dH, release, soil_moisture_0_to_7cm, soil_moisture_0_to_28cm
    """

    def __init__(
        self,
        n_stations: int,
        vars_per_station: int,
        future_precip_hours: int,
        n_horizons: int,
        hidden_size: int = 64,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        stations_with_q_indices: list[int] | None = None,
    ):
        super().__init__()
        self.n_stations = n_stations
        self.vars_per_station = vars_per_station
        self.hidden_size = hidden_size
        self.n_horizons = n_horizons
        # Indices (dans STATION_CODES) des stations avec Q
        if stations_with_q_indices is None:
            stations_with_q_indices = [i for i, c in enumerate(STATION_CODES) if c not in STATIONS_NO_Q]
        self.stations_with_q_indices = stations_with_q_indices
        # Boolean mask for building output (not traced by ONNX, used in loop)
        self._has_q = [i in set(stations_with_q_indices) for i in range(n_stations)]

        # Station encoder (LSTM partagé)
        self.station_lstm = nn.LSTM(
            vars_per_station, hidden_size, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
        )

        # Future precip encoder (Linear partagé)
        self.future_encoder = nn.Sequential(
            nn.Linear(future_precip_hours, hidden_size),
            nn.ReLU(),
        )

        # Cross-station attention
        self.attn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        for _ in range(n_attn_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
            )
            self.attn_norms.append(nn.LayerNorm(hidden_size))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
            ))
            self.ffn_norms.append(nn.LayerNorm(hidden_size))

        # Shared decoders
        self.decoder_h = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_horizons),
        )
        self.decoder_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_horizons),
        )

    def forward(self, past: torch.Tensor, future_precip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past: (batch, seq_len, n_stations * vars_per_station) — padded
            future_precip: (batch, n_stations * future_hours)

        Returns:
            (batch, n_outputs) — delta H (all stations) + delta Q (stations with Q)
        """
        batch_size = past.shape[0]
        seq_len = past.shape[1]

        # Reshape past par station : (batch, n_stations, seq, vars_per_station)
        past_per_station = past.reshape(batch_size, seq_len, self.n_stations, self.vars_per_station)
        past_per_station = past_per_station.permute(0, 2, 1, 3)
        past_flat = past_per_station.reshape(batch_size * self.n_stations, seq_len, self.vars_per_station)

        # Station encoder (LSTM partagé)
        _, (h_n, _) = self.station_lstm(past_flat)
        station_emb = h_n[-1].reshape(batch_size, self.n_stations, self.hidden_size)

        # Future precip encoder
        future = future_precip.reshape(batch_size, self.n_stations, -1)
        future_emb = self.future_encoder(future)

        # Fusion
        x = station_emb + future_emb

        # Cross-station attention
        for attn, norm1, ffn, norm2 in zip(self.attn_layers, self.attn_norms, self.ffn_layers, self.ffn_norms):
            attn_out, _ = attn(x, x, x)
            x = norm1(x + attn_out)
            ffn_out = ffn(x)
            x = norm2(x + ffn_out)

        # Decode H pour toutes les stations
        pred_h = self.decoder_h(x)  # (batch, n_stations, n_horizons)

        # Decode Q pour les stations avec Q (indices spécifiques)
        q_indices = torch.tensor(self.stations_with_q_indices, dtype=torch.long, device=x.device)
        q_stations_emb = x[:, q_indices]  # (batch, n_with_q, hidden)
        pred_q = self.decoder_q(q_stations_emb)  # (batch, n_with_q, n_horizons)

        # Assembler dans l'ordre output_map :
        # station0_H×5, station0_Q×5 (si Q), station1_H×5, station1_Q×5 (si Q), ...
        outputs = []
        q_idx = 0
        for s in range(self.n_stations):
            outputs.append(pred_h[:, s, :])
            if self._has_q[s]:
                outputs.append(pred_q[:, q_idx, :])
                q_idx += 1

        return torch.cat(outputs, dim=1)


def main():
    parser = argparse.ArgumentParser(description="Entraînement Station-Attention")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--hidden", type=int, default=64, help="Hidden size")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attn-layers", type=int, default=2)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement des données...")
    X_train_np = np.load(PROCESSED_DIR / "X_train.npy")
    fp_train_np = np.load(PROCESSED_DIR / "fp_train.npy")
    y_train_np = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = torch.from_numpy(np.load(PROCESSED_DIR / "X_val.npy"))
    fp_val = torch.from_numpy(np.load(PROCESSED_DIR / "fp_val.npy"))
    y_val = torch.from_numpy(np.load(PROCESSED_DIR / "y_val.npy"))
    X_test = torch.from_numpy(np.load(PROCESSED_DIR / "X_test.npy"))
    fp_test = torch.from_numpy(np.load(PROCESSED_DIR / "fp_test.npy"))
    y_test = torch.from_numpy(np.load(PROCESSED_DIR / "y_test.npy"))

    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)

    with open(PROCESSED_DIR / "norm_params.json") as f:
        norm_params = json.load(f)

    n_features = meta["n_features"]
    n_stations = meta["n_stations"]
    n_outputs = meta["n_outputs"]
    n_horizons = len(meta["forecast_horizons"])
    input_window = meta["input_window"]
    future_precip_hours = meta["future_precip_hours"]

    # Identifier les indices des stations with_q dans STATION_CODES order
    stations_with_q_indices = []
    stations_no_q_indices = []
    for i, code in enumerate(STATION_CODES):
        if code not in STATIONS_NO_Q:
            stations_with_q_indices.append(i)
        else:
            stations_no_q_indices.append(i)
    n_stations_with_q = len(stations_with_q_indices)

    print(f"Features: {n_features}, Stations: {n_stations}, Outputs: {n_outputs}")
    print(f"Stations avec Q: {n_stations_with_q}, sans Q: {len(stations_no_q_indices)}")
    print(f"Barrages: {BARRAGE_CODES}")

    # --- Suréchantillonnage des crues ---
    # Utiliser la station Châteaubourg comme référence
    target_code = "J706062001"
    target_h_col = f"{target_code}_h"
    target_idx = meta["feature_names"].index(target_h_col)
    np_t = norm_params[target_h_col]
    t_min, t_max = np_t["min"], np_t["max"]
    t_range = t_max - t_min

    h_last_norm = X_train_np[:, -1, target_idx]
    h_last_mm = h_last_norm * t_range + t_min

    FLOOD_THRESHOLD_1 = 800
    FLOOD_THRESHOLD_2 = 1200
    FLOOD_THRESHOLD_3 = 1500

    mask_1 = (h_last_mm >= FLOOD_THRESHOLD_1) & (h_last_mm < FLOOD_THRESHOLD_2)
    mask_2 = (h_last_mm >= FLOOD_THRESHOLD_2) & (h_last_mm < FLOOD_THRESHOLD_3)
    mask_3 = h_last_mm >= FLOOD_THRESHOLD_3

    n1, n2, n3 = mask_1.sum(), mask_2.sum(), mask_3.sum()
    print(f"\nSuréchantillonnage crues:")
    print(f"  H >= {FLOOD_THRESHOLD_1}mm: {n1} → x2")
    print(f"  H >= {FLOOD_THRESHOLD_2}mm: {n2} → x4")
    print(f"  H >= {FLOOD_THRESHOLD_3}mm: {n3} → x8")

    extra_X, extra_fp, extra_y = [], [], []
    if n1 > 0:
        extra_X.append(X_train_np[mask_1])
        extra_fp.append(fp_train_np[mask_1])
        extra_y.append(y_train_np[mask_1])
    if n2 > 0:
        for _ in range(3):
            extra_X.append(X_train_np[mask_2])
            extra_fp.append(fp_train_np[mask_2])
            extra_y.append(y_train_np[mask_2])
    if n3 > 0:
        for _ in range(7):
            extra_X.append(X_train_np[mask_3])
            extra_fp.append(fp_train_np[mask_3])
            extra_y.append(y_train_np[mask_3])

    if extra_X:
        X_train_np = np.concatenate([X_train_np] + extra_X)
        fp_train_np = np.concatenate([fp_train_np] + extra_fp)
        y_train_np = np.concatenate([y_train_np] + extra_y)
        print(f"  Dataset train: {len(X_train_np)} samples")

    X_train = torch.from_numpy(X_train_np)
    fp_train = torch.from_numpy(fp_train_np)
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

    # --- Reshape X pour le modèle ---
    # Le modèle attend (batch, seq, n_features) où n_features = n_stations * vars_per_station
    # Mais nos features ne sont PAS uniformes (certaines stations ont 3, 4 ou 5 vars).
    # On doit padder pour uniformiser à VARS_PER_STATION = 5 vars/station.
    def pad_features(X_tensor):
        """Pad X de (batch, seq, n_features_raw) à (batch, seq, n_stations * 7)."""
        batch_size, seq_len, _ = X_tensor.shape
        padded = torch.zeros(batch_size, seq_len, n_stations * VARS_PER_STATION, dtype=X_tensor.dtype)

        for s_idx, code in enumerate(STATION_CODES):
            sfm = meta["station_feature_map"][code]
            src_indices = sfm["indices"]
            src_vars = sfm["vars"]

            # Mapping des vars source vers slots de destination
            # Slots: 0=h, 1=q, 2=precip, 3=dh, 4=dq/release, 5=soil_moisture_0_to_7cm, 6=soil_moisture_0_to_28cm
            slot_map = {
                "h": 0, "q": 1, "precip": 2, "dh": 3, "dq": 4, "release": 4,
                "soil_moisture_0_to_7cm": 5, "soil_moisture_0_to_28cm": 6,
            }
            dst_base = s_idx * VARS_PER_STATION

            for src_i, var_name in zip(src_indices, src_vars):
                slot = slot_map.get(var_name, None)
                if slot is not None:
                    padded[:, :, dst_base + slot] = X_tensor[:, :, src_i]

        return padded

    print("\nPadding des features pour uniformiser à 7 vars/station...")
    X_train_pad = pad_features(X_train)
    X_val_pad = pad_features(X_val)
    X_test_pad = pad_features(X_test)
    print(f"  X padded: {X_train_pad.shape}")

    # Modèle
    model = StationAttentionModel(
        n_stations=n_stations,
        vars_per_station=VARS_PER_STATION,
        future_precip_hours=future_precip_hours,
        n_horizons=n_horizons,
        hidden_size=args.hidden,
        n_heads=args.attention_heads,
        n_attn_layers=args.attn_layers,
        lstm_layers=args.lstm_layers,
        dropout=0.1,
        stations_with_q_indices=stations_with_q_indices,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres: {total_params:,}")

    # Construire les indices de sortie H et Q pour la loss séparée
    h_output_indices = []
    q_output_indices = []
    for code in STATION_CODES:
        om = meta["output_map"][code]
        h_output_indices.extend(range(om["h_start"], om["h_end"]))
        if "q_start" in om:
            q_output_indices.extend(range(om["q_start"], om["q_end"]))

    h_output_indices = torch.tensor(h_output_indices, dtype=torch.long, device=device)
    q_output_indices = torch.tensor(q_output_indices, dtype=torch.long, device=device)

    # DataLoaders
    train_ds = torch.utils.data.TensorDataset(X_train_pad, fp_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val_pad, fp_val, y_val)
    test_ds = torch.utils.data.TensorDataset(X_test_pad, fp_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    mse = nn.MSELoss()

    print(f"\nEntraînement — {args.epochs} epochs max")
    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Val H':>10} {'Val Q':>10} {'LR':>10}")
    print("-" * 58)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n = 0
        for X_b, fp_b, y_b in train_loader:
            X_b, fp_b, y_b = X_b.to(device), fp_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b, fp_b)
            loss_h = mse(pred[:, h_output_indices], y_b[:, h_output_indices])
            loss_q = mse(pred[:, q_output_indices], y_b[:, q_output_indices])
            loss = loss_h + 0.3 * loss_q
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n += 1
        train_loss /= n

        # Val
        model.eval()
        val_loss = 0.0
        val_loss_h = 0.0
        val_loss_q = 0.0
        nv = 0
        with torch.no_grad():
            for X_b, fp_b, y_b in val_loader:
                X_b, fp_b, y_b = X_b.to(device), fp_b.to(device), y_b.to(device)
                pred = model(X_b, fp_b)
                lh = mse(pred[:, h_output_indices], y_b[:, h_output_indices]).item()
                lq = mse(pred[:, q_output_indices], y_b[:, q_output_indices]).item()
                val_loss += lh + 0.3 * lq
                val_loss_h += lh
                val_loss_q += lq
                nv += 1
        val_loss /= nv
        val_loss_h /= nv
        val_loss_q /= nv

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "station_attn_best.pt")
            marker = " ★"
        else:
            patience_counter += 1

        print(f"{epoch:>6} {train_loss:>10.6f} {val_loss:>10.6f} {val_loss_h:>10.6f} {val_loss_q:>10.6f} {lr:>10.2e}{marker}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping à l'epoch {epoch}")
            break

    # --- Évaluation ---
    print(f"\n{'='*60}")
    print("Évaluation Station-Attention")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "station_attn_best.pt", weights_only=True))
    model.eval()

    results = {"model": "StationAttention", "val": {}, "test": {}}

    # Indices de H et Q dans X non-paddé (pour récupérer h_last, q_last)
    h_feat_indices = {}
    q_feat_indices = {}
    for code in STATION_CODES:
        sfm = meta["station_feature_map"][code]
        for idx_raw, var in zip(sfm["indices"], sfm["vars"]):
            if var == "h":
                h_feat_indices[code] = idx_raw
            elif var == "q":
                q_feat_indices[code] = idx_raw

    for split_name, loader, X_raw in [("val", val_loader, X_val), ("test", test_loader, X_test)]:
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_b, fp_b, y_b in loader:
                X_b, fp_b = X_b.to(device), fp_b.to(device)
                all_preds.append(model(X_b, fp_b).cpu().numpy())
                all_targets.append(y_b.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        print(f"\n{split_name.upper()}:")

        # Per-station metrics (NSE sur valeurs absolues, comme l'ancien TFT)
        for code in STATION_CODES:
            om = meta["output_map"][code]
            np_h = norm_params[f"{code}_h"]
            h_min, h_max = np_h["min"], np_h["max"]
            h_range = h_max - h_min

            # h_last normalisé au dernier pas de chaque fenêtre (X non-paddé)
            h_last_norm = X_raw[:, -1, h_feat_indices[code]].numpy()

            for j, h in enumerate(FORECAST_HORIZONS):
                idx = om["h_start"] + j
                # Delta normalisé → valeur absolue en mm
                pred_abs_mm = (h_last_norm + y_pred[:, idx]) * h_range + h_min
                true_abs_mm = (h_last_norm + y_true[:, idx]) * h_range + h_min
                rmse = float(np.sqrt(np.mean((true_abs_mm - pred_abs_mm) ** 2)))
                mae = float(np.mean(np.abs(true_abs_mm - pred_abs_mm)))
                ss_res = np.sum((true_abs_mm - pred_abs_mm) ** 2)
                ss_tot = np.sum((true_abs_mm - np.mean(true_abs_mm)) ** 2)
                nse = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

                key = f"{code}_H_t+{h}h"
                results[split_name][key] = {"rmse": rmse, "mae": mae, "nse": nse}

            if "q_start" in om:
                np_q = norm_params[f"{code}_q"]
                q_min, q_max = np_q["min"], np_q["max"]
                q_range = q_max - q_min
                q_last_norm = X_raw[:, -1, q_feat_indices[code]].numpy()

                for j, h in enumerate(FORECAST_HORIZONS):
                    idx = om["q_start"] + j
                    pred_abs_ls = (q_last_norm + y_pred[:, idx]) * q_range + q_min
                    true_abs_ls = (q_last_norm + y_true[:, idx]) * q_range + q_min
                    rmse = float(np.sqrt(np.mean((true_abs_ls - pred_abs_ls) ** 2)))
                    mae = float(np.mean(np.abs(true_abs_ls - pred_abs_ls)))
                    ss_res = np.sum((true_abs_ls - pred_abs_ls) ** 2)
                    ss_tot = np.sum((true_abs_ls - np.mean(true_abs_ls)) ** 2)
                    nse = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

                    key = f"{code}_Q_t+{h}h"
                    results[split_name][key] = {"rmse": rmse, "mae": mae, "nse": nse}

        # Summary: avg NSE H across all stations at each horizon
        print("  Avg NSE H par horizon:")
        for h in FORECAST_HORIZONS:
            nses = [results[split_name][f"{c}_H_t+{h}h"]["nse"] for c in STATION_CODES]
            print(f"    t+{h}h: {np.mean(nses):.4f} (min={min(nses):.4f}, max={max(nses):.4f})")

        # Châteaubourg H specifically
        print(f"  Châteaubourg (J706062001) H:")
        for h in FORECAST_HORIZONS:
            m = results[split_name][f"J706062001_H_t+{h}h"]
            print(f"    t+{h}h: NSE={m['nse']:.4f}, RMSE={m['rmse']:.1f}mm")

    # Sauvegarder
    with open(CHECKPOINTS_DIR / "station_attn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Build RMSE dict per station/horizon (from test set) for confidence intervals
    rmse_per_station = {}
    for code in STATION_CODES:
        rmse_h = []
        for h in FORECAST_HORIZONS:
            key = f"{code}_H_t+{h}h"
            rmse_h.append(results["test"][key]["rmse"])
        entry = {"h": rmse_h}

        # Q if available
        om = meta["output_map"][code]
        if "q_start" in om:
            rmse_q = []
            for h in FORECAST_HORIZONS:
                key = f"{code}_Q_t+{h}h"
                rmse_q.append(results["test"][key]["rmse"])
            entry["q"] = rmse_q

        rmse_per_station[code] = entry

    model_config = {
        "n_stations": n_stations,
        "vars_per_station": VARS_PER_STATION,
        "future_precip_hours": future_precip_hours,
        "n_horizons": n_horizons,
        "hidden_size": args.hidden,
        "n_heads": args.attention_heads,
        "n_attn_layers": args.attn_layers,
        "lstm_layers": args.lstm_layers,
        "dropout": 0.1,
        "stations_with_q_indices": stations_with_q_indices,
        "n_outputs": n_outputs,
        "n_features": n_features,
        "rmse": rmse_per_station,
    }
    with open(CHECKPOINTS_DIR / "station_attn_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\n✓ Modèle Station-Attention dans {CHECKPOINTS_DIR}")


if __name__ == "__main__":
    main()
