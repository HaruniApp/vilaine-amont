# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

Bassin Vilaine Amont — Visualisation et prédiction des hauteurs d'eau et débits des 11 stations hydrométriques du bassin Vilaine Amont via les données ouvertes de [Hydro EauFrance](https://www.hydro.eaufrance.fr), avec un modèle ML de prévision de crues.

## Commandes

```bash
# Installer les dépendances (pas de package-lock root, installer chaque sous-projet)
npm -C backend install
npm -C frontend install

# Lancer tout (backend + frontend en parallèle via concurrently)
npm run dev

# Ou séparément
npm -C backend run dev    # Express sur :3001
npm -C frontend run dev   # Vite sur :5173
```

Pas de tests, pas de linter configurés.

## Déploiement

Architecture de production :
```
Browser → nginx (vilaine-amont.haruni.net:443)
              ├── / → fichiers statiques (frontend/dist/)
              └── /api → proxy_pass → localhost:3001 (Express via pm2)
```

```bash
# Build frontend
npm -C frontend run build

# Lancer le backend avec pm2
pm2 start ecosystem.config.cjs

# Copier la config nginx
sudo cp nginx/vilaine-amont.conf /etc/nginx/sites-available/vilaine-amont
sudo ln -s /etc/nginx/sites-available/vilaine-amont /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## CD

Push sur `main` → GitHub Actions déploie automatiquement sur haruni via SSH (`appleboy/ssh-action`).

Secrets GitHub à configurer (`Settings > Secrets and variables > Actions`) :
- `SSH_HOST` : IP ou hostname du serveur
- `SSH_KEY` : clé privée SSH pour l'user `sb`

## Architecture

Monorepo avec trois sous-projets :

- **backend/** — Express (Node.js) : proxy API Hydro EauFrance + inférence ONNX pour les prévisions
- **frontend/** — SPA SolidJS + uPlot pour visualiser les courbes hydrologiques
- **ml/** — Pipeline ML (PyTorch) pour entraîner le modèle de prédiction de crues

### Chaîne de requêtes en dev

```
Browser → Vite (:5173) → proxy /api → Express (:3001) → hydro.eaufrance.fr
                                                       → Open-Meteo (précip/sol)
                                                       → ONNX Runtime (prévisions)
```

Vite proxie `/api` vers le backend (configuré dans `frontend/vite.config.js`).

### Flux de données (observations)

1. `App.jsx` : formulaire station/dates → fetch parallèle des séries H (hauteur) et Q (débit)
2. L'API retourne `{ series: { data: [{ t, v }, ...] } }` — timestamps ISO + valeurs en mm (H) ou L/s (Q)
3. `HydroChart.jsx` : `buildData()` convertit en tableaux uPlot — timestamps Unix (secondes), valeurs en m (÷1000) et m³/s (÷1000). Les séries Q sont alignées sur les timestamps de H via une Map.
4. Graphique double axe : H à gauche (scale `"H"`, side 3), Q à droite (scale `"Q"`, side 1)

### Flux de données (prévisions ML)

1. `backend/src/forecast.js` : fetch 72h de données passées (11 stations H/Q + météo) → construit les tenseurs d'entrée paddés (7 vars/station)
2. Inférence ONNX : modèle Station-Attention → 1296 sorties (432 horizons × 3 quantiles q10/q50/q90, interleaved)
3. Dénormalisation : delta normalisé → valeur absolue en m (H) ou m³/s (Q), avec intervalles de confiance `v_lower`/`v_upper`
4. Frontend : `HydroChart.jsx` affiche les prévisions q50 avec bandes de confiance q10-q90

### Pipeline ML (ml/)

```bash
# Sur DGX : collecte → préparation → entraînement → export
python collect_hydro.py && python collect_meteo.py
python prepare_dataset.py
python train_station_attention.py --hidden 128 --lr 5e-4 --attention-heads 8 --attn-layers 3
python export_onnx.py --model station_attn

# Copier dans le backend
cp models/onnx/station_attn.onnx ../../backend/models/
cp models/onnx/station_attn_meta.json ../../backend/models/
cp models/onnx/norm_params.json ../../backend/models/
```

Modèle : Per-station LSTM → Cross-station Attention → Quantile Regression (q10/q50/q90) avec pinball loss asymétrique en crue.

## Stack technique

- **Backend** : Express 4, cors, onnxruntime-node, `node --watch` (pas de nodemon)
- **Frontend** : SolidJS, uPlot via `@dschz/solid-uplot` (plugins: cursor, tooltip, focusSeries), Vite 6
- **ML** : PyTorch, ONNX, Open-Meteo API (précip + humidité du sol)
- **Pas de TypeScript** — JS pur (ESM partout)

## Conventions

- Palette couleurs : teal `#0d9488` (Hauteur) / amber `#d97706` (Débit)
- Fuseau horaire : `Europe/Paris` (via `uPlot.tzDate`)
- Locale d'affichage : `fr-FR`
- Station par défaut dans le frontend : `J706062001` (Châteaubourg)
- Layout quantile interleaved : pour chaque horizon `[q10, q50, q90]`, indices via `output_map[station].h_start * nQuantiles + j * nQuantiles`
