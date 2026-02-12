# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

Vigicrue — Visualisation des hauteurs d'eau et débits des stations hydrométriques françaises via les données ouvertes de [Hydro EauFrance](https://www.hydro.eaufrance.fr).

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
Browser → nginx (vigicrue.haruni.net:443)
              ├── / → fichiers statiques (frontend/dist/)
              └── /api → proxy_pass → localhost:3001 (Express via pm2)
```

```bash
# Build frontend
npm -C frontend run build

# Lancer le backend avec pm2
pm2 start ecosystem.config.cjs

# Copier la config nginx (adapter le chemin root dans vigicrue.conf d'abord)
sudo cp nginx/vigicrue.conf /etc/nginx/sites-available/vigicrue
sudo ln -s /etc/nginx/sites-available/vigicrue /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## Architecture

Monorepo avec deux sous-projets indépendants (chacun a son propre `package.json`) :

- **backend/** — Proxy Express (Node.js) vers l'API Hydro EauFrance
- **frontend/** — SPA SolidJS + uPlot pour visualiser les courbes hydrologiques

### Chaîne de requêtes en dev

```
Browser → Vite (:5173) → proxy /api → Express (:3001) → hydro.eaufrance.fr
```

Vite proxie `/api` vers le backend (configuré dans `frontend/vite.config.js`). Le backend existe uniquement pour contourner les restrictions CORS de l'API EauFrance — il ajoute les headers nécessaires (`X-Requested-With`, `Referer`, etc.).

### Flux de données

1. `App.jsx` : formulaire station/dates → fetch parallèle des séries H (hauteur) et Q (débit)
2. L'API retourne `{ series: { data: [{ t, v }, ...] } }` — timestamps ISO + valeurs en mm (H) ou L/s (Q)
3. `HydroChart.jsx` : `buildData()` convertit en tableaux uPlot — timestamps Unix (secondes), valeurs en m (÷1000) et m³/s (÷1000). Les séries Q sont alignées sur les timestamps de H via une Map.
4. Graphique double axe : H à gauche (scale `"H"`, side 3), Q à droite (scale `"Q"`, side 1). Les splits de l'axe Q sont calés sur le nombre de splits de l'axe H pour un alignement visuel.

## Stack technique

- **Backend** : Express 4, cors, `node --watch` (pas de nodemon)
- **Frontend** : SolidJS, uPlot via `@dschz/solid-uplot` (plugins: cursor, tooltip, focusSeries), Vite 6
- **Pas de TypeScript** — JS pur (ESM partout)

## Conventions

- Palette couleurs : teal `#0d9488` (Hauteur) / amber `#d97706` (Débit)
- Fuseau horaire : `Europe/Paris` (via `uPlot.tzDate`)
- Locale d'affichage : `fr-FR`
- Station par défaut dans le frontend : `J706062001`
