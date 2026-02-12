# Vigicrue

Visualisation des hauteurs d'eau et debits des stations hydrometriques francaises, a partir des donnees ouvertes de [Hydro EauFrance](https://www.hydro.eaufrance.fr).

## Demarrage rapide

```bash
# Installer les dependances
npm -C backend install
npm -C frontend install

# Lancer le projet
npm run dev
```

Le frontend est accessible sur http://localhost:5173.

## Fonctionnalites

- Saisie d'un code station et d'une plage de dates
- Recuperation des series Hauteur (H) et Debit (Q) via l'API Hydro EauFrance
- Graphique interactif double axe avec tooltip
- Alignement automatique des echelles gauche (Hauteur) et droite (Debit)

## Architecture

```
vigicrue/
  backend/          Proxy Express (port 3001) vers l'API EauFrance
  frontend/         SPA SolidJS + uPlot (Vite, port 5173)
```

Le backend sert de proxy pour contourner les restrictions CORS de l'API EauFrance. Le frontend appelle `/api/station/:id/series` qui est relaye par Vite en dev.
