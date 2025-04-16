# FBref Data Scraper

Ce projet utilise la bibliothèque `soccerdata` pour collecter les statistiques des joueurs de football des principales ligues européennes sur les cinq dernières saisons.

## Objectif

Collecter toutes les données statistiques disponibles au niveau des joueurs pour un ensemble sélectionné de ligues européennes sur les cinq dernières saisons.

## Ligues ciblées

Le projet se concentre sur les premières divisions (et, lorsqu'elles sont disponibles, les deuxièmes divisions) des pays européens qui exportent fréquemment des joueurs vers les clubs de premier plan européens :

### Europe de l'Ouest
- France (Ligue 1, Ligue 2)
- Angleterre (Premier League, Championship)
- Espagne (La Liga, Segunda División)
- Allemagne (Bundesliga, 2. Bundesliga)
- Italie (Serie A, Serie B)

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Structure du projet

- `src/` - Code source du projet
  - `scraper/` - Scripts de scraping des données
  - `utils/` - Fonctions utilitaires
- `data/` - Données collectées
- `notebooks/` - Notebooks Jupyter pour l'exploration et l'analyse des données
- `tests/` - Tests unitaires

## Utilisation

[Instructions d'utilisation à venir]

## Licence

[Informations sur la licence à venir]
