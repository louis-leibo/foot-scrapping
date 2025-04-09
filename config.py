"""
Configuration des ligues et saisons cibles pour la collecte de données FBref.

Ce module définit les ligues et saisons disponibles pour la collecte de données
à partir de FBref via la bibliothèque soccerdata.
"""

# Ligues disponibles via l'API FBref
AVAILABLE_LEAGUES = [
    "ENG-Premier League",  # Angleterre - Premier League
    "ESP-La Liga",         # Espagne - La Liga
    "FRA-Ligue 1",         # France - Ligue 1
    "GER-Bundesliga",      # Allemagne - Bundesliga
    "ITA-Serie A",         # Italie - Serie A
]

# Saisons à collecter (les 5 dernières saisons)
# Note: FBref utilise le format "YYYY" pour les saisons (ex: 2021 pour la saison 2020-2021)
TARGET_SEASONS = [
    2020,  # Saison 2019-2020
    2021,  # Saison 2020-2021
    2022,  # Saison 2021-2022
    2023,  # Saison 2022-2023
    2024,  # Saison 2023-2024
]

# Types de statistiques à collecter
STAT_TYPES = [
    "standard",   # Statistiques standard (buts, passes décisives, etc.)
    "passing",    # Statistiques de passes
    "shooting",   # Statistiques de tirs
    "possession", # Statistiques de possession
    "defense",    # Statistiques défensives
    "misc"        # Statistiques diverses
]

# Mapping des ligues demandées aux ligues disponibles
# Pour référence des ligues qui étaient demandées mais ne sont pas disponibles
LEAGUE_MAPPING = {
    # Ligues disponibles
    "England": "ENG-Premier League",
    "Spain": "ESP-La Liga",
    "France": "FRA-Ligue 1",
    "Germany": "GER-Bundesliga",
    "Italy": "ITA-Serie A",
    
    # Ligues non disponibles
    "Croatia": None,
    "Czech Republic": None,
    "Serbia": None,
    "Ukraine": None,
    "Poland": None,
    "Romania": None,
    "Hungary": None,
    "Portugal": None,
    "Netherlands": None,
    
    # Deuxièmes divisions non disponibles
    "England 2": None,
    "Spain 2": None,
    "France 2": None,
    "Germany 2": None,
    "Italy 2": None,
    "Portugal 2": None,
    "Netherlands 2": None
}
