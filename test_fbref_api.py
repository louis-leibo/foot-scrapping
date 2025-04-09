import os
import pandas as pd
from pathlib import Path
import soccerdata as sd

# Script pour tester les capacités de l'API FBref via soccerdata
# Ce script explore les différentes méthodes disponibles et vérifie
# si nous pouvons accéder aux données des ligues européennes spécifiées

# Créer le dossier de sortie s'il n'existe pas
os.makedirs("data", exist_ok=True)

# Initialiser l'objet FBref
fbref = sd.FBref(leagues=["ENG-Premier League"], seasons=[2021])

# Tester la méthode read_player_season_stats avec différents types de statistiques
print("Test de read_player_season_stats avec stat_type='standard'")
try:
    player_stats_standard = fbref.read_player_season_stats(stat_type="standard")
    print(f"Colonnes disponibles: {player_stats_standard.columns.tolist()}")
    print(f"Nombre de joueurs: {len(player_stats_standard)}")
    # Sauvegarder un échantillon
    player_stats_standard.head(10).to_csv("data/player_stats_standard_sample.csv", index=False)
    print("Échantillon sauvegardé dans data/player_stats_standard_sample.csv")
except Exception as e:
    print(f"Erreur: {e}")

# Tester avec d'autres types de statistiques
stat_types = ["passing", "shooting", "possession", "defense", "misc"]
for stat_type in stat_types:
    print(f"\nTest de read_player_season_stats avec stat_type='{stat_type}'")
    try:
        player_stats = fbref.read_player_season_stats(stat_type=stat_type)
        print(f"Colonnes disponibles: {player_stats.columns.tolist()}")
        print(f"Nombre de joueurs: {len(player_stats)}")
        # Sauvegarder un échantillon
        player_stats.head(5).to_csv(f"data/player_stats_{stat_type}_sample.csv", index=False)
        print(f"Échantillon sauvegardé dans data/player_stats_{stat_type}_sample.csv")
    except Exception as e:
        print(f"Erreur avec {stat_type}: {e}")

# Tester la disponibilité des ligues
print("\nTest de disponibilité des ligues")
# Liste des ligues à tester
leagues_to_test = [
    # Europe de l'Ouest - Premières divisions
    "ENG-Premier League",
    "FRA-Ligue 1",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "POR-Primeira Liga",
    "NED-Eredivisie",
    
    # Europe de l'Ouest - Deuxièmes divisions
    "ENG-Championship",
    "FRA-Ligue 2",
    "ESP-Segunda División",
    "ITA-Serie B",
    "GER-2. Bundesliga",
    "POR-Liga Portugal 2",
    "NED-Eerste Divisie",
    
    # Europe Centrale et de l'Est
    "CRO-1. HNL",  # Croatie
    "CZE-Czech First League",  # République Tchèque
    "SRB-Serbian SuperLiga",  # Serbie
    "UKR-Ukrainian Premier League",  # Ukraine
    "POL-Ekstraklasa",  # Pologne
    "ROU-Liga I",  # Roumanie
    "HUN-NB I"  # Hongrie
]

league_availability = {}
for league in leagues_to_test:
    print(f"Test de la ligue: {league}")
    try:
        fbref_test = sd.FBref(leagues=[league], seasons=[2021])
        player_stats = fbref_test.read_player_season_stats(stat_type="standard")
        league_availability[league] = {
            "disponible": True,
            "nombre_joueurs": len(player_stats)
        }
        print(f"  ✓ Disponible - {len(player_stats)} joueurs")
    except Exception as e:
        league_availability[league] = {
            "disponible": False,
            "erreur": str(e)
        }
        print(f"  ✗ Non disponible - Erreur: {e}")

# Sauvegarder les résultats de disponibilité des ligues
league_availability_df = pd.DataFrame.from_dict(league_availability, orient="index")
league_availability_df.to_csv("data/league_availability.csv")
print("\nRésultats de disponibilité des ligues sauvegardés dans data/league_availability.csv")

print("\nTest terminé")
