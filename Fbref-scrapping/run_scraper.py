"""
Script principal pour exécuter la collecte de données FBref.

Ce script utilise le module scraper pour collecter les statistiques
des joueurs des principales ligues européennes sur les cinq dernières saisons.
"""

import sys
import os
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import du scraper
from scraper.scraper import FBrefScraper
from scraper.config import AVAILABLE_LEAGUES, TARGET_SEASONS, STAT_TYPES

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_test_scraping():
    """
    Exécute un test de scraping sur un échantillon réduit.
    """
    logger.info("Début du test de scraping sur un échantillon réduit")
    
    # Créer le répertoire de test s'il n'existe pas
    test_dir = Path("data/test")
    os.makedirs(test_dir, exist_ok=True)
    
    # Créer le scraper
    scraper = FBrefScraper(data_dir=str(test_dir))
    
    # Définir un échantillon réduit de ligues et saisons
    sample_leagues = ["ENG-Premier League", "ESP-La Liga"]
    sample_seasons = [2023]  # Juste la dernière saison pour le test
    sample_stat_types = ["standard", "passing"]  # Juste deux types de stats pour le test
    
    # Scraper l'échantillon
    logger.info(f"Scraping de {len(sample_leagues)} ligues, {len(sample_seasons)} saisons, {len(sample_stat_types)} types de stats")
    player_stats = scraper.scrape_all_leagues_seasons(
        leagues=sample_leagues,
        seasons=sample_seasons,
        stat_types=sample_stat_types
    )
    
    # Vérifier les résultats
    success = True
    for stat_type, df in player_stats.items():
        if df.empty:
            logger.error(f"Aucune donnée collectée pour le type de statistique {stat_type}")
            success = False
        else:
            logger.info(f"Collecté {len(df)} joueurs pour le type de statistique {stat_type}")
            
            # Sauvegarder un échantillon pour inspection
            sample_file = test_dir / f"sample_{stat_type}.csv"
            df.head(10).to_csv(sample_file, index=False)
            logger.info(f"Échantillon sauvegardé dans {sample_file}")
    
    # Tester la fusion des statistiques
    if success:
        logger.info("Test de la fusion des statistiques")
        merged_stats = scraper.merge_all_stats()
        
        if merged_stats.empty:
            logger.error("Échec de la fusion des statistiques")
            success = False
        else:
            logger.info(f"Fusion réussie, {len(merged_stats)} joueurs, {len(merged_stats.columns)} colonnes")
            
            # Sauvegarder un échantillon pour inspection
            sample_merged_file = test_dir / "sample_merged.csv"
            merged_stats.head(10).to_csv(sample_merged_file, index=False)
            logger.info(f"Échantillon fusionné sauvegardé dans {sample_merged_file}")
    
    # Résultat final
    if success:
        logger.info("Test de scraping réussi!")
        return True
    else:
        logger.error("Test de scraping échoué!")
        return False


def run_full_scraping():
    """
    Exécute la collecte complète des données.
    """
    logger.info("Début de la collecte complète des données")
    
    # Créer le scraper
    scraper = FBrefScraper()
    
    # Scraper toutes les données
    logger.info(f"Scraping de {len(AVAILABLE_LEAGUES)} ligues, {len(TARGET_SEASONS)} saisons, {len(STAT_TYPES)} types de stats")
    player_stats = scraper.scrape_all_leagues_seasons()
    
    # Sauvegarder les données
    csv_files = scraper.save_data_to_csv()
    
    # Fusionner toutes les statistiques
    logger.info("Fusion de toutes les statistiques")
    merged_stats = scraper.merge_all_stats()
    
    # Sauvegarder les statistiques fusionnées
    if not merged_stats.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_file = f"data/player_stats_all_{timestamp}.csv"
        merged_stats.to_csv(merged_file, index=False)
        logger.info(f"Statistiques fusionnées sauvegardées dans {merged_file}")
        return merged_file
    else:
        logger.error("Échec de la fusion des statistiques")
        return None


if __name__ == "__main__":
    # Exécuter d'abord le test
    if run_test_scraping():
        logger.info("Test réussi, lancement de la collecte complète")
        merged_file = run_full_scraping()
        if merged_file:
            logger.info(f"Collecte complète terminée avec succès. Fichier final: {merged_file}")
        else:
            logger.error("Échec de la collecte complète")
    else:
        logger.error("Test échoué, la collecte complète n'a pas été lancée")
