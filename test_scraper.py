"""
Script pour tester les fonctions de scraping sur un échantillon réduit.

Ce script utilise la classe FBrefScraper pour collecter un échantillon
de données afin de vérifier que tout fonctionne correctement avant
de lancer la collecte complète.
"""

import os
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Import du scraper
from .scraper import FBrefScraper

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_sample_scraping():
    """
    Teste le scraping sur un échantillon réduit de données.
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


if __name__ == "__main__":
    test_sample_scraping()
