"""
Module principal pour le scraping des données FBref.

Ce module contient les fonctions nécessaires pour collecter les statistiques
des joueurs des principales ligues européennes à partir de FBref.
"""

import os
import pandas as pd
import soccerdata as sd
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple

# Import de la configuration
from config import AVAILABLE_LEAGUES, TARGET_SEASONS, STAT_TYPES

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FBrefScraper:
    """
    Classe pour scraper les données de FBref en utilisant la bibliothèque soccerdata.
    """
    
    def __init__(self, data_dir: str = "data", no_cache: bool = False):
        """
        Initialise le scraper FBref.
        
        Args:
            data_dir: Répertoire où les données seront stockées
            no_cache: Si True, n'utilisera pas les données en cache
        """
        self.data_dir = Path(data_dir)
        self.no_cache = no_cache
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialiser les DataFrames pour stocker les données
        self.player_stats = {}
        
        logger.info(f"FBrefScraper initialisé avec data_dir={data_dir}, no_cache={no_cache}")
    
    def scrape_league_season_stats(self, league: str, season: int, stat_type: str) -> pd.DataFrame:
        """
        Scrape les statistiques des joueurs pour une ligue et une saison spécifiques.
        
        Args:
            league: Identifiant de la ligue (ex: "ENG-Premier League")
            season: Saison (ex: 2021 pour 2020-2021)
            stat_type: Type de statistiques à collecter
            
        Returns:
            DataFrame contenant les statistiques des joueurs
        """
        logger.info(f"Scraping des statistiques {stat_type} pour {league} saison {season}")
        
        try:
            # Initialiser l'objet FBref
            fbref = sd.FBref(leagues=[league], seasons=[season], no_cache=self.no_cache)
            
            # Collecter les statistiques des joueurs
            player_stats = fbref.read_player_season_stats(stat_type=stat_type)
            
            # Ajouter des colonnes pour faciliter l'identification
            player_stats['league'] = league
            player_stats['season'] = season
            player_stats['stat_type'] = stat_type
            
            logger.info(f"Collecté {len(player_stats)} joueurs pour {league} saison {season}, type {stat_type}")
            
            return player_stats
        
        except Exception as e:
            logger.error(f"Erreur lors du scraping de {league} saison {season}, type {stat_type}: {e}")
            return pd.DataFrame()
    
    def scrape_all_leagues_seasons(self, leagues: List[str] = None, 
                                  seasons: List[int] = None,
                                  stat_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Scrape les statistiques des joueurs pour toutes les ligues et saisons spécifiées.
        
        Args:
            leagues: Liste des ligues à scraper (utilise AVAILABLE_LEAGUES par défaut)
            seasons: Liste des saisons à scraper (utilise TARGET_SEASONS par défaut)
            stat_types: Liste des types de statistiques à collecter (utilise STAT_TYPES par défaut)
            
        Returns:
            Dictionnaire de DataFrames contenant les statistiques des joueurs par type de statistique
        """
        # Utiliser les valeurs par défaut si non spécifiées
        leagues = leagues or AVAILABLE_LEAGUES
        seasons = seasons or TARGET_SEASONS
        stat_types = stat_types or STAT_TYPES
        
        logger.info(f"Début du scraping pour {len(leagues)} ligues, {len(seasons)} saisons, {len(stat_types)} types de stats")
        
        # Initialiser les DataFrames pour chaque type de statistique
        for stat_type in stat_types:
            self.player_stats[stat_type] = pd.DataFrame()
        
        # Scraper les données pour chaque combinaison de ligue, saison et type de statistique
        for league in leagues:
            for season in seasons:
                for stat_type in stat_types:
                    try:
                        # Scraper les données
                        df = self.scrape_league_season_stats(league, season, stat_type)
                        
                        # Ajouter les données au DataFrame correspondant
                        if not df.empty:
                            if self.player_stats[stat_type].empty:
                                self.player_stats[stat_type] = df
                            else:
                                self.player_stats[stat_type] = pd.concat([self.player_stats[stat_type], df], ignore_index=True)
                        
                    except Exception as e:
                        logger.error(f"Erreur lors du scraping de {league} saison {season}, type {stat_type}: {e}")
        
        logger.info("Scraping terminé")
        
        return self.player_stats
    
    def save_data_to_csv(self, output_dir: str = None) -> Dict[str, str]:
        """
        Sauvegarde les données scrapées dans des fichiers CSV.
        
        Args:
            output_dir: Répertoire où sauvegarder les fichiers CSV (utilise self.data_dir par défaut)
            
        Returns:
            Dictionnaire des chemins des fichiers CSV créés
        """
        output_dir = Path(output_dir) if output_dir else self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_files = {}
        
        for stat_type, df in self.player_stats.items():
            if not df.empty:
                # Créer le nom du fichier
                filename = f"player_stats_{stat_type}_{timestamp}.csv"
                filepath = output_dir / filename
                
                # Sauvegarder le DataFrame
                df.to_csv(filepath, index=False)
                csv_files[stat_type] = str(filepath)
                
                logger.info(f"Données {stat_type} sauvegardées dans {filepath} ({len(df)} lignes)")
        
        return csv_files
    
    def merge_all_stats(self) -> pd.DataFrame:
        """
        Fusionne toutes les statistiques en un seul DataFrame.
        
        Returns:
            DataFrame contenant toutes les statistiques fusionnées
        """
        if not self.player_stats or all(df.empty for df in self.player_stats.values()):
            logger.warning("Aucune donnée à fusionner")
            return pd.DataFrame()
        
        logger.info("Fusion de toutes les statistiques")
        
        # Commencer avec les statistiques standard comme base
        if 'standard' in self.player_stats and not self.player_stats['standard'].empty:
            merged_df = self.player_stats['standard'].copy()
            
            # Colonnes d'identification pour la fusion
            id_cols = ['player', 'team', 'league', 'season']
            
            # Fusionner avec les autres types de statistiques
            for stat_type, df in self.player_stats.items():
                if stat_type != 'standard' and not df.empty:
                    # Supprimer les colonnes en double
                    cols_to_use = [col for col in df.columns if col not in id_cols or col in id_cols]
                    
                    # Fusionner les DataFrames
                    merged_df = pd.merge(
                        merged_df, 
                        df[cols_to_use], 
                        on=id_cols, 
                        how='outer',
                        suffixes=('', f'_{stat_type}')
                    )
            
            logger.info(f"Fusion terminée, DataFrame final contient {len(merged_df)} lignes et {len(merged_df.columns)} colonnes")
            return merged_df
        else:
            logger.warning("Statistiques standard non disponibles pour la fusion")
            return pd.DataFrame()


def main():
    """
    Fonction principale pour exécuter le scraping.
    """
    # Créer le scraper
    scraper = FBrefScraper()
    
    # Scraper toutes les données
    player_stats = scraper.scrape_all_leagues_seasons()
    
    # Sauvegarder les données
    csv_files = scraper.save_data_to_csv()
    
    # Fusionner toutes les statistiques
    merged_stats = scraper.merge_all_stats()
    
    # Sauvegarder les statistiques fusionnées
    if not merged_stats.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_file = f"data/player_stats_all_{timestamp}.csv"
        merged_stats.to_csv(merged_file, index=False)
        logger.info(f"Statistiques fusionnées sauvegardées dans {merged_file}")
    
    return csv_files


if __name__ == "__main__":
    main()
