import soccerdata as sd
import pandas as pd
import os
from datetime import datetime
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define leagues and seasons
LEAGUES = [
    "ENG-Premier League",  # England - Premier League
    "ESP-La Liga",         # Spain - La Liga
    "FRA-Ligue 1",         # France - Ligue 1
    "GER-Bundesliga",      # Germany - Bundesliga
    "ITA-Serie A",         # Italy - Serie A
]

# Last 5 seasons (2020-2024)
SEASONS = [2020, 2021, 2022, 2023, 2024]

# Types of statistics to collect
STAT_TYPES = [
    "standard",   # Standard stats (goals, assists, etc.)
    "passing",    # Passing stats
    "shooting",   # Shooting stats
    "possession", # Possession stats
    "defense",    # Defense stats
    "misc"        # Miscellaneous stats
]

def collect_all_data():
    """
    Collect data for all leagues, seasons, and stat types.
    """
    logger.info("Starting data collection for all leagues and seasons")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Initialize empty DataFrames for each stat type
    all_stats = {}
    
    # Collect data for each league, season, and stat type
    for league in LEAGUES:
        logger.info(f"Processing league: {league}")
        
        for season in SEASONS:
            logger.info(f"  Processing season: {season}")
            
            # Initialize FBref client for this league and season
            fbref = sd.FBref(
                leagues=[league],
                seasons=[season]
            )
            
            # Collect each type of statistic
            for stat_type in STAT_TYPES:
                logger.info(f"    Collecting {stat_type} stats")
                
                try:
                    # Get the stats
                    stats = fbref.read_player_season_stats(stat_type=stat_type)
                    
                    # Add league and season information
                    stats['league'] = league
                    stats['season'] = season
                    stats['stat_type'] = stat_type
                    
                    # Add to the appropriate DataFrame
                    if stat_type not in all_stats:
                        all_stats[stat_type] = stats
                    else:
                        all_stats[stat_type] = pd.concat([all_stats[stat_type], stats], ignore_index=True)
                    
                    logger.info(f"      Successfully retrieved {len(stats)} records")
                    
                    # Save intermediate results after each collection
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    intermediate_file = f"data/intermediate_{league}_{season}_{stat_type}_{timestamp}.csv"
                    stats.to_csv(intermediate_file, index=False)
                    logger.info(f"      Saved intermediate results to {intermediate_file}")
                    
                    # Sleep to avoid rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"      Error collecting {stat_type} stats for {league} {season}: {e}")
    
    # Merge all statistics
    logger.info("Merging all statistics")
    
    # Start with standard stats as the base
    if 'standard' in all_stats and not all_stats['standard'].empty:
        merged_df = all_stats['standard'].copy()
        
        # Columns to use for merging
        id_cols = ['player', 'team', 'league', 'season']
        
        # Merge with other stat types
        for stat_type, df in all_stats.items():
            if stat_type != 'standard' and not df.empty:
                logger.info(f"Merging {stat_type} stats")
                
                # Get columns that aren't in the ID columns
                cols_to_use = [col for col in df.columns if col not in id_cols or col in id_cols]
                
                # Merge the DataFrames
                merged_df = pd.merge(
                    merged_df, 
                    df[cols_to_use], 
                    on=id_cols, 
                    how='outer',
                    suffixes=('', f'_{stat_type}')
                )
        
        # Save the merged dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/complete_dataset_{timestamp}.csv"
        merged_df.to_csv(output_file, index=False)
        logger.info(f"Saved complete dataset to {output_file}")
        
        # Print dataset summary
        logger.info(f"Dataset contains {len(merged_df)} rows and {len(merged_df.columns)} columns")
        logger.info(f"Number of players: {merged_df['player'].nunique()}")
        logger.info(f"Number of seasons: {merged_df['season'].nunique()}")
        
        return output_file
    else:
        logger.error("No standard stats available for merging")
        return None

if __name__ == "__main__":
    output_file = collect_all_data()
    if output_file:
        print(f"\nData collection complete! Dataset saved to: {output_file}")
    else:
        print("\nData collection failed. Check the logs for details.") 