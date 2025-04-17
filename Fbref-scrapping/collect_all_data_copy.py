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

LEAGUES = [
    "ENG-Premier League"
    # "ESP-La Liga",         # Spain - La Liga
    # "FRA-Ligue 1",         # France - Ligue 1
    # "GER-Bundesliga",      # Germany - Bundesliga
    # "ITA-Serie A",         # Italy - Serie A
]

SEASONS = [2020, 2021, 2022, 2023, 2024]

STAT_TYPES = [
    "standard",
    "passing",
    "shooting",
    "possession",
    "defense",
    "misc"
]

def collect_all_data():
    logger.info("Starting data collection for all leagues and seasons")
    os.makedirs("data", exist_ok=True)
    all_stats = {}

    for league in LEAGUES:
        logger.info(f"Processing league: {league}")

        for season in SEASONS:
            logger.info(f"  Processing season: {season}")

            fbref = sd.FBref(leagues=[league], seasons=[season])

            for stat_type in STAT_TYPES:
                logger.info(f"    Collecting {stat_type} stats")

                try:
                    stats = fbref.read_player_season_stats(stat_type=stat_type)
                    
                    # Ensure player name is included by resetting index
                    stats = stats.reset_index()

                    # Validate that 'player' column exists
                    if 'player' not in stats.columns:
                        logger.warning(f"Missing 'player' column in stats for {league} {season} - {stat_type}")
                        stats.rename(columns={stats.columns[0]: 'player'}, inplace=True)

                    # Drop rows without player name
                    stats = stats[stats['player'].notna()]

                    # Add metadata
                    stats['league'] = league
                    stats['season'] = season
                    stats['stat_type'] = stat_type

                    # Move 'player' column to the front
                    columns = list(stats.columns)
                    if 'player' in columns:
                        columns.insert(0, columns.pop(columns.index('player')))
                        stats = stats[columns]

                    if stat_type not in all_stats:
                        all_stats[stat_type] = stats
                    else:
                        all_stats[stat_type] = pd.concat([all_stats[stat_type], stats], ignore_index=True)

                    logger.info(f"      Successfully retrieved {len(stats)} records")

                    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    intermediate_file = f"data/player_name_intermediate_{league}_{season}_{stat_type}.csv"
                    stats.to_csv(intermediate_file, index=False)
                    logger.info(f"      Saved intermediate results to {intermediate_file}")

                    time.sleep(2)

                except Exception as e:
                    logger.error(f"      Error collecting {stat_type} stats for {league} {season}: {e}")

    logger.info("Merging all statistics")

    if 'standard' in all_stats and not all_stats['standard'].empty:
        merged_df = all_stats['standard'].copy()
        id_cols = ['player', 'team', 'league', 'season']

        for stat_type, df in all_stats.items():
            if stat_type != 'standard' and not df.empty:
                logger.info(f"Merging {stat_type} stats")
                cols_to_use = [col for col in df.columns if col not in id_cols or col in id_cols]
                merged_df = pd.merge(
                    merged_df,
                    df[cols_to_use],
                    on=id_cols,
                    how='outer',
                    suffixes=('', f'_{stat_type}')
                )

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"Fbref-data/complete_dataset_{league}_{season}.csv"
        merged_df.to_csv(output_file, index=False)
        logger.info(f"Saved complete dataset to {output_file}")
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
        print(f"\n✅ Data collection complete! Dataset saved to: {output_file}")
    else:
        print("\n❌ Data collection failed. Check the logs for details.")

