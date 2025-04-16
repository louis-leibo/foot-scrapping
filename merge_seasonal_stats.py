import os
import pandas as pd
import logging
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
FILENAME_PATTERN = r"player_name_intermediate_(?P<league>.+?)_(?P<season>\d{4})_(?P<stat_type>\w+)\.csv"

def collect_all_data():
    logger.info("🔍 Scanning CSV files...")

    # Collect files grouped by league and season
    files_by_league_season = defaultdict(lambda: defaultdict(str))

    for file in os.listdir(DATA_DIR):
        logger.debug(f"Checking file: {file}")
        match = re.match(FILENAME_PATTERN, file)
        if match:
            league = match.group("league")
            season = match.group("season")
            stat_type = match.group("stat_type")
            files_by_league_season[(league, season)][stat_type] = os.path.join(DATA_DIR, file)
        else:
            logger.debug(f"Skipping unmatched file: {file}")

    all_outputs = []

    for (league, season), stat_files in files_by_league_season.items():
        logger.info(f"\n📦 Merging stats for {league} - {season}")

        all_stats = {}
        for stat_type, path in stat_files.items():
            try:
                df = pd.read_csv(path)
                all_stats[stat_type] = df
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                continue

        if 'standard' in all_stats and not all_stats['standard'].empty:
            merged_df = all_stats['standard'].copy()

            id_cols = ['player', 'team', 'league', 'season']

            for stat_type, df in all_stats.items():
                if stat_type != 'standard' and not df.empty:
                    logger.info(f"➕ Merging {stat_type} stats")

                    # Keep only non-ID columns to avoid duplication of keys
                    cols_to_use = [col for col in df.columns if col not in id_cols]
                    
                    # Keep the ID columns + the rest for merge
                    df_to_merge = df[id_cols + cols_to_use]

                    merged_df = pd.merge(
                        merged_df,
                        df_to_merge,
                        on=id_cols,
                        how='outer',
                        suffixes=('', f'_{stat_type}')
                    )


            # Create safe filename (replace spaces/dashes with underscores)
            league_safe = league.replace(" ", "_").replace("-", "_")
            output_file = f"{DATA_DIR}/complete_dataset_{league_safe}_{season}.csv"
            merged_df.to_csv(output_file, index=False)
            logger.info(f"✅ Saved: {output_file}")
            all_outputs.append(output_file)
        else:
            logger.warning(f"⚠️ No standard stats found for {league} - {season}, skipping...")

    return all_outputs


if __name__ == "__main__":
    outputs = collect_all_data()
    if outputs:
        print("\n✅ Data collection complete!")
        for f in outputs:
            print(f"📁 {f}")
    else:
        print("\n❌ Data collection failed. Check the logs for details.")
