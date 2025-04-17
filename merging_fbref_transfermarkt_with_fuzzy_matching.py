import pandas as pd
from rapidfuzz import process, fuzz
import unidecode
import os 

'''
# === File paths ===
performance_file = "Fbref-data/Cleaned_final_data/merged_serieA_2020_2024.csv"
market_value_file = "Transfermarkt-data/cleaned_final_data/serieA_cleaned.csv"
output_file = "merged_serieA_with_market_value.csv"
unmatched_output = "unmatched_players_serieA.csv"
'''

# === League configuration ===
leagues = ["serieA", "ligue1", "premier_league", "bundesliga", "laliga"]

# === Directories ===
perf_dir = "Fbref-data/Cleaned_final_data"
market_dir = "Transfermarkt-data/cleaned_final_data"
output_dir = "merged_with_market_value"
os.makedirs(output_dir, exist_ok=True)

# === Normalize player names ===
def normalize_name(name):
    return unidecode.unidecode(str(name).lower().strip())

# === Fuzzy match function ===
def fuzzy_match(name, choices, threshold=80):
    match, score, _ = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    return match if score >= threshold else None

# === Process each league ===
for league in leagues:
    print(f"\n🔁 Processing: {league.upper()}")

    # File paths
    performance_file = f"{perf_dir}/merged_{league}_2020_2024.csv"
    market_file = f"{market_dir}/{league}_cleaned.csv"
    output_file = f"{output_dir}/merged_{league}_with_market_value.csv"
    unmatched_output = f"{output_dir}/unmatched_players_{league}.csv"

    # Load data
    try:
        df_perf = pd.read_csv(performance_file)
        df_market = pd.read_csv(market_file)
    except Exception as e:
        print(f"⚠️ Failed to load files for {league}: {e}")
        continue

    # Standardize columns
    df_market = df_market.rename(columns={"Nom": "player", "Saison": "season"})
    df_market["season"] = df_market["season"].astype(str).str[:4]
    df_perf["season"] = df_perf["season"].astype(str)

    # Normalize names
    df_perf["player_norm"] = df_perf["player"].apply(normalize_name)
    df_market["player_norm"] = df_market["player"].apply(normalize_name)

    # Build fuzzy name map
    market_names = df_market["player_norm"].unique()
    name_map = {
        name: fuzzy_match(name, market_names)
        for name in df_perf["player_norm"].unique()
    }
    df_perf["player_matched"] = df_perf["player_norm"].map(name_map)

    # Merge on matched name + season
    df_merged = pd.merge(
        df_perf,
        df_market[["player_norm", "season", "Valeur marchande (euros)"]],
        left_on=["player_matched", "season"],
        right_on=["player_norm", "season"],
        how="left"
    )

    # Drop unmatched
    initial_rows = len(df_merged)
    df_merged = df_merged[df_merged["Valeur marchande (euros)"].notna()]
    final_rows = len(df_merged)
    removed_rows = initial_rows - final_rows

    print(f"🧹 Removed {removed_rows} rows (missing market value). Final size: {final_rows} rows.")

    # Save merged data
    df_merged.to_csv(output_file, index=False)
    print(f"✅ Merged file saved: {output_file}")
