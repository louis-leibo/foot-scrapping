import pandas as pd
import os

# Folder containing the final merged datasets
input_dir = "Data_merged_with_market_value"
output_file = "Data_merged_with_market_value/all_leagues_merged_final_df.csv"

# List of expected files per league
file_names = [
    "merged_ligue1_with_market_value.csv",
    "merged_serieA_with_market_value.csv",
    "merged_premier_league_with_market_value.csv",
    "merged_bundesliga_with_market_value.csv",
    "merged_laliga_with_market_value.csv"
]

# Load and combine all DataFrames
dfs = []
for file_name in file_names:
    file_path = os.path.join(input_dir, file_name)
    try:
        df = pd.read_csv(file_path)
        dfs.append(df)
        print(f"✅ Loaded: {file_name}")
    except Exception as e:
        print(f"⚠️ Could not load {file_name}: {e}")

# Concatenate into one big DataFrame
df_combined = pd.concat(dfs, ignore_index=True)

# Ensure 'season' is string and sorted numerically
df_combined["season"] = df_combined["season"].astype(int)
df_combined.sort_values(by="season", inplace=True)

# Save final merged file
df_combined.to_csv(output_file, index=False)
print(f"📦 Final combined file saved to: {output_file}")
