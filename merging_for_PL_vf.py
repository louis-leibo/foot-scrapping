import pandas as pd
import glob
import os

# Function to create composite key and move it to the front
def create_composite_key(df):
    df["composite_key"] = (
        df["nation"].astype(str).str.strip() + "_" +
        df["pos"].astype(str).str.strip() + "_" +
        df["age"].astype(str).str.strip() + "_" +
        df["born"].astype(str).str.strip()
    )
    cols = ["composite_key"] + [col for col in df.columns if col != "composite_key"]
    return df[cols]

# Function to preprocess individual DataFrame
def preprocess(df):
    df = create_composite_key(df)
    df = df.drop_duplicates(subset="composite_key", keep="first")
    return df

# Function to load a file by type using wildcard
def load_file(season, stat_type):
    pattern = f"Data/intermediate_ENG-Premier League_{season}_{stat_type}_*.csv"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file found for {season} - {stat_type}")
    return pd.read_csv(files[0])

# Loop through all seasons
for season in range(2020, 2025):
    print(f"🔄 Processing season {season}...")

    # Load all stat files using wildcards
    standard   = preprocess(load_file(season, "standard"))
    shooting   = preprocess(load_file(season, "shooting"))
    passing    = preprocess(load_file(season, "passing"))
    possession = preprocess(load_file(season, "possession"))
    defense    = preprocess(load_file(season, "defense"))
    misc       = preprocess(load_file(season, "misc"))

    # Merge all files
    merged = standard
    for df in [shooting, passing, possession, defense, misc]:
        merged = merged.merge(df, on="composite_key", how="outer", suffixes=('', '_dup'))

    # Drop full duplicate rows
    merged.drop_duplicates(inplace=True)

    # Optional: keep only one row per player
    merged = merged.sort_values(by="composite_key")
    merged = merged.drop_duplicates(subset="composite_key", keep="first")

    # Drop constant columns (except for a few key metadata ones)
    constant_cols = []
    for col in merged.columns:
        try:
            if merged[col].nunique(dropna=False) <= 1:
                constant_cols.append(col)
        except Exception:
            continue

    for key_col in ['league', 'season', 'stat_type']:
        if key_col in constant_cols:
            constant_cols.remove(key_col)

    merged.drop(columns=constant_cols, inplace=True)

    # Drop rows with missing values
    merged.dropna(inplace=True)

    # Save final cleaned dataset
    output_path = f"Data_merged/merged_premier_league_{season}.csv"
    merged.to_csv(output_path, index=False)
    print(f"✅ Season {season} complete. File saved as '{output_path}'")

print("🎉 All seasons processed and saved!")