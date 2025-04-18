import pandas as pd

# === File paths ===
performance_file = "Fbref-data/Cleaned_final_data/merged_serieA_2020_2024.csv"
market_value_file = "Transfermarkt-data/cleaned_final_data/serieA_cleaned.csv"
output_file = "merged_serieA_with_market_value.csv"

# === Load data ===
print("📥 Loading datasets...")
df_perf = pd.read_csv(performance_file)
df_market = pd.read_csv(market_value_file)

# === Standardize column names for merging ===
df_market = df_market.rename(columns={"Nom": "player", "Saison": "season"})

# ✅ Normalize season format in market value dataset (e.g., '2020/2021' → '2020')
df_market["season"] = df_market["season"].astype(str).str[:4]

# ✅ Ensure season in performance dataset is also a string
df_perf["season"] = df_perf["season"].astype(str)

# === Merge datasets ===
print("🔄 Merging on ['player', 'season']...")
df_merged = pd.merge(
    df_perf,
    df_market[["player", "season", "Valeur marchande (euros)"]],
    on=["player", "season"],
    how="left"
)

# ✅ Drop rows where 'Valeur marchande (euros)' is missing
initial_rows = len(df_merged)
df_merged = df_merged[df_merged["Valeur marchande (euros)"].notna()]
final_rows = len(df_merged)
removed_rows = initial_rows - final_rows

print(f"🧹 Removed {removed_rows} rows with missing market value.")
print(f"✅ Final dataset size: {final_rows} rows.")

# === Save result ===
df_merged.to_csv(output_file, index=False)
print(f"📦 Merged file saved to: {output_file}")