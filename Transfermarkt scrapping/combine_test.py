import pandas as pd

# Load files
market_df = pd.read_csv("valeurs_marchandes_premier_league_2020.csv")
stats_df = pd.read_csv("intermediate_ENG-Premier_League.csv")

# Clean player names
market_df["Nom"] = market_df["Nom"].astype(str).str.strip().str.lower()
stats_df["nation"] = stats_df["nation"].astype(str).str.strip().str.lower()
stats_df["pos"] = stats_df["pos"].astype(str).str.strip().str.lower()
stats_df["born"] = stats_df["born"].astype(str).str.strip()
stats_df["age"] = stats_df["age"].astype(str).str.strip()

# Create a helper dict from stats rows
stats_lookup = stats_df.to_dict(orient="records")

combined_rows = []

for idx, mv_row in market_df.iterrows():
    found = None
    for stat_row in stats_lookup:
        # Match based on demo: pos, nation, age, born
        if (
            stat_row["pos"] == stat_row.get("pos") and
            stat_row["nation"] == stat_row.get("nation") and
            stat_row["age"] == stat_row.get("age") and
            stat_row["born"] == stat_row.get("born")
        ):
            found = stat_row
            break

    if found:
        combined = found.copy()
        combined["Nom"] = mv_row["Nom"]
        combined["Valeur marchande"] = mv_row["Valeur marchande"]
        combined["Valeur marchande (euros)"] = mv_row["Valeur marchande (euros)"]
        combined_rows.append(combined)

# Save the result
combined_df = pd.DataFrame(combined_rows)
combined_df.to_csv("combined_defense_with_market_value.csv", index=False, encoding="utf-8")
print("✅ Fichier combiné sauvegardé : combined_defense_with_market_value.csv")
