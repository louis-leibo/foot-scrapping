from scrape_market_values import scrape_ligue1_market_values

df = scrape_ligue1_market_values(start_season=2020, end_season=2024)

if df.empty:
    print("❌ Aucun joueur n'a été récupéré. Vérifie les URLs ou si tu es bloqué par Transfermarkt.")
else:
    df.to_csv("valeurs_marchandes_ligue1_name.csv", index=False, encoding="utf-8")
    print("✅ Fichier sauvegardé : valeurs_marchandes_ligue1_name.csv")
