import pandas as pd
from scraper_utils import get_soup
import time
import random

def get_players_with_urls(team_url):
    soup = get_soup(team_url)
    if not soup:
        return []

    table = soup.find("table", class_="items")
    if not table:
        return []

    rows = table.find_all("tr", class_=["odd", "even"])
    players = []

    for row in rows:
        name_td = row.find("td", class_="hauptlink")
        if name_td and name_td.find("a"):
            link = name_td.find("a")
            name = link.text.strip()
            href = link.get("href")
            player_url = "https://www.transfermarkt.fr" + href
            players.append((name.lower(), player_url))

    return players

def update_ligue1_csv_with_urls():
    input_file = "valeurs_marchandes_ligue1.csv"
    output_file = "valeurs_marchandes_ligue1_with_urls.csv"

    df = pd.read_csv(input_file)

    if "Lien" in df.columns:
        print("✅ La colonne 'Lien' existe déjà. Rien à faire.")
        return

    print(f"📥 Chargement du fichier : {input_file}")
    all_players = df["Nom"].str.lower()
    unique_seasons = df["Saison"].unique()
    unique_teams = df["Équipe"].unique()

    player_url_map = {}

    for season in unique_seasons:
        year = season.split("/")[0]
        print(f"\n🔄 Saison {season}")
        for team_slug in unique_teams:
            team_url = f"https://www.transfermarkt.fr/{team_slug}/startseite/verein/0/saison_id/{year}"
            print(f"    🔍 {team_url}")
            players = get_players_with_urls(team_url)
            for name, url in players:
                key = f"{name}_{team_slug}_{season}"
                player_url_map[key] = url
            time.sleep(random.uniform(2, 4))

    liens = []
    for idx, row in df.iterrows():
        key = f"{row['Nom'].lower()}_{row['Équipe']}_{row['Saison']}"
        lien = player_url_map.get(key)
        liens.append(lien)

    df["Lien"] = liens
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n✅ Fichier enrichi sauvegardé : {output_file}")
