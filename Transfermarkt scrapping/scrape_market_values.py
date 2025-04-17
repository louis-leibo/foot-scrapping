import pandas as pd
from scraper_utils import get_soup
from get_team_urls import get_teams_urls

def get_players_from_team(team_url):
    soup = get_soup(team_url)
    if not soup:
        print(f"❌ Failed to load team page: {team_url}")
        return []

    table = soup.find("table", class_="items")
    if not table:
        print(f"⚠️ No player table found for {team_url}")
        return []

    rows = table.find_all("tr", class_=["odd", "even"])
    players = []

    for row in rows:
        name_tag = row.find("td", class_="hauptlink")
        value_tag = row.find("td", class_="rechts hauptlink")

        if name_tag and value_tag:
            name = name_tag.get_text(strip=True)
            value = value_tag.get_text(strip=True)
            players.append({
                "Nom": name,
                "Valeur marchande": value
            })

    return players

def convert_market_value(val_str):
    try:
        val_str = val_str.replace("€", "").replace("m", "000000").replace("k", "000").replace(",", ".").strip()
        if val_str.lower() in ["-", ""]:
            return None
        if "." in val_str:
            return int(float(val_str) * 1_000_000)
        return int(val_str)
    except:
        return None

def scrape_ligue1_market_values(start_season=2020, end_season=2024):
    all_data = []

    for year in range(start_season, end_season):
        print(f"\n🔍 Scraping saison {year}...")
        team_urls = get_teams_urls(year)
        print(f"  ✅ {len(team_urls)} équipes trouvées")

        for team_url in team_urls:
            team_name = team_url.split("/")[3]  # -3 to get team id
            print(f"    🟦 {team_name}")
            players = get_players_from_team(team_url)

            for player in players:
                all_data.append({
                    "Saison": f"{year}/{year+1}",
                    "Équipe": team_name,
                    "Nom": player["Nom"],
                    "Valeur marchande": player["Valeur marchande"]
                })

    df = pd.DataFrame(all_data)
    if not df.empty:
        df["Valeur marchande (euros)"] = df["Valeur marchande"].apply(convert_market_value)
    return df
