from scraper_utils import get_soup

def get_teams_urls(season_year):
    base_url = f"https://www.transfermarkt.fr/ligue-1/startseite/wettbewerb/FR1/saison_id/{season_year}"
    # base_url = f"https://www.transfermarkt.fr/premier-league/startseite/wettbewerb/GB1/saison_id/{season_year}"
    # base_url = f"https://www.transfermarkt.fr/la-liga/startseite/wettbewerb/ES1/saison_id/{season_year}"
    # base_url = f"https://www.transfermarkt.fr/bundesliga/startseite/wettbewerb/L1/saison_id/{season_year}"
    # base_url = f"https://www.transfermarkt.fr/ligue-2/startseite/wettbewerb/FR2/saison_id/{season_year}"
    # base_url = f"https://www.transfermarkt.fr/serie-a/startseite/wettbewerb/IT1/saison_id/{season_year}"
    soup = get_soup(base_url)

    if not soup:
        print(f"❌ Page non chargée pour la saison {season_year}")
        return []

    table = soup.find("table", class_="items")
    if not table:
        print(f"⚠️ Table introuvable dans la page pour la saison {season_year}")
        return []

    urls = []
    rows = table.find_all("tr", class_=["odd", "even"])
    for row in rows:
        team_cell = row.find("td", class_="hauptlink")
        if team_cell:
            link = team_cell.find("a", href=True)
            if link:
                href = link['href']
                if "/startseite/verein/" in href:
                    full_url = "https://www.transfermarkt.fr" + href
                    urls.append(full_url)

    print(f"✅ {len(urls)} équipes trouvées pour la saison {season_year}")
    return urls
