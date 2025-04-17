import requests
from bs4 import BeautifulSoup
import random
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def get_soup(url):
    time.sleep(random.uniform(2, 4))  # Be polite
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"⚠️ Error {response.status_code} when requesting {url}")
        return None
    return BeautifulSoup(response.text, "html.parser")
