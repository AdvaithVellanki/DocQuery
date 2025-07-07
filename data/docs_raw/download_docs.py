# This script fetches specific Python documentation pages and saves them as text files.
import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://docs.python.org/3/library/"
PAGES = ["asyncio.html", "json.html", "collections.html"]
OUTPUT_DIR = "data/docs_raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_and_save_page(slug):
    url = BASE_URL + slug
    response = requests.get(url)
    if response.status_code == 200:
        filename = os.path.join(OUTPUT_DIR, slug.replace(".html", ".txt"))
        soup = BeautifulSoup(response.text, "html.parser")
        body_text = soup.get_text()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(body_text)
        print(f"[✔] Saved {slug}")
    else:
        print(f"[✘] Failed to fetch {slug}: {response.status_code}")


if __name__ == "__main__":
    for page in PAGES:
        fetch_and_save_page(page)
