import requests
import xml.etree.ElementTree as ET
import hashlib
import hmac
import os
from typing import List
import gzip
import pandas as pd
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup
import json

DATA_PATH = "data.jsonl"

CURRENT_DATA = pd.read_json(DATA_PATH, lines=True)
LATEST_PUBLISHED_TIME = datetime.fromisoformat(CURRENT_DATA["published_at"].max())

EXISTING_HASHES = set(CURRENT_DATA["hash"])

ZENN_SITEMAPS_URL = "https://zenn.dev/sitemaps/_index.xml"
SALT = os.environ.get("HASH_SALT")

TIME_INTERVAL = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ZennTopicSuggester/1.0 (Contact: 196759132+northward1@users.noreply.github.com)"
}


def get_url_hash(url: str) -> str:
    return hmac.new(
        SALT.encode("utf-8"), url.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def fetch_sitemaps() -> List[str]:
    time.sleep(TIME_INTERVAL)
    r = requests.get(ZENN_SITEMAPS_URL, headers=HEADERS)

    r.raise_for_status()

    root = ET.fromstring(r.text)

    article_urls = []

    for loc in root.findall(".//{*}loc"):
        url = loc.text

        if "article" in url:
            article_urls.append(url)

    return article_urls


def fetch_article_sitemap(article_sitemap_url: str) -> List[str]:
    time.sleep(TIME_INTERVAL)
    r = requests.get(article_sitemap_url, headers=HEADERS)

    r.raise_for_status()

    bytes = gzip.decompress(r.content)
    xml = bytes.decode("utf-8")
    root = ET.fromstring(xml)

    locs = [loc.text for loc in root.findall(".//{*}loc")]
    lastmods = [lastmod.text for lastmod in root.findall(".//{*}lastmod")]

    urls = []

    for i in range(len(locs)):
        loc = locs[i]
        lastmod = lastmods[i]
        lastmod = datetime.fromisoformat(lastmod)
        lastmod = pd.to_datetime(lastmod)

        if loc.endswith("?locale=en"):
            continue

        hash = get_url_hash(loc)

        if hash in EXISTING_HASHES:
            continue

        if lastmod < LATEST_PUBLISHED_TIME - timedelta(days=180):
            continue

        urls.append(loc)

    return urls


def fetch_article(article_url: str) -> dict:
    time.sleep(TIME_INTERVAL)
    r = requests.get(article_url, headers=HEADERS)

    r.raise_for_status()

    soup = BeautifulSoup(r.text, features="html.parser")
    json_elem = soup.select_one("#__NEXT_DATA__")
    dict = json.loads(json_elem.get_text(strip=True))

    published_at = dict["props"]["pageProps"]["article"]["publishedAt"]
    hash = get_url_hash(article_url)
    topics_dict = dict["props"]["pageProps"]["topics"]
    topics = [d["name"] for d in topics_dict]

    return {"published_at": published_at, "hash": hash, "topics": topics}


def main():
    article_sitemaps = fetch_sitemaps()

    urls = []

    for sitemap in article_sitemaps:
        urls.extend(fetch_article_sitemap(sitemap))

    for url in urls:
        d = fetch_article(url)

        with open(DATA_PATH, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
