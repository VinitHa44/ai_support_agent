"""
Ampere Docs Scraper
Dynamically crawls https://docs.ampere.sh/docs and extracts structured text for the RAG knowledge base.
"""

import json
import logging
import os
import re
import urllib.request
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://docs.ampere.sh"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def discover_doc_urls(start_url: str = "https://docs.ampere.sh/docs") -> List[str]:
    """
    Dynamically crawls documentation pages starting from start_url.
    Follows internal links under /docs.
    """
    logger.info(f"Dynamically discovering documentation pages from {start_url}...")
    headers = {"User-Agent": "Mozilla/5.0 AmpereSupportBot/1.0"}
    visited: Set[str] = set()
    to_visit: List[str] = [start_url]
    discovered: List[str] = []

    while to_visit:
        curr_url = to_visit.pop(0)
        if curr_url in visited:
            continue
        visited.add(curr_url)

        try:
            req = urllib.request.Request(curr_url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                if resp.status == 200:
                    discovered.append(curr_url)
                    html = resp.read().decode("utf-8")
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        full = urljoin(curr_url, href)
                        p = urlparse(full)
                        if p.netloc == "docs.ampere.sh" and p.path.startswith("/docs"):
                            clean_link = f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")
                            if clean_link not in visited and clean_link not in to_visit:
                                to_visit.append(clean_link)
        except Exception as e:
            logger.debug(f"Skipping {curr_url}: {e}")

    logger.info(f"Discovered {len(discovered)} documentation URLs dynamically.")
    return sorted(discovered)


def scrape_doc_page(full_url: str) -> Dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0 AmpereSupportBot/1.0"}
    req = urllib.request.Request(full_url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8")

    soup = BeautifulSoup(html, "html.parser")

    # Find main article or content container
    article = soup.find("article", id="nd-page")
    if not article:
        article = soup.find("main") or soup.find("body")

    # Extract title
    h1 = article.find("h1") if article else None
    title = h1.get_text().strip() if h1 else full_url.split("/")[-1].replace("-", " ").title()

    # Extract description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else ""

    # Remove non-content elements
    sections = []
    if article:
        for tag in article.find_all(["script", "style", "button", "svg", "nav"]):
            tag.decompose()

        # Extract structured semantic sections by headings
        current_heading = title
        current_texts = []
        for el in article.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "pre", "table", "blockquote"]):
            text = clean_text(el.get_text(" ", strip=True))
            if not text:
                continue
            if el.name in ["h1", "h2", "h3", "h4"]:
                if current_texts:
                    sections.append({
                        "heading": current_heading,
                        "content": " ".join(current_texts),
                    })
                    current_texts = []
                current_heading = text
            else:
                current_texts.append(text)

        if current_texts:
            sections.append({
                "heading": current_heading,
                "content": " ".join(current_texts),
            })

        body_text = clean_text(article.get_text())
    else:
        body_text = clean_text(soup.get_text())
        sections.append({
            "heading": title,
            "content": body_text,
        })

    path = urlparse(full_url).path
    return {
        "path": path,
        "url": full_url,
        "title": title,
        "description": description,
        "content": body_text,
        "sections": sections,
    }


def scrape_all_docs(output_path: str = "data/docs/scraped_docs.json") -> List[Dict[str, str]]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    urls = discover_doc_urls()
    results = []

    for url in urls:
        try:
            doc = scrape_doc_page(url)
            results.append(doc)
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully scraped {len(results)} docs to {output_path}")
    return results


if __name__ == "__main__":
    scrape_all_docs()
