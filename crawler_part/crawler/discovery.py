from __future__ import annotations

from bs4 import BeautifulSoup

from crawler.config import CrawlConfig
from crawler.utils import normalize_url

# 函数逻辑是从指定页面的aside节点中爬取ahref，然后进行normalize和去重
def extract_framework_links(html: str, config: CrawlConfig) -> list[str]:
    # 将 html变成可查询的 DOM
    soup = BeautifulSoup(html, "lxml")
    sidebar = soup.select_one(config.selectors.sidebar_container)
    if sidebar is None:
        return []
    links: list[str] = []
    seen: set[str] = set()

    for link in sidebar.select(config.selectors.sidebar_link_selector):
        href = link.get("href")
        text = link.get_text(" ", strip=True)
        if not href or not text:
            continue
        normalized = normalize_url(href, config.reference_url)
        if "/miniprogram/dev/reference/" not in normalized:
            continue
        if normalized not in seen:
            seen.add(normalized)
            links.append(normalized)
    return links
