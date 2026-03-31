'''
该代码的主要作用是从侧边栏提取出所有需要爬取的子节点目录
'''
from __future__ import annotations

from bs4 import BeautifulSoup

from crawler.config import CrawlConfig
from crawler.utils import normalize_url

def extract_subnavbar_links(html: str, config: CrawlConfig) -> list[str]:
    """从 framework 页面顶部 subnavbar 中提取目标栏目入口。"""
    soup = BeautifulSoup(html, "lxml")
    subnavbar = soup.select_one(config.selectors.subnavbar_container)
    if subnavbar is None:
        return []

    links: list[str] = []
    seen: set[str] = set()
    allowed_labels = {label.strip() for label in config.selectors.discovery_section_labels}

    for link in subnavbar.select(config.selectors.subnavbar_link_selector):
        href = link.get("href")
        text = link.get_text(" ", strip=True)
        if not href or text not in allowed_labels:
            continue
        normalized = normalize_url(href, config.framework_url)
        if normalized not in seen:
            seen.add(normalized)
            links.append(normalized)
    return links


def _is_same_section_url(url: str, section_url: str) -> bool:
    normalized_section = normalize_url(section_url)
    return url == normalized_section or url.startswith(f"{normalized_section}/")


# 函数逻辑是从指定页面的aside节点中爬取ahref，然后进行normalize和去重
def extract_framework_links(html: str, config: CrawlConfig, base_url: str | None = None) -> list[str]:
    # 将 html变成可查询的 DOM
    soup = BeautifulSoup(html, "lxml")
    sidebar = soup.select_one(config.selectors.sidebar_container)
    if sidebar is None:
        return []

    section_url = normalize_url(base_url or config.reference_url)
    links: list[str] = []
    seen: set[str] = set()

    for link in sidebar.select(config.selectors.sidebar_link_selector):
        href = link.get("href")
        text = link.get_text(" ", strip=True)
        if not href or not text:
            continue
        normalized = normalize_url(href, section_url)
        if not _is_same_section_url(normalized, section_url):
            continue
        if normalized not in seen:
            seen.add(normalized)
            links.append(normalized)
    return links
