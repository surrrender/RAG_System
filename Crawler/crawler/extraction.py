from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup, Tag

from crawler.config import CrawlConfig
from crawler.models import ExtractedPage
from crawler.utils import clean_text


HEADING_NAMES = {"h2", "h3", "h4"}
SECTION_HEADING_NAMES = {"h2", "h3", "h4"}
IGNORED_CONTAINER_NAMES = {"footer"}


def _pick_content_node(soup: BeautifulSoup, selectors: Iterable[str]) -> Tag:
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            return node
    if soup.body is None:
        raise ValueError("missing body node")
    return soup.body


def _extract_nav_path(soup: BeautifulSoup, config: CrawlConfig) -> list[str]:
    separators = {">", "/", "›", "»", "|", "→"}
    for selector in config.selectors.breadcrumb_selectors:
        node = soup.select_one(selector)
        if node is not None:
            item_nodes = node.select(".breadcrumb-item .breadcrumb-inner, .breadcrumb-item a, .breadcrumb-item span")
            if item_nodes:
                parts = [clean_text(item.get_text(" ", strip=True)) for item in item_nodes]
                return [part for part in parts if part and part not in separators]
            parts = [clean_text(text) for text in node.stripped_strings]
            return [part for part in parts if part and part not in separators]
    return []


def _extract_updated_at(soup: BeautifulSoup, config: CrawlConfig) -> str | None:
    text = clean_text(soup.get_text("\n"))
    for pattern in config.selectors.update_time_patterns:
        regex = rf"{re.escape(pattern)}[:：]?\s*([0-9]{{4}}[-/.][0-9]{{1,2}}[-/.][0-9]{{1,2}}(?:\s+[0-9:]{{4,8}})?)"
        match = re.search(regex, text)
        if match:
            return match.group(1)
    return None


def extract_page_content(html: str, url: str, config: CrawlConfig) -> ExtractedPage:
    soup = BeautifulSoup(html, "lxml")
    # 获取main节点的body
    content = _pick_content_node(soup, config.selectors.content_selectors)
    # TODO：这里存疑不确实是否要flip，而且不确定这里的titlenode是作为数据源还是单纯作为展示title
    title_node = content.find("h1") or soup.find("title")
    title = clean_text(title_node.get_text(" ", strip=True) if title_node else "")
    if not title:
        raise ValueError("title not found")

    code_blocks: list[str] = []
    for code in content.select("pre, code"):
        if code.name == "code" and code.parent and code.parent.name == "pre":
            continue
        block = code.get_text(" ", strip=False)

        if block and block not in code_blocks:
            code_blocks.append(block)

    # 在页面级别获取 raw_text 的时候的确已经将代码包含了进去
    raw_text = clean_text(content.get_text("\n", strip=True))
    if not raw_text:
        raise ValueError("content text empty")

    return ExtractedPage(
        url=url,
        title=title,
        nav_path=_extract_nav_path(soup, config),
        updated_at=_extract_updated_at(soup, config),
        raw_text=raw_text,
        code_blocks=code_blocks if config.include_code else [],
    )


def extract_heading_blocks(html: str, config: CrawlConfig) -> list[dict[str, object]]:
    """
    提取标题块，按照标题进行拆分的核心逻辑
    """
    soup = BeautifulSoup(html, "lxml")
    content = _pick_content_node(soup, config.selectors.content_selectors)
    blocks: list[dict[str, object]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_codes: list[str] = []

    def flush() -> None:
        text = clean_text("\n".join(current_lines))
        if text or current_codes:
            blocks.append(
                {
                    "section_path": heading_stack.copy(),
                    "text": text,
                    "code_blocks": current_codes.copy(),
                }
            )
        current_lines.clear()
        current_codes.clear()

    for node in content.descendants:
        # 对于Tag标签的内容不予理睬
        if not isinstance(node, Tag):
            continue
        name = node.name.lower()
        if name in IGNORED_CONTAINER_NAMES or node.find_parent(IGNORED_CONTAINER_NAMES) is not None:
            continue
        if name in HEADING_NAMES:
            flush()
            level = int(name[1])
            heading_text = clean_text(node.get_text(" ", strip=True))
            if not heading_text:
                continue
            heading_stack[:] = heading_stack[: max(0, level - 1)]
            heading_stack.append(heading_text)
            continue
        if name == "pre":
            if not heading_stack:
                continue
            code_text = clean_text(node.get_text(" ", strip=True))
            if code_text:
                current_codes.append(code_text)
            continue
        if name == "code" and node.parent and node.parent.name == "pre":
            continue
        if name in {"p", "li", "td", "th", "blockquote"}:
            if not heading_stack:
                continue
            text = clean_text(node.get_text(" ", strip=True))
            if text:
                current_lines.append(text)

    flush()
    return blocks


def extract_heading_blocks_with_code_and_text_handling_together(html: str, config: CrawlConfig) -> list[dict[str, object]]:
    """
    提取标题块，按照标题进行拆分的核心逻辑;将文本和相关代码块作为一个整体进行提取,而不是分开
    """
    soup = BeautifulSoup(html, "lxml")
    content = _pick_content_node(soup, config.selectors.content_selectors)
    blocks: list[dict[str, object]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_codes: list[str] = []

    def flush() -> None:
        text = clean_text("\n".join(current_lines))
        if text or current_codes:
            blocks.append(
                {
                    "section_path": heading_stack.copy(),
                    "text": text,
                    "code_blocks": current_codes.copy(),# 这里返回的 code_blocks 会一直为空
                }
            )
        current_lines.clear()
        current_codes.clear()

    for node in content.descendants:
        # 对于Tag标签的内容不予理睬
        if not isinstance(node, Tag):
            continue
        name = node.name.lower()
        if name in IGNORED_CONTAINER_NAMES or node.find_parent(IGNORED_CONTAINER_NAMES) is not None:
            continue
        if name in HEADING_NAMES:
            flush()
            level = int(name[1])
            heading_text = clean_text(node.get_text(" ", strip=True))
            if not heading_text:
                continue
            heading_stack[:] = heading_stack[: max(0, level - 1)]
            heading_stack.append(heading_text)
            continue
        if name == "pre":
            if not heading_stack:
                continue
            code_text = clean_text(node.get_text(" ", strip=True))
            if code_text:
                #TODO:这里是不是加上一些标识代码的语义文本会比较好
                current_lines.append(code_text)
                # current_codes.append(code_text)
            continue
        if name == "code" and node.parent and node.parent.name == "pre":
            continue
        if name in {"p", "li", "td", "th", "blockquote"}:
            if not heading_stack:
                continue
            text = clean_text(node.get_text(" ", strip=True))
            if text:
                current_lines.append(text)

    flush()
    return blocks