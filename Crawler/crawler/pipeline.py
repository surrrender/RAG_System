from __future__ import annotations

import asyncio
from dataclasses import dataclass

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from crawler.chunking import build_chunks
from crawler.config import CrawlConfig
from crawler.discovery import extract_framework_links
from crawler.extraction import extract_heading_blocks, extract_page_content, extract_heading_blocks_with_code_and_text_handling_together
from crawler.models import FailureRecord, PageRecord
from crawler.storage import ensure_runtime_dirs, load_fingerprints, save_fingerprints, write_jsonl
from crawler.utils import compute_fingerprint, make_doc_id, normalize_url, utc_now_iso


@dataclass(slots=True)
class FetchedPage:
    page: PageRecord
    heading_blocks: list[dict[str, object]]


async def _new_page(context: BrowserContext, config: CrawlConfig) -> Page:
    """Create a new browser page with configured timeouts.

    Args:
        context: The browser context to create the page from.
        config: The crawl configuration containing timeout settings.

    Returns:
        A new Page instance with default timeouts set.
    """
    page = await context.new_page()
    page.set_default_timeout(config.timeout_ms)
    page.set_default_navigation_timeout(config.timeout_ms)
    return page


async def discover_urls(context: BrowserContext, config: CrawlConfig) -> list[str]:
    """Discover and extract URLs from the reference page.

    Navigates to the reference URL and extracts framework-specific links
    from the sidebar container.

    Args:
        context: The browser context to use for navigation.
        config: The crawl configuration containing reference URL and selectors.

    Returns:
        A list of normalized URLs discovered from the reference page.
    """
    page = await _new_page(context, config)
    try:
        await page.goto(config.reference_url, wait_until="networkidle")
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_selector(config.selectors.sidebar_container)
        html = await page.content()
    finally:
        await page.close()

    links = extract_framework_links(html, config)
    return [normalize_url(url, config.reference_url) for url in links]


async def _fetch_html(context: BrowserContext, url: str, config: CrawlConfig) -> str:
    """Fetch the HTML content of a given URL.

    Args:
        context: The browser context to use for fetching.
        url: The URL to fetch.
        config: The crawl configuration containing timeout settings.

    Returns:
        The HTML content of the page.
    """
    page = await _new_page(context, config)
    try:
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_load_state("domcontentloaded")
        return await page.content()
    finally:
        await page.close()


async def _fetch_one(context: BrowserContext, url: str, config: CrawlConfig) -> tuple[FetchedPage | None, FailureRecord | None]:
    """Fetch a single URL with retry logic.

    Attempts to fetch and extract content from a URL, with exponential
    backoff retry on failure.

    Args:
        context: The browser context to use for fetching.
        url: The URL to fetch.
        config: The crawl configuration containing retry settings.

    Returns:
        A tuple of (FetchedPage, None) on success, or (None, FailureRecord) on failure.
    """
    last_error = ""
    for attempt in range(1, config.retries + 1):
        try:
            # 获取raw html
            html = await _fetch_html(context, url, config)
            # 
            extracted = extract_page_content(html, url, config)
            page_record = PageRecord(
                doc_id=make_doc_id(url),
                url=url,
                title=extracted.title,
                nav_path=extracted.nav_path,
                raw_text=extracted.raw_text,
                code_blocks=extracted.code_blocks,
                source=config.source_name,
                fetched_at=utc_now_iso(),
                updated_at=extracted.updated_at,
            )
            return FetchedPage(page=page_record, heading_blocks=extract_heading_blocks_with_code_and_text_handling_together(html, config)), None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < config.retries:
                await asyncio.sleep(config.base_delay_seconds * (2 ** (attempt - 1)))

    return None, FailureRecord(
        url=url,
        stage="fetch",
        error=last_error,
        retry_count=config.retries,
        failed_at=utc_now_iso(),
    )


async def _fetch_all(context: BrowserContext, urls: list[str], config: CrawlConfig) -> tuple[list[FetchedPage], list[FailureRecord]]:
    """Fetch multiple URLs concurrently with controlled concurrency.

    Uses a semaphore to limit the number of concurrent fetches and
    collects both successful fetches and failures.

    Args:
        context: The browser context to use for fetching.
        urls: The list of URLs to fetch.
        config: The crawl configuration containing concurrency settings.

    Returns:
        A tuple of (list of FetchedPage, list of FailureRecord).
    """
    semaphore = asyncio.Semaphore(config.max_concurrency)
    fetched_pages: list[FetchedPage] = []
    failures: list[FailureRecord] = []

    async def runner(url: str) -> None:
        async with semaphore:
            fetched_page, failure = await _fetch_one(context, url, config)
            if failure is not None:
                failures.append(failure)
                return
            assert fetched_page is not None
            fetched_pages.append(fetched_page)

    await asyncio.gather(*(runner(url) for url in urls))
    return fetched_pages, failures


def _select_changed_pages(fetched_pages: list[FetchedPage], known_fingerprints: dict[str, str]) -> list[FetchedPage]:
    """Select pages whose content has changed since last crawl.

    Compares fingerprints of fetched pages against known fingerprints
    to identify new or modified pages.

    Args:
        fetched_pages: The list of pages that were fetched.
        known_fingerprints: A dictionary mapping URLs to their known fingerprints.

    Returns:
        A list of FetchedPage objects whose content has changed.
    """
    changed: list[FetchedPage] = []
    for fetched_page in fetched_pages:
        page = fetched_page.page
        fingerprint = compute_fingerprint(page.title, page.raw_text, page.updated_at)
        if known_fingerprints.get(page.url) != fingerprint:
            changed.append(fetched_page)
    return changed


def _merge_fingerprints(existing: dict[str, str], pages: list[PageRecord]) -> dict[str, str]:
    """Merge new page fingerprints with existing ones.

    Updates the fingerprint dictionary with new or changed page fingerprints.

    Args:
        existing: The existing fingerprint dictionary.
        pages: The list of pages to compute fingerprints for.

    Returns:
        A merged dictionary containing all fingerprints.
    """
    merged = existing.copy()
    for page in pages:
        merged[page.url] = compute_fingerprint(page.title, page.raw_text, page.updated_at)
    return merged


async def run_crawl(config: CrawlConfig) -> dict[str, int]:
    """Run the main crawl pipeline.

    Orchestrates the entire crawling process: discovering URLs, fetching pages,
    selecting changed pages (in incremental mode), building chunks, and writing
    output files.

    Args:
        config: The crawl configuration containing all settings.

    Returns:
        A dictionary with statistics about the crawl (discovered, fetched,
        chunks, failed counts).
    """
    ensure_runtime_dirs(config.output_dir, config.state_dir)
    existing_fingerprints = load_fingerprints(config.fingerprint_path) if config.mode == "incremental" else {}

    async with async_playwright() as playwright:
        browser: Browser = await playwright.chromium.launch(headless=config.headless)
        context = await browser.new_context()
        try:
            discovered_urls = await discover_urls(context, config)
            unique_urls = list(dict.fromkeys(discovered_urls))
            ## fetch_all逻辑是什么
            fetched_pages, failures = await _fetch_all(context, unique_urls, config)
        finally:
            await context.close()
            await browser.close()

    selected_pages = fetched_pages if config.mode == "full" else _select_changed_pages(fetched_pages, existing_fingerprints)
    page_records = [item.page for item in selected_pages]
    chunk_records = []
    for item in selected_pages:
        chunk_records.extend(chunk.to_dict() for chunk in build_chunks(item.page, item.heading_blocks))

    write_jsonl(config.pages_output_path, [page.to_dict() for page in page_records])
    write_jsonl(config.chunks_output_path, chunk_records)
    write_jsonl(config.failed_output_path, [failure.to_dict() for failure in failures])
    save_fingerprints(config.fingerprint_path, _merge_fingerprints(existing_fingerprints, [item.page for item in fetched_pages]))

    return {
        "discovered": len(unique_urls),
        "fetched": len(page_records),
        "chunks": len(chunk_records),
        "failed": len(failures),
    }
