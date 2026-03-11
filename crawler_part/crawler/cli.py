from __future__ import annotations

import asyncio

import typer

from crawler.config import CrawlConfig
from crawler.pipeline import run_crawl


app = typer.Typer(add_completion=False, help="WeChat Mini Program framework docs crawler.")


@app.command("run")
def run(
    mode: str = typer.Option("full", help="Crawl mode: full or incremental."),
    headless: bool = typer.Option(True, help="Run browser in headless mode."),
    max_concurrency: int = typer.Option(4, help="Maximum concurrent fetches."),
    timeout_ms: int = typer.Option(15_000, help="Per-page timeout in milliseconds."),
    include_code: bool = typer.Option(True, help="Keep code blocks in outputs."),
) -> None:
    if mode not in {"full", "incremental"}:
        raise typer.BadParameter("mode must be 'full' or 'incremental'")

    config = CrawlConfig(
        mode=mode,
        headless=headless,
        max_concurrency=max_concurrency,
        timeout_ms=timeout_ms,
        include_code=include_code,
    )
    summary = asyncio.run(run_crawl(config))
    typer.echo(
        "crawl complete: "
        f"discovered={summary['discovered']} "
        f"fetched={summary['fetched']} "
        f"chunks={summary['chunks']} "
        f"failed={summary['failed']}"
    )


if __name__ == "__main__":
    app()
