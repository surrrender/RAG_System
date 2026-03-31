from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMEWORK_URL = "https://developers.weixin.qq.com/miniprogram/dev/framework/"
REFERENCE_URL = "https://developers.weixin.qq.com/miniprogram/dev/reference/"
SITE_ROOT = "https://developers.weixin.qq.com"
SOURCE_NAME = "wechat-miniprogram-framework-docs"


@dataclass(slots=True)
class SelectorConfig:
    subnavbar_container: str = ".subnavbar"
    subnavbar_link_selector: str = "a[href]"
    sidebar_container: str = "aside"
    # TODO：不确定这里还要不要保留，寻找链接的逻辑已经从文字锚点转换为aside节点寻找
    framework_label: str = "框架"
    sidebar_link_selector: str = "a[href]"
    discovery_section_labels: tuple[str, ...] = ("指南", "框架", "组件", "API")
    # TODO：content的内容就是在main节点下，后续这里可以简化逻辑
    content_selectors: tuple[str, ...] = (
        "main",
        ".markdown-doc",
        ".doc-content",
        ".markdown-body",
        "article",
    )
    breadcrumb_selectors: tuple[str, ...] = (
        ".breadcrumb",
        ".Breadcrumb",
        ".bread-crumb",
        "nav[aria-label='breadcrumb']",
        "[class*='breadcrumb' i]",
    )
    update_time_patterns: tuple[str, ...] = (
        "更新时间",
        "最近更新时间",
        "Last updated",
    )


@dataclass(slots=True)
class CrawlConfig:
    mode: str = "full"
    headless: bool = True
    max_concurrency: int = 4
    timeout_ms: int = 15_000
    include_code: bool = True
    retries: int = 3
    base_delay_seconds: float = 1.0
    framework_url: str = FRAMEWORK_URL
    reference_url: str = REFERENCE_URL
    source_name: str = SOURCE_NAME
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    state_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "state")
    selectors: SelectorConfig = field(default_factory=SelectorConfig)

    @property
    def pages_output_path(self) -> Path:
        return self.output_dir / "framework_pages.jsonl"

    @property
    def chunks_output_path(self) -> Path:
        return self.output_dir / "framework_chunks.jsonl"

    @property
    def failed_output_path(self) -> Path:
        return self.output_dir / "failed.jsonl"

    @property
    def fingerprint_path(self) -> Path:
        return self.state_dir / "page_fingerprints.json"
