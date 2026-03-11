from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urlunparse

from crawler.config import SITE_ROOT


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_url(url: str, base_url: str = SITE_ROOT) -> str:
    absolute = urljoin(base_url, url.strip())
    parsed = urlparse(absolute)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=path.rstrip("/") if path != "/" else path,
        params="",
        query="",
        fragment="",
    )
    return urlunparse(normalized)


def make_doc_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def make_chunk_id(doc_id: str, section_path: list[str], chunk_text: str) -> str:
    payload = f"{doc_id}|{' > '.join(section_path)}|{chunk_text[:200]}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{doc_id}-{digest}"


def compute_fingerprint(title: str, raw_text: str, updated_at: str | None) -> str:
    payload = f"{title}\n{updated_at or ''}\n{raw_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clean_text(value: str) -> str:
    value = value.replace("\xa0", " ") # 将不间断空格转换为普通空格
    value = re.sub(r"[ \t]+", " ", value) # 将多个空格和制表符转换为一个空格
    value = re.sub(r"\n{3,}", "\n\n", value) # 将多行换行符转换为两行
    return value.strip() # 去除收尾空白


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)
