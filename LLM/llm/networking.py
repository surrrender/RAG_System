from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}


def normalize_local_service_url(url: str | None) -> str | None:
    if url is None:
        return None

    trimmed = url.strip()
    if not trimmed:
        return None

    parsed = urlsplit(trimmed)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and hostname in LOCAL_HOSTS:
        return urlunsplit(("http", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
    return trimmed


def protocol_hint(url: str | None, service_name: str) -> str:
    normalized = normalize_local_service_url(url)
    if not normalized:
        return f"{service_name} connection failed."

    parsed = urlsplit(normalized)
    host = parsed.hostname or normalized
    if host.lower() in LOCAL_HOSTS:
        return (
            f"{service_name} connection failed. Check whether the service is running and whether "
            f"the URL scheme should be http:// instead of https://."
        )
    return f"{service_name} connection failed. Check whether the configured URL is reachable."


def is_local_service_url(url: str | None) -> bool:
    normalized = normalize_local_service_url(url)
    if not normalized:
        return False
    parsed = urlsplit(normalized)
    return (parsed.hostname or "").lower() in LOCAL_HOSTS
