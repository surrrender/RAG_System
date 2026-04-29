#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_QUESTION = "小程序 App 生命周期是什么？"
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_USER_COUNT = 4
DEFAULT_CONVERSATIONS_PER_USER = 2


@dataclass(slots=True)
class BenchmarkCase:
    request_id: str
    user_id: str
    conversation_id: str
    question: str


@dataclass(slots=True)
class BenchmarkResult:
    request_id: str
    user_id: str
    conversation_id: str
    ok: bool
    status_code: int | None
    error: str | None
    first_char_latency_ms: float | None
    total_duration_ms: float
    server_retrieval_ms: float | None
    server_first_token_ms: float | None
    server_total_ms: float | None


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    total_requests = args.user_count * args.conversations_per_user
    if total_requests <= 0:
        parser.error("user-count * conversations-per-user must be greater than 0")

    cases = prepare_cases(
        base_url=args.base_url.rstrip("/"),
        question=args.question,
        timeout=args.timeout,
        user_count=args.user_count,
        conversations_per_user=args.conversations_per_user,
    )
    barrier = threading.Barrier(len(cases))
    results: list[BenchmarkResult] = []

    suite_started_at = time.perf_counter()
    threads = [
        threading.Thread(
            target=_run_case,
            name=f"qa-bench-{case.request_id}",
            args=(args.base_url.rstrip("/"), args.top_k, args.timeout, barrier, case, results),
        )
        for case in cases
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall_clock_ms = _elapsed_ms(suite_started_at)

    results.sort(key=lambda item: item.request_id)
    summary = build_summary(
        base_url=args.base_url.rstrip("/"),
        top_k=args.top_k,
        wall_clock_ms=wall_clock_ms,
        user_count=args.user_count,
        conversations_per_user=args.conversations_per_user,
        results=results,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"summary": summary, "results": [asdict(item) for item in results]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print_summary(summary)
    if args.verbose or any(not item.ok for item in results):
        print_request_results(results)

    return 0 if summary["failed_requests"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Concurrent /qa/stream benchmark for multi-user multi-conversation QA.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"LLM API base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question sent by every concurrent request.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"top_k sent to /qa/stream. Default: {DEFAULT_TOP_K}")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--user-count",
        type=int,
        default=DEFAULT_USER_COUNT,
        help=f"Number of distinct user_id values. Default: {DEFAULT_USER_COUNT}",
    )
    parser.add_argument(
        "--conversations-per-user",
        type=int,
        default=DEFAULT_CONVERSATIONS_PER_USER,
        help=f"Number of conversations created for each user. Default: {DEFAULT_CONVERSATIONS_PER_USER}",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path for structured JSON results.")
    parser.add_argument("--verbose", action="store_true", help="Print per-request result lines even when all requests succeed.")
    return parser


def prepare_cases(
    *,
    base_url: str,
    question: str,
    timeout: float,
    user_count: int,
    conversations_per_user: int,
) -> list[BenchmarkCase]:
    run_id = uuid.uuid4().hex[:8]
    cases: list[BenchmarkCase] = []
    for user_index in range(user_count):
        user_id = f"bench-user-{run_id}-{user_index + 1}"
        for conversation_index in range(conversations_per_user):
            response_payload = _post_json(
                f"{base_url}/conversations",
                {"user_id": user_id},
                timeout=timeout,
            )
            conversation_id = str(response_payload["id"])
            request_id = f"u{user_index + 1:02d}-c{conversation_index + 1:02d}"
            cases.append(
                BenchmarkCase(
                    request_id=request_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    question=question,
                )
            )
    return cases


def _run_case(
    base_url: str,
    top_k: int,
    timeout: float,
    barrier: threading.Barrier,
    case: BenchmarkCase,
    results: list[BenchmarkResult],
) -> None:
    barrier.wait()
    result = _stream_request(base_url=base_url, case=case, top_k=top_k, timeout=timeout)
    results.append(result)


def _stream_request(base_url: str, case: BenchmarkCase, top_k: int, timeout: float) -> BenchmarkResult:
    started_at = time.perf_counter()
    request = urllib.request.Request(
        f"{base_url}/qa/stream",
        data=json.dumps(
            {
                "user_id": case.user_id,
                "conversation_id": case.conversation_id,
                "question": case.question,
                "top_k": top_k,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    first_char_latency_ms: float | None = None
    server_retrieval_ms: float | None = None
    server_first_token_ms: float | None = None
    server_total_ms: float | None = None
    status_code: int | None = None
    error_message: str | None = None
    saw_done = False

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = getattr(response, "status", None)
            event_name: str | None = None
            data_lines: list[str] = []

            for raw_line in response:
                line = raw_line.decode("utf-8")
                if line in {"\n", "\r\n"}:
                    event_name, data_lines, first_char_latency_ms, server_retrieval_ms, server_first_token_ms, server_total_ms, error_message, saw_done = _consume_sse_event(
                        event_name=event_name,
                        data_lines=data_lines,
                        started_at=started_at,
                        first_char_latency_ms=first_char_latency_ms,
                        server_retrieval_ms=server_retrieval_ms,
                        server_first_token_ms=server_first_token_ms,
                        server_total_ms=server_total_ms,
                        error_message=error_message,
                        saw_done=saw_done,
                    )
                    if error_message is not None or saw_done:
                        break
                    continue

                if line.startswith("event:"):
                    event_name = line.partition(":")[2].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line.partition(":")[2].strip())

            if error_message is None and not saw_done and data_lines:
                _, _, first_char_latency_ms, server_retrieval_ms, server_first_token_ms, server_total_ms, error_message, saw_done = _consume_sse_event(
                    event_name=event_name,
                    data_lines=data_lines,
                    started_at=started_at,
                    first_char_latency_ms=first_char_latency_ms,
                    server_retrieval_ms=server_retrieval_ms,
                    server_first_token_ms=server_first_token_ms,
                    server_total_ms=server_total_ms,
                    error_message=error_message,
                    saw_done=saw_done,
                )
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        error_message = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        error_message = str(exc)

    total_duration_ms = _elapsed_ms(started_at)
    if error_message is None and not saw_done:
        error_message = "stream ended before done event"

    return BenchmarkResult(
        request_id=case.request_id,
        user_id=case.user_id,
        conversation_id=case.conversation_id,
        ok=error_message is None,
        status_code=status_code,
        error=error_message,
        first_char_latency_ms=first_char_latency_ms,
        total_duration_ms=total_duration_ms,
        server_retrieval_ms=server_retrieval_ms,
        server_first_token_ms=server_first_token_ms,
        server_total_ms=server_total_ms,
    )


def _consume_sse_event(
    *,
    event_name: str | None,
    data_lines: list[str],
    started_at: float,
    first_char_latency_ms: float | None,
    server_retrieval_ms: float | None,
    server_first_token_ms: float | None,
    server_total_ms: float | None,
    error_message: str | None,
    saw_done: bool,
) -> tuple[str | None, list[str], float | None, float | None, float | None, float | None, str | None, bool]:
    payload = _decode_sse_payload(data_lines)
    if event_name == "meta":
        retrieval_value = payload.get("retrieval_finished_at_ms")
        if retrieval_value is not None:
            server_retrieval_ms = float(retrieval_value)
    elif event_name == "delta":
        text = str(payload.get("text") or "")
        if first_char_latency_ms is None and text:
            first_char_latency_ms = _elapsed_ms(started_at)
        first_token_value = payload.get("server_first_token_at_ms")
        if first_token_value is not None:
            server_first_token_ms = float(first_token_value)
    elif event_name == "done":
        total_value = payload.get("server_completed_at_ms")
        if total_value is not None:
            server_total_ms = float(total_value)
        saw_done = True
    elif event_name == "error":
        error_message = str(payload.get("message") or "unknown stream error")

    return None, [], first_char_latency_ms, server_retrieval_ms, server_first_token_ms, server_total_ms, error_message, saw_done


def _decode_sse_payload(data_lines: list[str]) -> dict[str, object]:
    if not data_lines:
        return {}
    text = "\n".join(data_lines)
    if not text:
        return {}
    return json.loads(text)


def _post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_summary(
    *,
    base_url: str,
    top_k: int,
    wall_clock_ms: float,
    user_count: int,
    conversations_per_user: int,
    results: list[BenchmarkResult],
) -> dict[str, object]:
    total_requests = len(results)
    failed_requests = sum(1 for item in results if not item.ok)
    successful_results = [item for item in results if item.ok]
    failure_rate = round((failed_requests / total_requests) * 100, 2) if total_requests else 0.0

    return {
        "base_url": base_url,
        "top_k": top_k,
        "user_count": user_count,
        "conversations_per_user": conversations_per_user,
        "total_requests": total_requests,
        "successful_requests": len(successful_results),
        "failed_requests": failed_requests,
        "failure_rate_percent": failure_rate,
        "wall_clock_ms": wall_clock_ms,
        "first_char_latency_ms": _summarize_metric([item.first_char_latency_ms for item in successful_results]),
        "total_duration_ms": _summarize_metric([item.total_duration_ms for item in successful_results]),
        "server_retrieval_ms": _summarize_metric([item.server_retrieval_ms for item in successful_results]),
        "server_first_token_ms": _summarize_metric([item.server_first_token_ms for item in successful_results]),
        "server_total_ms": _summarize_metric([item.server_total_ms for item in successful_results]),
    }


def _summarize_metric(values: list[float | None]) -> dict[str, float | int] | None:
    filtered = sorted(float(value) for value in values if value is not None)
    if not filtered:
        return None
    return {
        "count": len(filtered),
        "avg": round(statistics.fmean(filtered), 3),
        "p50": round(_percentile(filtered, 50), 3),
        "p95": round(_percentile(filtered, 95), 3),
        "max": round(filtered[-1], 3),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (percentile / 100)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(values) - 1)
    fraction = rank - lower_index
    return values[lower_index] + (values[upper_index] - values[lower_index]) * fraction


def print_summary(summary: dict[str, object]) -> None:
    print("Concurrent /qa/stream benchmark")
    print(f"base_url: {summary['base_url']}")
    print(f"users x conversations: {summary['user_count']} x {summary['conversations_per_user']}")
    print(f"total_requests: {summary['total_requests']}")
    print(f"failures: {summary['failed_requests']} ({summary['failure_rate_percent']}%)")
    print(f"wall_clock_ms: {round(float(summary['wall_clock_ms']), 3)}")
    _print_metric_block("first_char_latency_ms", summary["first_char_latency_ms"])
    _print_metric_block("total_duration_ms", summary["total_duration_ms"])
    _print_metric_block("server_retrieval_ms", summary["server_retrieval_ms"])
    _print_metric_block("server_first_token_ms", summary["server_first_token_ms"])
    _print_metric_block("server_total_ms", summary["server_total_ms"])


def _print_metric_block(name: str, metric: object) -> None:
    if metric is None:
        print(f"{name}: n/a")
        return
    payload = dict(metric)
    print(
        f"{name}: count={payload['count']} avg={payload['avg']} p50={payload['p50']} "
        f"p95={payload['p95']} max={payload['max']}"
    )


def print_request_results(results: list[BenchmarkResult]) -> None:
    print("\nPer-request results")
    for item in results:
        status = "ok" if item.ok else f"failed ({item.error})"
        first_char = "n/a" if item.first_char_latency_ms is None else f"{item.first_char_latency_ms:.3f}"
        print(
            f"{item.request_id} user={item.user_id} conversation={item.conversation_id} "
            f"status={status} first_char_ms={first_char} total_ms={item.total_duration_ms:.3f}"
        )


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


if __name__ == "__main__":
    sys.exit(main())
