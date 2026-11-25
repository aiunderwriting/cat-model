#!/usr/bin/env python3
"""Resolve monthly arXiv category-listing totals for the cs.AI and cs.CR seeds.

Source and canonical URLs: arXiv monthly listing pages at
https://arxiv.org/list/{category}/{YYYY-MM}?skip=0&show=25, plus
https://arxiv.org/robots.txt. For each seed, the filter is the category path
(cs.AI or cs.CR) and the three page months 2026-07, 2026-08, and 2026-09.
The date field is the submission/identifier month selected by the listing URL;
the parsed value is the page's visible “Total of N entries” count. The monthly
page text presents one Authors and titles total for the selected category and
month rather than separate primary-submission and cross-list totals. arXiv
describes cross-listed articles as appearing in the regular listing for the
target category in its cross-list section, so this total includes matching
cross-lists as well as primary-category submissions. Each identifier is
counted once within each category/month listing. Later replacement versions
do not become separate new submission identifiers. There is no title-level or
author-level filter.

The script saves the robots response and every requested HTML page byte-for-
byte, and records URL, SHA-256, byte count, and retrieval time for every file.
The robots policy specifies a 15-second crawl delay; requests, including
retries, are spaced by at least that delay. Listing counts are a snapshot and
can change if arXiv revises or reclassifies an entry; submission-month
attribution and the total can therefore differ from later page versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

SOURCE_ID = "arxiv"
SCRIPT_NAME = Path(__file__).name
USER_AGENT = "AIUC calibration-seed resolver (research)"
MIN_CRAWL_DELAY_SECONDS = 15.0
RETRY_DELAYS_SECONDS = (2, 4, 8, 16, 32)
TIMEOUT = (30, 120)

MONTHS = [
    f"{year:04d}-{month:02d}"
    for year in range(2024, 2027)
    for month in range(1, 13)
    if (year, month) <= (2026, 9)
]
CATEGORIES = ("cs.AI", "cs.CR")
ROBOTS_URL = "https://arxiv.org/robots.txt"

SEEDS: list[dict[str, Any]] = [
    {
        "id": "arxiv-cs-ai-2026q3",
        "role": "primary",
        "window": {"start": "2026-07-01", "end": "2026-09-30"},
        "history_granularity": "quarter",
        "filter": {
            "category": "cs.AI",
            "months": ["2026-07", "2026-08", "2026-09"],
            "page_total_text": "Total of N entries",
        },
    },
    {
        "id": "arxiv-cs-cr-2026q3",
        "role": "primary",
        "window": {"start": "2026-07-01", "end": "2026-09-30"},
        "history_granularity": "quarter",
        "filter": {
            "category": "cs.CR",
            "months": ["2026-07", "2026-08", "2026-09"],
            "page_total_text": "Total of N entries",
        },
    },
]


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def utc_text(value: datetime) -> str:
    return (
        value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def parse_utc_text(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError(f"Expected UTC timestamp ending in Z: {value!r}")
    return datetime.fromisoformat(value[:-1] + "+00:00")


def source_directory(data_dir: Path, offline: bool) -> Path:
    source_dir = data_dir / SOURCE_ID
    if offline:
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Offline snapshot directory does not exist: {source_dir}"
            )
        return source_dir
    if source_dir.exists():
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Snapshot path is not a directory: {source_dir}")
        if any(source_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to download into non-empty snapshot directory: {source_dir}"
            )
    else:
        source_dir.mkdir(parents=True)
    return source_dir


def expected_pages() -> list[tuple[str, str]]:
    result = [("robots.txt", ROBOTS_URL)]
    for month in MONTHS:
        for category in CATEGORIES:
            filename = f"list_{category}_{month}.html"
            url = f"https://arxiv.org/list/{category}/{month}?skip=0&show=25"
            result.append((filename, url))
    return result


def crawl_delay_from_robots(content: bytes) -> float:
    text = content.decode("utf-8", errors="replace")
    delays = []
    for match in re.finditer(
        r"(?im)^\s*Crawl-delay\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:#.*)?$",
        text,
    ):
        delays.append(float(match.group(1)))
    return max([MIN_CRAWL_DELAY_SECONDS, *delays])


class SpacedSession:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.last_completed = None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "identity",
            }
        )

    def _wait_for_spacing(self) -> None:
        if self.last_completed is None:
            return
        remaining = self.delay_seconds - (time.monotonic() - self.last_completed)
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _proxy_connect_refused(error: BaseException) -> bool:
        message = str(error).casefold()
        return "403" in message and ("connect" in message or "tunnel" in message)

    def get(self, url: str) -> requests.Response:
        for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
            self._wait_for_spacing()
            try:
                response = self.session.get(url, timeout=TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as exc:
                self.last_completed = time.monotonic()
                if self._proxy_connect_refused(exc):
                    raise RuntimeError(
                        f"HTTPS CONNECT was refused with HTTP 403 for {url}: {exc}"
                    ) from exc
                if attempt >= len(RETRY_DELAYS_SECONDS):
                    raise RuntimeError(
                        f"Connection failed after retries for {url}: {exc}"
                    ) from exc
                time.sleep(max(RETRY_DELAYS_SECONDS[attempt], self.delay_seconds))
                continue

            self.last_completed = time.monotonic()
            if response.status_code == 403:
                response.close()
                raise RuntimeError(
                    f"HTTP 403 while fetching {url}; stopping as required"
                )
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                status = response.status_code
                response.close()
                if attempt >= len(RETRY_DELAYS_SECONDS):
                    raise RuntimeError(
                        f"HTTP {status} after retries for {url}; stopping"
                    )
                time.sleep(max(RETRY_DELAYS_SECONDS[attempt], self.delay_seconds))
                continue
            if not 200 <= response.status_code < 300:
                response.close()
                raise RuntimeError(
                    f"HTTP {response.status_code} while fetching {url}; stopping"
                )
            return response
        raise AssertionError("unreachable retry loop")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_retrieved_at(path: Path) -> str:
    return utc_text(datetime.fromtimestamp(path.stat().st_mtime, UTC))


def set_file_time(path: Path, retrieved_at: str) -> None:
    timestamp = parse_utc_text(retrieved_at).timestamp()
    os.utime(path, (timestamp, timestamp))


def record_saved_file(
    source_dir: Path,
    filename: str,
    url: str,
    content: bytes,
    retrieved_at: str,
) -> dict[str, str | int]:
    path = source_dir / filename
    path.write_bytes(content)
    set_file_time(path, retrieved_at)
    digest = sha256_bytes(content)
    item = {
        "url": url,
        "file": filename,
        "sha256": digest,
        "bytes": len(content),
        "retrieved_at": retrieved_at,
    }
    return item


def parse_total(content: bytes, label: str) -> int:
    page_text = BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
    matches = re.findall(
        r"\bTotal\s+of\s+([0-9][0-9,]*)\s+entries\b",
        page_text,
        flags=re.IGNORECASE,
    )
    totals = {int(value.replace(",", "")) for value in matches}
    if len(totals) != 1:
        raise ValueError(
            f"{label}: expected one unambiguous 'Total of N entries' count; "
            f"found {sorted(totals)}"
        )
    return totals.pop()


def load_prior_metadata(
    out_path: Path,
) -> tuple[str | None, dict[str, dict[str, object]]]:
    if not out_path.is_file():
        return None, {}
    try:
        previous = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read prior result metadata from {out_path}: {exc}"
        ) from exc
    if previous.get("source_id") != SOURCE_ID or previous.get("script") != SCRIPT_NAME:
        raise RuntimeError(
            f"Existing --out is not a prior {SOURCE_ID} result for {SCRIPT_NAME}: {out_path}"
        )
    by_file = {}
    for item in previous.get("snapshots", []):
        if isinstance(item, dict) and isinstance(item.get("file"), str):
            by_file[item["file"]] = item
    return previous.get("retrieved_at"), by_file


def read_snapshot_files(
    source_dir: Path,
    expected: list[tuple[str, str]],
    offline: bool,
    out_path: Path,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], bytes]]:
    existing = {path.name for path in source_dir.iterdir() if path.is_file()}
    nested = [path.name for path in source_dir.iterdir() if not path.is_file()]
    expected_names = {filename for filename, _ in expected}
    missing = sorted(expected_names - existing)
    extra = sorted(existing - expected_names)
    if missing or extra or nested:
        raise RuntimeError(
            f"Snapshot files do not match expected arXiv crawl; "
            f"missing={missing}, extra={extra}, nested={nested}"
        )

    old_retrieved_at, old_by_file = (
        load_prior_metadata(out_path) if offline else (None, {})
    )
    snapshots = []
    content_by_url: dict[tuple[str, str], bytes] = {}
    for filename, url in expected:
        path = source_dir / filename
        content = path.read_bytes()
        digest = sha256_bytes(content)
        prior = old_by_file.get(filename)
        if prior is not None:
            if prior.get("url") != url:
                raise RuntimeError(f"Prior metadata URL mismatch for {filename}")
            if prior.get("sha256") != digest or prior.get("bytes") != len(content):
                raise RuntimeError(f"Snapshot hash/size mismatch for {filename}")
            retrieved_at = prior.get("retrieved_at")
            if not isinstance(retrieved_at, str):
                raise RuntimeError(f"Prior metadata lacks retrieved_at for {filename}")
        else:
            retrieved_at = file_retrieved_at(path)
        item = {
            "url": url,
            "file": filename,
            "sha256": digest,
            "bytes": len(content),
            "retrieved_at": retrieved_at,
        }
        snapshots.append(item)
        content_by_url[(filename, url)] = content

    if offline and old_retrieved_at is not None:
        parse_utc_text(old_retrieved_at)
    return snapshots, content_by_url


def snapshot_digest(snapshots: list[dict[str, object]]) -> str:
    lines = [
        f"{item['sha256']}  {item['file']}"
        for item in sorted(snapshots, key=lambda item: str(item["file"]))
    ]
    manifest = ("\n".join(lines) + "\n").encode("utf-8")
    return sha256_bytes(manifest)


def quarter_parts(period_month: str) -> tuple[int, int]:
    year_text, month_text = period_month.split("-")
    month = int(month_text)
    return int(year_text), (month - 1) // 3 + 1


def quarterly_history(
    category: str, monthly_totals: dict[tuple[str, str], int]
) -> list[dict[str, object]]:
    output = []
    for year in range(2024, 2027):
        for quarter in range(1, 5):
            period = f"{year}-Q{quarter}"
            if period >= "2026-Q3":
                continue
            value = 0
            for month in MONTHS:
                if quarter_parts(month) == (year, quarter):
                    value += monthly_totals[(category, month)]
            output.append({"period": period, "value": value})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()

    source_dir = source_directory(args.data_dir, args.offline)
    expected = expected_pages()
    if args.offline:
        snapshots, content_by_url = read_snapshot_files(
            source_dir, expected, True, args.out
        )
    else:
        session = SpacedSession(MIN_CRAWL_DELAY_SECONDS)
        snapshots = []
        content_by_url = {}
        crawl_delay = MIN_CRAWL_DELAY_SECONDS
        for filename, url in expected:
            session.delay_seconds = crawl_delay
            response = session.get(url)
            content = response.content
            response.close()
            retrieved_at = utc_text(utc_now())
            item = record_saved_file(source_dir, filename, url, content, retrieved_at)
            snapshots.append(item)
            content_by_url[(filename, url)] = content
            print(f"{item['sha256']}  {item['file']}", flush=True)
            if filename == "robots.txt":
                crawl_delay = crawl_delay_from_robots(content)
                print(f"arXiv crawl delay: {crawl_delay:g} seconds", flush=True)

    if args.offline:
        for item in snapshots:
            print(f"{item['sha256']}  {item['file']}", flush=True)

    monthly_totals: dict[tuple[str, str], int] = {}
    for month in MONTHS:
        for category in CATEGORIES:
            filename = f"list_{category}_{month}.html"
            url = f"https://arxiv.org/list/{category}/{month}?skip=0&show=25"
            content = content_by_url[(filename, url)]
            monthly_totals[(category, month)] = parse_total(
                content, f"{category} {month}"
            )

    output_seeds = {}
    for seed in SEEDS:
        category = seed["filter"]["category"]
        window_months = seed["filter"]["months"]
        value = sum(monthly_totals[(category, month)] for month in window_months)
        output_seeds[seed["id"]] = {
            "role": seed["role"],
            "value": value,
            "window_start": seed["window"]["start"],
            "window_end": seed["window"]["end"],
            "history_granularity": seed["history_granularity"],
            "history": quarterly_history(category, monthly_totals),
            "detail": {
                "window_breakdown": {
                    month: monthly_totals[(category, month)] for month in window_months
                }
            },
            "notes": (
                "Value and history are sums of the monthly arXiv category-listing "
                "page totals. Counts are the page's Total of N entries, including "
                "cross-listings shown in that category/month listing; the listing "
                "month is the submission/identifier month. Counts reflect this "
                "retrieval snapshot and can change after revisions or reclassification."
            ),
        }

    snapshot_sha256 = snapshot_digest(snapshots)
    first_retrieved_at = min(
        (str(item["retrieved_at"]) for item in snapshots),
        key=parse_utc_text,
    )
    result = {
        "source_id": SOURCE_ID,
        "script": SCRIPT_NAME,
        "retrieved_at": first_retrieved_at,
        "snapshots": snapshots,
        "commit": None,
        "snapshot_sha256": snapshot_sha256,
        "seeds": output_seeds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"snapshot_sha256  {snapshot_sha256}", flush=True)
    print(f"runtime_seconds  {time.monotonic() - started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"arxiv resolver failed: {exc}", file=sys.stderr)
        raise
