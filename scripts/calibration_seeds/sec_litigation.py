#!/usr/bin/env python3
"""Resolve SEC litigation-release counts from the SEC listing pages.

Source and canonical URL: SEC Litigation Releases,
https://www.sec.gov/enforcement-litigation/litigation-releases. For each month
from 2024-01 through 2026-10, request the listing with year=YYYY and month=M,
then follow its page= pagination links. The release date is the date printed in
the listing row's visible time element; the release identifier is the LR-NNNNN
number printed in that row (some older SEC rows omit the hyphen or include a
space after it; both are normalized to the same number). The primary seed sec-litigation-releases-2026q3
counts distinct LR numbers printed with dates 2026-07-01 through 2026-09-30;
its quarterly history counts the same rows in 2024-Q1 through 2026-Q2. The
primary seed sec-litigation-releases-2026-10 counts distinct LR numbers printed
with dates 2026-10-01 through 2026-10-31; its monthly history counts the same
rows in 2024-01 through 2026-06. There is no subject or release-type filter.
Snapshot rule: save every raw listing response, redirect response, and
transient HTTP response body byte-for-byte. Page completeness is checked using
the listing pager and contiguous page indexes. LR number gaps and duplicate
rows are reported by queried month. These are publication-dated records, but
the SEC can correct or republish listing rows after the displayed release date.

SEC access requires SEC_EDGAR_USER_AGENT in the environment; the script never
supplies a requester email itself. Requests are spaced to stay below five per
second.
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

SOURCE_ID = "sec_litigation"
SCRIPT_NAME = "sec_litigation.py"
BASE_URL = "https://www.sec.gov/enforcement-litigation/litigation-releases"
RETRY_DELAYS = (2, 4, 8, 16, 32)
MIN_REQUEST_INTERVAL_SECONDS = 0.21
MAX_RESULT_BYTES = 300 * 1024
SEEDS: list[dict[str, Any]] = [
    {
        "id": "sec-litigation-releases-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "history_granularity": "quarter",
        "filter": {
            "release_number": "distinct LR number",
            "date_field": "printed listing date",
        },
    },
    {
        "id": "sec-litigation-releases-2026-10",
        "role": "primary",
        "window_start": "2026-10-01",
        "window_end": "2026-10-31",
        "history_granularity": "month",
        "filter": {
            "release_number": "distinct LR number",
            "date_field": "printed listing date",
        },
    },
]


def utc_now():
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def manifest_sha256(snapshots):
    text = "".join(
        f"{entry['sha256']}  {entry['file']}\n"
        for entry in sorted(snapshots, key=lambda item: item["file"])
    )
    return sha256_bytes(text.encode("utf-8"))


def prepare_directory(path, offline):
    if offline:
        if not path.is_dir():
            raise SystemExit(f"offline data directory is missing: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise SystemExit(f"refusing online download into non-empty directory: {path}")


def record_response(directory, snapshots, filename, url, content):
    relative = Path(filename).as_posix()
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    snapshots.append(
        {
            "url": url,
            "file": relative,
            "sha256": sha256_bytes(content),
            "bytes": len(content),
            "retrieved_at": utc_now(),
        }
    )


def is_connect_403(exc):
    message = str(exc).lower()
    return "403" in message and (
        "tunnel" in message or "connect" in message or "proxy" in message
    )


def throttle(request_clock):
    now = time.monotonic()
    if request_clock[0] is not None:
        remaining = MIN_REQUEST_INTERVAL_SECONDS - (now - request_clock[0])
        if remaining > 0:
            time.sleep(remaining)
    request_clock[0] = time.monotonic()


def request_page(
    session, year, month, page_number, directory, snapshots, request_clock
):
    stem = f"{year:04d}-{month:02d}_page_{page_number:03d}"
    base_filename = stem + ".html"
    params = {"year": year, "month": month}
    if page_number:
        params["page"] = page_number
    for attempt in range(len(RETRY_DELAYS) + 1):
        throttle(request_clock)
        try:
            response = session.get(
                BASE_URL,
                params=params,
                timeout=(30, 120),
                allow_redirects=True,
            )
        except requests.exceptions.ProxyError as exc:
            if is_connect_403(exc):
                raise SystemExit(
                    f"proxy refused HTTPS CONNECT to www.sec.gov (HTTP 403): {exc}"
                ) from exc
            if attempt >= len(RETRY_DELAYS):
                raise SystemExit(
                    f"SEC listing request failed after retries: {exc}"
                ) from exc
            time.sleep(RETRY_DELAYS[attempt])
            continue
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt >= len(RETRY_DELAYS):
                raise SystemExit(
                    f"SEC listing request failed after retries: {exc}"
                ) from exc
            time.sleep(RETRY_DELAYS[attempt])
            continue

        chain = [*list(response.history), response]
        for index, item in enumerate(chain):
            if item.status_code in (301, 302, 303, 307, 308):
                filename = f"{stem}_attempt{attempt}_redirect{index}_http{item.status_code}.bin"
            elif item.status_code >= 400:
                filename = f"{stem}_attempt{attempt}_http{item.status_code}.bin"
            elif index == len(chain) - 1:
                filename = base_filename
            else:
                filename = f"{stem}_attempt{attempt}_redirect{index}_http{item.status_code}.bin"
            record_response(directory, snapshots, filename, item.url, item.content)

        if response.status_code == 403:
            raise SystemExit(
                "www.sec.gov returned HTTP 403; stopping without further requests"
            )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt >= len(RETRY_DELAYS):
                raise SystemExit(
                    f"SEC listing returned HTTP {response.status_code} after retries"
                )
            time.sleep(RETRY_DELAYS[attempt])
            continue
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise SystemExit(f"SEC listing request failed: {exc}") from exc
        return response.content
    raise SystemExit("SEC listing retry loop ended unexpectedly")


def parse_page_index(href):
    values = parse_qs(urlparse(href).query).get("page", [])
    indices = []
    for value in values:
        for component in value.split(","):
            if component.isdigit():
                indices.append(int(component))
    return indices


def advertised_last_page(soup):
    last_values = []
    all_values = []
    for link in soup.select("a[href]"):
        indices = parse_page_index(link.get("href", ""))
        if not indices:
            continue
        all_values.extend(indices)
        classes = " ".join(link.get("class", [])).lower()
        parent_classes = (
            " ".join(link.parent.get("class", [])).lower() if link.parent else ""
        )
        rel = " ".join(link.get("rel", [])).lower()
        if "last" in classes or "last" in parent_classes or "last" in rel:
            last_values.extend(indices)
    if last_values:
        return max(last_values)
    return max(all_values, default=0)


def parse_printed_date(value, page_name):
    normalized = re.sub(r"\s+", " ", value.strip()).replace(".", "")
    normalized = re.sub(r"^Sept\b", "Sep", normalized)
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            pass
    raise SystemExit(
        f"could not parse SEC listing's printed release date {value!r} in {page_name}"
    )


def parse_listing_page(raw, page_name):
    soup = BeautifulSoup(raw, "html.parser")
    releases = []
    for row in soup.select("tr.pr-list-page-row"):
        time_tag = row.select_one(
            "td.views-field-field-publish-date time"
        ) or row.select_one("time")
        if time_tag is None:
            raise SystemExit(f"SEC release row lacks its printed date in {page_name}")
        release_date = parse_printed_date(time_tag.get_text(" ", strip=True), page_name)
        release_number_node = row.select_one(
            ".view-table_subfield_release_number .view-table_subfield_value"
        )
        visible = (
            release_number_node.get_text(" ", strip=True)
            if release_number_node
            else row.get_text(" ", strip=True)
        )
        match = re.search(r"\bLR\s*-?\s*(\d{5})\b", visible, flags=re.IGNORECASE)
        if not match:
            raise SystemExit(
                f"SEC release row lacks an LR-NNNNN number in {page_name}: {visible[:180]!r}"
            )
        releases.append(
            {"lr_number": int(match.group(1)), "release_date": release_date.isoformat()}
        )
    return releases, advertised_last_page(soup)


def month_labels():
    labels = []
    year, month = 2024, 1
    while (year, month) <= (2026, 10):
        labels.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return labels


def fetch_month(session, directory, year, month, snapshots, request_clock):
    pages = {}
    releases = []
    last_page = 0
    page_number = 0
    while page_number <= last_page:
        raw = request_page(
            session, year, month, page_number, directory, snapshots, request_clock
        )
        filename = f"{year:04d}-{month:02d}_page_{page_number:03d}.html"
        page_releases, advertised = parse_listing_page(raw, filename)
        if page_number > 0 and not page_releases:
            raise SystemExit(
                f"SEC pager advertised empty page {page_number} for {year:04d}-{month:02d}"
            )
        pages[page_number] = len(page_releases)
        releases.extend(page_releases)
        last_page = max(last_page, advertised)
        page_number += 1
    if sorted(pages) != list(range(last_page + 1)):
        raise SystemExit(
            f"SEC listing pagination is not contiguous for {year:04d}-{month:02d}"
        )
    return releases, {
        "pages_retrieved": len(pages),
        "last_page_index": last_page,
        "rows_retrieved": len(releases),
        "page_row_counts": pages,
    }


def load_offline_manifest(out_path, directory):
    if not out_path.is_file():
        raise SystemExit(
            f"offline mode needs the online result file for snapshot metadata: {out_path}"
        )
    try:
        previous = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read prior online result: {exc}") from exc
    if previous.get("source_id") != SOURCE_ID or not isinstance(
        previous.get("snapshots"), list
    ):
        raise SystemExit("prior result has the wrong source id or no snapshots list")
    snapshots = previous["snapshots"]
    listed = set()
    for entry in snapshots:
        relative = Path(entry.get("file", ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise SystemExit(
                f"unsafe snapshot path in prior result: {entry.get('file')!r}"
            )
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            raise SystemExit(
                f"offline snapshot is missing or not a regular file: {path}"
            )
        raw = path.read_bytes()
        if sha256_bytes(raw) != entry.get("sha256") or len(raw) != entry.get("bytes"):
            raise SystemExit(f"offline snapshot integrity check failed: {path}")
        listed.add(relative.as_posix())
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != listed:
        raise SystemExit(
            f"offline files differ from prior result (extra={sorted(actual - listed)}, missing={sorted(listed - actual)})"
        )
    expected_hash = manifest_sha256(snapshots)
    if expected_hash != previous.get("snapshot_sha256"):
        raise SystemExit("prior result snapshot_sha256 does not match its manifest")
    return previous


def parse_saved_pages(directory, snapshots):
    pages_by_month = defaultdict(dict)
    for entry in snapshots:
        relative = Path(entry.get("file", ""))
        match = re.fullmatch(r"(\d{4}-\d{2})_page_(\d{3})\.html", relative.name)
        if not match:
            continue
        month_label = match.group(1)
        page_number = int(match.group(2))
        pages_by_month[month_label][page_number] = directory / relative
    expected_months = month_labels()
    if set(pages_by_month) != set(expected_months):
        raise SystemExit(
            "offline SEC listing snapshots do not cover exactly 2024-01 through 2026-10"
        )

    releases_by_month = {}
    month_checks = {}
    for label in expected_months:
        page_map = pages_by_month[label]
        if 0 not in page_map:
            raise SystemExit(f"offline SEC listing pages lack page zero for {label}")
        page_results = {}
        advertised_values = []
        for page_number, path in sorted(page_map.items()):
            page_releases, advertised = parse_listing_page(path.read_bytes(), path.name)
            page_results[page_number] = page_releases
            advertised_values.append(advertised)
        last_page = max(advertised_values, default=0)
        expected_pages = list(range(last_page + 1))
        if sorted(page_map) != expected_pages:
            raise SystemExit(
                f"offline SEC listing pagination mismatch for {label}: "
                f"expected pages {expected_pages}, found {sorted(page_map)}"
            )
        releases_by_month[label] = [
            release
            for page_number in expected_pages
            for release in page_results[page_number]
        ]
        month_checks[label] = {
            "pages_retrieved": len(expected_pages),
            "last_page_index": last_page,
            "rows_retrieved": len(releases_by_month[label]),
            "page_row_counts": {
                str(page): len(page_results[page]) for page in expected_pages
            },
        }
    return releases_by_month, month_checks


def compact_lr_ranges(numbers):
    if not numbers:
        return []
    sorted_numbers = sorted(numbers)
    ranges = []
    start = previous = sorted_numbers[0]
    for current in sorted_numbers[1:]:
        if current == previous + 1:
            previous = current
            continue
        ranges.append((start, previous))
        start = previous = current
    ranges.append((start, previous))
    return [
        f"LR-{start:05d}" if start == end else f"LR-{start:05d}..LR-{end:05d}"
        for start, end in ranges
    ]


def monthly_sequence_checks(releases_by_month, month_checks):
    output = {}
    number_months = defaultdict(set)
    for label in month_labels():
        releases = releases_by_month[label]
        counts = Counter(item["lr_number"] for item in releases)
        distinct = sorted(counts)
        gaps = (
            set(range(distinct[0], distinct[-1] + 1)) - set(distinct)
            if distinct
            else set()
        )
        duplicates = {
            f"LR-{number:05d}": count
            for number, count in sorted(counts.items())
            if count > 1
        }
        for number in distinct:
            number_months[number].add(label)
        output[label] = {
            "listing_rows": len(releases),
            "distinct_lr_numbers": len(distinct),
            "min_lr": f"LR-{distinct[0]:05d}" if distinct else None,
            "max_lr": f"LR-{distinct[-1]:05d}" if distinct else None,
            "gap_count": len(gaps),
            "gap_ranges": compact_lr_ranges(gaps),
            "duplicate_lr_numbers": duplicates,
            "pages_retrieved": month_checks[label]["pages_retrieved"],
            "page_row_counts": month_checks[label]["page_row_counts"],
        }
    cross_month = {
        f"LR-{number:05d}": sorted(labels)
        for number, labels in sorted(number_months.items())
        if len(labels) > 1
    }
    return output, cross_month


def period_label(day, granularity):
    if granularity == "month":
        return day.strftime("%Y-%m")
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


def history_periods(granularity):
    if granularity == "month":
        return [label for label in month_labels() if label <= "2026-06"]
    return [
        f"{year}-Q{quarter}"
        for year in range(2024, 2027)
        for quarter in range(1, 5)
        if f"{year}-Q{quarter}" <= "2026-Q2"
    ]


def build_seeds(releases_by_month, month_checks):
    all_releases = [
        release for month in month_labels() for release in releases_by_month[month]
    ]
    sequence_checks, cross_month_duplicates = monthly_sequence_checks(
        releases_by_month, month_checks
    )
    results = {}
    for seed in SEEDS:
        start = date.fromisoformat(seed["window_start"])
        end = date.fromisoformat(seed["window_end"])
        granularity = seed["history_granularity"]
        window_releases = [
            release
            for release in all_releases
            if start <= date.fromisoformat(release["release_date"]) <= end
        ]
        history = []
        for label in history_periods(granularity):
            values = {
                release["lr_number"]
                for release in all_releases
                if date.fromisoformat(release["release_date"]) < start
                and period_label(
                    date.fromisoformat(release["release_date"]), granularity
                )
                == label
            }
            history.append({"period": label, "value": len(values)})
        window_months = []
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            window_months.append(f"{y:04d}-{m:02d}")
            m += 1
            if m == 13:
                y, m = y + 1, 1
        window_sets = {
            month: {
                release["lr_number"]
                for release in releases_by_month[month]
                if start <= date.fromisoformat(release["release_date"]) <= end
            }
            for month in window_months
        }
        details = {
            "window_breakdown": {
                month: len(window_sets[month]) for month in window_months
            },
            "window_listing_rows": sum(1 for release in window_releases),
            "sequence_checks_by_month": sequence_checks,
            "cross_month_duplicate_lr_numbers": cross_month_duplicates,
            "history_vintage": {
                "method": "publication_dated",
                "note": "The SEC release date is the date printed on its listing row; later corrections or republication can change the listing.",
            },
            "pagination_complete": all(
                month_checks[month]["pages_retrieved"]
                == month_checks[month]["last_page_index"] + 1
                for month in month_labels()
            ),
        }
        results[seed["id"]] = {
            "role": seed["role"],
            "value": len({release["lr_number"] for release in window_releases}),
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": granularity,
            "history": history,
            "detail": details,
            "notes": "Counts distinct LR numbers using the visible release date printed in the SEC monthly litigation-release listing; listing corrections remain possible.",
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Resolve SEC Litigation Release counts."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "")
    if not user_agent.strip():
        raise SystemExit(
            "SEC_EDGAR_USER_AGENT must be set to a non-empty requester/contact User-Agent"
        )

    directory = args.data_dir / SOURCE_ID
    prepare_directory(directory, args.offline)
    previous = load_offline_manifest(args.out, directory) if args.offline else None
    if args.offline:
        assert previous is not None
        snapshots = previous["snapshots"]
        retrieved_at = previous["retrieved_at"]
        releases_by_month, month_checks = parse_saved_pages(directory, snapshots)
    else:
        snapshots = []
        request_clock = [None]
        releases_by_month = {}
        month_checks = {}
        with requests.Session() as session:
            session.headers.update(
                {
                    "User-Agent": user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Encoding": "identity",
                }
            )
            for label in month_labels():
                year, month = map(int, label.split("-"))
                releases_by_month[label], month_checks[label] = fetch_month(
                    session, directory, year, month, snapshots, request_clock
                )
        if not snapshots:
            raise SystemExit("SEC listing request produced no snapshots")
        retrieved_at = min(item["retrieved_at"] for item in snapshots)

    seeds = build_seeds(releases_by_month, month_checks)
    snapshot_hash = manifest_sha256(snapshots)
    result = {
        "source_id": SOURCE_ID,
        "script": SCRIPT_NAME,
        "retrieved_at": retrieved_at,
        "snapshots": snapshots,
        "commit": None,
        "snapshot_sha256": snapshot_hash,
        "seeds": seeds,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if len(rendered.encode("utf-8")) >= MAX_RESULT_BYTES:
        raise SystemExit(
            f"results file would exceed 300 KB ({len(rendered.encode('utf-8'))} bytes)"
        )
    if args.offline and previous is not None and previous.get("seeds") != seeds:
        raise SystemExit("offline recomputation differs from online seed results")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")

    for item in sorted(snapshots, key=lambda entry: entry["file"]):
        print(f"{item['sha256']}  {item['file']}")
    print(f"snapshot_sha256  {snapshot_hash}")
    for seed_id, value in seeds.items():
        print(f"{seed_id}: {value['value']}")
        for month, check in value["detail"]["sequence_checks_by_month"].items():
            if check["gap_count"] or check["duplicate_lr_numbers"]:
                print(
                    f"{seed_id} monthly-sequence {month}: gaps={check['gap_ranges']} "
                    f"duplicates={check['duplicate_lr_numbers']}"
                )
    print(
        "pagination_months="
        f"{len(month_checks)} pages={sum(check['pages_retrieved'] for check in month_checks.values())} "
        f"listing_rows={sum(check['rows_retrieved'] for check in month_checks.values())}"
    )


if __name__ == "__main__":
    main()
