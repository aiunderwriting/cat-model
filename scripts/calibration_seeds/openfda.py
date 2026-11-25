#!/usr/bin/env python3
"""Resolve openFDA device and drug recall event seeds.

Canonical sources: https://api.fda.gov/device/enforcement.json and
https://api.fda.gov/drug/enforcement.json. For each endpoint, every quarter
2024-Q1 through 2026-Q3 is queried with report_date:[YYYYMMDD TO YYYYMMDD],
limit=1000, and skip paging. The date field is report_date, the weekly
Enforcement Report date. The device and drug event seeds count distinct
non-empty event_id values in Q3 2026. The device software seed uses the
case-insensitive substring "software" in reason_for_recall and counts distinct
non-empty event_id values among matching records. Records without event_id are
excluded from event counts and reported separately. Every raw JSON response,
redirect, or transient HTTP response body is saved byte-for-byte. Each quarter
is paged completely and checked against meta.results.total; skip is limited to
the API's 25000 maximum. All seeds are publication-dated by report_date, though
the FDA can revise records later and meta.last_updated is retained per quarter.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "openfda"
USER_AGENT = "AIUC calibration-seed resolver (research)"
BASE_URLS = {
    "device": "https://api.fda.gov/device/enforcement.json",
    "drug": "https://api.fda.gov/drug/enforcement.json",
}
RETRY_DELAYS = (2, 4, 8, 16, 32)
PAGE_SIZE = 1000
MAX_SKIP = 25000
MIN_REQUEST_INTERVAL_SECONDS = 0.26
MAX_RESULT_BYTES = 300 * 1024
LAST_REQUEST_STARTED = None

SEEDS: list[dict[str, Any]] = [
    {
        "id": "fda-device-recall-events-2026q3",
        "source": "device",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"kind": "distinct_event_id", "date_field": "report_date"},
        "history_granularity": "quarter",
    },
    {
        "id": "fda-device-software-recall-events-2026q3",
        "source": "device",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {
            "kind": "distinct_event_id",
            "date_field": "report_date",
            "reason_for_recall_contains_case_insensitive": "software",
        },
        "history_granularity": "quarter",
    },
    {
        "id": "fda-drug-recall-events-2026q3",
        "source": "drug",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"kind": "distinct_event_id", "date_field": "report_date"},
        "history_granularity": "quarter",
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


def prepare_directory(data_root, offline):
    directory = data_root / SOURCE_ID
    if offline:
        if not directory.is_dir():
            raise RuntimeError(f"offline data directory does not exist: {directory}")
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise RuntimeError(
            f"refusing online download into non-empty directory: {directory}"
        )
    return directory


def record_response(directory, snapshots, filename, url, content):
    path = directory / filename
    path.write_bytes(content)
    snapshots.append(
        {
            "url": url,
            "file": filename,
            "sha256": sha256_bytes(content),
            "bytes": len(content),
            "retrieved_at": utc_now(),
        }
    )


def proxy_refusal(exc):
    message = str(exc).lower()
    return "403" in message and (
        "connect" in message or "tunnel" in message or "proxy" in message
    )


def enforce_rate_limit():
    global LAST_REQUEST_STARTED
    now = time.monotonic()
    if LAST_REQUEST_STARTED is not None:
        delay = MIN_REQUEST_INTERVAL_SECONDS - (now - LAST_REQUEST_STARTED)
        if delay > 0:
            time.sleep(delay)
    LAST_REQUEST_STARTED = time.monotonic()


def request_bytes(session, url, directory, filename, snapshots):
    stem = Path(filename).stem
    for attempt in range(len(RETRY_DELAYS) + 1):
        response = None
        enforce_rate_limit()
        try:
            response = session.get(url, timeout=(30, 180), allow_redirects=True)
        except requests.exceptions.ProxyError as exc:
            if proxy_refusal(exc):
                raise RuntimeError(
                    f"HTTP 403 on proxy CONNECT while requesting {url}: {exc}"
                ) from exc
            if attempt >= len(RETRY_DELAYS):
                raise RuntimeError(
                    f"request failed after retries for {url}: {exc}"
                ) from exc
            time.sleep(RETRY_DELAYS[attempt])
            continue
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt >= len(RETRY_DELAYS):
                raise RuntimeError(
                    f"request failed after retries for {url}: {exc}"
                ) from exc
            time.sleep(RETRY_DELAYS[attempt])
            continue

        chain = [*list(response.history), response]
        for index, item in enumerate(chain):
            if item.status_code in (301, 302, 303, 307, 308):
                saved_name = f"{stem}_attempt{attempt}_redirect{index}_http{item.status_code}.bin"
            elif item.status_code >= 400:
                saved_name = f"{stem}_attempt{attempt}_http{item.status_code}.bin"
            elif index == len(chain) - 1:
                saved_name = filename
            else:
                saved_name = f"{stem}_attempt{attempt}_redirect{index}_http{item.status_code}.bin"
            record_response(directory, snapshots, saved_name, item.url, item.content)

        if response.status_code == 403:
            response.close()
            raise RuntimeError(f"HTTP 403 from {url}; host access is refused")
        if response.status_code == 429 or response.status_code >= 500:
            status = response.status_code
            response.close()
            if attempt >= len(RETRY_DELAYS):
                raise RuntimeError(f"transient HTTP {status} after retries for {url}")
            time.sleep(RETRY_DELAYS[attempt])
            continue
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            response.close()
            raise RuntimeError(f"HTTP request failed for {url}: {exc}") from exc
        content = response.content
        response.close()
        return content
    raise RuntimeError(f"request retry loop ended unexpectedly for {url}")


def quarter_window(year, quarter):
    first_month = 3 * quarter - 2
    start = date(year, first_month, 1)
    next_start = (
        date(year + 1, 1, 1) if quarter == 4 else date(year, first_month + 3, 1)
    )
    return start, date.fromordinal(next_start.toordinal() - 1)


def quarter_labels(start_year=2024, start_quarter=1, end_year=2026, end_quarter=3):
    year, quarter = start_year, start_quarter
    output = []
    while (year, quarter) <= (end_year, end_quarter):
        output.append((year, quarter, f"{year}-Q{quarter}"))
        quarter += 1
        if quarter == 5:
            year += 1
            quarter = 1
    return output


def query_url(source, year, quarter, skip):
    start, end = quarter_window(year, quarter)
    params = {
        "search": f"report_date:[{start:%Y%m%d} TO {end:%Y%m%d}]",
        "limit": PAGE_SIZE,
        "skip": skip,
    }
    return requests.Request("GET", BASE_URLS[source], params=params).prepare().url


def parse_api_response(raw, page_name, expected_quarter):
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"openFDA response {page_name} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected openFDA response shape in {page_name}")
    meta = payload.get("meta")
    results_meta = meta.get("results") if isinstance(meta, dict) else None
    total = results_meta.get("total") if isinstance(results_meta, dict) else None
    records = payload.get("results")
    if not isinstance(total, int) or not isinstance(records, list):
        raise RuntimeError(
            f"openFDA response {page_name} lacks meta.results.total or results"
        )
    if not all(isinstance(record, dict) for record in records):
        raise RuntimeError(
            f"openFDA results contain a non-object record in {page_name}"
        )
    last_updated = meta.get("last_updated") if isinstance(meta, dict) else None
    start, end = quarter_window(*expected_quarter)
    for record in records:
        value = str(record.get("report_date", "")).strip()
        if not re.fullmatch(r"\d{8}", value):
            raise RuntimeError(f"invalid report_date {value!r} in {page_name}")
        report_day = datetime.strptime(value, "%Y%m%d").date()
        if not start <= report_day <= end:
            raise RuntimeError(f"openFDA returned {value} outside {expected_quarter}")
    return total, records, last_updated


def retrieve_quarter(session, directory, source, year, quarter, label, snapshots):
    total_expected = None
    last_updated_values = set()
    all_records = []
    downloaded = []
    skip = 0
    page_count = 0
    while True:
        if skip > MAX_SKIP:
            raise RuntimeError(
                f"openFDA {source} {label} needs skip={skip}, above the API maximum 25000"
            )
        url = query_url(source, year, quarter, skip)
        filename = f"{source}_enforcement_{label}_skip_{skip:05d}.json"
        content = request_bytes(session, url, directory, filename, snapshots)
        downloaded.append(filename)
        total, records, last_updated = parse_api_response(
            content, filename, (year, quarter)
        )
        if total_expected is None:
            total_expected = total
        elif total != total_expected:
            raise RuntimeError(
                f"openFDA total changed while paging {source} {label}: {total_expected} -> {total}"
            )
        if last_updated is not None:
            last_updated_values.add(last_updated)
        if len(records) > PAGE_SIZE:
            raise RuntimeError(
                f"openFDA returned more than {PAGE_SIZE} records for {source} {label}"
            )
        if not records and len(all_records) < total_expected:
            raise RuntimeError(
                f"openFDA returned an empty page before reaching total for {source} {label}"
            )
        if len(all_records) + len(records) > total_expected:
            raise RuntimeError(
                f"openFDA rows exceed reported total for {source} {label}"
            )
        all_records.extend(records)
        page_count += 1
        if len(all_records) == total_expected:
            break
        if not records:
            raise RuntimeError(
                f"openFDA pagination made no progress for {source} {label}"
            )
        skip += len(records)
    if len(all_records) != total_expected:
        raise RuntimeError(
            f"openFDA pagination incomplete for {source} {label}: {len(all_records)} of {total_expected}"
        )
    if len(last_updated_values) > 1:
        raise RuntimeError(
            f"openFDA meta.last_updated changed across pages for {source} {label}"
        )
    return all_records, {
        "records": total_expected,
        "rows_retrieved": len(all_records),
        "meta_last_updated": next(iter(last_updated_values), None),
        "pages_retrieved": page_count,
        "files": downloaded,
    }


def normalized_event_id(record):
    value = record.get("event_id")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def record_is_software(record):
    reason = record.get("reason_for_recall")
    return isinstance(reason, str) and "software" in reason.casefold()


def quarter_metrics(records):
    event_ids = set()
    software_ids = set()
    software_rows = 0
    missing = 0
    for record in records:
        event_id = normalized_event_id(record)
        software = record_is_software(record)
        if software:
            software_rows += 1
        if event_id is None:
            missing += 1
            continue
        event_ids.add(event_id)
        if software:
            software_ids.add(event_id)
    return {
        "records": len(records),
        "distinct_events": len(event_ids),
        "software_distinct_events": len(software_ids),
        "software_record_count": software_rows,
        "records_missing_event_id": missing,
    }


def load_offline_manifest(out_path, directory):
    if not out_path.is_file():
        raise RuntimeError(f"offline mode needs the online result file: {out_path}")
    try:
        previous = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read prior openFDA result: {exc}") from exc
    if previous.get("source_id") != SOURCE_ID or not isinstance(
        previous.get("snapshots"), list
    ):
        raise RuntimeError(
            "prior openFDA result has the wrong source id or no snapshots list"
        )
    listed = set()
    for entry in previous["snapshots"]:
        relative = entry.get("file")
        if (
            not isinstance(relative, str)
            or Path(relative).name != relative
            or relative in ("", ".", "..")
        ):
            raise RuntimeError(f"unsafe openFDA snapshot path: {relative!r}")
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"offline openFDA snapshot is missing: {path}")
        raw = path.read_bytes()
        if sha256_bytes(raw) != entry.get("sha256") or len(raw) != entry.get("bytes"):
            raise RuntimeError(
                f"offline openFDA snapshot failed integrity check: {relative}"
            )
        listed.add(relative)
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != listed:
        raise RuntimeError(
            f"offline openFDA files differ from prior manifest (extra={sorted(actual - listed)}, "
            f"missing={sorted(listed - actual)})"
        )
    if manifest_sha256(previous["snapshots"]) != previous.get("snapshot_sha256"):
        raise RuntimeError(
            "offline openFDA manifest hash does not match the prior result"
        )
    return previous


def load_offline_quarters(directory, snapshots):
    grouped = defaultdict(list)
    for entry in snapshots:
        filename = entry.get("file", "")
        match = re.fullmatch(
            r"(device|drug)_enforcement_(\d{4})-Q([1-4])_skip_(\d{5})\.json", filename
        )
        if match:
            source, year, quarter, skip = match.groups()
            grouped[(source, int(year), int(quarter))].append(
                (int(skip), directory / filename)
            )
    expected = {
        (source, year, quarter)
        for source in BASE_URLS
        for year, quarter, _label in quarter_labels()
    }
    if set(grouped) != expected:
        raise RuntimeError(
            "offline openFDA files do not cover both endpoints from 2024-Q1 through 2026-Q3"
        )

    records_by_source_quarter = {}
    stats_by_source_quarter = {}
    for source in BASE_URLS:
        for year, quarter, label in quarter_labels():
            pages = sorted(grouped[(source, year, quarter)])
            records = []
            total_expected = None
            last_updated_values = set()
            next_skip = 0
            file_names = []
            for skip, path in pages:
                if skip != next_skip:
                    raise RuntimeError(
                        f"offline openFDA page gap for {source} {label}: expected skip={next_skip}"
                    )
                page_total, page_records, last_updated = parse_api_response(
                    path.read_bytes(), path.name, (year, quarter)
                )
                if total_expected is None:
                    total_expected = page_total
                elif page_total != total_expected:
                    raise RuntimeError(
                        f"offline openFDA totals disagree across pages for {source} {label}"
                    )
                if last_updated is not None:
                    last_updated_values.add(last_updated)
                records.extend(page_records)
                next_skip += len(page_records)
                file_names.append(path.name)
            if len(records) != total_expected:
                raise RuntimeError(
                    f"offline openFDA pages contain {len(records)} of {total_expected} rows for {source} {label}"
                )
            if len(last_updated_values) > 1:
                raise RuntimeError(
                    f"offline openFDA meta.last_updated changed across pages for {source} {label}"
                )
            records_by_source_quarter[(source, label)] = records
            metrics = quarter_metrics(records)
            metrics.update(
                {
                    "records": total_expected,
                    "rows_retrieved": len(records),
                    "meta_last_updated": next(iter(last_updated_values), None),
                    "pages_retrieved": len(pages),
                    "files": file_names,
                }
            )
            stats_by_source_quarter[(source, label)] = metrics
    return records_by_source_quarter, stats_by_source_quarter


def build_results(args):
    global LAST_REQUEST_STARTED
    directory = prepare_directory(args.data_dir, args.offline)
    if args.offline:
        previous = load_offline_manifest(args.out, directory)
        snapshots = previous["snapshots"]
        retrieved_at = previous["retrieved_at"]
        records_by_source_quarter, stats_by_source_quarter = load_offline_quarters(
            directory, snapshots
        )
    else:
        snapshots = []
        records_by_source_quarter = {}
        stats_by_source_quarter = {}
        LAST_REQUEST_STARTED = None
        with requests.Session() as session:
            session.headers.update(
                {
                    "User-Agent": USER_AGENT,
                    "Accept-Encoding": "identity",
                    "Accept": "application/json",
                }
            )
            for source in BASE_URLS:
                for year, quarter, label in quarter_labels():
                    records, stats = retrieve_quarter(
                        session, directory, source, year, quarter, label, snapshots
                    )
                    records_by_source_quarter[(source, label)] = records
                    metrics = quarter_metrics(records)
                    metrics.update(stats)
                    stats_by_source_quarter[(source, label)] = metrics
        if not snapshots:
            raise RuntimeError("openFDA produced no snapshots")
        retrieved_at = min(item["retrieved_at"] for item in snapshots)

    history_labels = [
        label for _year, _quarter, label in quarter_labels(2024, 1, 2026, 2)
    ]
    quarter_stats_by_source = {
        source: {
            label: stats_by_source_quarter[(source, label)]
            for _year, _quarter, label in quarter_labels()
        }
        for source in BASE_URLS
    }
    seed_results = {}
    for seed in SEEDS:
        source = seed["source"]
        is_software = (
            seed["filter"].get("reason_for_recall_contains_case_insensitive")
            == "software"
        )
        q3_label = "2026-Q3"
        q3_records = records_by_source_quarter[(source, q3_label)]
        overall_ids = set()
        software_ids = set()
        missing_ids = 0
        records_by_month = {"2026-07": 0, "2026-08": 0, "2026-09": 0}
        overall_ids_by_month = {month: set() for month in records_by_month}
        software_ids_by_month = {month: set() for month in records_by_month}
        software_rows_by_month = {month: 0 for month in records_by_month}
        for record in q3_records:
            report_day = datetime.strptime(str(record["report_date"]), "%Y%m%d").date()
            month = report_day.strftime("%Y-%m")
            records_by_month[month] += 1
            event_id = normalized_event_id(record)
            if event_id is None:
                missing_ids += 1
                continue
            overall_ids.add(event_id)
            overall_ids_by_month[month].add(event_id)
            if record_is_software(record):
                software_ids.add(event_id)
                software_ids_by_month[month].add(event_id)
                software_rows_by_month[month] += 1

        selected_ids = software_ids if is_software else overall_ids
        selected_by_month = (
            software_ids_by_month if is_software else overall_ids_by_month
        )
        history_key = "software_distinct_events" if is_software else "distinct_events"
        quarterly_detail = {
            label: {
                "records": quarter_stats_by_source[source][label]["records"],
                "rows_retrieved": quarter_stats_by_source[source][label][
                    "rows_retrieved"
                ],
                "pages_retrieved": quarter_stats_by_source[source][label][
                    "pages_retrieved"
                ],
                "distinct_events": quarter_stats_by_source[source][label][
                    "distinct_events"
                ],
                "software_distinct_events": quarter_stats_by_source[source][label][
                    "software_distinct_events"
                ],
                "records_missing_event_id": quarter_stats_by_source[source][label][
                    "records_missing_event_id"
                ],
                "meta_last_updated": quarter_stats_by_source[source][label][
                    "meta_last_updated"
                ],
            }
            for _year, _quarter, label in quarter_labels()
        }
        detail = {
            "window_breakdown": {
                month: len(selected_by_month[month]) for month in records_by_month
            },
            "window_record_count": len(q3_records),
            "window_records_by_month": records_by_month,
            "window_software_record_count": sum(software_rows_by_month.values()),
            "window_software_records_by_month": software_rows_by_month,
            "window_records_missing_event_id": missing_ids,
            "window_distinct_events": len(overall_ids),
            "window_software_distinct_events": len(software_ids),
            "meta_last_updated": quarter_stats_by_source[source][q3_label][
                "meta_last_updated"
            ],
            "distinct_event_counts_by_quarter": {
                label: quarter_stats_by_source[source][label][history_key]
                for label in history_labels
            },
            "all_distinct_event_counts_by_quarter": {
                label: quarter_stats_by_source[source][label]["distinct_events"]
                for _year, _quarter, label in quarter_labels()
            },
            "software_distinct_event_counts_by_quarter": {
                label: quarter_stats_by_source[source][label][
                    "software_distinct_events"
                ]
                for _year, _quarter, label in quarter_labels()
            },
            "quarterly_api_totals_and_counts": quarterly_detail,
            "pagination_crosscheck_by_quarter": {
                label: {
                    "api_total": quarter_stats_by_source[source][label]["records"],
                    "rows_retrieved": quarter_stats_by_source[source][label][
                        "rows_retrieved"
                    ],
                    "pages_retrieved": quarter_stats_by_source[source][label][
                        "pages_retrieved"
                    ],
                }
                for _year, _quarter, label in quarter_labels()
            },
            "pagination_crosscheck": {
                "q3_meta_results_total": quarter_stats_by_source[source][q3_label][
                    "records"
                ],
                "q3_rows_retrieved": len(q3_records),
                "q3_pages_retrieved": quarter_stats_by_source[source][q3_label][
                    "pages_retrieved"
                ],
            },
            "history_vintage": {
                "method": "publication_dated",
                "note": "report_date is the weekly Enforcement Report date; later FDA revisions can change records and meta.last_updated is retained.",
            },
        }
        seed_results[seed["id"]] = {
            "role": seed["role"],
            "value": len(selected_ids),
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": seed["history_granularity"],
            "history": [
                {
                    "period": label,
                    "value": quarter_stats_by_source[source][label][history_key],
                }
                for label in history_labels
            ],
            "detail": detail,
            "notes": (
                "Counts distinct non-empty event_id values among report_date records whose reason_for_recall "
                "contains the case-insensitive substring software. Records without event_id are omitted from "
                "the count and reported separately. report_date is the weekly Enforcement Report date."
                if is_software
                else "Counts distinct non-empty event_id values among report_date records; records without "
                "event_id are omitted from the count and reported separately. report_date is the weekly "
                "Enforcement Report date."
            ),
        }

    snapshots = sorted(snapshots, key=lambda item: item["file"])
    snapshot_hash = manifest_sha256(snapshots)
    result = {
        "source_id": SOURCE_ID,
        "script": Path(__file__).name,
        "retrieved_at": retrieved_at,
        "snapshots": snapshots,
        "commit": None,
        "snapshot_sha256": snapshot_hash,
        "seeds": seed_results,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if len(rendered.encode("utf-8")) >= MAX_RESULT_BYTES:
        raise RuntimeError(
            f"results file would exceed 300 KB ({len(rendered.encode('utf-8'))} bytes)"
        )
    if args.offline and previous.get("seeds") != seed_results:
        raise RuntimeError(
            "offline recomputation differs from online openFDA seed results"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    for item in snapshots:
        print(f"{item['sha256']}  {item['file']}")
    print(f"snapshot_sha256  {snapshot_hash}")
    for seed_id, seed_result in seed_results.items():
        print(f"{seed_id}: {seed_result['value']}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Resolve openFDA device and drug recall event seeds."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        build_results(args)
    except Exception as exc:
        print(f"openFDA resolver failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
