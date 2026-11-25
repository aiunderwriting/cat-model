#!/usr/bin/env python3
"""Resolve CFPB complaint-count seeds from the public Complaint Database API.

Canonical source: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/
Exact filters and inclusive UTC windows (all use date_received; no company or
other exclusions):
- cfpb-fraud-scam-complaints-2026q3: issue = "Fraud or scam", all products,
  2026-07-01 through 2026-09-30.
- cfpb-money-transfer-complaints-2026q3: product =
  "Money transfer, virtual currency, or money service", all issues,
  2026-07-01 through 2026-09-30.
- cfpb-fraud-alert-complaints-2026q3: issue =
  "Problem with fraud alerts or security freezes", all products,
  2026-07-01 through 2026-09-30.

Monthly history uses date_received from 2024-01 through 2026-05. Each filtered
monthly count uses size=0, no_aggs=true, and inclusive date_received_min/max;
exact hit totals are required. The script also saves a single-day min=max
probe and an unfiltered July 2026 aggregation response to verify date-boundary
inclusion and every exact filter (including the new issue string). Each raw
successful response is saved byte-for-byte with a deterministic filename.

CFPB publishes complaints after the company responds or after 15 days, so
June 2026 and later are guarded out of history for the 2026-06-30 information
vintage. Complaint counts can still change as records are published or the API
index is revised.
"""

import argparse
import calendar
import datetime as dt
import email.utils
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "cfpb"
SCRIPT_NAME = "cfpb.py"
API_URL = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
)
USER_AGENT = "AIUC calibration-seed resolver (research)"
TRANSIENT_RETRY_DELAYS = (2, 4, 8, 16, 32)
REFUSAL_RETRY_DELAYS = (60, 120)
START_MONTH = (2024, 1)
LAST_HISTORY_MONTH = (2026, 5)
PROBE_DATE = "2026-09-01"
VERIFY_MONTH = (2026, 7)
HISTORY_VINTAGE_REASON = (
    "CFPB publishes complaints after the company responds or after 15 days, "
    "so June 2026 was not guaranteed public by the 2026-06-30 information date."
)

SEEDS: list[dict[str, Any]] = [
    {
        "id": "cfpb-fraud-scam-complaints-2026q3",
        "role": "primary",
        "window": {"start": "2026-07-01", "end": "2026-09-30"},
        "filter": {"issue": "Fraud or scam"},
    },
    {
        "id": "cfpb-money-transfer-complaints-2026q3",
        "role": "primary",
        "window": {"start": "2026-07-01", "end": "2026-09-30"},
        "filter": {"product": "Money transfer, virtual currency, or money service"},
    },
    {
        "id": "cfpb-fraud-alert-complaints-2026q3",
        "role": "primary",
        "window": {"start": "2026-07-01", "end": "2026-09-30"},
        "filter": {"issue": "Problem with fraud alerts or security freezes"},
    },
]


class RequestLimiter:
    """Space request starts at least one second apart."""

    def __init__(self):
        self.last_started = None

    def wait(self):
        now = time.monotonic()
        if self.last_started is not None:
            remaining = 1.0 - (now - self.last_started)
            if remaining > 0:
                time.sleep(remaining)
        self.last_started = time.monotonic()


def utc_now():
    return (
        dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def utc_date(value):
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"invalid UTC retrieval timestamp: {value!r}")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RuntimeError(f"invalid UTC retrieval timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"retrieval timestamp has no timezone: {value!r}")
    return parsed.astimezone(dt.UTC).date()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def retry_after_seconds(value):
    if value is None:
        return None
    try:
        return max(0.0, float(value.strip()))
    except (AttributeError, ValueError):
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return max(
        0.0, (parsed.astimezone(dt.UTC) - dt.datetime.now(dt.UTC)).total_seconds()
    )


def get_json(session, params, limiter, status_events):
    transient_retries = 0
    refusal_retries = 0
    while True:
        limiter.wait()
        try:
            response = session.get(API_URL, params=params, timeout=(15, 90))
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if transient_retries >= len(TRANSIENT_RETRY_DELAYS):
                raise RuntimeError(
                    f"CFPB request failed after transient retries: {exc}"
                ) from exc
            delay = TRANSIENT_RETRY_DELAYS[transient_retries]
            transient_retries += 1
            print(
                f"cfpb.py: transient connection failure; retrying in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)
            continue

        status = response.status_code
        if status in (403, 429):
            retry_after = response.headers.get("Retry-After")
            status_events.append(
                {
                    "status_code": status,
                    "url": response.url,
                    "retrieved_at": utc_now(),
                    "retry_after": retry_after,
                }
            )
            if refusal_retries >= len(REFUSAL_RETRY_DELAYS):
                url = response.url
                response.close()
                raise RuntimeError(
                    f"CFPB request failed after {len(REFUSAL_RETRY_DELAYS)} retries with HTTP {status}: {url}"
                )
            delay = float(REFUSAL_RETRY_DELAYS[refusal_retries])
            retry_value = retry_after_seconds(retry_after)
            if retry_value is not None:
                delay = max(delay, retry_value)
            refusal_retries += 1
            url = response.url
            print(
                f"cfpb.py: received HTTP {status} from {url}; retrying in {delay:g}s"
                + (f" (Retry-After={retry_after})" if retry_after is not None else ""),
                file=sys.stderr,
            )
            response.close()
            time.sleep(delay)
            continue

        if 500 <= status <= 599:
            if transient_retries >= len(TRANSIENT_RETRY_DELAYS):
                url = response.url
                response.close()
                raise RuntimeError(
                    f"CFPB request failed after transient retries with HTTP {status}: {url}"
                )
            delay = TRANSIENT_RETRY_DELAYS[transient_retries]
            transient_retries += 1
            url = response.url
            print(
                f"cfpb.py: received HTTP {status} from {url}; retrying in {delay}s",
                file=sys.stderr,
            )
            response.close()
            time.sleep(delay)
            continue

        if not 200 <= status <= 299:
            url = response.url
            response.close()
            raise RuntimeError(f"CFPB request failed with HTTP {status}: {url}")

        body = response.content
        url = response.url
        retrieved_at = utc_now()
        response.close()
        return body, url, retrieved_at


def save_snapshot(root, filename, url, body, retrieved_at, snapshots):
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    snapshots.append(
        {
            "url": url,
            "file": filename,
            "sha256": sha256_bytes(body),
            "bytes": len(body),
            "retrieved_at": retrieved_at,
        }
    )
    try:
        return json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"CFPB response is not valid JSON: {filename}") from exc


def load_offline_snapshots(root, out_path):
    if not root.is_dir():
        raise RuntimeError(f"offline data directory does not exist: {root}")
    if not out_path.is_file():
        raise RuntimeError(
            f"--offline requires prior snapshot metadata in --out: {out_path}"
        )
    previous = json.loads(out_path.read_text(encoding="utf-8"))
    if previous.get("source_id") != SOURCE_ID or not isinstance(
        previous.get("snapshots"), list
    ):
        raise RuntimeError(
            f"--out does not contain prior {SOURCE_ID} snapshot metadata"
        )

    snapshots = previous["snapshots"]
    if not snapshots:
        raise RuntimeError("prior CFPB result has an empty snapshots list")
    payloads = {}
    listed = set()
    for entry in snapshots:
        if not isinstance(entry, dict):
            raise RuntimeError("invalid snapshot metadata entry")
        relative = Path(entry.get("file", ""))
        if not entry.get("file") or relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"unsafe snapshot path in prior result: {entry.get('file')!r}"
            )
        name = relative.as_posix()
        if name in listed:
            raise RuntimeError(f"duplicate snapshot path in prior result: {name}")
        listed.add(name)
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"offline snapshot is missing: {path}")
        body = path.read_bytes()
        if len(body) != entry.get("bytes") or sha256_bytes(body) != entry.get("sha256"):
            raise RuntimeError(f"offline snapshot checksum/size mismatch: {path}")
        try:
            payloads[name] = json.loads(body.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"offline snapshot is not valid JSON: {path}") from exc
        utc_date(entry.get("retrieved_at"))

    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual != listed:
        raise RuntimeError(
            "offline directory contents differ from the saved snapshot list "
            f"(extra={sorted(actual - listed)}, missing={sorted(listed - actual)})"
        )
    return snapshots, payloads


def month_tuples(first, last):
    year, month = first
    last_year, last_month = last
    result = []
    while (year, month) <= (last_year, last_month):
        result.append((year, month))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return result


def history_months():
    return month_tuples(START_MONTH, LAST_HISTORY_MONTH)


def window_months(window):
    start = dt.date.fromisoformat(window["start"])
    end = dt.date.fromisoformat(window["end"])
    if start.day != 1 or end.day != calendar.monthrange(end.year, end.month)[1]:
        raise RuntimeError(f"window is not a complete calendar-month range: {window}")
    return month_tuples((start.year, start.month), (end.year, end.month))


def month_bounds(year, month):
    return (
        dt.date(year, month, 1).isoformat(),
        dt.date(year, month, calendar.monthrange(year, month)[1]).isoformat(),
    )


def filter_identity(filter_spec):
    if len(filter_spec) != 1:
        raise RuntimeError(
            f"each CFPB seed must have exactly one filter: {filter_spec!r}"
        )
    return next(iter(filter_spec.items()))


def filter_token(filter_spec):
    key, value = filter_identity(filter_spec)
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return f"{key}__{slug}"


def month_filename(filter_spec, year, month):
    return f"month__{filter_token(filter_spec)}__{year:04d}-{month:02d}.json"


def monthly_params(filter_spec, year, month):
    start, end = month_bounds(year, month)
    key, value = filter_identity(filter_spec)
    return {
        "size": 0,
        "no_aggs": "true",
        "date_received_min": start,
        "date_received_max": end,
        key: value,
    }


def expected_month_files():
    filenames = set()
    for seed in SEEDS:
        needed = set(history_months())
        needed.update(window_months(seed["window"]))
        for year, month in needed:
            filenames.add(month_filename(seed["filter"], year, month))
    return filenames


def request_plan():
    plan = {}
    for seed in SEEDS:
        needed = set(history_months())
        needed.update(window_months(seed["window"]))
        filter_key, filter_value = filter_identity(seed["filter"])
        for year, month in needed:
            query_key = (year, month, filter_key, filter_value)
            plan.setdefault(
                query_key,
                (
                    month_filename(seed["filter"], year, month),
                    monthly_params(seed["filter"], year, month),
                ),
            )
    return sorted(plan.items(), key=lambda item: item[0])


def total_hits(payload):
    total = payload.get("hits", {}).get("total")
    if isinstance(total, dict):
        if total.get("relation", "eq") != "eq":
            raise RuntimeError(f"CFPB returned a non-exact total: {total}")
        total = total.get("value")
    if not isinstance(total, int) or isinstance(total, bool):
        raise RuntimeError(f"CFPB response has no integer hits.total.value: {total!r}")
    return total


def find_bucket(payload, aggregation, key):
    node = payload.get("aggregations", {}).get(aggregation, {}).get(aggregation, {})
    buckets = node.get("buckets", [])
    for bucket in buckets:
        if bucket.get("key") == key:
            count = bucket.get("doc_count")
            if not isinstance(count, int) or isinstance(count, bool):
                raise RuntimeError(
                    f"CFPB aggregation bucket has an invalid doc_count: {bucket!r}"
                )
            return count
    return None


def manifest_sha256(snapshots):
    lines = [
        f"{item['sha256']}  {item['file']}"
        for item in sorted(snapshots, key=lambda item: item["file"])
    ]
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def download_online(root):
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise RuntimeError(f"refusing to download into non-empty directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    snapshots = []
    payloads = {}
    status_events = []
    limiter = RequestLimiter()
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

        probe_params = {
            "size": 0,
            "no_aggs": "true",
            "date_received_min": PROBE_DATE,
            "date_received_max": PROBE_DATE,
            "issue": "Fraud or scam",
        }
        body, url, retrieved_at = get_json(
            session, probe_params, limiter, status_events
        )
        probe_name = f"probe__max-inclusive-{PROBE_DATE}.json"
        payloads[probe_name] = save_snapshot(
            root, probe_name, url, body, retrieved_at, snapshots
        )
        probe_total = total_hits(payloads[probe_name])
        if probe_total <= 0:
            raise RuntimeError(
                "single-day CFPB min=max probe returned no records; date_received_max inclusivity is unverified"
            )

        verify_start, verify_end = month_bounds(*VERIFY_MONTH)
        verify_params = {
            "size": 0,
            "date_received_min": verify_start,
            "date_received_max": verify_end,
        }
        body, url, retrieved_at = get_json(
            session, verify_params, limiter, status_events
        )
        verify_name = f"verify__unfiltered-aggregation-{VERIFY_MONTH[0]:04d}-{VERIFY_MONTH[1]:02d}.json"
        payloads[verify_name] = save_snapshot(
            root, verify_name, url, body, retrieved_at, snapshots
        )
        total_hits(payloads[verify_name])

        # Fail early if an exact category is missing from the API aggregation.
        for seed in SEEDS:
            filter_key, filter_value = filter_identity(seed["filter"])
            if find_bucket(payloads[verify_name], filter_key, filter_value) is None:
                raise RuntimeError(
                    f"CFPB {filter_key} aggregation is missing exact category {filter_value!r}"
                )

        for _, (filename, params) in request_plan():
            body, url, retrieved_at = get_json(session, params, limiter, status_events)
            payloads[filename] = save_snapshot(
                root, filename, url, body, retrieved_at, snapshots
            )

    return snapshots, payloads, status_events


def validate_snapshot_set(root, snapshots, payloads):
    required = {
        f"probe__max-inclusive-{PROBE_DATE}.json",
        f"verify__unfiltered-aggregation-{VERIFY_MONTH[0]:04d}-{VERIFY_MONTH[1]:02d}.json",
    } | expected_month_files()
    listed = {entry.get("file") for entry in snapshots}
    if listed != required:
        raise RuntimeError(
            f"snapshot list does not match required query set "
            f"(extra={sorted(listed - required)}, missing={sorted(required - listed)})"
        )
    if set(payloads) != required:
        raise RuntimeError(
            f"loaded response set does not match required query set "
            f"(extra={sorted(set(payloads) - required)}, missing={sorted(required - set(payloads))})"
        )
    if root is not None and root.exists():
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if actual != required:
            raise RuntimeError(
                f"data directory contents differ from required snapshot set "
                f"(extra={sorted(actual - required)}, missing={sorted(required - actual)})"
            )


def build_result(snapshots, payloads):
    if not snapshots:
        raise RuntimeError("no CFPB snapshots were recorded")
    retrieved_at = snapshots[0]["retrieved_at"]
    validate_snapshot_set(None, snapshots, payloads)

    probe_name = f"probe__max-inclusive-{PROBE_DATE}.json"
    verify_name = f"verify__unfiltered-aggregation-{VERIFY_MONTH[0]:04d}-{VERIFY_MONTH[1]:02d}.json"
    probe_total = total_hits(payloads[probe_name])
    if probe_total <= 0:
        raise RuntimeError(
            "single-day CFPB min=max probe returned no records; date_received_max inclusivity is unverified"
        )
    verification_payload = payloads[verify_name]
    total_hits(verification_payload)

    month_counts = {}
    latest_meta_by_filter = {}
    for seed in SEEDS:
        filter_key, filter_value = filter_identity(seed["filter"])
        needed = set(history_months())
        needed.update(window_months(seed["window"]))
        filter_key_tuple = (filter_key, filter_value)
        for year, month in needed:
            filename = month_filename(seed["filter"], year, month)
            if filename not in payloads:
                raise RuntimeError(
                    f"missing required monthly CFPB response: {filename}"
                )
            count = total_hits(payloads[filename])
            month_counts[(filter_key, filter_value, year, month)] = count
            old_latest = latest_meta_by_filter.get(filter_key_tuple)
            if old_latest is None or (year, month) > old_latest[0]:
                meta = payloads[filename].get("_meta", {})
                latest_meta_by_filter[filter_key_tuple] = (
                    (year, month),
                    {
                        key: meta.get(key)
                        for key in ("last_indexed", "last_updated", "license")
                    },
                )

    verification_by_filter = {}
    for seed in SEEDS:
        filter_key, filter_value = filter_identity(seed["filter"])
        filter_id = (filter_key, filter_value)
        if filter_id in verification_by_filter:
            continue
        filtered_total = month_counts[(filter_key, filter_value, *VERIFY_MONTH)]
        bucket_total = find_bucket(verification_payload, filter_key, filter_value)
        if bucket_total is None:
            raise RuntimeError(
                f"CFPB {filter_key} aggregation is missing exact category {filter_value!r}"
            )
        if filtered_total != bucket_total:
            raise RuntimeError(
                f"CFPB filter verification mismatch for {filter_key}={filter_value!r} in {VERIFY_MONTH}: "
                f"filtered total {filtered_total}, unfiltered bucket total {bucket_total}"
            )
        verification_by_filter[filter_id] = {
            "aggregation": filter_key,
            "filter": {filter_key: filter_value},
            "filtered_total": filtered_total,
            "unfiltered_bucket_total": bucket_total,
        }

    seed_results = {}
    for seed in SEEDS:
        seed_id = seed["id"]
        filter_key, filter_value = filter_identity(seed["filter"])
        history = [
            {
                "period": f"{year:04d}-{month:02d}",
                "value": month_counts[(filter_key, filter_value, year, month)],
            }
            for year, month in history_months()
        ]
        detail = {
            "history_vintage": {
                "method": "lag_guard",
                "last_period": "2026-05",
                "reason": HISTORY_VINTAGE_REASON,
            },
            "date_received_max_inclusive": True,
            "single_day_min_equals_max_probe": {
                "date": PROBE_DATE,
                "filter": {"issue": "Fraud or scam"},
                "total": probe_total,
            },
            "filter_verification_month": f"{VERIFY_MONTH[0]:04d}-{VERIFY_MONTH[1]:02d}",
            "filter_verification": verification_by_filter[(filter_key, filter_value)],
        }
        latest_meta = latest_meta_by_filter.get((filter_key, filter_value), (None, {}))[
            1
        ]
        detail.update(latest_meta)

        window_breakdown = {}
        for year, month in window_months(seed["window"]):
            window_breakdown[f"{year:04d}-{month:02d}"] = month_counts[
                (filter_key, filter_value, year, month)
            ]
        value = sum(window_breakdown.values())
        detail["window_breakdown"] = window_breakdown

        notes = (
            "Counts use date_received with inclusive date_received_min/date_received_max; a single-day min=max "
            "query confirmed the upper bound. CFPB publishes complaints after the company responds or after "
            "15 days, so history is lag-guarded through 2026-05. Earlier counts can change as complaints are "
            "published or the API index is revised."
        )
        seed_results[seed_id] = {
            "role": seed["role"],
            "value": value,
            "window_start": seed["window"]["start"],
            "window_end": seed["window"]["end"],
            "history_granularity": "month",
            "history": history,
            "detail": detail,
            "notes": notes,
        }

    return {
        "source_id": SOURCE_ID,
        "script": SCRIPT_NAME,
        "retrieved_at": retrieved_at,
        "snapshots": snapshots,
        "commit": None,
        "snapshot_sha256": manifest_sha256(snapshots),
        "seeds": seed_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Resolve CFPB complaint-count seeds")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    root = args.data_dir / SOURCE_ID
    if args.offline:
        snapshots, payloads = load_offline_snapshots(root, args.out)
        status_events = []
    else:
        snapshots, payloads, status_events = download_online(root)

    result = build_result(snapshots, payloads)
    write_result(args.out, result)
    for item in sorted(snapshots, key=lambda entry: entry["file"]):
        print(f"{item['sha256']}  {item['file']}")
    print(f"snapshot_sha256  {result['snapshot_sha256']}")
    print(
        "seed_values  "
        + json.dumps(
            {seed_id: item["value"] for seed_id, item in result["seeds"].items()},
            sort_keys=True,
        )
    )
    alert_id = "cfpb-fraud-alert-complaints-2026q3"
    alert_verification = result["seeds"][alert_id]["detail"]["filter_verification"]
    print(
        "issue_aggregation_verified  "
        + json.dumps(alert_verification, ensure_ascii=False, sort_keys=True)
    )
    print(
        "http_403_429_responses  "
        + json.dumps(status_events, ensure_ascii=False, sort_keys=True)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"cfpb.py: {exc}", file=sys.stderr)
        raise
