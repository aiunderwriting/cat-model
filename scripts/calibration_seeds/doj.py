#!/usr/bin/env python3
"""Resolve DOJ press-release topic-count seeds from the DOJ News API.

Canonical source: https://www.justice.gov/api/v1/press_releases.json .
The primary seeds match an exact topic name in the release's topic list:
doj-cyber-press-releases-2026q3 uses "Cybercrime" and
doj-fraud-press-releases-2026q3 uses "Financial Fraud". The date field is the
Unix-seconds release date, converted to America/New_York before inclusion. Both
include dates 2026-07-01 through 2026-09-30 and have no other exclusions. The
history is quarterly from 2024-Q1 through 2026-Q2. The API's parameters[date]
filter is queried one local date at a time, with all pages saved byte-for-byte;
topic matching is exact and local because the API's parameters[topic] probe is
checked for ignored/unfiltered results. Page size is 50, sorted by date DESC,
with at least one second between request starts. DOJ may add or revise releases;
the snapshot records the API state encountered during retrieval. The observed
API may repeat identical UUID rows and can return a blank placeholder without
a UUID; identical repeats are deduplicated and the blank placeholder is excluded.
"""

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

SOURCE_ID = "doj"
SCRIPT_NAME = "doj.py"
API_URL = "https://www.justice.gov/api/v1/press_releases.json"
USER_AGENT = "AIUC calibration-seed resolver (research)"
RETRY_DELAYS = (2, 4, 8, 16, 32)
PAGE_SIZE = 50
MIN_REQUEST_INTERVAL = 1.05
EASTERN = ZoneInfo("America/New_York")
START_DATE = dt.date(2024, 1, 1)
END_DATE = dt.date(2026, 9, 30)
WINDOW_START = dt.date(2026, 7, 1)
WINDOW_END = dt.date(2026, 9, 30)
TOPIC_PROBE_DATE = dt.date(2026, 6, 30)

SEEDS: list[dict[str, Any]] = [
    {
        "id": "doj-cyber-press-releases-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"topic_name": "Cybercrime", "match": "exact"},
    },
    {
        "id": "doj-fraud-press-releases-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"topic_name": "Financial Fraud", "match": "exact"},
    },
]


class TopicHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.names = []
        self.in_anchor = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self.in_anchor = True
            self.parts = []

    def handle_data(self, data):
        if self.in_anchor:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.in_anchor:
            name = "".join(self.parts).strip()
            if name:
                self.names.append(name)
            self.in_anchor = False
            self.parts = []


def utc_now():
    return (
        dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def epoch_for_query_date(day):
    # Noon is safely inside the requested local calendar date across DST changes.
    return int(
        dt.datetime(day.year, day.month, day.day, 12, 0, tzinfo=EASTERN).timestamp()
    )


def request_body(session, params, request_state):
    for attempt in range(len(RETRY_DELAYS) + 1):
        elapsed = None
        if request_state["last_start"] is not None:
            elapsed = time.monotonic() - request_state["last_start"]
        if elapsed is not None and elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        request_state["last_start"] = time.monotonic()
        try:
            response = session.get(API_URL, params=params, timeout=(20, 120))
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt == len(RETRY_DELAYS):
                raise RuntimeError(f"DOJ request failed after retries: {exc}") from exc
            time.sleep(RETRY_DELAYS[attempt])
            continue

        if response.status_code == 403:
            raise RuntimeError(
                f"DOJ host refused the request with HTTP 403: {response.url}"
            )
        if response.status_code == 429 or 500 <= response.status_code <= 599:
            if attempt == len(RETRY_DELAYS):
                raise RuntimeError(
                    f"DOJ request failed after retries with HTTP {response.status_code}: {response.url}"
                )
            response.close()
            time.sleep(RETRY_DELAYS[attempt])
            continue
        if not 200 <= response.status_code <= 299:
            raise RuntimeError(
                f"DOJ request failed with HTTP {response.status_code}: {response.url}"
            )
        return response.content, response.url, utc_now()
    raise AssertionError("unreachable retry state")


def save_snapshot(root, filename, url, body, retrieved_at, snapshots):
    path = root / filename
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
    return json.loads(body.decode("utf-8-sig"))


def load_offline_snapshots(root, out_path):
    if not root.is_dir():
        raise RuntimeError(f"offline data directory does not exist: {root}")
    if out_path.is_file():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        if previous.get("source_id") != SOURCE_ID or not isinstance(
            previous.get("snapshots"), list
        ):
            raise RuntimeError(
                f"--out does not contain prior {SOURCE_ID} snapshot metadata"
            )
        snapshots = previous["snapshots"]
    else:
        # Recover complete downloaded data when processing failed before a result
        # was written. URLs are deterministic; file mtimes provide UTC seconds.
        snapshots = []
        pattern = re.compile(r"day_(\d{4}-\d{2}-\d{2})_page_(\d{3})\.json\Z")
        for path in sorted(root.iterdir()):
            if not path.is_file():
                raise RuntimeError(
                    f"unexpected non-file in DOJ offline directory: {path}"
                )
            name = path.name
            match = pattern.fullmatch(name)
            if match:
                day = dt.date.fromisoformat(match.group(1))
                page = int(match.group(2))
                params = {
                    "pagesize": PAGE_SIZE,
                    "fields": "date,topic,uuid",
                    "sort": "date",
                    "direction": "DESC",
                    "parameters[date]": str(epoch_for_query_date(day)),
                    "page": page,
                }
            elif (
                name
                == f"probe__topic-filter-Cybercrime-{TOPIC_PROBE_DATE.isoformat()}-page-000.json"
            ):
                params = {
                    "pagesize": PAGE_SIZE,
                    "fields": "date,topic,uuid",
                    "sort": "date",
                    "direction": "DESC",
                    "parameters[date]": str(epoch_for_query_date(TOPIC_PROBE_DATE)),
                    "parameters[topic]": "Cybercrime",
                    "page": 0,
                }
            else:
                raise RuntimeError(f"unexpected DOJ offline snapshot filename: {name}")
            body = path.read_bytes()
            retrieved_at = (
                dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            snapshots.append(
                {
                    "url": requests.Request("GET", API_URL, params=params)
                    .prepare()
                    .url,
                    "file": name,
                    "sha256": sha256_bytes(body),
                    "bytes": len(body),
                    "retrieved_at": retrieved_at,
                }
            )
    listed = set()
    payloads = {}
    for entry in snapshots:
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe snapshot path in prior result: {entry['file']}")
        name = relative.as_posix()
        if name in listed:
            raise RuntimeError(f"duplicate snapshot path in prior result: {name}")
        listed.add(name)
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"offline snapshot is missing: {path}")
        body = path.read_bytes()
        if len(body) != entry["bytes"] or sha256_bytes(body) != entry["sha256"]:
            raise RuntimeError(f"offline snapshot checksum/size mismatch: {path}")
        payloads[name] = json.loads(body.decode("utf-8-sig"))
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != listed:
        raise RuntimeError(
            f"offline directory contents differ from the saved snapshot list (extra={sorted(actual - listed)}, missing={sorted(listed - actual)})"
        )
    return snapshots, payloads


def expected_days():
    day = START_DATE
    while day <= END_DATE:
        yield day
        day += dt.timedelta(days=1)


def resultset_info(payload, requested_page):
    try:
        resultset = payload["metadata"]["resultset"]
        total = int(resultset["count"])
        page_size = int(resultset["pagesize"])
        page = int(resultset["page"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"DOJ response lacks usable metadata.resultset: {payload.get('metadata')!r}"
        ) from exc
    if total < 0 or page_size < 1 or page != requested_page:
        raise RuntimeError(
            f"DOJ returned inconsistent page metadata: requested={requested_page}, resultset={resultset}"
        )
    return total, page_size, page


def parse_local_date(release):
    try:
        timestamp = int(release["date"])
        return dt.datetime.fromtimestamp(timestamp, EASTERN).date()
    except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
        raise RuntimeError(f"DOJ release has invalid Unix date: {release!r}") from exc


def topic_names(topic_field):
    if isinstance(topic_field, list):
        result = []
        for item in topic_field:
            if (
                isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and item["name"].strip()
            ):
                result.append(item["name"].strip())
        return result, "list"
    if isinstance(topic_field, str):
        if not topic_field.strip():
            return [], "empty_string"
        parser = TopicHTMLParser()
        parser.feed(topic_field)
        if parser.names:
            return parser.names, "html_string"
        value = html.unescape(topic_field).strip()
        return ([value] if value else []), "string"
    if topic_field is None:
        return [], "null"
    raise RuntimeError(
        f"unsupported DOJ topic representation: {type(topic_field).__name__}"
    )


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


def parse_snapshot_files(root, snapshots, payloads):
    expected = {day.isoformat(): [] for day in expected_days()}
    probe_name = (
        f"probe__topic-filter-Cybercrime-{TOPIC_PROBE_DATE.isoformat()}-page-000.json"
    )
    probe_payload = payloads.get(probe_name)
    if probe_payload is None:
        raise RuntimeError(f"missing DOJ topic-filter probe snapshot: {probe_name}")

    pattern = re.compile(r"day_(\d{4}-\d{2}-\d{2})_page_(\d{3})\.json\Z")
    for item in snapshots:
        filename = item["file"]
        match = pattern.fullmatch(filename)
        if not match:
            if filename == probe_name:
                continue
            raise RuntimeError(f"unexpected DOJ snapshot filename: {filename}")
        day_text, page_text = match.groups()
        if day_text not in expected:
            raise RuntimeError(
                f"DOJ snapshot date is outside the requested range: {filename}"
            )
        expected[day_text].append((int(page_text), filename))

    all_records = []
    api_daily_totals = {}
    local_daily_uuid_sets = {}
    total_api_rows = 0
    total_retrieved_rows = 0
    global_ids = set()
    global_records_by_uuid = {}
    global_duplicate_count = 0
    duplicate_uuid_rows = 0
    duplicate_uuid_dates = {}
    blank_uuid_rows = 0
    representation_counts = Counter()
    topic_uuids = defaultdict(set)

    for day_text, files in expected.items():
        day = dt.date.fromisoformat(day_text)
        files.sort()
        if not files or files[0][0] != 0:
            raise RuntimeError(f"DOJ data is missing first page for {day_text}")
        page_zero = payloads[files[0][1]]
        reported_total, page_size, _ = resultset_info(page_zero, 0)
        if page_size != PAGE_SIZE:
            raise RuntimeError(
                f"DOJ page size changed for {day_text}: expected {PAGE_SIZE}, got {page_size}"
            )
        expected_page_count = max(1, math.ceil(reported_total / PAGE_SIZE))
        expected_pages = list(range(expected_page_count))
        actual_pages = [page for page, _ in files]
        if actual_pages != expected_pages:
            raise RuntimeError(
                f"DOJ pagination is incomplete for {day_text}: expected pages {expected_pages}, got {actual_pages}"
            )

        day_records = []
        day_raw_row_count = 0
        for page, filename in files:
            payload = payloads[filename]
            page_total, page_size_check, actual_page = resultset_info(payload, page)
            if (
                page_total != reported_total
                or page_size_check != PAGE_SIZE
                or actual_page != page
            ):
                raise RuntimeError(
                    f"DOJ pagination metadata changed on {filename}: {payload.get('metadata', {}).get('resultset')}"
                )
            rows = payload.get("results")
            if not isinstance(rows, list):
                raise RuntimeError(f"DOJ page has no results list: {filename}")
            expected_rows = min(PAGE_SIZE, max(0, reported_total - page * PAGE_SIZE))
            if len(rows) != expected_rows:
                raise RuntimeError(
                    f"DOJ page row count mismatch for {filename}: expected {expected_rows}, got {len(rows)}"
                )
            for release in rows:
                day_raw_row_count += 1
                if (
                    isinstance(release, dict)
                    and release.get("uuid") == ""
                    and release.get("date") == ""
                    and release.get("topic") == ""
                ):
                    blank_uuid_rows += 1
                    continue
                if (
                    not isinstance(release, dict)
                    or not isinstance(release.get("uuid"), str)
                    or not release["uuid"]
                ):
                    raise RuntimeError(
                        f"DOJ release lacks uuid in {filename}: {release!r}"
                    )
                local_day = parse_local_date(release)
                if local_day != day:
                    raise RuntimeError(
                        f"DOJ date filter returned {release['uuid']} for {day_text}, but its America/New_York date is {local_day}"
                    )
                day_records.append(release)

        if day_raw_row_count != reported_total:
            raise RuntimeError(
                f"DOJ total cross-check failed for {day_text}: API count {reported_total}, "
                f"rows retrieved {day_raw_row_count}"
            )
        unique_by_uuid = {}
        duplicates_for_day = 0
        for release in day_records:
            release_id = release["uuid"]
            previous = unique_by_uuid.get(release_id)
            if previous is not None:
                if any(
                    previous.get(field) != release.get(field)
                    for field in ("date", "topic", "uuid")
                ):
                    raise RuntimeError(
                        f"DOJ returned conflicting records for duplicate UUID {release_id} on {day_text}"
                    )
                duplicates_for_day += 1
                continue
            unique_by_uuid[release_id] = release
        if duplicates_for_day:
            duplicate_uuid_rows += duplicates_for_day
            duplicate_uuid_dates[day_text] = duplicates_for_day

        unique_day_records = list(unique_by_uuid.values())
        for release in unique_day_records:
            topic_field = release.get("topic", "")
            names, representation = topic_names(topic_field)
            representation_counts[representation] += 1
            release["_resolver_topic_names"] = sorted(set(names))
            release["_resolver_local_date"] = day.isoformat()
            for topic in set(names):
                if isinstance(topic_field, list):
                    for entry in topic_field:
                        if (
                            isinstance(entry, dict)
                            and entry.get("name") == topic
                            and isinstance(entry.get("uuid"), str)
                        ):
                            topic_uuids[topic].add(entry["uuid"])

        api_daily_totals[day_text] = reported_total
        local_daily_uuid_sets[day_text] = set(unique_by_uuid)
        total_api_rows += reported_total
        total_retrieved_rows += day_raw_row_count
        for release in unique_day_records:
            release_id = release["uuid"]
            if release_id in global_ids:
                global_duplicate_count += 1
                previous = global_records_by_uuid.get(release_id)
                if previous is not None and any(
                    previous.get(field) != release.get(field)
                    for field in ("date", "topic", "uuid")
                ):
                    raise RuntimeError(
                        f"DOJ returned conflicting cross-date records for duplicate UUID {release_id}"
                    )
            else:
                global_ids.add(release_id)
                global_records_by_uuid[release_id] = release
                all_records.append(release)

    if total_api_rows != total_retrieved_rows:
        raise RuntimeError(
            f"DOJ total row cross-check failed: API page totals {total_api_rows}, rows retrieved {total_retrieved_rows}"
        )

    probe_total, probe_page_size, probe_page = resultset_info(probe_payload, 0)
    if probe_page_size != PAGE_SIZE or probe_page != 0:
        raise RuntimeError(
            f"DOJ topic-filter probe has inconsistent pagination metadata: {probe_payload.get('metadata', {}).get('resultset')}"
        )
    probe_rows = probe_payload.get("results")
    if not isinstance(probe_rows, list) or len(probe_rows) != min(
        PAGE_SIZE, probe_total
    ):
        raise RuntimeError("DOJ topic-filter probe response has incomplete page rows")
    probe_unmatched = []
    for release in probe_rows:
        names, _ = topic_names(release.get("topic", ""))
        if "Cybercrime" not in names:
            probe_unmatched.append(release.get("uuid"))
    june_30_count = api_daily_totals[TOPIC_PROBE_DATE.isoformat()]
    topic_filter_ignored = probe_total == june_30_count and bool(probe_unmatched)

    june = dt.date(2026, 6, 1)
    june_end = dt.date(2026, 6, 30)
    june_api_ids = set().union(
        *(
            local_daily_uuid_sets[day.isoformat()]
            for day in (
                june + dt.timedelta(days=offset)
                for offset in range((june_end - june).days + 1)
            )
        )
    )
    june_local_ids = {
        release["uuid"]
        for release in all_records
        if dt.date.fromisoformat(release["_resolver_local_date"]).year == 2026
        and dt.date.fromisoformat(release["_resolver_local_date"]).month == 6
    }
    if june_api_ids != june_local_ids:
        raise RuntimeError(
            "DOJ June 2026 date-filter results do not match local America/New_York date filtering"
        )
    date_filter_validation = {
        "month": "2026-06",
        "server_filtered_api_total": sum(
            api_daily_totals[day.isoformat()]
            for day in (
                june + dt.timedelta(days=offset)
                for offset in range((june_end - june).days + 1)
            )
        ),
        "server_filtered_distinct_uuids": len(june_api_ids),
        "locally_filtered_distinct_uuids": len(june_local_ids),
        "matches": True,
    }

    topic_counts_q2 = Counter()
    q2_release_count = 0
    for release in all_records:
        local_day = dt.date.fromisoformat(release["_resolver_local_date"])
        if dt.date(2026, 4, 1) <= local_day <= dt.date(2026, 6, 30):
            q2_release_count += 1
            topic_counts_q2.update(release["_resolver_topic_names"])
    top_topics = [
        {"name": name, "count": count}
        for name, count in sorted(
            topic_counts_q2.items(), key=lambda item: (-item[1], item[0])
        )[:15]
    ]

    monthly_counts = {seed["id"]: Counter() for seed in SEEDS}
    quarterly_counts = {seed["id"]: Counter() for seed in SEEDS}
    selected_topic_uuids = {seed["id"]: set() for seed in SEEDS}
    for release in all_records:
        release_id = release["uuid"]
        local_day = dt.date.fromisoformat(release["_resolver_local_date"])
        names = set(release["_resolver_topic_names"])
        for seed in SEEDS:
            topic = seed["filter"]["topic_name"]
            if topic in names:
                month_label = f"{local_day.year:04d}-{local_day.month:02d}"
                quarter_label = (
                    f"{local_day.year:04d}-Q{(local_day.month - 1) // 3 + 1}"
                )
                monthly_counts[seed["id"]][month_label] += 1
                quarterly_counts[seed["id"]][quarter_label] += 1
                selected_topic_uuids[seed["id"]].update(topic_uuids.get(topic, set()))

    range_daily = len(api_daily_totals)
    if range_daily != (END_DATE - START_DATE).days + 1:
        raise RuntimeError(
            f"DOJ daily coverage incomplete: expected {(END_DATE - START_DATE).days + 1}, got {range_daily}"
        )

    return {
        "all_records": all_records,
        "api_daily_totals": api_daily_totals,
        "total_api_rows": total_api_rows,
        "total_retrieved_rows": total_retrieved_rows,
        "distinct_uuid_count": len(global_ids),
        "global_duplicate_count": global_duplicate_count,
        "duplicate_uuid_rows_deduplicated": duplicate_uuid_rows,
        "duplicate_uuid_dates": duplicate_uuid_dates,
        "blank_uuid_placeholder_rows_excluded": blank_uuid_rows,
        "topic_representation_counts": dict(sorted(representation_counts.items())),
        "topic_filter_probe": {
            "date": TOPIC_PROBE_DATE.isoformat(),
            "requested_topic_name": "Cybercrime",
            "resultset_total": probe_total,
            "unfiltered_day_total": june_30_count,
            "page_rows": len(probe_rows),
            "unmatched_topic_rows_on_probe_page": len(probe_unmatched),
            "server_side_topic_filter_ignored": topic_filter_ignored,
        },
        "date_filter_validation": date_filter_validation,
        "q2_release_count": q2_release_count,
        "top_topics_q2": top_topics,
        "monthly_counts": monthly_counts,
        "quarterly_counts": quarterly_counts,
        "selected_topic_uuids": selected_topic_uuids,
    }


def run_online(root):
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to download into non-empty directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    request_state = {"last_start": None}
    snapshots = []

    for day_number, day in enumerate(expected_days(), start=1):
        day_text = day.isoformat()
        date_timestamp = str(epoch_for_query_date(day))
        page = 0
        expected_total = None
        while True:
            params = {
                "pagesize": PAGE_SIZE,
                "fields": "date,topic,uuid",
                "sort": "date",
                "direction": "DESC",
                "parameters[date]": date_timestamp,
                "page": page,
            }
            body, url, retrieved_at = request_body(session, params, request_state)
            filename = f"day_{day_text}_page_{page:03d}.json"
            payload = save_snapshot(root, filename, url, body, retrieved_at, snapshots)
            total, returned_page_size, _returned_page = resultset_info(payload, page)
            if returned_page_size != PAGE_SIZE:
                raise RuntimeError(
                    f"DOJ API did not honor pagesize=50 for {day_text}: {returned_page_size}"
                )
            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise RuntimeError(
                    f"DOJ API total changed while paging {day_text}: {expected_total} -> {total}"
                )
            results = payload.get("results")
            if not isinstance(results, list):
                raise RuntimeError(
                    f"DOJ API has no results list for {day_text} page {page}"
                )
            expected_rows = min(PAGE_SIZE, max(0, total - page * PAGE_SIZE))
            if len(results) != expected_rows:
                raise RuntimeError(
                    f"DOJ API page size mismatch for {day_text} page {page}: expected {expected_rows}, got {len(results)}"
                )
            if page + 1 >= max(1, math.ceil(total / PAGE_SIZE)):
                break
            page += 1

        if day_number % 10 == 0:
            print(
                f"DOJ download progress: {day_text}; {day_number}/{(END_DATE - START_DATE).days + 1} dates, {len(snapshots)} files",
                file=sys.stderr,
                flush=True,
            )

    probe_params = {
        "pagesize": PAGE_SIZE,
        "fields": "date,topic,uuid",
        "sort": "date",
        "direction": "DESC",
        "parameters[date]": str(epoch_for_query_date(TOPIC_PROBE_DATE)),
        "parameters[topic]": "Cybercrime",
        "page": 0,
    }
    body, url, retrieved_at = request_body(session, probe_params, request_state)
    probe_filename = (
        f"probe__topic-filter-Cybercrime-{TOPIC_PROBE_DATE.isoformat()}-page-000.json"
    )
    save_snapshot(root, probe_filename, url, body, retrieved_at, snapshots)
    return snapshots


def main():
    parser = argparse.ArgumentParser(
        description="Resolve DOJ press-release topic-count seeds"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    root = args.data_dir / SOURCE_ID
    if args.offline:
        snapshots, payloads = load_offline_snapshots(root, args.out)
    else:
        snapshots = run_online(root)
        payloads = {}
        for item in snapshots:
            payloads[item["file"]] = json.loads(
                (root / item["file"]).read_bytes().decode("utf-8-sig")
            )

    computed = parse_snapshot_files(root, snapshots, payloads)
    seeds_out = {}
    for seed in SEEDS:
        seed_id = seed["id"]
        topic = seed["filter"]["topic_name"]
        history = []
        for year in range(2024, 2027):
            last_quarter = 2 if year == 2026 else 4
            for quarter in range(1, last_quarter + 1):
                period = f"{year:04d}-Q{quarter}"
                history.append(
                    {
                        "period": period,
                        "value": computed["quarterly_counts"][seed_id][period],
                    }
                )
        window_breakdown = {
            f"2026-{month:02d}": computed["monthly_counts"][seed_id][
                f"2026-{month:02d}"
            ]
            for month in (7, 8, 9)
        }
        detail = {
            "topic_name_exact_match": topic,
            "topic_uuids_observed_for_exact_name": sorted(
                computed["selected_topic_uuids"][seed_id]
            ),
            "date_field": "date (Unix seconds converted to America/New_York)",
            "date_filter_validation": computed["date_filter_validation"],
            "topic_filter_probe": computed["topic_filter_probe"],
            "topic_field_representation_counts": computed[
                "topic_representation_counts"
            ],
            "q2_2026_release_count_used_for_topic_review": computed["q2_release_count"],
            "most_frequent_topics_2026_q2": computed["top_topics_q2"],
            "window_breakdown": window_breakdown,
            "pagination": {
                "page_size": PAGE_SIZE,
                "reported_rows_across_daily_queries": computed["total_api_rows"],
                "retrieved_rows_across_daily_queries": computed["total_retrieved_rows"],
                "distinct_uuids_retrieved": computed["distinct_uuid_count"],
                "cross_date_duplicate_uuids": computed["global_duplicate_count"],
                "duplicate_uuid_rows_deduplicated": computed[
                    "duplicate_uuid_rows_deduplicated"
                ],
                "duplicate_uuid_dates": computed["duplicate_uuid_dates"],
                "blank_uuid_placeholder_rows_excluded": computed[
                    "blank_uuid_placeholder_rows_excluded"
                ],
                "date_count": len(computed["api_daily_totals"]),
            },
        }
        notes = (
            f"Count distinct press-release UUIDs with an exact topic-name match for {topic!r}. "
            "Release date Unix seconds are converted to America/New_York before quarter/window assignment. "
            "The API date filter was applied one day at a time; all returned rows matched the requested local date and "
            "daily metadata totals were cross-checked against retrieved rows. Duplicate UUID rows were deduplicated as "
            "required by the seed's distinct-UUID definition and recorded in pagination detail; a blank row with no UUID "
            "was excluded. Topic filtering was "
            "performed locally after inspecting the API's topic list representation and testing parameters[topic]."
        )
        seeds_out[seed_id] = {
            "role": seed["role"],
            "value": sum(window_breakdown.values()),
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": "quarter",
            "history": history,
            "detail": detail,
            "notes": notes,
        }

    result = {
        "source_id": SOURCE_ID,
        "script": SCRIPT_NAME,
        "retrieved_at": snapshots[0]["retrieved_at"] if snapshots else None,
        "snapshots": snapshots,
        "commit": None,
        "snapshot_sha256": manifest_sha256(snapshots),
        "seeds": seeds_out,
    }
    write_result(args.out, result)
    for item in sorted(snapshots, key=lambda entry: entry["file"]):
        print(f"{item['sha256']}  {item['file']}")
    print(f"snapshot_sha256  {result['snapshot_sha256']}")
    print(
        "seed_values  "
        + json.dumps(
            {seed_id: item["value"] for seed_id, item in seeds_out.items()},
            sort_keys=True,
        )
    )
    print(
        "doj_topic_names_2026_q2_top_15  "
        + json.dumps(computed["top_topics_q2"], ensure_ascii=False)
    )
    print("doj_api_rows  " + str(computed["total_api_rows"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"doj.py: {exc}", file=sys.stderr)
        raise
