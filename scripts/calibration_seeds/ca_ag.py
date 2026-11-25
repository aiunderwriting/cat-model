#!/usr/bin/env python3
"""Resolve California AG breach-notice counts from its canonical CSV export.

Source: California Attorney General data security breach list.
Canonical URL: https://oag.ca.gov/privacy/databreach/list-export
Filter: ca-ag-breach-notices-2026q3 includes each CSV row with Reported Date
from 2026-07-01 through 2026-09-30. No organization, breach-type, or other
row filters are applied. Date field: Reported Date
(MM/DD/YYYY), interpreted as a UTC calendar date. Inclusion/exclusion: count
one per posted notice; discard rows whose Reported Date cannot be parsed from
window/history counts and report those rows. Snapshot rule: save every HTTP
response body, including redirects and transient HTTP errors, as exact bytes.
Known lags/revisions: notices can be posted after their reported dates, so the
history uses the specified lag guard; the CSV is a current snapshot and may be
revised.
"""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "ca_ag"
SCRIPT_NAME = "ca_ag.py"
EXPORT_URL = "https://oag.ca.gov/privacy/databreach/list-export"
USER_AGENT = "AIUC calibration-seed resolver (research)"
RETRY_DELAYS = (2, 4, 8, 16, 32)
SEEDS: list[dict[str, Any]] = [
    {
        "id": "ca-ag-breach-notices-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"reported_date": "2026-Q3"},
        "history_granularity": "month",
        "history_end": "2026-05",
    },
]


def utc_now():
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare_directory(path, offline):
    if offline:
        if not path.is_dir() or not any(path.iterdir()):
            raise SystemExit(f"offline data directory is missing or empty: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise SystemExit(f"refusing online download into non-empty directory: {path}")


def load_offline_manifest(out_path, data_dir):
    if not out_path.is_file():
        raise SystemExit(
            f"offline mode needs the online results file for snapshot metadata: {out_path}"
        )
    try:
        previous = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"cannot read prior online results for offline metadata: {exc}"
        ) from exc
    snapshots = previous.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise SystemExit("prior results do not contain a usable snapshots list")
    listed = set()
    for item in snapshots:
        rel = Path(item.get("file", ""))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise SystemExit(
                f"unsafe snapshot path in prior results: {item.get('file')!r}"
            )
        path = data_dir / rel
        if path.is_symlink() or not path.is_file():
            raise SystemExit(
                f"offline snapshot file is missing or not a regular file: {path}"
            )
        raw = path.read_bytes()
        if sha256_bytes(raw) != item.get("sha256") or len(raw) != item.get("bytes"):
            raise SystemExit(f"offline snapshot integrity check failed: {path}")
        listed.add(rel.as_posix())
    actual = {
        p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*") if p.is_file()
    }
    if actual != listed:
        raise SystemExit(
            f"offline data files do not match prior snapshot manifest (extra={sorted(actual - listed)}, missing={sorted(listed - actual)})"
        )
    if not previous.get("retrieved_at"):
        raise SystemExit("prior results do not contain retrieved_at")
    return previous


def record_response(data_dir, snapshots, filename, url, content):
    path = data_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    snapshots.append(
        {
            "url": url,
            "file": Path(filename).as_posix(),
            "sha256": sha256_bytes(content),
            "bytes": len(content),
            "retrieved_at": utc_now(),
        }
    )


def is_connect_403(exc):
    message = str(exc).lower()
    return "403" in message and ("tunnel" in message or "connect" in message)


def fetch_csv(session, data_dir, snapshots, last_request):
    """GET with a 10-second crawl interval, retry policy, and byte snapshots."""
    for attempt in range(len(RETRY_DELAYS) + 1):
        if last_request[0] is not None:
            remaining = 10.0 - (time.monotonic() - last_request[0])
            if remaining > 0:
                time.sleep(remaining)
        last_request[0] = time.monotonic()
        try:
            response = session.get(EXPORT_URL, timeout=120, allow_redirects=True)
        except requests.exceptions.ProxyError as exc:
            if is_connect_403(exc):
                raise SystemExit(
                    f"proxy refused HTTPS CONNECT to oag.ca.gov (HTTP 403): {exc}"
                ) from exc
            if attempt >= len(RETRY_DELAYS):
                raise SystemExit(
                    f"network request failed after retries: {exc}"
                ) from exc
            time.sleep(RETRY_DELAYS[attempt])
            continue
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt >= len(RETRY_DELAYS):
                raise SystemExit(
                    f"network request failed after retries: {exc}"
                ) from exc
            time.sleep(RETRY_DELAYS[attempt])
            continue

        chain = [*list(response.history), response]
        for i, item in enumerate(chain):
            if item.status_code in (301, 302, 303, 307, 308):
                name = f"ca_ag_attempt{attempt}_redirect{i}_http{item.status_code}.bin"
            elif item.status_code >= 400:
                name = f"ca_ag_attempt{attempt}_http{item.status_code}.bin"
            else:
                name = (
                    "ca_ag_export.csv"
                    if i == len(chain) - 1
                    else f"ca_ag_attempt{attempt}_redirect{i}_http{item.status_code}.bin"
                )
            record_response(data_dir, snapshots, name, item.request.url, item.content)

        if response.status_code == 403:
            raise SystemExit(
                "oag.ca.gov returned HTTP 403; stopping without further requests"
            )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt >= len(RETRY_DELAYS):
                raise SystemExit(
                    f"oag.ca.gov returned HTTP {response.status_code} after retries"
                )
            time.sleep(RETRY_DELAYS[attempt])
            continue
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise SystemExit(f"oag.ca.gov request failed: {exc}") from exc
        return response
    raise SystemExit("request retry loop ended unexpectedly")


def parse_reported_date(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%m/%d/%Y").date()
    except ValueError:
        return None


def period_label(day, granularity):
    if granularity == "month":
        return day.strftime("%Y-%m")
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


def history_periods(granularity, start="2024-01", end="2026-05"):
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    current_y, current_m = sy, sm
    labels = []
    while (current_y, current_m) <= (ey, em):
        if granularity == "month":
            labels.append(f"{current_y:04d}-{current_m:02d}")
        elif current_m in (1, 4, 7, 10):
            labels.append(f"{current_y:04d}-Q{(current_m - 1) // 3 + 1}")
        current_m += 1
        if current_m == 13:
            current_y += 1
            current_m = 1
    return labels


def build_seeds(rows, unparseable):
    # Each parsed row has its original row number, organization, date, and date text.
    dates = [row["reported_date"] for row in rows]
    key_counts = Counter((row["organization"], row["reported_date"]) for row in rows)
    output = {}
    for seed in SEEDS:
        start = date.fromisoformat(seed["window_start"])
        end = date.fromisoformat(seed["window_end"])
        window_rows = [row for row in rows if start <= row["reported_date"] <= end]
        duplicate_rows = [
            row
            for row in window_rows
            if key_counts[(row["organization"], row["reported_date"])] > 1
        ]
        granularity = seed["history_granularity"]
        labels = history_periods(granularity, end=seed["history_end"])
        history = []
        for label in labels:
            count = sum(
                1
                for day in dates
                if period_label(day, granularity) == label and day < start
            )
            history.append({"period": label, "value": count})
        if history and any(
            item["period"] >= period_label(start, granularity) for item in history
        ):
            raise SystemExit(f"history included the seed window for {seed['id']}")
        breakdown = Counter(
            row["reported_date"].strftime("%Y-%m") for row in window_rows
        )
        month_labels = []
        cursor = start
        while cursor <= end:
            label = cursor.strftime("%Y-%m")
            if label not in month_labels:
                month_labels.append(label)
            if cursor.month == 12:
                cursor = date(cursor.year + 1, 1, 1)
            else:
                cursor = date(cursor.year, cursor.month + 1, 1)
        detail = {
            "window_breakdown": {
                month: breakdown.get(month, 0) for month in month_labels
            },
            "history_vintage": {
                "method": "lag_guard",
                "last_period": seed["history_end"],
                "reason": "California AG notices may be posted after their Reported Date, so history ends at the specified lag-guard month.",
            },
            "rows_sharing_organization_reported_date": len(duplicate_rows),
            "duplicate_organization_reported_date_keys": len(
                {(row["organization"], row["reported_date"]) for row in duplicate_rows}
            ),
            "unparseable_reported_date_rows": len(unparseable),
        }
        if unparseable:
            detail["unparseable_dates"] = unparseable
        output[seed["id"]] = {
            "role": seed["role"],
            "value": len(window_rows),
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": granularity,
            "history": history,
            "detail": detail,
            "notes": "Each CSV row is one posted notice; unparseable Reported Date rows are excluded from counts.",
        }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Resolve California AG breach-notice counts"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir / SOURCE_ID
    prepare_directory(data_dir, args.offline)
    previous = load_offline_manifest(args.out, data_dir) if args.offline else None
    if args.offline:
        assert previous is not None
        snapshots = previous["snapshots"]
        retrieved_at = previous["retrieved_at"]
        export_path = data_dir / "ca_ag_export.csv"
        if not export_path.is_file():
            raise SystemExit(f"offline CSV snapshot is missing: {export_path}")
        raw = export_path.read_bytes()
    else:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        snapshots = []
        response = fetch_csv(session, data_dir, snapshots, [None])
        retrieved_at = snapshots[0]["retrieved_at"] if snapshots else utc_now()
        raw = response.content

    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        required = [
            "Organization Name",
            "Date(s) of Breach  (if known)",
            "Reported Date",
        ]
        if reader.fieldnames != required:
            raise ValueError(f"unexpected CSV columns: {reader.fieldnames!r}")
        rows = []
        unparseable = []
        for row_number, record in enumerate(reader, start=2):
            organization = (record.get("Organization Name") or "").strip()
            reported_raw = record.get("Reported Date") or ""
            reported_date = parse_reported_date(reported_raw)
            if reported_date is None:
                unparseable.append(
                    {
                        "row": row_number,
                        "organization": organization,
                        "reported_date": reported_raw.strip(),
                    }
                )
                continue
            rows.append(
                {
                    "organization": organization,
                    "reported_date": reported_date,
                    "reported_date_raw": reported_raw.strip(),
                }
            )
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise SystemExit(f"could not parse California AG CSV: {exc}") from exc

    seeds = build_seeds(rows, unparseable)
    snapshot_text = "".join(
        f"{item['sha256']}  {item['file']}\n"
        for item in sorted(snapshots, key=lambda item: item["file"])
    )
    snapshot_sha = sha256_bytes(snapshot_text.encode("utf-8"))
    result = {
        "source_id": SOURCE_ID,
        "script": SCRIPT_NAME,
        "retrieved_at": retrieved_at,
        "snapshots": snapshots,
        "commit": None,
        "snapshot_sha256": snapshot_sha,
        "seeds": seeds,
    }
    if args.offline and previous is not None and previous.get("seeds") != seeds:
        raise SystemExit("offline recomputation differs from online seed results")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for item in sorted(snapshots, key=lambda item: item["file"]):
        print(f"{item['sha256']}  {item['file']}")
    print(f"snapshot_sha256  {snapshot_sha}")
    for seed_id, value in seeds.items():
        print(f"{seed_id}: {value['value']}")
    print(f"rows={len(rows)} unparseable_reported_dates={len(unparseable)}")
    for seed_id, value in seeds.items():
        print(
            f"{seed_id} duplicate_key_rows={value['detail']['rows_sharing_organization_reported_date']}"
        )


if __name__ == "__main__":
    main()
