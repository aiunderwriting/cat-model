#!/usr/bin/env python3
"""Resolve Washington AG breach-notice counts from the Socrata dataset.

Source: Washington State Attorney General, Data Breach Notifications Affecting
Washington Residents, Socrata dataset sb4j-ca4h.
Canonical URLs: https://data.wa.gov/api/views/sb4j-ca4h/rows.csv?accessType=DOWNLOAD
and https://data.wa.gov/api/views/sb4j-ca4h.json; row-count check uses
https://data.wa.gov/resource/sb4j-ca4h.json?$select=count(*).
Filters: wa-ag-ransomware-notices-2026q3 counts rows dated 2026-07-01 through
2026-09-30 whose cyberattacktype equals "Ransomware" case-insensitively;
wa-ag-largest-notice-2026q3 takes the maximum washingtoniansaffected among all
rows dated 2026-07-01 through 2026-09-30. Date field: datesubmitted, interpreted as a UTC date. Inclusion/
exclusion: each dataset row is one notification; invalid dates are excluded
from windows and histories and reported. Snapshot rule: save the complete CSV,
metadata JSON, and SODA count response as exact response bytes. Known
lags/revisions: Socrata rows may be posted after datesubmitted or updated;
metadata's rowsUpdatedAt is recorded, and histories use the specified lag
guards.
"""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "wa_ag"
SCRIPT_NAME = "wa_ag.py"
DATASET_ID = "sb4j-ca4h"
CSV_URL = "https://data.wa.gov/api/views/sb4j-ca4h/rows.csv?accessType=DOWNLOAD"
METADATA_URL = "https://data.wa.gov/api/views/sb4j-ca4h.json"
COUNT_URL = "https://data.wa.gov/resource/sb4j-ca4h.json"
USER_AGENT = "AIUC calibration-seed resolver (research)"
RETRY_DELAYS = (2, 4, 8, 16, 32)
SEEDS: list[dict[str, Any]] = [
    {
        "id": "wa-ag-ransomware-notices-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {
            "date_field": "datesubmitted",
            "cyberattacktype": "Ransomware (case-insensitive)",
        },
        "history_granularity": "month",
        "history_end": "2026-05",
    },
    {
        "id": "wa-ag-largest-notice-2026q3",
        "role": "secondary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {
            "date_field": "datesubmitted",
            "aggregate": "maximum washingtoniansaffected",
        },
        "history_granularity": "quarter",
        "history_end": "2026-Q1",
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


def fetch(session, url, filename, params, data_dir, snapshots, request_clock):
    for attempt in range(len(RETRY_DELAYS) + 1):
        if request_clock[0] is not None:
            wait = 0.3 - (time.monotonic() - request_clock[0])
            if wait > 0:
                time.sleep(wait)
        request_clock[0] = time.monotonic()
        try:
            response = session.get(
                url, params=params, timeout=120, allow_redirects=True
            )
        except requests.exceptions.ProxyError as exc:
            if is_connect_403(exc):
                raise SystemExit(
                    f"proxy refused HTTPS CONNECT to {url}: HTTP 403 ({exc})"
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
                name = f"{filename}_attempt{attempt}_redirect{i}_http{item.status_code}.bin"
            elif item.status_code >= 400:
                name = f"{filename}_attempt{attempt}_http{item.status_code}.bin"
            else:
                name = (
                    filename
                    if i == len(chain) - 1
                    else f"{filename}_attempt{attempt}_redirect{i}_http{item.status_code}.bin"
                )
            record_response(data_dir, snapshots, name, item.request.url, item.content)

        if response.status_code == 403:
            raise SystemExit(
                f"{url} returned HTTP 403; stopping without further requests"
            )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt >= len(RETRY_DELAYS):
                raise SystemExit(
                    f"{url} returned HTTP {response.status_code} after retries"
                )
            time.sleep(RETRY_DELAYS[attempt])
            continue
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise SystemExit(f"request failed: {exc}") from exc
        return response
    raise SystemExit("request retry loop ended unexpectedly")


def parse_timestamp(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 100_000_000_000:
            timestamp /= 1000.0
        try:
            return (
                datetime.fromtimestamp(timestamp, tz=UTC)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if text.isdigit():
        return parse_timestamp(int(text))
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (
        parsed.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def extract_metadata_summary(metadata):
    license_fields = {}
    for key, value in metadata.items():
        if "license" in key.casefold():
            license_fields[key] = value
    return {
        "rowsUpdatedAt_utc": parse_timestamp(metadata.get("rowsUpdatedAt")),
        "license_fields": license_fields,
        "license_fields_present": bool(license_fields),
        "rights": metadata.get("rights"),
        "attribution": metadata.get("attribution"),
        "attributionLink": metadata.get("attributionLink"),
    }


def parse_day(raw):
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                pass
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC)
    return parsed.date()


def parse_affected(raw):
    text = (raw or "").strip().replace(",", "")
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if number != number.to_integral_value():
        return None
    return int(number)


def history_periods(granularity, end_label):
    if granularity == "month":
        labels = []
        y, m = 2024, 1
        end_y, end_m = map(int, end_label.split("-"))
        while (y, m) <= (end_y, end_m):
            labels.append(f"{y:04d}-{m:02d}")
            m += 1
            if m == 13:
                y, m = y + 1, 1
        return labels
    labels = []
    end_y, end_q = int(end_label[:4]), int(end_label[-1])
    for y in range(2024, end_y + 1):
        for q in range(1, 5):
            label = f"{y}-Q{q}"
            if (y, q) <= (end_y, end_q):
                labels.append(label)
    return labels


def period_label(day, granularity):
    if granularity == "month":
        return day.strftime("%Y-%m")
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


def parse_count_response(raw):
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot parse SODA count response: {exc}") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
        raise SystemExit(f"unexpected SODA count response shape: {payload!r}")
    for key, value in payload[0].items():
        if key.casefold().startswith("count"):
            try:
                return int(value)
            except (TypeError, ValueError):
                break
    raise SystemExit(f"SODA count response has no integer count field: {payload!r}")


def parse_csv(raw):
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        fieldnames = reader.fieldnames or []
        fields = {name.strip().casefold() for name in fieldnames}
        required = {
            "datesubmitted",
            "cyberattacktype",
            "washingtoniansaffected",
            "id",
            "name",
        }
        if not required.issubset(fields):
            raise ValueError(
                f"missing fields: expected {sorted(required)}, got {reader.fieldnames!r}"
            )
        normalized = {name.strip().casefold(): name for name in fieldnames}
        rows = []
        invalid_dates = []
        for row_number, record in enumerate(reader, start=2):
            raw_date = record.get(normalized["datesubmitted"]) or ""
            day = parse_day(raw_date)
            if day is None:
                invalid_dates.append(
                    {
                        "row": row_number,
                        "id": record.get(normalized["id"]),
                        "datesubmitted": raw_date,
                    }
                )
            rows.append(
                {
                    "day": day,
                    "attack_type": (
                        record.get(normalized["cyberattacktype"]) or ""
                    ).strip()
                    or None,
                    "affected": parse_affected(
                        record.get(normalized["washingtoniansaffected"])
                    ),
                    "id": (record.get(normalized["id"]) or "").strip(),
                    "name": (record.get(normalized["name"]) or "").strip(),
                }
            )
        return rows, invalid_dates
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise SystemExit(f"could not parse Washington Socrata CSV: {exc}") from exc


def build_seeds(rows, invalid_dates, source_metadata):
    valid_rows = [row for row in rows if row["day"] is not None]
    distinct_types = sorted(
        {row["attack_type"] for row in rows},
        key=lambda value: (value is not None, (value or "").casefold()),
    )
    type_values = [value for value in distinct_types]
    output = {}
    for seed in SEEDS:
        start = date.fromisoformat(seed["window_start"])
        end = date.fromisoformat(seed["window_end"])
        window_rows = [row for row in valid_rows if start <= row["day"] <= end]
        if seed["id"] == "wa-ag-ransomware-notices-2026q3":
            selected = [
                row
                for row in window_rows
                if (row["attack_type"] or "").casefold() == "ransomware"
            ]
            metric = "count"
        else:
            selected = window_rows
            metric = "max" if seed["role"] == "secondary" else "count"
        granularity = seed["history_granularity"]
        history = []
        for label in history_periods(granularity, seed["history_end"]):
            period_rows = [
                row
                for row in valid_rows
                if row["day"] < start and period_label(row["day"], granularity) == label
            ]
            if metric == "max":
                values = [
                    row["affected"]
                    for row in period_rows
                    if row["affected"] is not None
                ]
                value = max(values) if values else None
            elif seed["id"] == "wa-ag-ransomware-notices-2026q3":
                value = sum(
                    1
                    for row in period_rows
                    if (row["attack_type"] or "").casefold() == "ransomware"
                )
            else:
                value = len(period_rows)
            history.append({"period": label, "value": value})
        month_counts = Counter(row["day"].strftime("%Y-%m") for row in selected)
        value = (
            max(
                (row["affected"] for row in selected if row["affected"] is not None),
                default=None,
            )
            if metric == "max"
            else len(selected)
        )
        detail = {
            "window_breakdown": {
                month: month_counts.get(month, 0)
                for month in history_periods("month", seed["window_end"][:7])[-3:]
            },
            "distinct_cyberattacktype_values": type_values,
            "unparseable_dates": len(invalid_dates),
            "source_metadata": source_metadata,
            "history_vintage": {
                "method": "lag_guard",
                "last_period": seed["history_end"],
                "reason": "Socrata may post notices after datesubmitted, so history ends at the specified lag-guard period.",
            },
        }
        output[seed["id"]] = {
            "role": seed["role"],
            "value": value,
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": granularity,
            "history": history,
            "detail": detail,
            "notes": f"One dataset row is one notification; history uses datesubmitted and stops at {seed['history_end']} under the lag guard.",
        }
    return output, type_values


def main():
    parser = argparse.ArgumentParser(
        description="Resolve Washington AG breach-notice counts"
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
        csv_path = data_dir / "wa_ag_rows.csv"
        metadata_path = data_dir / "wa_ag_metadata.json"
        count_path = data_dir / "wa_ag_count.json"
        for path in (csv_path, metadata_path, count_path):
            if not path.is_file():
                raise SystemExit(f"offline source snapshot is missing: {path}")
        csv_raw, metadata_raw, count_raw = (
            path.read_bytes() for path in (csv_path, metadata_path, count_path)
        )
    else:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        snapshots = []
        request_clock = [None]
        csv_response = fetch(
            session, CSV_URL, "wa_ag_rows.csv", None, data_dir, snapshots, request_clock
        )
        metadata_response = fetch(
            session,
            METADATA_URL,
            "wa_ag_metadata.json",
            None,
            data_dir,
            snapshots,
            request_clock,
        )
        count_response = fetch(
            session,
            COUNT_URL,
            "wa_ag_count.json",
            {"$select": "count(*)"},
            data_dir,
            snapshots,
            request_clock,
        )
        retrieved_at = snapshots[0]["retrieved_at"] if snapshots else utc_now()
        csv_raw, metadata_raw, count_raw = (
            csv_response.content,
            metadata_response.content,
            count_response.content,
        )

    try:
        metadata = json.loads(metadata_raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not parse Socrata metadata JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SystemExit("Socrata metadata JSON is not an object")
    source_metadata = extract_metadata_summary(metadata)
    if source_metadata["rowsUpdatedAt_utc"] is None:
        raise SystemExit(
            f"Socrata metadata lacks a parseable rowsUpdatedAt value: {metadata.get('rowsUpdatedAt')!r}"
        )

    rows, invalid_dates = parse_csv(csv_raw)
    api_count = parse_count_response(count_raw)
    if len(rows) != api_count:
        raise SystemExit(
            f"CSV row count {len(rows)} does not match SODA count(*) {api_count}"
        )
    seeds, distinct_types = build_seeds(rows, invalid_dates, source_metadata)

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
    print(
        f"csv_rows={len(rows)} soda_count={api_count} unparseable_dates={len(invalid_dates)}"
    )
    print(
        f"distinct_cyberattacktype_values={json.dumps(distinct_types, ensure_ascii=False)}"
    )
    for seed_id, value in seeds.items():
        print(f"{seed_id}: {value['value']}")


if __name__ == "__main__":
    main()
