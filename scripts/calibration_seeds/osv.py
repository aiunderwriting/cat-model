#!/usr/bin/env python3
"""Resolve Q3 2026 OSV advisory counts from pinned bulk-export snapshots.

Source: OSV.dev public bulk ZIP exports at
https://osv-vulnerabilities.storage.googleapis.com/<ECOSYSTEM>/all.zip for npm,
PyPI, crates.io, and Go. The seeds use the UTC calendar date of each record's
published field, count distinct IDs, and exclude records with a non-null
withdrawn field. Prefix filters are MAL- for npm, PyPI, and crates.io; GO- for
Go; and RUSTSEC- for crates.io. RustSec histories use a monthly lag guard through
2026-05 because advisories can be added after their published date; MAL- and
GO- histories remain quarterly and publication-dated. Raw ZIP response bytes
are kept without extraction; one canonical JSONL per ecosystem records the
used ID, published value, and withdrawn flag for the requested prefixes. ZIP
generation, ETag, and Last-Modified headers are retained when supplied. Bulk
exports may revise published dates and withdrawn status in later snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "osv"
BASE_URL = "https://osv-vulnerabilities.storage.googleapis.com"
USER_AGENT = "AIUC calibration-seed resolver (research)"
ECOSYSTEMS = ("npm", "PyPI", "crates.io", "Go")
PREFIXES_BY_ECOSYSTEM = {
    "npm": ("MAL-",),
    "PyPI": ("MAL-",),
    "crates.io": ("MAL-", "RUSTSEC-"),
    "Go": ("GO-",),
}
HEADER_NAMES = ("x-goog-generation", "ETag", "Last-Modified")
HISTORY_PERIODS = tuple(
    f"{year:04d}-Q{quarter}"
    for year in range(2024, 2027)
    for quarter in range(1, 5)
    if (year, quarter) <= (2026, 2)
)

# All seed filters are declared here so the source/date/prefix rules stay visible.
SEEDS: list[dict[str, Any]] = [
    {
        "id": "osv-mal-npm-2026q3",
        "role": "primary",
        "ecosystem": "npm",
        "prefix": "MAL-",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"id_prefix": "MAL-", "withdrawn": False},
        "date_field": "published",
        "history_granularity": "quarter",
        "history_end": "2026-Q2",
    },
    {
        "id": "osv-mal-pypi-2026q3",
        "role": "primary",
        "ecosystem": "PyPI",
        "prefix": "MAL-",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"id_prefix": "MAL-", "withdrawn": False},
        "date_field": "published",
        "history_granularity": "quarter",
        "history_end": "2026-Q2",
    },
    {
        "id": "osv-mal-crates-2026q3",
        "role": "primary",
        "ecosystem": "crates.io",
        "prefix": "MAL-",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"id_prefix": "MAL-", "withdrawn": False},
        "date_field": "published",
        "history_granularity": "quarter",
        "history_end": "2026-Q2",
    },
    {
        "id": "go-vulndb-reports-2026q3",
        "role": "primary",
        "ecosystem": "Go",
        "prefix": "GO-",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"id_prefix": "GO-", "withdrawn": False},
        "date_field": "published",
        "history_granularity": "quarter",
        "history_end": "2026-Q2",
    },
    {
        "id": "rustsec-advisories-2026q3",
        "role": "primary",
        "ecosystem": "crates.io",
        "prefix": "RUSTSEC-",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"id_prefix": "RUSTSEC-", "withdrawn": False},
        "date_field": "published",
        "history_granularity": "month",
        "history_end": "2026-05",
    },
]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iso_from_timestamp(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_utc_date(value: Any) -> date | None:
    parsed = parse_utc_datetime(value)
    if parsed is not None:
        return parsed.date()
    if isinstance(value, str) and len(value.strip()) == 10:
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def quarter_label(day: date) -> str:
    return f"{day.year:04d}-Q{(day.month - 1) // 3 + 1}"


def history_periods(granularity: str, end_label: str) -> tuple[str, ...]:
    if granularity == "month":
        end_year, end_month = map(int, end_label.split("-"))
        year, month = 2024, 1
        labels = []
        while (year, month) <= (end_year, end_month):
            labels.append(f"{year:04d}-{month:02d}")
            month += 1
            if month == 13:
                year, month = year + 1, 1
        return tuple(labels)
    return tuple(period for period in HISTORY_PERIODS if period <= end_label)


def period_counts(
    records: list[dict[str, Any]],
    include_withdrawn: bool,
    granularity: str,
    end_label: str,
) -> list[dict[str, Any]]:
    periods = history_periods(granularity, end_label)
    counts = {period: 0 for period in periods}
    for record in records:
        if not include_withdrawn and record["withdrawn"]:
            continue
        day = record["published_date"]
        if day is None:
            continue
        label = day.strftime("%Y-%m") if granularity == "month" else quarter_label(day)
        if label in counts:
            counts[label] += 1
    return [{"period": period, "value": counts[period]} for period in periods]


def load_prior_result(path: Path) -> dict[str, Any] | None:
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        candidate.get("source_id") != SOURCE_ID
        or candidate.get("script") != Path(__file__).name
    ):
        return None
    return candidate


def prior_snapshots(prior: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not prior:
        return {}
    return {
        entry.get("file"): entry
        for entry in prior.get("snapshots", [])
        if isinstance(entry, dict) and isinstance(entry.get("file"), str)
    }


def prior_zip_headers(prior: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    if not prior:
        return found
    for seed in prior.get("seeds", {}).values():
        if not isinstance(seed, dict):
            continue
        detail = seed.get("detail") or {}
        ecosystem = detail.get("ecosystem")
        headers = detail.get("zip_headers")
        if isinstance(ecosystem, str) and isinstance(headers, dict):
            found[ecosystem] = {str(key): str(value) for key, value in headers.items()}
    return found


def prepare_online_dir(data_root: Path) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    source_root = data_root / SOURCE_ID
    if source_root.exists():
        if not source_root.is_dir():
            raise RuntimeError(
                f"refusing to download: {source_root} exists and is not a directory"
            )
        if any(source_root.iterdir()):
            raise RuntimeError(
                f"refusing to download into non-empty directory: {source_root}"
            )
    else:
        source_root.mkdir()
    return source_root


def proxy_403(message: str) -> bool:
    lowered = message.lower()
    return "403" in lowered and any(
        token in lowered for token in ("connect", "tunnel", "proxy")
    )


def transient_request_error(error: BaseException) -> bool:
    if proxy_403(str(error)):
        return False
    return isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ContentDecodingError,
        ),
    )


def download_zip(
    session: requests.Session, url: str, target: Path
) -> tuple[str, dict[str, str]]:
    waits = (2, 4, 8, 16, 32)
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(len(waits) + 1):
        if target.exists():
            target.unlink()
        requested_at = utc_now()
        try:
            with session.get(
                url, headers=headers, stream=True, timeout=(30, 300)
            ) as response:
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    message = f"HTTP {response.status_code} downloading {url}"
                    response.close()
                    if attempt == len(waits):
                        raise RuntimeError(
                            f"transient failure exhausted retries: {message}"
                        )
                    time.sleep(waits[attempt])
                    continue
                if response.status_code >= 400:
                    response.raise_for_status()
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
                selected_headers: dict[str, str] = {}
                for requested_name in HEADER_NAMES:
                    value = response.headers.get(requested_name)
                    if value is not None:
                        selected_headers[requested_name] = value
                return requested_at, selected_headers
        except requests.exceptions.RequestException as error:
            if proxy_403(str(error)):
                raise RuntimeError(
                    f"host refused with HTTP 403 on CONNECT; stopping: {url}: {error}"
                ) from error
            if attempt == len(waits) or not transient_request_error(error):
                raise RuntimeError(f"download failed for {url}: {error}") from error
            time.sleep(waits[attempt])
        except RuntimeError:
            raise
    raise AssertionError("unreachable")


def read_zip_records(
    archive_path: Path, ecosystem: str
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    tracked_prefixes = PREFIXES_BY_ECOSYSTEM[ecosystem]
    tracked: dict[str, dict[str, Any]] = {}
    stats = {
        "json_members_read": 0,
        "json_members_without_id": 0,
        "tracked_record_occurrences": 0,
        "duplicate_record_occurrences": 0,
        "conflicting_duplicate_occurrences": 0,
        "tracked_records_missing_published": 0,
    }
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".json"):
                    continue
                with archive.open(info, "r") as member:
                    try:
                        item = json.loads(member.read())
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise RuntimeError(
                            f"invalid JSON member {info.filename} in {archive_path}: {error}"
                        ) from error
                stats["json_members_read"] += 1
                if not isinstance(item, dict):
                    continue
                identifier = item.get("id")
                published = item.get("published")
                withdrawn_flag = item.get("withdrawn") is not None
                modified = item.get("modified")
                if not isinstance(identifier, str) or not identifier:
                    stats["json_members_without_id"] += 1
                    continue
                if not identifier.startswith(tracked_prefixes):
                    continue
                stats["tracked_record_occurrences"] += 1
                if published is None:
                    stats["tracked_records_missing_published"] += 1
                candidate = {
                    "id": identifier,
                    "published": published,
                    "withdrawn": withdrawn_flag,
                    "modified": modified,
                    "published_date": parse_utc_date(published),
                    "modified_datetime": parse_utc_datetime(modified),
                }
                previous = tracked.get(identifier)
                if previous is None:
                    tracked[identifier] = candidate
                    continue
                stats["duplicate_record_occurrences"] += 1
                comparison_fields = ("published", "withdrawn")
                if any(
                    previous[field] != candidate[field] for field in comparison_fields
                ):
                    stats["conflicting_duplicate_occurrences"] += 1
                    old_modified = previous["modified_datetime"]
                    new_modified = candidate["modified_datetime"]
                    if new_modified is not None and (
                        old_modified is None or new_modified > old_modified
                    ):
                        tracked[identifier] = candidate
                    elif new_modified == old_modified:
                        raise RuntimeError(
                            f"conflicting duplicate OSV id {identifier} with tied modified timestamps in {archive_path}"
                        )
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"invalid ZIP file {archive_path}: {error}") from error
    return tracked, stats


def canonical_extract(
    source_root: Path, ecosystem: str, records: dict[str, dict[str, Any]]
) -> Path:
    rows = [
        {
            "id": record["id"],
            "published": record["published"],
            "withdrawn": record["withdrawn"],
        }
        for record in records.values()
    ]
    rows.sort(key=lambda row: row["id"])
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    path = source_root / f"{ecosystem}_canonical.jsonl"
    path.write_bytes(payload)
    return path


def add_snapshot(
    snapshots: list[dict[str, Any]],
    source_root: Path,
    file_path: Path,
    url: str,
    retrieved_at: str,
    previous: dict[str, dict[str, Any]],
) -> None:
    relative = file_path.relative_to(source_root).as_posix()
    old = previous.get(relative, {})
    snapshots.append(
        {
            "url": old.get("url", url),
            "file": relative,
            "sha256": sha256_file(file_path),
            "bytes": file_path.stat().st_size,
            "retrieved_at": old.get("retrieved_at", retrieved_at),
        }
    )


def snapshot_hash(snapshots: list[dict[str, Any]]) -> str:
    lines = [
        f"{item['sha256']}  {item['file']}"
        for item in sorted(snapshots, key=lambda entry: entry["file"])
    ]
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def window_count(
    records: list[dict[str, Any]], start: date, end: date, include_withdrawn: bool
) -> int:
    return sum(
        1
        for record in records
        if (include_withdrawn or not record["withdrawn"])
        and record["published_date"] is not None
        and start <= record["published_date"] <= end
    )


def monthly_window_breakdown(
    records: list[dict[str, Any]], start: date, end: date
) -> dict[str, int]:
    month_labels: list[str] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        month_labels.append(f"{cursor.year:04d}-{cursor.month:02d}")
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    counts = {label: 0 for label in month_labels}
    for record in records:
        if record["withdrawn"] or record["published_date"] is None:
            continue
        day = record["published_date"]
        if start <= day <= end:
            label = f"{day.year:04d}-{day.month:02d}"
            counts[label] += 1
    return counts


def build_seeds(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for spec in SEEDS:
        ecosystem = spec["ecosystem"]
        prefix = spec["prefix"]
        dataset = datasets[ecosystem]
        records = [
            record
            for identifier, record in dataset["records"].items()
            if identifier.startswith(prefix)
        ]
        start = date.fromisoformat(spec["window_start"])
        end = date.fromisoformat(spec["window_end"])
        value = window_count(records, start, end, include_withdrawn=False)
        granularity = spec["history_granularity"]
        history_end = spec["history_end"]
        history = period_counts(
            records,
            include_withdrawn=False,
            granularity=granularity,
            end_label=history_end,
        )
        detail: dict[str, Any] = {
            "ecosystem": ecosystem,
            "id_prefix": prefix,
            "zip_headers": dataset["headers"],
            "json_members_read": dataset["stats"]["json_members_read"],
            "tracked_record_occurrences": dataset["stats"][
                "tracked_record_occurrences"
            ],
            "distinct_ids_for_prefix": len(records),
            "duplicate_record_occurrences_in_export": dataset["stats"][
                "duplicate_record_occurrences"
            ],
            "conflicting_duplicate_occurrences_in_export": dataset["stats"][
                "conflicting_duplicate_occurrences"
            ],
            "tracked_records_missing_published": sum(
                1 for record in records if record["published"] is None
            ),
        }
        if spec["prefix"] == "RUSTSEC-":
            detail["history_vintage"] = {
                "method": "lag_guard",
                "last_period": history_end,
                "reason": "RustSec advisories can enter the OSV export after their published date, so history stops at the specified lag-guard month.",
            }
        else:
            detail["history_vintage"] = {
                "method": "publication_dated",
                "note": "Records use their own published date; later OSV bulk exports may revise published or withdrawn fields.",
            }
        detail["window_breakdown"] = monthly_window_breakdown(records, start, end)
        detail["value_including_withdrawn"] = window_count(
            records, start, end, include_withdrawn=True
        )
        detail["history_including_withdrawn"] = period_counts(
            records,
            include_withdrawn=True,
            granularity=granularity,
            end_label=history_end,
        )
        detail["withdrawn_ids_in_window"] = sum(
            1
            for record in records
            if record["withdrawn"]
            and record["published_date"] is not None
            and start <= record["published_date"] <= end
        )
        notes = (
            "Counts are distinct IDs by UTC date of published; records with a non-null withdrawn field are excluded. "
            "All JSON members in the bulk ZIP were read; the export provides no separate total-count value to cross-check."
        )
        if dataset["stats"]["duplicate_record_occurrences"]:
            notes += (
                " Duplicate IDs were deduplicated; when published/withdrawn fields conflicted, the record with the latest "
                "parseable modified timestamp was retained."
            )
        results[spec["id"]] = {
            "role": spec["role"],
            "value": value,
            "window_start": spec["window_start"],
            "window_end": spec["window_end"],
            "history_granularity": granularity,
            "history": history,
            "detail": detail,
            "notes": notes,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve OSV ecosystem seed counts from bulk exports."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    prior = load_prior_result(args.out) if args.offline else None
    old_snapshots = prior_snapshots(prior)
    old_headers = prior_zip_headers(prior)
    if args.offline:
        source_root = args.data_dir / SOURCE_ID
        if not source_root.is_dir():
            raise RuntimeError(
                f"offline source directory does not exist: {source_root}"
            )
    else:
        source_root = prepare_online_dir(args.data_dir)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    datasets: dict[str, dict[str, Any]] = {}
    snapshot_times: dict[str, str] = {}
    first_download_at = prior.get("retrieved_at") if prior else None

    for ecosystem in ECOSYSTEMS:
        url = f"{BASE_URL}/{ecosystem}/all.zip"
        zip_path = source_root / f"{ecosystem}_all.zip"
        if args.offline:
            if not zip_path.is_file():
                raise RuntimeError(f"offline ZIP snapshot is missing: {zip_path}")
            previous = old_snapshots.get(
                zip_path.relative_to(source_root).as_posix(), {}
            )
            zip_retrieved_at = previous.get("retrieved_at") or iso_from_timestamp(
                zip_path.stat().st_mtime
            )
            headers = old_headers.get(ecosystem, {})
        else:
            zip_retrieved_at, headers = download_zip(session, url, zip_path)
            if first_download_at is None:
                first_download_at = zip_retrieved_at
        snapshot_times[ecosystem] = zip_retrieved_at
        records, stats = read_zip_records(zip_path, ecosystem)
        datasets[ecosystem] = {"records": records, "stats": stats, "headers": headers}

    snapshots: list[dict[str, Any]] = []
    for ecosystem in ECOSYSTEMS:
        url = f"{BASE_URL}/{ecosystem}/all.zip"
        zip_path = source_root / f"{ecosystem}_all.zip"
        extract_path = canonical_extract(
            source_root, ecosystem, datasets[ecosystem]["records"]
        )
        add_snapshot(
            snapshots,
            source_root,
            zip_path,
            url,
            snapshot_times[ecosystem],
            old_snapshots,
        )
        add_snapshot(
            snapshots,
            source_root,
            extract_path,
            url,
            snapshot_times[ecosystem],
            old_snapshots,
        )
    snapshots.sort(key=lambda entry: entry["file"])
    for entry in snapshots:
        print(f"{entry['sha256']}  {entry['file']}")
    digest = snapshot_hash(snapshots)
    print(f"snapshot_sha256  {digest}")

    if not first_download_at:
        first_download_at = (
            min(snapshot_times.values()) if snapshot_times else utc_now()
        )
    output = {
        "source_id": SOURCE_ID,
        "script": Path(__file__).name,
        "retrieved_at": first_download_at,
        "snapshots": snapshots,
        "commit": None,
        "snapshot_sha256": digest,
        "seeds": build_seeds(datasets),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    session.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"osv.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
