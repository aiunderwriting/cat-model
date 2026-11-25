"""Resolve NHTSA SGO report counts from the current amended-order ADS and ADAS CSVs.

Source and canonical page: https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting
Canonical CSVs: https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv
and https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv.
The date field is the NHTSA-generated ``Report Submission Date`` (month-year); ``Report Type`` is used
to identify updates. ``nhtsa-sgo-ads-reports-2026-07-08`` counts distinct Report IDs in the ADS CSV and
``nhtsa-sgo-adas-reports-2026-07-08`` counts distinct Report IDs in the ADAS CSV, each by the earliest
submission month among rows whose type is not ``Update`` and within JUL-2026 and AUG-2026.
IDs with only Update rows are excluded from values and counted separately by earliest update
month in detail. No report is excluded for version, reporting entity, or outcome. Raw page and CSV
response bytes are saved unchanged, with per-file hashes and a manifest hash. The 2025 amendment took
effect June 16, 2025; updates can revise prior report details, and the monthly files lag publication by
roughly one month. Current report submission months end partway through September 2026.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, date, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

SOURCE_ID = "nhtsa_sgo"
USER_AGENT = "AIUC calibration-seed resolver (research)"
PAGE_URL = (
    "https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting"
)
ADS_URL = (
    "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv"
)
ADAS_URL = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv"
PAGE_FILE = "standing_general_order_crash_reporting.html"
ADS_FILE = "SGO-2021-01_Incident_Reports_ADS.csv"
ADAS_FILE = "SGO-2021-01_Incident_Reports_ADAS.csv"
METADATA_FILE = ".snapshot_metadata.json"
RETRY_DELAYS = (2, 4, 8, 16, 32)

SEEDS: list[dict[str, Any]] = [
    {
        "id": "nhtsa-sgo-ads-reports-2026-07-08",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-08-31",
        "filter": {
            "file": ADS_FILE,
            "initial_submission_months": ["2026-07", "2026-08"],
        },
    },
    {
        "id": "nhtsa-sgo-adas-reports-2026-07-08",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-08-31",
        "filter": {
            "file": ADAS_FILE,
            "initial_submission_months": ["2026-07", "2026-08"],
        },
    },
]


class AnchorParser(HTMLParser):
    """Collect anchor labels and hrefs from the saved NHTSA page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.anchors = []
        self._href = None
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._parts = []

    def handle_data(self, data):
        if self._href is not None:
            value = data.strip()
            if value:
                self._parts.append(value)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._href is not None:
            label = " ".join(self._parts)
            self.anchors.append((label, self._href))
            self._href = None
            self._parts = []


def utc_now_string():
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path):
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def manifest_sha256(snapshots):
    body = "".join(
        f"{entry['sha256']}  {entry['file']}\n"
        for entry in sorted(snapshots, key=lambda item: item["file"])
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def prepare_data_directory(data_root, offline):
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


def retryable_proxy_refusal(exc):
    message = str(exc).lower()
    return "403" in message and (
        "connect" in message or "tunnel" in message or "proxy" in message
    )


def download_file(session, url, directory, filename):
    destination = directory / filename
    partial = directory / (filename + ".part")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    last_error = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        response = None
        try:
            response = session.get(url, headers=headers, stream=True, timeout=(30, 300))
            if response.status_code == 403:
                raise RuntimeError(
                    f"HTTP 403 while requesting {url}; host access is refused"
                )
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                last_error = RuntimeError(
                    f"transient HTTP {response.status_code} for {url}"
                )
                response.close()
                response = None
                if attempt == len(RETRY_DELAYS):
                    break
                time.sleep(RETRY_DELAYS[attempt])
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code} while requesting {url}"
                )
            response.raise_for_status()
            digest = hashlib.sha256()
            byte_count = 0
            with partial.open("wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        output.write(block)
                        digest.update(block)
                        byte_count += len(block)
            response.close()
            response = None
            os.replace(partial, destination)
            return {
                "url": url,
                "file": filename,
                "sha256": digest.hexdigest(),
                "bytes": byte_count,
                "retrieved_at": utc_now_string(),
            }
        except RuntimeError:
            partial.unlink(missing_ok=True)
            raise
        except requests.exceptions.RequestException as exc:
            if response is not None:
                response.close()
            partial.unlink(missing_ok=True)
            if retryable_proxy_refusal(exc):
                raise RuntimeError(
                    f"HTTP 403 on proxy CONNECT while requesting {url}: {exc}"
                ) from exc
            last_error = exc
            if attempt == len(RETRY_DELAYS):
                break
            time.sleep(RETRY_DELAYS[attempt])
        except OSError as exc:
            if response is not None:
                response.close()
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"failed to save response for {url}: {exc}") from exc
    raise RuntimeError(
        f"download failed after retries for {url}: {last_error}"
    ) from last_error


def download_page_once(session, directory):
    """Try the descriptive page once; a non-200 response is metadata, not a blocker."""
    destination = directory / PAGE_FILE
    partial = directory / (PAGE_FILE + ".part")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    for attempt in range(len(RETRY_DELAYS) + 1):
        response = None
        try:
            response = session.get(
                PAGE_URL, headers=headers, stream=True, timeout=(30, 300)
            )
            page_status = response.status_code
            if page_status != 200:
                retrieved_at = utc_now_string()
                response.close()
                return {
                    "page_status": page_status,
                    "page_saved": False,
                    "retrieved_at": retrieved_at,
                    "snapshot": None,
                }

            digest = hashlib.sha256()
            byte_count = 0
            retrieved_at = utc_now_string()
            with partial.open("wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        output.write(block)
                        digest.update(block)
                        byte_count += len(block)
            response.close()
            response = None
            os.replace(partial, destination)
            snapshot = {
                "url": PAGE_URL,
                "file": PAGE_FILE,
                "sha256": digest.hexdigest(),
                "bytes": byte_count,
                "retrieved_at": retrieved_at,
            }
            return {
                "page_status": page_status,
                "page_saved": True,
                "retrieved_at": retrieved_at,
                "snapshot": snapshot,
            }
        except requests.exceptions.RequestException as exc:
            if response is not None:
                response.close()
            partial.unlink(missing_ok=True)
            if retryable_proxy_refusal(exc):
                raise RuntimeError(
                    f"HTTP 403 on proxy CONNECT while requesting {PAGE_URL}: {exc}"
                ) from exc
            if attempt == len(RETRY_DELAYS):
                raise RuntimeError(
                    f"page request failed after retries for {PAGE_URL}: {exc}"
                ) from exc
            time.sleep(RETRY_DELAYS[attempt])
        except OSError as exc:
            if response is not None:
                response.close()
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"failed to save page response for {PAGE_URL}: {exc}"
            ) from exc
    raise AssertionError("unreachable")


def save_snapshot_metadata(
    directory, snapshots, first_retrieved_at, page_status, page_saved
):
    sorted_snapshots = sorted(snapshots, key=lambda item: item["file"])
    snapshot_hash = manifest_sha256(sorted_snapshots)
    payload = {
        "source_id": SOURCE_ID,
        "retrieved_at": first_retrieved_at,
        "page_status": page_status,
        "page_saved": page_saved,
        "snapshots": sorted_snapshots,
        "snapshot_sha256": snapshot_hash,
    }
    temporary = directory / (METADATA_FILE + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, directory / METADATA_FILE)
    return sorted_snapshots, snapshot_hash, first_retrieved_at, page_status, page_saved


def load_snapshot_metadata(directory):
    metadata_path = directory / METADATA_FILE
    if not metadata_path.is_file():
        raise RuntimeError(f"offline snapshot metadata is missing: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read offline snapshot metadata: {exc}") from exc
    if payload.get("source_id") != SOURCE_ID or not isinstance(
        payload.get("snapshots"), list
    ):
        raise RuntimeError("offline snapshot metadata has the wrong source id or shape")
    snapshots = payload["snapshots"]
    expected_files = {METADATA_FILE}
    for entry in snapshots:
        relative = entry.get("file")
        if (
            not isinstance(relative, str)
            or Path(relative).name != relative
            or relative in ("", ".", "..")
        ):
            raise RuntimeError(
                f"unsafe snapshot path in offline metadata: {relative!r}"
            )
        expected_files.add(relative)
        path = directory / relative
        if not path.is_file():
            raise RuntimeError(f"offline snapshot is missing: {path}")
        actual_hash, actual_bytes = sha256_file(path)
        if actual_hash != entry.get("sha256") or actual_bytes != entry.get("bytes"):
            raise RuntimeError(f"offline snapshot failed integrity check: {relative}")
    actual_files = {entry.name for entry in directory.iterdir()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"offline directory contents differ from its manifest; expected {sorted(expected_files)}, "
            f"found {sorted(actual_files)}"
        )
    snapshot_hash = manifest_sha256(snapshots)
    if snapshot_hash != payload.get("snapshot_sha256"):
        raise RuntimeError("offline snapshot manifest hash does not match metadata")
    page_saved = payload.get("page_saved")
    if not isinstance(page_saved, bool):
        page_saved = PAGE_FILE in expected_files
    page_status = payload.get("page_status")
    if page_status is None and page_saved:
        page_status = 200
    return (
        snapshots,
        snapshot_hash,
        payload.get("retrieved_at"),
        page_status,
        page_saved,
    )


def print_snapshot_hashes(snapshots, snapshot_hash):
    for entry in sorted(snapshots, key=lambda item: item["file"]):
        print(f"{entry['sha256']}  {entry['file']}")
    print(f"snapshot_sha256  {snapshot_hash}")


def parse_submission_month(value):
    value = (value or "").strip().upper()
    match = re.fullmatch(r"([A-Z]{3})-(\d{4})", value)
    if not match:
        return None
    try:
        parsed = datetime.strptime(value, "%b-%Y").date()
    except ValueError:
        return None
    return parsed.replace(day=1)


def months_inclusive(start, end):
    current = date(start.year, start.month, 1)
    final = date(end.year, end.month, 1)
    result = []
    while current <= final:
        result.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return result


def parse_page_details(page_bytes):
    html_text = page_bytes.decode("utf-8", errors="replace")
    parser = AnchorParser()
    parser.feed(html_text)
    page_text = unescape(
        " ".join(
            re.sub(r"\s+", " ", part) for part in re.findall(r">([^<>]+)<", html_text)
        )
    )
    page_text = re.sub(r"\s+", " ", page_text)

    range_match = re.search(
        r"data files contain incident report data from\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\s+to\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
        page_text,
        flags=re.IGNORECASE,
    )
    if range_match is None:
        range_match = re.search(
            r"through\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
            page_text,
            flags=re.IGNORECASE,
        )
        if range_match is None:
            raise RuntimeError(
                "could not find NHTSA SGO data-through date on the saved page"
            )
        through_text = range_match.group(1)
        start_text = None
    else:
        start_text, through_text = range_match.groups()

    try:
        through_date = datetime.strptime(through_text, "%B %d, %Y").date().isoformat()
        start_date = (
            datetime.strptime(start_text, "%B %d, %Y").date().isoformat()
            if start_text
            else None
        )
    except ValueError as exc:
        raise RuntimeError(f"unrecognized NHTSA SGO data range date: {exc}") from exc

    cadence = (
        "monthly" if re.search(r"\bmonthly\b", page_text, flags=re.IGNORECASE) else None
    )
    if cadence is None:
        raise RuntimeError(
            "could not find the NHTSA SGO release cadence on the saved page"
        )

    other_files = []
    selected_urls = {ADS_URL, ADAS_URL}
    seen = set()
    for label, href in parser.anchors:
        normalized_label = re.sub(r"\s+", " ", label).strip()
        if not href or "incident report data" not in normalized_label.lower():
            continue
        absolute_url = urljoin(PAGE_URL, href)
        if absolute_url in selected_urls:
            continue
        key = (normalized_label, absolute_url)
        if key in seen:
            continue
        seen.add(key)
        other_files.append({"label": normalized_label, "url": absolute_url})
    other_files.sort(key=lambda item: (item["label"].lower(), item["url"]))
    return {
        "data_start_date": start_date,
        "data_through_date": through_date,
        "release_cadence": cadence,
        "other_downloadable_sgo_files": other_files,
    }


def read_report_initial_months(path):
    reports = {}
    row_count = 0
    missing_report_id_rows = 0
    invalid_submission_date_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Report ID", "Report Type", "Report Submission Date"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(
                f"unexpected SGO CSV headers in {path.name}: {reader.fieldnames}"
            )
        for row in reader:
            row_count += 1
            report_id = (row.get("Report ID") or "").strip()
            if not report_id:
                missing_report_id_rows += 1
                continue
            month = parse_submission_month(row.get("Report Submission Date"))
            if month is None:
                invalid_submission_date_rows += 1
            report_type = (row.get("Report Type") or "").strip().casefold()
            aggregate = reports.setdefault(
                report_id,
                {
                    "earliest_any": None,
                    "earliest_non_update": None,
                    "has_non_update": False,
                    "rows": 0,
                },
            )
            aggregate["rows"] += 1
            if report_type != "update":
                aggregate["has_non_update"] = True
            if month is not None:
                if (
                    aggregate["earliest_any"] is None
                    or month < aggregate["earliest_any"]
                ):
                    aggregate["earliest_any"] = month
                if report_type != "update" and (
                    aggregate["earliest_non_update"] is None
                    or month < aggregate["earliest_non_update"]
                ):
                    aggregate["earliest_non_update"] = month

    initial_months = {}
    update_only_months = {}
    unresolvable_ids = 0
    for aggregate in reports.values():
        if aggregate["has_non_update"]:
            month = aggregate["earliest_non_update"]
            if month is None:
                unresolvable_ids += 1
                continue
            label = month.strftime("%Y-%m")
            initial_months[label] = initial_months.get(label, 0) + 1
            continue

        # Update-only IDs are excluded from resolved counts but shown by month.
        month = aggregate["earliest_any"]
        if month is None:
            unresolvable_ids += 1
            continue
        label = month.strftime("%Y-%m")
        update_only_months[label] = update_only_months.get(label, 0) + 1
    return {
        "initial_months": initial_months,
        "update_only_months": update_only_months,
        "row_count": row_count,
        "report_id_count": len(reports),
        "missing_report_id_rows": missing_report_id_rows,
        "invalid_submission_date_rows": invalid_submission_date_rows,
        "unresolvable_report_ids": unresolvable_ids,
    }


def build_results(data_root, out_path, offline):
    directory = prepare_data_directory(data_root, offline)
    if offline:
        snapshots, snapshot_hash, first_retrieved_at, page_status, page_saved = (
            load_snapshot_metadata(directory)
        )
    else:
        with requests.Session() as session:
            page_result = download_page_once(session, directory)
            downloaded = []
            if page_result["snapshot"] is not None:
                downloaded.append(page_result["snapshot"])
            downloaded.append(download_file(session, ADS_URL, directory, ADS_FILE))
            downloaded.append(download_file(session, ADAS_URL, directory, ADAS_FILE))
        snapshots, snapshot_hash, first_retrieved_at, page_status, page_saved = (
            save_snapshot_metadata(
                directory,
                downloaded,
                page_result["retrieved_at"],
                page_result["page_status"],
                page_result["page_saved"],
            )
        )

    page_details = (
        parse_page_details((directory / PAGE_FILE).read_bytes()) if page_saved else None
    )
    parsed_files = {
        ADS_FILE: read_report_initial_months(directory / ADS_FILE),
        ADAS_FILE: read_report_initial_months(directory / ADAS_FILE),
    }
    seed_results = {}
    for seed in SEEDS:
        source_file = seed["filter"]["file"]
        parsed = parsed_files[source_file]
        window_months = seed["filter"]["initial_submission_months"]
        date.fromisoformat(seed["window_start"])
        date.fromisoformat(seed["window_end"])
        history_granularity = "month"
        history_periods = months_inclusive(date(2025, 7, 1), date(2026, 4, 30))
        history_values = {
            period: parsed["initial_months"].get(period, 0)
            for period in history_periods
        }
        value = sum(parsed["initial_months"].get(month, 0) for month in window_months)
        month_breakdown = {
            month: parsed["initial_months"].get(month, 0) for month in window_months
        }
        update_only_breakdown = {
            month: parsed["update_only_months"].get(month, 0) for month in window_months
        }

        detail = {
            "report_rows": parsed["row_count"],
            "distinct_report_ids": parsed["report_id_count"],
            "missing_report_id_rows": parsed["missing_report_id_rows"],
            "invalid_submission_date_rows": parsed["invalid_submission_date_rows"],
            "report_ids_without_usable_submission_month": parsed[
                "unresolvable_report_ids"
            ],
            "page_status": page_status,
            "page_saved": page_saved,
            "page_data_start_date": page_details["data_start_date"]
            if page_details
            else None,
            "page_data_through_date": page_details["data_through_date"]
            if page_details
            else None,
            "page_release_cadence": page_details["release_cadence"]
            if page_details
            else None,
            "other_downloadable_sgo_files": page_details["other_downloadable_sgo_files"]
            if page_details
            else None,
            "history_vintage": {
                "method": "lag_guard",
                "last_period": history_periods[-1],
                "reason": "NHTSA SGO releases lag by roughly one month, so May and June 2026 were not public by the 2026-06-30 information date.",
            },
            "window_breakdown": month_breakdown,
            "update_only_ids_in_window": sum(update_only_breakdown.values()),
            "update_only_ids_by_month": update_only_breakdown,
        }
        notes = (
            "Counts distinct Report ID values by the earliest Report Submission Date among non-Update rows; "
            "IDs with only Update rows are excluded and reported separately by month in detail. "
            "The current amended-order files begin 2025-06-16; history starts at the first complete month, 2025-07. "
            "NHTSA updates the files monthly, and later Update rows can revise report details."
        )
        if seed["role"] == "primary":
            july_count = parsed["initial_months"].get("2026-07", 0)
            august_count = parsed["initial_months"].get("2026-08", 0)
            september_count = parsed["initial_months"].get("2026-09", 0)
            notes += (
                f" September 2026 is visibly partial in this snapshot: July {july_count}, "
                f"August {august_count}, September {september_count} initial-submission Report IDs; "
                "the seed window ends in August for that reason."
            )
        if not page_saved:
            notes += f" The NHTSA information page returned HTTP {page_status}; its page metadata was unavailable."
        seed_results[seed["id"]] = {
            "role": seed["role"],
            "value": value,
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": history_granularity,
            "history": [
                {"period": period, "value": history_values[period]}
                for period in history_periods
            ],
            "detail": detail,
            "notes": notes,
        }

    result = {
        "source_id": SOURCE_ID,
        "script": Path(__file__).name,
        "retrieved_at": first_retrieved_at,
        "snapshots": snapshots,
        "commit": None,
        "snapshot_sha256": snapshot_hash,
        "seeds": seed_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print_snapshot_hashes(snapshots, snapshot_hash)
    for seed_id, seed_result in seed_results.items():
        print(f"{seed_id}: {seed_result['value']}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Resolve NHTSA Standing General Order report-count seeds."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        build_results(args.data_dir, args.out, args.offline)
    except Exception as exc:
        print(f"nhtsa_sgo resolver failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
