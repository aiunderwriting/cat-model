"""Resolve NHTSA vehicle-recall and defect-investigation seeds from official flat files.

Canonical recall ZIP: https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip; field definitions:
https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt. Canonical investigation ZIP:
https://static.nhtsa.gov/odi/ffdd/inv/FLAT_INV.zip; field definitions:
https://static.nhtsa.gov/odi/ffdd/inv/INV.txt. Recall dates are RCDATE (field 16, YYYYMMDD), record
creation dates are DATEA (field 17), campaign IDs are CAMPNO (field 2), component names are COMPNAME
(field 7), recall type is RCLTYPECD (field 11), and potential units are POTAFF (field 12). An eligible
vehicle campaign has at least one row with RCLTYPECD exactly ``V``; it is assigned to the earliest valid
RCDATE among all of its rows. If rows disagree, the earliest date is used and the disagreement is reported.
The software seed includes a campaign if any campaign row's COMPNAME contains ``SOFTWARE`` case-insensitively.
The secondary seed takes the per-campaign maximum POTAFF, then the maximum of those values by quarter.
Investigation dates are ODATE (field 7, YYYYMMDD); the seed counts distinct NHTSA ACTION NUMBER values and
assigns duplicates with inconsistent ODATE to their earliest valid date. There are no additional exclusions.
All four downloaded source files are saved byte-for-byte and hashed. The recall ZIP is read in place; the
390 MB investigation member is streamed from its ZIP without extraction. NHTSA flat files are revised over
time; the campaign posting-lag detail uses one earliest DATEA minus earliest RCDATE value per 2026 vehicle
campaign to avoid giving multi-row campaigns extra weight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "nhtsa_defects"
USER_AGENT = "AIUC calibration-seed resolver (research)"
RCL_ZIP_URL = "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip"
RCL_FIELDS_URL = "https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt"
INV_ZIP_URL = "https://static.nhtsa.gov/odi/ffdd/inv/FLAT_INV.zip"
INV_FIELDS_URL = "https://static.nhtsa.gov/odi/ffdd/inv/INV.txt"
RCL_ZIP_FILE = "FLAT_RCL_POST_2010.zip"
RCL_FIELDS_FILE = "RCL.txt"
INV_ZIP_FILE = "FLAT_INV.zip"
INV_FIELDS_FILE = "INV.txt"
METADATA_FILE = ".snapshot_metadata.json"
RETRY_DELAYS = (2, 4, 8, 16, 32)

SEEDS: list[dict[str, Any]] = [
    {
        "id": "nhtsa-vehicle-recalls-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {
            "kind": "vehicle_recall_count",
            "rcltypecd": "V",
            "date_field": "RCDATE",
        },
    },
    {
        "id": "nhtsa-software-recalls-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {
            "kind": "vehicle_recall_count",
            "rcltypecd": "V",
            "date_field": "RCDATE",
            "component_contains_case_insensitive": "SOFTWARE",
        },
    },
    {
        "id": "nhtsa-odi-investigations-2026-07-10",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-10-31",
        "filter": {"kind": "distinct_investigation_count", "date_field": "ODATE"},
    },
    {
        "id": "nhtsa-largest-vehicle-recall-2026q3",
        "role": "secondary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {
            "kind": "vehicle_recall_max_potaaf",
            "rcltypecd": "V",
            "date_field": "RCDATE",
            "campaign_aggregation": "maximum POTAFF across rows",
        },
    },
]


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
            response = session.get(url, headers=headers, stream=True, timeout=(30, 600))
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
            if response is not None:
                response.close()
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


def save_snapshot_metadata(directory, snapshots, first_retrieved_at):
    sorted_snapshots = sorted(snapshots, key=lambda item: item["file"])
    snapshot_hash = manifest_sha256(sorted_snapshots)
    payload = {
        "source_id": SOURCE_ID,
        "retrieved_at": first_retrieved_at,
        "snapshots": sorted_snapshots,
        "snapshot_sha256": snapshot_hash,
    }
    temporary = directory / (METADATA_FILE + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, directory / METADATA_FILE)
    return sorted_snapshots, snapshot_hash, first_retrieved_at


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
    return snapshots, snapshot_hash, payload.get("retrieved_at")


def print_snapshot_hashes(snapshots, snapshot_hash):
    for entry in sorted(snapshots, key=lambda item: item["file"]):
        print(f"{entry['sha256']}  {entry['file']}")
    print(f"snapshot_sha256  {snapshot_hash}")


def parse_date_yyyymmdd(value):
    value = (value or "").strip()
    if len(value) != 8 or not value.isdigit() or value == "00000000":
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def parse_potaaf(value):
    value = (value or "").strip().replace(",", "")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def find_text_member(archive, expected_prefix):
    candidates = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and info.filename.lower().endswith(".txt")
        and Path(info.filename).name.lower().startswith(expected_prefix.lower())
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one {expected_prefix} text member in ZIP, found {candidates}"
        )
    return candidates[0]


def aggregate_recalls(zip_path):
    campaigns: dict[str, dict[str, Any]] = {}
    row_count = 0
    short_rows = 0
    rows_missing_campaign = 0
    invalid_rcdate_rows = 0
    invalid_datea_rows = 0
    invalid_potaaf_rows = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        member_name = find_text_member(archive, "FLAT_RCL")
        with (
            archive.open(member_name, "r") as raw,
            io.TextIOWrapper(raw, encoding="latin-1", newline="") as text,
        ):
            reader = csv.reader(text, delimiter="\t", quotechar='"', doublequote=True)
            for row in reader:
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                row_count += 1
                if len(row) < 17:
                    short_rows += 1
                    continue
                campaign_id = row[1].strip().upper()
                if not campaign_id or campaign_id.upper() == "CAMPNO":
                    rows_missing_campaign += 1
                    continue
                recall_date = parse_date_yyyymmdd(row[15])
                date_a = parse_date_yyyymmdd(row[16])
                if recall_date is None:
                    invalid_rcdate_rows += 1
                if date_a is None:
                    invalid_datea_rows += 1
                affected = parse_potaaf(row[11])
                if row[11].strip() and affected is None:
                    invalid_potaaf_rows += 1
                component = row[6].strip()
                recall_type = row[10].strip().upper()
                aggregate = campaigns.get(campaign_id)
                if aggregate is None:
                    new_aggregate: dict[str, Any] = {
                        "rc_min": None,
                        "rc_max": None,
                        "rc_disagree": False,
                        "datea_for_min_rc": None,
                        "vehicle": False,
                        "software": False,
                        "potaaf_min": None,
                        "potaaf_max": None,
                        "potaaf_disagree": False,
                        "row_count": 0,
                    }
                    campaigns[campaign_id] = new_aggregate
                    aggregate = new_aggregate
                aggregate["row_count"] += 1
                if recall_type == "V":
                    aggregate["vehicle"] = True
                if "SOFTWARE" in component.upper():
                    aggregate["software"] = True
                if recall_date is not None:
                    if aggregate["rc_min"] is None or recall_date < aggregate["rc_min"]:
                        aggregate["rc_min"] = recall_date
                        aggregate["datea_for_min_rc"] = date_a
                    elif (
                        recall_date == aggregate["rc_min"]
                        and date_a is not None
                        and (
                            aggregate["datea_for_min_rc"] is None
                            or date_a < aggregate["datea_for_min_rc"]
                        )
                    ):
                        aggregate["datea_for_min_rc"] = date_a
                    if aggregate["rc_max"] is None or recall_date > aggregate["rc_max"]:
                        aggregate["rc_max"] = recall_date
                    if aggregate["rc_min"] != aggregate["rc_max"]:
                        aggregate["rc_disagree"] = True
                if affected is not None:
                    if (
                        aggregate["potaaf_min"] is None
                        or affected < aggregate["potaaf_min"]
                    ):
                        aggregate["potaaf_min"] = affected
                    if (
                        aggregate["potaaf_max"] is None
                        or affected > aggregate["potaaf_max"]
                    ):
                        aggregate["potaaf_max"] = affected
                    if aggregate["potaaf_min"] != aggregate["potaaf_max"]:
                        aggregate["potaaf_disagree"] = True
    return campaigns, {
        "member": member_name,
        "row_count": row_count,
        "distinct_campaigns": len(campaigns),
        "rows_shorter_than_17_fields": short_rows,
        "rows_missing_campaign_id": rows_missing_campaign,
        "rows_with_invalid_rcdate": invalid_rcdate_rows,
        "rows_with_invalid_datea": invalid_datea_rows,
        "rows_with_invalid_potaaf": invalid_potaaf_rows,
    }


def aggregate_investigations(zip_path):
    actions = {}
    row_count = 0
    short_rows = 0
    missing_action_rows = 0
    invalid_odate_rows = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        member_name = find_text_member(archive, "FLAT_INV")
        with (
            archive.open(member_name, "r") as raw,
            io.TextIOWrapper(raw, encoding="latin-1", newline="") as text,
        ):
            reader = csv.reader(text, delimiter="\t", quotechar='"', doublequote=True)
            for row in reader:
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                row_count += 1
                if len(row) < 7:
                    short_rows += 1
                    continue
                action_number = row[0].strip().upper()
                if not action_number or action_number == "NHTSA ACTION NUMBER":
                    missing_action_rows += 1
                    continue
                opened = parse_date_yyyymmdd(row[6])
                if opened is None:
                    invalid_odate_rows += 1
                    continue
                prefix_match = re.match(r"([A-Z]+)", action_number)
                prefix = prefix_match.group(1) if prefix_match else "OTHER"
                aggregate = actions.get(action_number)
                if aggregate is None:
                    actions[action_number] = {
                        "date": opened,
                        "date_min": opened,
                        "date_max": opened,
                        "date_disagree": False,
                        "prefix": prefix,
                        "rows": 1,
                    }
                else:
                    aggregate["rows"] += 1
                    if opened < aggregate["date"]:
                        aggregate["date"] = opened
                    if opened < aggregate["date_min"]:
                        aggregate["date_min"] = opened
                    if opened > aggregate["date_max"]:
                        aggregate["date_max"] = opened
                    if aggregate["date_min"] != aggregate["date_max"]:
                        aggregate["date_disagree"] = True
                    if prefix != aggregate["prefix"]:
                        aggregate["prefix"] = "OTHER"
    return actions, {
        "member": member_name,
        "row_count": row_count,
        "distinct_action_numbers": len(actions),
        "rows_shorter_than_7_fields": short_rows,
        "rows_missing_action_number": missing_action_rows,
        "rows_with_invalid_odate": invalid_odate_rows,
    }


def quarter_for_date(value):
    quarter = ((value.month - 1) // 3) + 1
    return f"{value.year}-Q{quarter}"


def month_labels(start, end):
    current = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    labels = []
    while current <= last:
        labels.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return labels


def quarter_labels(start, end):
    year = start.year
    quarter = ((start.month - 1) // 3) + 1
    last_year = end.year
    last_quarter = ((end.month - 1) // 3) + 1
    labels = []
    while (year, quarter) <= (last_year, last_quarter):
        labels.append(f"{year}-Q{quarter}")
        quarter += 1
        if quarter == 5:
            year += 1
            quarter = 1
    return labels


def lag_statistics(lags):
    if not lags:
        return {
            "basis": "one value per distinct vehicle campaign: earliest DATEA minus earliest RCDATE",
            "campaign_count": 0,
            "missing_datea_campaigns": 0,
            "min_days": None,
            "median_days": None,
            "p90_days": None,
            "mean_days": None,
            "max_days": None,
            "bucket_counts": {},
        }
    ordered = sorted(lags)
    length = len(ordered)
    if length % 2:
        median = float(ordered[length // 2])
    else:
        median = (ordered[length // 2 - 1] + ordered[length // 2]) / 2
    p90 = ordered[max(0, (9 * length + 9) // 10 - 1)]
    buckets = {"negative": 0, "0": 0, "1-7": 0, "8-30": 0, "31-90": 0, ">90": 0}
    for lag in lags:
        if lag < 0:
            buckets["negative"] += 1
        elif lag == 0:
            buckets["0"] += 1
        elif lag <= 7:
            buckets["1-7"] += 1
        elif lag <= 30:
            buckets["8-30"] += 1
        elif lag <= 90:
            buckets["31-90"] += 1
        else:
            buckets[">90"] += 1
    return {
        "basis": "one value per distinct vehicle campaign: earliest DATEA minus earliest RCDATE",
        "campaign_count": length,
        "missing_datea_campaigns": None,
        "min_days": ordered[0],
        "median_days": median,
        "p90_days": p90,
        "mean_days": round(sum(lags) / length, 2),
        "max_days": ordered[-1],
        "bucket_counts": buckets,
    }


def resolve_recall_seeds(campaigns, recall_parse_detail):
    history_periods = quarter_labels(date(2024, 1, 1), date(2026, 6, 30))
    all_history_counts = {period: 0 for period in history_periods}
    software_history_counts = {period: 0 for period in history_periods}
    all_history_maxima = {period: None for period in history_periods}
    quarter_campaigns = {}
    vehicle_2026 = []
    missing_lag_datea = 0
    for campaign_id, aggregate in campaigns.items():
        recall_date = aggregate["rc_min"]
        if not aggregate["vehicle"] or recall_date is None:
            continue
        if recall_date.year == 2026:
            vehicle_2026.append((campaign_id, aggregate))
        quarter = quarter_for_date(recall_date)
        if quarter in all_history_counts:
            all_history_counts[quarter] += 1
            if aggregate["software"]:
                software_history_counts[quarter] += 1
            affected = aggregate["potaaf_max"]
            if affected is not None and (
                all_history_maxima[quarter] is None
                or affected > all_history_maxima[quarter]
            ):
                all_history_maxima[quarter] = affected
        if recall_date.year == 2026 and recall_date.month in (7, 8, 9):
            quarter_campaigns[campaign_id] = aggregate

    q3_months = ["2026-07", "2026-08", "2026-09"]
    q3_count_breakdown = {month: 0 for month in q3_months}
    q3_software_breakdown = {month: 0 for month in q3_months}
    q3_max_breakdown = {month: None for month in q3_months}
    q3_rcdate_disagreements = 0
    q3_potaaf_disagreements = 0
    q3_software_campaigns = 0
    q3_potaaf_missing_campaigns = 0
    for aggregate in quarter_campaigns.values():
        month = aggregate["rc_min"].strftime("%Y-%m")
        q3_count_breakdown[month] += 1
        if aggregate["software"]:
            q3_software_breakdown[month] += 1
            q3_software_campaigns += 1
        if aggregate["rc_disagree"]:
            q3_rcdate_disagreements += 1
        if aggregate["potaaf_disagree"]:
            q3_potaaf_disagreements += 1
        affected = aggregate["potaaf_max"]
        if affected is None:
            q3_potaaf_missing_campaigns += 1
        elif q3_max_breakdown[month] is None or affected > q3_max_breakdown[month]:
            q3_max_breakdown[month] = affected

    lags = []
    for _campaign_id, aggregate in vehicle_2026:
        if aggregate["datea_for_min_rc"] is None:
            missing_lag_datea += 1
            continue
        lags.append((aggregate["datea_for_min_rc"] - aggregate["rc_min"]).days)
    lag_detail = lag_statistics(lags)
    lag_detail["missing_datea_campaigns"] = missing_lag_datea
    lag_detail["vehicle_campaigns_with_rcdate_disagreement"] = sum(
        1 for _, aggregate in vehicle_2026 if aggregate["rc_disagree"]
    )
    lag_detail["vehicle_campaigns_in_2026"] = len(vehicle_2026)

    q3_count = len(quarter_campaigns)
    q3_max = max(
        (
            aggregate["potaaf_max"]
            for aggregate in quarter_campaigns.values()
            if aggregate["potaaf_max"] is not None
        ),
        default=None,
    )
    q3_largest_campaigns = [
        campaign_id
        for campaign_id, aggregate in quarter_campaigns.items()
        if aggregate["potaaf_max"] == q3_max and q3_max is not None
    ]
    q3_details = {
        "window_breakdown": q3_count_breakdown,
        "software_window_breakdown": q3_software_breakdown,
        "largest_recall_window_breakdown": q3_max_breakdown,
        "vehicle_campaigns_with_disagreeing_rcdate_in_window": q3_rcdate_disagreements,
        "vehicle_campaigns_with_disagreeing_potaaf_in_window": q3_potaaf_disagreements,
        "vehicle_campaigns_missing_potaaf_in_window": q3_potaaf_missing_campaigns,
        "software_vehicle_campaigns_in_window": q3_software_campaigns,
        "largest_recall_campaign_ids": sorted(q3_largest_campaigns),
        "posting_lag_2026_vehicle_campaigns": lag_detail,
        "recall_source_rows": recall_parse_detail["row_count"],
        "recall_campaign_ids_in_source": recall_parse_detail["distinct_campaigns"],
        "recall_rows_with_invalid_rcdate": recall_parse_detail[
            "rows_with_invalid_rcdate"
        ],
    }
    results = {}
    for seed in SEEDS:
        if seed["filter"]["kind"] not in (
            "vehicle_recall_count",
            "vehicle_recall_max_potaaf",
        ):
            continue
        kind = seed["filter"]["kind"]
        if kind == "vehicle_recall_count":
            is_software = (
                seed["filter"].get("component_contains_case_insensitive") == "SOFTWARE"
            )
            value = q3_software_campaigns if is_software else q3_count
            month_values = q3_software_breakdown if is_software else q3_count_breakdown
            detail = dict(q3_details)
            detail["window_breakdown"] = month_values
            detail["count_definition"] = (
                "distinct CAMPNO; any RCLTYPECD=V row; earliest valid RCDATE per campaign"
            )
            if is_software:
                detail["software_filter"] = (
                    "any COMPNAME contains SOFTWARE, case-insensitive"
                )
            else:
                detail["software_filter"] = None
        else:
            value = q3_max
            detail = dict(q3_details)
            detail["window_breakdown"] = q3_max_breakdown
            detail["campaign_aggregation"] = (
                "maximum POTAFF across rows of each vehicle campaign"
            )
            detail["campaigns_tied_for_largest_recall"] = len(q3_largest_campaigns)
        results[seed["id"]] = {
            "role": seed["role"],
            "value": value,
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": "quarter",
            "history": [
                {
                    "period": period,
                    "value": (
                        all_history_maxima[period]
                        if kind == "vehicle_recall_max_potaaf"
                        else software_history_counts[period]
                        if seed["filter"].get("component_contains_case_insensitive")
                        == "SOFTWARE"
                        else all_history_counts[period]
                    ),
                }
                for period in history_periods
            ],
            "detail": detail,
            "notes": (
                "Recall campaigns are grouped by CAMPNO. A campaign is a vehicle campaign if any row has "
                "RCLTYPECD exactly V; if its rows disagree on RCDATE, its earliest valid RCDATE assigns it "
                "to the quarter. POTAFF is the maximum valid row value per campaign. Posting-lag statistics "
                "use one earliest DATEA minus earliest RCDATE value per distinct 2026 vehicle campaign."
            ),
        }
    return results


def resolve_investigation_seed(actions, investigation_parse_detail):
    seed = next(
        item
        for item in SEEDS
        if item["filter"]["kind"] == "distinct_investigation_count"
    )
    history_periods = month_labels(date(2024, 1, 1), date(2026, 6, 30))
    counts = {month: 0 for month in history_periods}
    window_months = month_labels(date(2026, 7, 1), date(2026, 10, 31))
    window_breakdown = {month: 0 for month in window_months}
    prefix_breakdown = {}
    duplicate_numbers = 0
    for _action_number, aggregate in actions.items():
        opened = aggregate["date"]
        month = opened.strftime("%Y-%m")
        if month in counts:
            counts[month] += 1
        if month in window_breakdown:
            window_breakdown[month] += 1
            prefix = aggregate["prefix"]
            prefix_breakdown[prefix] = prefix_breakdown.get(prefix, 0) + 1
        if aggregate["rows"] > 1:
            duplicate_numbers += 1
    value = sum(window_breakdown.values())
    return {
        seed["id"]: {
            "role": seed["role"],
            "value": value,
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": "month",
            "history": [
                {"period": month, "value": counts[month]} for month in history_periods
            ],
            "detail": {
                "window_breakdown": window_breakdown,
                "action_number_prefix_breakdown": dict(
                    sorted(prefix_breakdown.items())
                ),
                "distinct_action_numbers_in_source": investigation_parse_detail[
                    "distinct_action_numbers"
                ],
                "investigation_source_rows": investigation_parse_detail["row_count"],
                "action_numbers_with_multiple_rows": duplicate_numbers,
                "action_numbers_with_disagreeing_odate": sum(
                    1 for aggregate in actions.values() if aggregate["date_disagree"]
                ),
                "rows_with_invalid_odate": investigation_parse_detail[
                    "rows_with_invalid_odate"
                ],
            },
            "notes": (
                "Counts distinct NHTSA ACTION NUMBER values by earliest valid ODATE (field 7); duplicates "
                "with inconsistent dates are assigned to the earliest date. Prefixes are the leading letters "
                "of the action number; unexpected formats are grouped as OTHER."
            ),
        }
    }


def build_results(data_root, out_path, offline):
    directory = prepare_data_directory(data_root, offline)
    if offline:
        snapshots, snapshot_hash, first_retrieved_at = load_snapshot_metadata(directory)
    else:
        download_order = [
            (RCL_ZIP_URL, RCL_ZIP_FILE),
            (RCL_FIELDS_URL, RCL_FIELDS_FILE),
            (INV_ZIP_URL, INV_ZIP_FILE),
            (INV_FIELDS_URL, INV_FIELDS_FILE),
        ]
        with requests.Session() as session:
            downloaded = [
                download_file(session, url, directory, filename)
                for url, filename in download_order
            ]
        snapshots, snapshot_hash, first_retrieved_at = save_snapshot_metadata(
            directory, downloaded, downloaded[0]["retrieved_at"]
        )

    recall_campaigns, recall_parse_detail = aggregate_recalls(directory / RCL_ZIP_FILE)
    actions, investigation_parse_detail = aggregate_investigations(
        directory / INV_ZIP_FILE
    )
    seed_results = resolve_recall_seeds(recall_campaigns, recall_parse_detail)
    seed_results.update(resolve_investigation_seed(actions, investigation_parse_detail))

    result = {
        "source_id": SOURCE_ID,
        "script": Path(__file__).name,
        "retrieved_at": first_retrieved_at,
        "snapshots": snapshots,
        "commit": None,
        "snapshot_sha256": snapshot_hash,
        "seeds": {seed["id"]: seed_results[seed["id"]] for seed in SEEDS},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print_snapshot_hashes(snapshots, snapshot_hash)
    for seed_id, seed_result in result["seeds"].items():
        print(f"{seed_id}: {seed_result['value']}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Resolve NHTSA recalls and investigation seed counts."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        build_results(args.data_dir, args.out, args.offline)
    except Exception as exc:
        print(f"nhtsa_defects resolver failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
