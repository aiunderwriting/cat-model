#!/usr/bin/env python3
"""Resolve OFAC SDN addition counts from the Sanctions List Service deltas.

Source: OFAC Sanctions List Service. Annual publication lists are downloaded
from https://sanctionslistservice.ofac.treas.gov/changes/history/<year> and
contain publicationID/datePublished entries used to find delta XML at
https://sanctionslistservice.ofac.treas.gov/changes/<publicationID>.
The date in each XML delta's <publicationInfo><datePublished> assigns its
publication period, converted to its America/New_York calendar date (a
timestamp without an offset is treated as local Eastern time). The history
JSON date is retained only for backfill diagnostics. Publications whose XML
date is before 2024-01-01 fall outside all reported periods. For each
publication dated 2024-01-01 through 2026-10-31, inspect entity action and list
membership. Count distinct entity IDs with action=add that belong to the SDN
list; if a delta has no list-membership field, count all its add actions and
record that fallback. Also extract sanctions programs for counted additions;
program totals are non-exclusive when one entity lists multiple programs.
The primary seed ofac-sdn-additions-2026q3 covers 2026-07-01 through
2026-09-30 with quarterly history 2024-Q1 through 2026-Q2. The primary seed
ofac-sdn-additions-2026-10 covers 2026-10-01 through 2026-10-31 with monthly
history 2024-01 through 2026-06. All 54 history-list/XML date mismatches have
a later history-list date. Of these, 28 XML dates are in October-December
2023 and their history dates are in January 2024; the other 26 XML dates are
earlier dates within 2024. None of the mismatches have history-list dates in
2025 or 2026. Every history
JSON response, delta XML response, redirect, and transient HTTP response is
saved byte-for-byte. OFAC data are U.S. Government public information; the
downloaded payloads do not provide a separate license field. These are
publication-dated records, though OFAC can revise published data. Some XML
deltas use action=remove for deletions; only top-level entity action=add is
counted. Requests start at least three seconds apart; HTTP 429 and 503
responses honor Retry-After or use 30, 60, 120, and 240 second backoffs before
failing.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import requests

SOURCE_ID = "ofac"
SCRIPT_NAME = "ofac.py"
BASE_URL = "https://sanctionslistservice.ofac.treas.gov"
HISTORY_URL = BASE_URL + "/changes/history/{year}"
DELTA_URL = BASE_URL + "/changes/{publication_id}"
EASTERN = ZoneInfo("America/New_York")
USER_AGENT = "AIUC calibration-seed resolver (research)"
RETRY_DELAYS = (2, 4, 8, 16, 32)
RATE_LIMIT_RETRY_DELAYS = (30, 60, 120, 240)
MIN_REQUEST_INTERVAL_SECONDS = 3.0
MAX_RESULT_BYTES = 300 * 1024
LEGACY_HISTORY_VINTAGE_NOTE = (
    "The annual history JSON datePublished, converted to an America/New_York calendar date, "
    "defines the publication period; XML delta datePublished can be older on backfilled records, "
    "and OFAC may revise published deltas."
)
YEARS = (2024, 2025, 2026)
WINDOW_END = date(2026, 10, 31)
WINDOW_START = date(2024, 1, 1)
SEEDS: list[dict[str, Any]] = [
    {
        "id": "ofac-sdn-additions-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "history_granularity": "quarter",
        "filter": {"action": "add", "list": "SDN"},
    },
    {
        "id": "ofac-sdn-additions-2026-10",
        "role": "primary",
        "window_start": "2026-10-01",
        "window_end": "2026-10-31",
        "history_granularity": "month",
        "filter": {"action": "add", "list": "SDN"},
    },
]
LAST_REQUEST_STARTED = None


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


def retry_after_seconds(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[0-9]+", text):
        return float(int(text))
    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at.astimezone(UTC) - datetime.now(UTC)).total_seconds())


def throttle():
    global LAST_REQUEST_STARTED
    now = time.monotonic()
    if LAST_REQUEST_STARTED is not None:
        remaining = MIN_REQUEST_INTERVAL_SECONDS - (now - LAST_REQUEST_STARTED)
        if remaining > 0:
            time.sleep(remaining)
    LAST_REQUEST_STARTED = time.monotonic()


def request_bytes(session, url, directory, filename, snapshots, allow_404=False):
    stem = Path(filename).stem
    attempt = 0
    retry_attempt = 0
    rate_limit_retry_attempt = 0
    while True:
        current_url = url
        redirect_index = 0
        retry = False
        while True:
            throttle()
            try:
                response = session.get(
                    current_url, timeout=(30, 180), allow_redirects=False
                )
            except requests.exceptions.ProxyError as exc:
                if proxy_refusal(exc):
                    raise RuntimeError(
                        f"HTTP 403 on proxy CONNECT while requesting {current_url}: {exc}"
                    ) from exc
                if retry_attempt >= len(RETRY_DELAYS):
                    raise RuntimeError(
                        f"OFAC request failed after retries for {current_url}: {exc}"
                    ) from exc
                time.sleep(RETRY_DELAYS[retry_attempt])
                retry_attempt += 1
                attempt += 1
                retry = True
                break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as exc:
                if retry_attempt >= len(RETRY_DELAYS):
                    raise RuntimeError(
                        f"OFAC request failed after retries for {current_url}: {exc}"
                    ) from exc
                time.sleep(RETRY_DELAYS[retry_attempt])
                retry_attempt += 1
                attempt += 1
                retry = True
                break

            status = response.status_code
            if status in (301, 302, 303, 307, 308):
                saved_name = (
                    f"{stem}_attempt{attempt}_redirect{redirect_index}_http{status}.bin"
                )
                # Keep the stable OFAC request URL in the manifest. The signed
                # object-storage redirect URL is temporary and can be very long.
                record_response(directory, snapshots, saved_name, url, response.content)
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise RuntimeError(
                        f"OFAC redirect from {current_url} had no Location header"
                    )
                redirect_index += 1
                if redirect_index > 10:
                    raise RuntimeError(
                        f"OFAC redirect chain exceeded ten hops for {url}"
                    )
                current_url = urljoin(current_url, location)
                continue

            content = response.content
            if status == 403:
                saved_name = f"{stem}_attempt{attempt}_http403.bin"
                record_response(directory, snapshots, saved_name, url, content)
                response.close()
                raise RuntimeError(
                    f"HTTP 403 from OFAC while requesting {current_url}; stopping"
                )
            if status == 404 and allow_404:
                record_response(
                    directory, snapshots, f"{stem}_http404.bin", url, content
                )
                response.close()
                return content, 404
            if status in (429, 503):
                record_response(
                    directory,
                    snapshots,
                    f"{stem}_attempt{attempt}_http{status}.bin",
                    url,
                    content,
                )
                retry_after = retry_after_seconds(response.headers.get("Retry-After"))
                response.close()
                if rate_limit_retry_attempt >= len(RATE_LIMIT_RETRY_DELAYS):
                    raise RuntimeError(
                        f"transient HTTP {status} after {rate_limit_retry_attempt} retries for {current_url}"
                    )
                delay = retry_after
                if delay is None:
                    delay = RATE_LIMIT_RETRY_DELAYS[rate_limit_retry_attempt]
                rate_limit_retry_attempt += 1
                attempt += 1
                time.sleep(delay)
                retry = True
                break
            if status >= 500:
                record_response(
                    directory,
                    snapshots,
                    f"{stem}_attempt{attempt}_http{status}.bin",
                    url,
                    content,
                )
                response.close()
                if retry_attempt >= len(RETRY_DELAYS):
                    raise RuntimeError(
                        f"transient HTTP {status} after retries for {current_url}"
                    )
                time.sleep(RETRY_DELAYS[retry_attempt])
                retry_attempt += 1
                attempt += 1
                retry = True
                break
            if status >= 400:
                record_response(
                    directory,
                    snapshots,
                    f"{stem}_attempt{attempt}_http{status}.bin",
                    url,
                    content,
                )
                response.close()
                raise RuntimeError(
                    f"HTTP {status} from OFAC while requesting {current_url}"
                )

            record_response(directory, snapshots, filename, url, content)
            response.close()
            return content, 200
        if retry:
            continue
    raise RuntimeError(f"OFAC request retry loop ended unexpectedly for {url}")


def parse_publication_date(value):
    text = str(value or "").strip()
    if not text:
        raise RuntimeError("publication record has no datePublished")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RuntimeError(f"unparseable datePublished value {text!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=EASTERN)
    else:
        parsed = parsed.astimezone(EASTERN)
    return parsed.date()


def parse_history_list(raw, year):
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"OFAC {year} history response is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"OFAC {year} history response is not a JSON list")
    records = []
    seen_ids = set()
    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError(f"OFAC {year} history contains a non-object record")
        publication_id = str(item.get("publicationID", "")).strip()
        if not publication_id:
            raise RuntimeError(f"OFAC {year} history record has no publicationID")
        if publication_id in seen_ids:
            raise RuntimeError(
                f"OFAC {year} history repeats publicationID {publication_id}"
            )
        seen_ids.add(publication_id)
        published = str(item.get("datePublished", "")).strip()
        published_date = parse_publication_date(published)
        if abs(published_date.year - year) > 1:
            raise RuntimeError(
                f"OFAC {year} history contains publication {publication_id} dated {published_date.isoformat()}"
            )
        records.append(
            {
                "publication_id": publication_id,
                "date_published": published,
                "local_date": published_date.isoformat(),
            }
        )
    return records


def unique_publications(history_records):
    by_id = {}
    for records in history_records.values():
        for record in records:
            prior = by_id.get(record["publication_id"])
            if prior is not None and prior["local_date"] != record["local_date"]:
                raise RuntimeError(
                    f"OFAC publication {record['publication_id']} appears with conflicting local dates"
                )
            by_id[record["publication_id"]] = record
    return sorted(
        by_id.values(),
        key=lambda record: (record["local_date"], record["publication_id"]),
    )


def local_name(tag):
    return tag.rsplit("}", 1)[-1].casefold()


def normalized_label(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def membership_from_label(value):
    label = normalized_label(value)
    if not label:
        return None
    if (
        "nonsdn" in label
        or "nonsd" in label
        or "consolidated" in label
        or label.startswith("ns")
    ):
        return "non_sdn"
    if (
        label == "sdn"
        or label.startswith("sdnlist")
        or "speciallydesignatednationals" in label
    ):
        return "sdn"
    return None


def tag_membership(tag):
    label = normalized_label(tag)
    if "nonsdn" in label or "nonsd" in label or "consolidatedlist" in label:
        return "non_sdn"
    if label == "sdnlist" or label == "sdn" or "speciallydesignatednationals" in label:
        return "sdn"
    return None


def classify_entity_list(entity, parent_map):
    candidates = []
    node = entity
    while node is not None:
        tag_class = tag_membership(local_name(node.tag))
        if tag_class:
            candidates.append(tag_class)
        for key, value in node.attrib.items():
            key_name = normalized_label(local_name(key))
            if key_name in {
                "list",
                "listtype",
                "listname",
                "listcode",
                "sanctionslist",
                "sanctionslistname",
            }:
                label_class = membership_from_label(value)
                if label_class is None:
                    if str(value).strip():
                        raise RuntimeError(
                            f"unrecognized OFAC list-membership label {value!r}"
                        )
                else:
                    candidates.append(label_class)
        node = parent_map.get(node)

    for descendant in entity.iter():
        if descendant is entity:
            continue
        tag = normalized_label(local_name(descendant.tag))
        if tag in {
            "list",
            "listtype",
            "listname",
            "listcode",
            "sanctionslist",
            "sanctionslistname",
        }:
            value = "".join(descendant.itertext()).strip()
            label_class = membership_from_label(value)
            if label_class is None:
                if value:
                    raise RuntimeError(
                        f"unrecognized OFAC list-membership label {value!r}"
                    )
            else:
                candidates.append(label_class)

    if "sdn" in candidates:
        return "sdn"
    if "non_sdn" in candidates:
        return "non_sdn"
    return None


def find_first_text(root, wanted_names):
    for node in root.iter():
        if local_name(node.tag) in wanted_names:
            value = "".join(node.itertext()).strip()
            if value:
                return value
    return None


def parse_delta(raw, publication_id, history_date):
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise RuntimeError(
            f"OFAC delta {publication_id} is not valid XML: {exc}"
        ) from exc
    if local_name(root.tag) in {"html", "error"}:
        raise RuntimeError(
            f"OFAC delta {publication_id} has an error document root {root.tag!r}"
        )

    actual_id = find_first_text(root, {"publicationid"})
    if actual_id is not None and actual_id != publication_id:
        raise RuntimeError(
            f"OFAC delta URL {publication_id} returned publicationID {actual_id}"
        )
    publication_info = next(
        (node for node in root if local_name(node.tag) == "publicationinfo"),
        None,
    )
    published = (
        find_first_text(publication_info, {"datepublished"})
        if publication_info is not None
        else None
    )
    if published is None:
        raise RuntimeError(
            f"OFAC delta {publication_id} has no publicationInfo/datePublished"
        )
    actual_date = parse_publication_date(published)

    parent_map = {child: parent for parent in root.iter() for child in parent}
    actions = Counter()
    add_records = []
    for entity in root.iter():
        if local_name(entity.tag) != "entity" or "action" not in entity.attrib:
            continue
        raw_action = str(entity.attrib["action"]).strip().casefold()
        if raw_action not in {"add", "delete", "remove", "update"}:
            raise RuntimeError(
                f"OFAC entity has unexpected action {raw_action!r} in delta {publication_id}"
            )
        action = "delete" if raw_action == "remove" else raw_action
        actions[action] += 1
        if action != "add":
            continue
        entity_id = str(entity.attrib.get("id", "")).strip()
        if not entity_id:
            raise RuntimeError(f"OFAC add entity has no id in delta {publication_id}")
        membership = classify_entity_list(entity, parent_map)
        programs = set()
        for descendant in entity.iter():
            if local_name(descendant.tag) in {
                "program",
                "programname",
                "sanctionsprogram",
            }:
                program = " ".join("".join(descendant.itertext()).split())
                if program:
                    programs.add(program)
        add_records.append(
            {
                "id": entity_id,
                "membership": membership,
                "programs": sorted(programs),
            }
        )

    if not any(local_name(node.tag) == "entities" for node in root.iter()):
        raise RuntimeError(f"OFAC delta {publication_id} has no entities section")

    by_id = {}
    for record in add_records:
        item = by_id.setdefault(record["id"], {"memberships": set(), "programs": set()})
        if record["membership"]:
            item["memberships"].add(record["membership"])
        item["programs"].update(record["programs"])

    if not by_id:
        count = 0
        basis = "no_additions"
        fallback = False
        selected_ids = set()
    else:
        has_membership = [bool(item["memberships"]) for item in by_id.values()]
        if any(has_membership) and not all(has_membership):
            raise RuntimeError(
                f"OFAC delta {publication_id} has mixed present/missing list membership on add entities"
            )
        fallback = not any(has_membership)
        if fallback:
            selected_ids = set(by_id)
            basis = "all_adds_membership_not_available"
        else:
            selected_ids = {
                entity_id
                for entity_id, item in by_id.items()
                if "sdn" in item["memberships"]
            }
            basis = "sdn_membership"
        count = len(selected_ids)

    program_counts = Counter()
    for entity_id in selected_ids:
        for program in by_id[entity_id]["programs"]:
            program_counts[program] += 1
    return {
        "publication_id": publication_id,
        "date": actual_date.isoformat(),
        "history_date": history_date,
        "delta_date": actual_date.isoformat(),
        "date_mismatch": actual_date.isoformat() != history_date,
        "count": count,
        "basis": basis,
        "fallback_used": fallback,
        "add_entity_ids": sorted(by_id),
        "all_add_ids": len(by_id),
        "sdn_add_ids": sum(
            1 for item in by_id.values() if "sdn" in item["memberships"]
        ),
        "non_sdn_add_ids": sum(
            1
            for item in by_id.values()
            if item["memberships"] and "sdn" not in item["memberships"]
        ),
        "action_counts": {
            action: actions.get(action, 0) for action in ("add", "delete", "update")
        },
        "program_counts": dict(sorted(program_counts.items())),
        "counted_adds_without_program": sum(
            1 for entity_id in selected_ids if not by_id[entity_id]["programs"]
        ),
        "duplicate_add_action_rows": len(add_records) - len(by_id),
    }


def load_offline_manifest(out_path, directory):
    if not out_path.is_file():
        raise RuntimeError(f"offline mode needs the online result file: {out_path}")
    try:
        previous = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read prior OFAC result: {exc}") from exc
    if previous.get("source_id") != SOURCE_ID or not isinstance(
        previous.get("snapshots"), list
    ):
        raise RuntimeError(
            "prior OFAC result has the wrong source id or no snapshots list"
        )
    listed = set()
    for entry in previous["snapshots"]:
        relative = Path(entry.get("file", ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RuntimeError(f"unsafe OFAC snapshot path {entry.get('file')!r}")
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"offline OFAC snapshot is missing: {path}")
        raw = path.read_bytes()
        if sha256_bytes(raw) != entry.get("sha256") or len(raw) != entry.get("bytes"):
            raise RuntimeError(
                f"offline OFAC snapshot failed integrity check: {relative}"
            )
        listed.add(relative.as_posix())
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != listed:
        raise RuntimeError(
            f"offline OFAC files differ from prior manifest (extra={sorted(actual - listed)}, "
            f"missing={sorted(listed - actual)})"
        )
    if manifest_sha256(previous["snapshots"]) != previous.get("snapshot_sha256"):
        raise RuntimeError("offline OFAC manifest hash does not match the prior result")
    return previous


def uses_legacy_history_date_rule(result):
    seeds = result.get("seeds")
    expected_seed_ids = {seed["id"] for seed in SEEDS}
    if not isinstance(seeds, dict) or set(seeds) != expected_seed_ids:
        return False
    return all(
        seed.get("detail", {}).get("history_vintage", {}).get("note")
        == LEGACY_HISTORY_VINTAGE_NOTE
        for seed in seeds.values()
    )


def load_histories_offline(directory, snapshots):
    history_records = {}
    history_statuses = {}
    for year in YEARS:
        expected_name = f"history_{year}.json"
        missing_name = f"history_{year}_http404.bin"
        files = {entry["file"] for entry in snapshots}
        if expected_name in files:
            records = parse_history_list((directory / expected_name).read_bytes(), year)
            history_records[year] = records
            history_statuses[year] = 200
        elif year == 2024 and missing_name in files:
            history_records[year] = []
            history_statuses[year] = 404
        else:
            raise RuntimeError(
                f"offline OFAC snapshot lacks the expected history list for {year}"
            )
    expected_delta_ids = {
        record["publication_id"]
        for record in unique_publications(history_records)
        if WINDOW_START <= date.fromisoformat(record["local_date"]) <= WINDOW_END
    }
    delta_files = {
        Path(entry["file"]).stem.removeprefix("delta_"): entry["file"]
        for entry in snapshots
        if re.fullmatch(r"delta_[A-Za-z0-9_-]+\.xml", Path(entry["file"]).name)
    }
    if set(delta_files) != expected_delta_ids:
        raise RuntimeError(
            f"offline OFAC delta files differ from history lists "
            f"(extra={sorted(set(delta_files) - expected_delta_ids)}, missing={sorted(expected_delta_ids - set(delta_files))})"
        )
    delta_metrics = {}
    for record in unique_publications(history_records):
        published_date = date.fromisoformat(record["local_date"])
        if not WINDOW_START <= published_date <= WINDOW_END:
            continue
        filename = delta_files[record["publication_id"]]
        delta_metrics[record["publication_id"]] = parse_delta(
            (directory / filename).read_bytes(),
            record["publication_id"],
            record["local_date"],
        )
    return history_records, history_statuses, delta_metrics


def download_histories(session, directory, snapshots):
    history_records = {}
    history_statuses = {}
    for year in YEARS:
        url = HISTORY_URL.format(year=year)
        filename = f"history_{year}.json"
        raw, status = request_bytes(
            session,
            url,
            directory,
            filename,
            snapshots,
            allow_404=(year == 2024),
        )
        if status == 404:
            history_records[year] = []
            history_statuses[year] = 404
            continue
        records = parse_history_list(raw, year)
        if year != 2024 and not records:
            raise RuntimeError(
                f"OFAC history endpoint returned no records for required year {year}"
            )
        history_records[year] = records
        history_statuses[year] = 200
    return history_records, history_statuses


def download_deltas(session, directory, snapshots, history_records):
    needed = [
        record
        for record in unique_publications(history_records)
        if WINDOW_START <= date.fromisoformat(record["local_date"]) <= WINDOW_END
    ]
    metrics = {}
    for record in needed:
        publication_id = record["publication_id"]
        url = DELTA_URL.format(publication_id=publication_id)
        filename = f"delta_{publication_id}.xml"
        raw, status = request_bytes(session, url, directory, filename, snapshots)
        if status != 200:
            raise RuntimeError(f"OFAC delta {publication_id} returned HTTP {status}")
        metrics[publication_id] = parse_delta(raw, publication_id, record["local_date"])
    return metrics


def history_start_year(history_records, history_statuses):
    covered = []
    for year in YEARS:
        if history_statuses.get(year) != 200:
            continue
        if any(
            date.fromisoformat(record["local_date"]).year == year
            for record in history_records[year]
        ):
            covered.append(year)
    if not covered:
        raise RuntimeError("OFAC history endpoint has no covered publication year")
    return min(covered)


def history_periods(granularity, first_year):
    if granularity == "month":
        output = []
        year, month = first_year, 1
        while (year, month) <= (2026, 6):
            output.append(f"{year:04d}-{month:02d}")
            month += 1
            if month == 13:
                year, month = year + 1, 1
        return output
    return [
        f"{year}-Q{quarter}"
        for year in range(first_year, 2027)
        for quarter in range(1, 5)
        if f"{year}-Q{quarter}" <= "2026-Q2"
    ]


def period_label(day, granularity):
    if granularity == "month":
        return day.strftime("%Y-%m")
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


def build_seeds(delta_metrics, first_history_year):
    results = {}
    date_mismatches = [
        metric for metric in delta_metrics.values() if metric["date_mismatch"]
    ]
    mismatch_counts_by_year = Counter(
        date.fromisoformat(metric["history_date"]).year for metric in date_mismatches
    )
    mismatch_examples = [
        {
            "publication_id": metric["publication_id"],
            "history_date": metric["history_date"],
            "delta_xml_date": metric["delta_date"],
        }
        for metric in date_mismatches[:20]
    ]
    for seed in SEEDS:
        start = date.fromisoformat(seed["window_start"])
        end = date.fromisoformat(seed["window_end"])
        granularity = seed["history_granularity"]
        window_metrics = [
            metric
            for metric in delta_metrics.values()
            if start <= date.fromisoformat(metric["date"]) <= end
        ]
        window_metrics.sort(key=lambda item: (item["date"], item["publication_id"]))
        history = []
        for label in history_periods(granularity, first_history_year):
            count = sum(
                metric["count"]
                for metric in delta_metrics.values()
                if date.fromisoformat(metric["date"]) < start
                and period_label(date.fromisoformat(metric["date"]), granularity)
                == label
            )
            history.append({"period": label, "value": count})
        months = []
        year, month = start.year, start.month
        while (year, month) <= (end.year, end.month):
            months.append(f"{year:04d}-{month:02d}")
            month += 1
            if month == 13:
                year, month = year + 1, 1
        additions_by_month = {month: 0 for month in months}
        publications_by_month = {month: [] for month in months}
        action_counts = Counter()
        program_counts = Counter()
        for metric in window_metrics:
            month = metric["date"][:7]
            additions_by_month[month] += metric["count"]
            publications_by_month[month].append(
                {
                    "publication_id": metric["publication_id"],
                    "date": metric["date"],
                    "count": metric["count"],
                    "basis": metric["basis"],
                }
            )
            action_counts.update(metric["action_counts"])
            program_counts.update(metric["program_counts"])
        fallback_publications = sum(
            1 for metric in window_metrics if metric["fallback_used"]
        )
        sdn_publications = sum(
            1 for metric in window_metrics if metric["basis"] == "sdn_membership"
        )
        no_add_publications = sum(
            1 for metric in window_metrics if metric["basis"] == "no_additions"
        )
        detail = {
            "window_breakdown": additions_by_month,
            "window_publications": len(window_metrics),
            "publications_by_month": publications_by_month,
            "window_action_record_counts": {
                action: action_counts.get(action, 0)
                for action in ("add", "delete", "update")
            },
            "sdn_membership_classification": {
                "publications_with_sdn_membership": sdn_publications,
                "publications_using_all_adds_fallback": fallback_publications,
                "publications_with_no_additions": no_add_publications,
                "counted_additions_by_program": dict(sorted(program_counts.items())),
                "program_counts_are_nonexclusive": True,
                "counted_additions_without_program": sum(
                    metric["counted_adds_without_program"] for metric in window_metrics
                ),
            },
            "history_coverage_start": (
                f"{first_history_year}-01"
                if granularity == "month"
                else f"{first_history_year}-Q1"
            ),
            "history_vintage": {
                "method": "publication_dated",
                "note": "The XML delta publicationInfo/datePublished, converted to an America/New_York calendar date, defines the publication period; timestamps without an offset are treated as Eastern. The annual history JSON datePublished is retained for mismatch diagnostics, and OFAC may revise published deltas.",
            },
            "history_list_delta_date_mismatches": {
                "count": len(date_mismatches),
                "by_history_year": {
                    str(year): mismatch_counts_by_year.get(year, 0) for year in YEARS
                },
                "mismatches_with_history_date_in_2025_or_2026": sum(
                    mismatch_counts_by_year.get(year, 0) for year in (2025, 2026)
                ),
                "first_examples": mismatch_examples,
            },
            "terms_statement": (
                "OFAC data are U.S. Government public information; the downloaded history and delta payloads "
                "contained no separate license field."
            ),
        }
        fallback_note = (
            " For publications whose delta had no list-membership field, all add actions were counted as "
            "allowed by the seed specification."
            if fallback_publications
            else ""
        )
        coverage_note = (
            f" The history endpoint did not provide a covered 2024 publication year; history begins at "
            f"{detail['history_coverage_start']}."
            if first_history_year > 2024
            else ""
        )
        mismatch_note = (
            f" The delta XML datePublished differed from the annual history JSON datePublished for "
            f"{len(date_mismatches)} publication(s); periods use the XML date. The mismatches were "
            f"listed in {mismatch_counts_by_year.get(2024, 0)} publications in 2024, "
            f"{mismatch_counts_by_year.get(2025, 0)} in 2025, and "
            f"{mismatch_counts_by_year.get(2026, 0)} in 2026, with examples recorded in detail."
            if date_mismatches
            else ""
        )
        results[seed["id"]] = {
            "role": seed["role"],
            "value": sum(metric["count"] for metric in window_metrics),
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": granularity,
            "history": history,
            "detail": detail,
            "notes": (
                "Counts distinct entity IDs with action=add on the SDN list, grouped by the Eastern calendar "
                "date in the delta XML publicationInfo/datePublished; timestamps without an offset are treated "
                "as Eastern. Sanctions program additions are reported as non-exclusive counts."
                + fallback_note
                + coverage_note
                + mismatch_note
            ),
        }
    return results


def main():
    global LAST_REQUEST_STARTED
    parser = argparse.ArgumentParser(
        description="Resolve OFAC SDN list addition counts."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    directory = prepare_directory(args.data_dir, args.offline)
    if args.offline:
        previous = load_offline_manifest(args.out, directory)
        snapshots = previous["snapshots"]
        retrieved_at = previous["retrieved_at"]
        history_records, history_statuses, delta_metrics = load_histories_offline(
            directory, snapshots
        )
    else:
        snapshots = []
        LAST_REQUEST_STARTED = None
        with requests.Session() as session:
            session.headers.update(
                {
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json, application/xml, text/xml",
                    "Accept-Encoding": "identity",
                }
            )
            history_records, history_statuses = download_histories(
                session, directory, snapshots
            )
            delta_metrics = download_deltas(
                session, directory, snapshots, history_records
            )
        if not snapshots:
            raise RuntimeError("OFAC produced no snapshots")
        retrieved_at = min(item["retrieved_at"] for item in snapshots)

    first_year = history_start_year(history_records, history_statuses)
    seeds = build_seeds(delta_metrics, first_year)
    snapshots = sorted(snapshots, key=lambda item: item["file"])
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
        raise RuntimeError(
            f"results file would exceed 300 KB ({len(rendered.encode('utf-8'))} bytes)"
        )
    if (
        args.offline
        and previous.get("seeds") != seeds
        and not uses_legacy_history_date_rule(previous)
    ):
        raise RuntimeError(
            "offline recomputation differs from online OFAC seed results"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    for item in snapshots:
        print(f"{item['sha256']}  {item['file']}")
    print(f"snapshot_sha256  {snapshot_hash}")
    for seed_id, seed in seeds.items():
        print(f"{seed_id}: {seed['value']}")
        print(
            f"{seed_id} history_start={seed['detail']['history_coverage_start']} "
            f"membership_fallbacks={seed['detail']['sdn_membership_classification']['publications_using_all_adds_fallback']}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"OFAC resolver failed: {exc}", file=sys.stderr)
        raise
