#!/usr/bin/env python3
"""Resolve HHS OCR breach counts from the portal's two CSV lists.

Source: HHS Office for Civil Rights breach portal for breaches of unsecured
protected health information affecting 500 or more individuals.
Canonical URL: https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf
Filters: hhs-ocr-breaches-2026q3 counts each unique union row with Breach
Submission Date 2026-07-01..2026-09-30; hhs-ocr-breaches-2026q1 counts each
unique union row dated 2026-01-01..2026-03-31; hhs-ocr-largest-breach-2026q3
takes the maximum Individuals Affected among Q3 union rows. Date field:
Breach Submission Date (MM/DD/YYYY).
Inclusion/exclusion: combine Under Investigation and Archive CSVs and drop
exact duplicate rows across all columns; a non-exact repeat is retained.
Snapshot rule: save every landing, navigation, AJAX, export, redirect, and
transient HTTP response as exact bytes; discover all JSF/PrimeFaces component
ids from the live page at runtime. Known lags/revisions: OCR posts reports
after submission, so histories use the per-seed lag-guard periods. The current
under-investigation list is limited to reports from the last 24 months; archive
records closed cases and both lists can change as OCR updates case status.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

SOURCE_ID = "hhs_ocr"
SCRIPT_NAME = "hhs_ocr.py"
LANDING_URL = "https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf"
USER_AGENT = "AIUC calibration-seed resolver (research)"
RETRY_DELAYS = (2, 4, 8, 16, 32)
SEEDS: list[dict[str, Any]] = [
    {
        "id": "hhs-ocr-breaches-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"submission_date": "2026-Q3", "individuals_affected": "any"},
        "history_granularity": "month",
        "history_end": "2026-05",
    },
    {
        "id": "hhs-ocr-breaches-2026q1",
        "role": "control",
        "window_start": "2026-01-01",
        "window_end": "2026-03-31",
        "filter": {"submission_date": "2026-Q1", "individuals_affected": "any"},
        "history_granularity": "month",
        "history_end": "2025-11",
    },
    {
        "id": "hhs-ocr-largest-breach-2026q3",
        "role": "secondary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {
            "submission_date": "2026-Q3",
            "aggregate": "maximum individuals affected",
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


class HHSClient:
    """Requests session that saves each HTTP hop and waits two seconds between requests."""

    def __init__(self, session, data_dir, snapshots):
        self.session = session
        self.data_dir = data_dir
        self.snapshots = snapshots
        self.last_started = None

    def request(
        self, method, url, stem, extension, *, files=None, data=None, headers=None
    ):
        current_url = url
        current_method = method.upper()
        current_files = files
        current_data = data
        current_headers = dict(headers or {})
        redirects = 0
        attempt = 0
        while attempt <= len(RETRY_DELAYS):
            if self.last_started is not None:
                remaining = 2.0 - (time.monotonic() - self.last_started)
                if remaining > 0:
                    time.sleep(remaining)
            self.last_started = time.monotonic()
            try:
                response = self.session.request(
                    current_method,
                    current_url,
                    files=current_files,
                    data=current_data,
                    headers=current_headers,
                    timeout=120,
                    allow_redirects=False,
                )
            except requests.exceptions.ProxyError as exc:
                if is_connect_403(exc):
                    raise SystemExit(
                        f"proxy refused HTTPS CONNECT to ocrportal.hhs.gov (HTTP 403): {exc}"
                    ) from exc
                if attempt >= len(RETRY_DELAYS):
                    raise SystemExit(
                        f"HHS request failed after retries: {exc}"
                    ) from exc
                time.sleep(RETRY_DELAYS[attempt])
                attempt += 1
                continue
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as exc:
                if attempt >= len(RETRY_DELAYS):
                    raise SystemExit(
                        f"HHS request failed after retries: {exc}"
                    ) from exc
                time.sleep(RETRY_DELAYS[attempt])
                attempt += 1
                continue

            status = response.status_code
            if status in (301, 302, 303, 307, 308):
                suffix = f"_retry{attempt}" if attempt else ""
                filename = f"{stem}{suffix}_redirect{redirects}_http{status}.bin"
                record_response(
                    self.data_dir,
                    self.snapshots,
                    filename,
                    response.request.url,
                    response.content,
                )
                location = response.headers.get("Location")
                if not location:
                    raise SystemExit(
                        f"HHS redirect from {response.url} did not include Location"
                    )
                if redirects >= 10:
                    raise SystemExit("HHS exceeded ten redirects")
                current_url = urljoin(response.url, location)
                if status in (301, 302, 303) and current_method != "HEAD":
                    current_method = "GET"
                    current_files = None
                    current_data = None
                redirects += 1
                continue

            if status == 429 or status >= 500:
                filename = f"{stem}_retry{attempt}_http{status}.bin"
                record_response(
                    self.data_dir,
                    self.snapshots,
                    filename,
                    response.request.url,
                    response.content,
                )
                if attempt >= len(RETRY_DELAYS):
                    raise SystemExit(f"HHS returned HTTP {status} after retries")
                time.sleep(RETRY_DELAYS[attempt])
                attempt += 1
                continue
            if status == 403:
                filename = f"{stem}_http403.bin"
                record_response(
                    self.data_dir,
                    self.snapshots,
                    filename,
                    response.request.url,
                    response.content,
                )
                raise SystemExit(
                    "ocrportal.hhs.gov returned HTTP 403; stopping without further requests"
                )
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                filename = f"{stem}_http{status}.bin"
                record_response(
                    self.data_dir,
                    self.snapshots,
                    filename,
                    response.request.url,
                    response.content,
                )
                raise SystemExit(f"HHS request failed: {exc}") from exc

            filename = f"{stem}{extension}"
            record_response(
                self.data_dir,
                self.snapshots,
                filename,
                response.request.url,
                response.content,
            )
            return response
        raise SystemExit("HHS retry loop ended unexpectedly")


def page_soup(response):
    return BeautifulSoup(response.content, "html.parser")


def get_viewstate(soup):
    form = soup.find("form", id="ocrForm")
    if form is None:
        raise SystemExit("HHS page has no ocrForm form")
    field = form.find("input", attrs={"name": "javax.faces.ViewState"})
    if field is None or not field.get("value"):
        raise SystemExit("HHS ocrForm has no javax.faces.ViewState")
    return field["value"], form


def mojarra_field(anchor):
    onclick = anchor.get("onclick", "")
    match = re.search(r"\{\s*['\"]([^'\"]+)['\"]\s*:\s*['\"]([^'\"]+)['\"]", onclick)
    if not match:
        raise SystemExit(
            f"could not discover JSF command field from onclick={onclick!r}"
        )
    key, value = match.groups()
    if key != value:
        raise SystemExit(f"unexpected JSF command mapping {key!r}: {value!r}")
    return key, value


def multipart_fields(mapping):
    return {key: (None, str(value)) for key, value in mapping.items()}


def find_report_link(soup):
    for anchor in soup.find_all("a"):
        text = " ".join(anchor.stripped_strings)
        if text == "View HIPAA Breach Reports" and "mojarra.jsfcljs" in anchor.get(
            "onclick", ""
        ):
            return anchor
    raise SystemExit("could not find the 'View HIPAA Breach Reports' JSF link")


def find_export_anchor(soup):
    image = soup.find("img", attrs={"title": "Export as CSV"})
    if image is None or image.find_parent("a") is None:
        raise SystemExit("could not find the CSV export link")
    anchor = image.find_parent("a")
    if "mojarra.jsfcljs" not in anchor.get("onclick", ""):
        raise SystemExit("CSV export anchor has no mojarra.jsfcljs command")
    return anchor


def discover_tabview(soup):
    for div in soup.find_all("div", attrs={"data-widget": True}):
        if not str(div.get("data-widget", "")).startswith("widget_ocrForm_"):
            continue
        if div.find("a", href=re.compile("underInvestigationTab")) and div.find(
            "a", href=re.compile("archiveTab")
        ):
            tab_id = div.get("id")
            if tab_id:
                return tab_id
    raise SystemExit("could not discover PrimeFaces TabView id and its archive tabs")


def parse_partial_updates(raw):
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise SystemExit(
            f"HHS archive tab response was not valid JSF partial XML: {exc}"
        ) from exc
    result_markup = None
    state = None
    for update in root.findall(".//update"):
        update_id = update.attrib.get("id", "")
        text = update.text or ""
        if update_id == "ocrForm:results" or update_id.endswith(":results"):
            result_markup = text
        if "javax.faces.ViewState" in update_id and text.strip():
            state = text.strip()
    return result_markup, state


def parse_csv(raw, list_name):
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text, newline=""))
        records = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SystemExit(f"could not parse HHS {list_name} CSV: {exc}") from exc
    if not records:
        raise SystemExit(f"HHS {list_name} CSV was empty")
    if len(records[0]) != 9:
        raise SystemExit(
            f"HHS {list_name} CSV had {len(records[0])} columns, expected 9"
        )
    rows = []
    invalid_dates = []
    for line_number, values in enumerate(records[1:], start=2):
        if not values or (len(values) == 1 and values[0] == ""):
            continue
        if len(values) != 9:
            raise SystemExit(
                f"HHS {list_name} CSV row {line_number} had {len(values)} columns, expected 9"
            )
        row = tuple(values)
        try:
            day = datetime.strptime(row[4].strip(), "%m/%d/%Y").date()
        except ValueError:
            day = None
            invalid_dates.append(
                {"row": line_number, "name": row[0], "submission_date": row[4]}
            )
        try:
            affected_text = row[3].replace(",", "").strip()
            affected = int(affected_text) if affected_text else None
        except ValueError:
            affected = None
        rows.append({"raw": row, "day": day, "name": row[0], "affected": affected})
    return rows, invalid_dates


def quarter_label(day):
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


def quarter_periods(end_label):
    result = []
    for year in range(2024, 2027):
        for quarter in range(1, 5):
            label = f"{year}-Q{quarter}"
            if label <= end_label:
                result.append(label)
    return result


def month_periods(end_label):
    end_year, end_month = map(int, end_label.split("-"))
    year, month = 2024, 1
    result = []
    while (year, month) <= (end_year, end_month):
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return result


def month_breakdown(rows, start, end, filter_func):
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    current_year, current_month = first.year, first.month
    labels = []
    while (current_year, current_month) <= (last.year, last.month):
        labels.append(f"{current_year:04d}-{current_month:02d}")
        current_month += 1
        if current_month == 13:
            current_year += 1
            current_month = 1
    counts = Counter(
        row["day"].strftime("%Y-%m")
        for row in rows
        if row["day"] is not None and first <= row["day"] <= last and filter_func(row)
    )
    return {label: counts.get(label, 0) for label in labels}


def source_checks(under_rows, archive_rows, union_rows, invalid_under, invalid_archive):
    under_keys = Counter(
        (row["name"].strip(), row["day"])
        for row in under_rows
        if row["day"] is not None
    )
    archive_keys = Counter(
        (row["name"].strip(), row["day"])
        for row in archive_rows
        if row["day"] is not None
    )
    overlap = set(under_keys) & set(archive_keys)
    archive_dates = [row["day"] for row in archive_rows if row["day"] is not None]
    under_dates = [row["day"] for row in under_rows if row["day"] is not None]
    return {
        "under_investigation_rows": len(under_rows),
        "archive_rows": len(archive_rows),
        "union_rows_after_exact_deduplication": len(union_rows),
        "exact_duplicate_rows_removed": len(under_rows)
        + len(archive_rows)
        - len(union_rows),
        "cross_list_name_submission_date_keys": len(overlap),
        "under_rows_with_cross_list_key": sum(under_keys[key] for key in overlap),
        "archive_rows_with_cross_list_key": sum(archive_keys[key] for key in overlap),
        "under_earliest_submission_date": min(under_dates).isoformat()
        if under_dates
        else None,
        "archive_earliest_submission_date": min(archive_dates).isoformat()
        if archive_dates
        else None,
        "archive_csv_differs_from_under_investigation": bool(archive_rows)
        and archive_rows != under_rows,
        "invalid_under_investigation_dates": len(invalid_under),
        "invalid_archive_dates": len(invalid_archive),
    }


def build_seeds(rows, checks):
    output = {}
    for seed in SEEDS:
        start = date.fromisoformat(seed["window_start"])
        end = date.fromisoformat(seed["window_end"])
        threshold = seed["filter"].get("individuals_affected_minimum")
        is_max = seed["filter"].get("aggregate") == "maximum individuals affected"

        def selected(row, start=start, end=end, threshold=threshold):
            if row["day"] is None or not start <= row["day"] <= end:
                return False
            if threshold is not None:
                return row["affected"] is not None and row["affected"] >= threshold
            return True

        window_rows = [row for row in rows if selected(row)]
        if is_max:
            value = max(
                (row["affected"] for row in window_rows if row["affected"] is not None),
                default=None,
            )
        else:
            value = len(window_rows)

        granularity = seed["history_granularity"]
        period_labels = (
            month_periods(seed["history_end"])
            if granularity == "month"
            else quarter_periods(seed["history_end"])
        )
        history = []
        for label in period_labels:
            if granularity == "month":
                year, month = map(int, label.split("-"))
                period_start = date(year, month, 1)
                next_month = (
                    date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
                )
                period_end = date.fromordinal(next_month.toordinal() - 1)
            else:
                year_text, quarter_text = label.split("-Q")
                year, quarter = int(year_text), int(quarter_text)
                period_start = date(year, (quarter - 1) * 3 + 1, 1)
                period_end_month = quarter * 3
                period_end = date(
                    year,
                    period_end_month,
                    30 if period_end_month in (4, 6, 9, 11) else 31,
                )
            if period_end >= start:
                raise SystemExit(
                    f"history period {label} overlaps or follows window for {seed['id']}"
                )
            period_rows = [
                row
                for row in rows
                if row["day"] is not None and period_start <= row["day"] <= period_end
            ]
            if threshold is not None:
                period_rows = [
                    row
                    for row in period_rows
                    if row["affected"] is not None and row["affected"] >= threshold
                ]
            if is_max:
                value_for_period = max(
                    (
                        row["affected"]
                        for row in period_rows
                        if row["affected"] is not None
                    ),
                    default=None,
                )
            else:
                value_for_period = len(period_rows)
            history.append({"period": label, "value": value_for_period})

        detail = {
            "source_checks": checks,
            "history_vintage": {
                "method": "lag_guard",
                "last_period": seed["history_end"],
                "reason": f"OCR may post reports after submission, so this history ends at the specified lag-guard period {seed['history_end']}.",
            },
            "window_breakdown": month_breakdown(
                rows, seed["window_start"], seed["window_end"], selected
            ),
        }
        notes = (
            "Counts use the exact-deduplicated union of the current Under Investigation and Archive CSVs; "
            "all columns must match to drop a duplicate. Archive records can overlap by name and submission date."
        )
        output[seed["id"]] = {
            "role": seed["role"],
            "value": value,
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": granularity,
            "history": history,
            "detail": detail,
            "notes": notes,
        }
    return output


def download_lists(data_dir, snapshots):
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    client = HHSClient(session, data_dir, snapshots)

    landing = client.request("GET", LANDING_URL, "hhs_01_landing", ".html")
    landing_soup = page_soup(landing)
    viewstate, landing_form = get_viewstate(landing_soup)
    link = find_report_link(landing_soup)
    link_field, link_value = mojarra_field(link)
    landing_action = urljoin(landing.url, landing_form.get("action") or "")
    nav_fields = multipart_fields(
        {
            "ocrForm": "ocrForm",
            link_field: link_value,
            "javax.faces.ViewState": viewstate,
        }
    )
    report_page = client.request(
        "POST",
        landing_action,
        "hhs_02_open_reports",
        ".html",
        files=nav_fields,
        headers={"Referer": landing.url},
    )
    report_soup = page_soup(report_page)
    viewstate, report_form = get_viewstate(report_soup)
    report_action = urljoin(report_page.url, report_form.get("action") or "")
    export_anchor = find_export_anchor(report_soup)
    export_field, export_value = mojarra_field(export_anchor)
    under_fields = multipart_fields(
        {
            "ocrForm": "ocrForm",
            export_field: export_value,
            "javax.faces.ViewState": viewstate,
        }
    )
    under_csv_response = client.request(
        "POST",
        report_action,
        "hhs_03_under_investigation_csv",
        ".csv",
        files=under_fields,
        headers={"Referer": report_page.url},
    )
    under_csv = under_csv_response.content

    tab_id = discover_tabview(report_soup)
    archive_payload = [
        ("javax.faces.partial.ajax", "true"),
        ("javax.faces.source", tab_id),
        ("javax.faces.partial.execute", tab_id),
        ("javax.faces.partial.render", "ocrForm:breachReports ocrForm:results"),
        ("javax.faces.behavior.event", "tabChange"),
        ("javax.faces.partial.event", "tabChange"),
        (f"{tab_id}_newTab", f"{tab_id}:archiveTab"),
        (f"{tab_id}_tabindex", "1"),
        (f"{tab_id}_activeIndex", "1"),
        ("ocrForm", "ocrForm"),
        ("javax.faces.ViewState", viewstate),
    ]
    archive_tab_response = client.request(
        "POST",
        report_action,
        "hhs_04_archive_tab_ajax",
        ".xml",
        data=archive_payload,
        headers={"Faces-Request": "partial/ajax", "Referer": report_page.url},
    )
    archive_markup, archive_viewstate = parse_partial_updates(
        archive_tab_response.content
    )
    if archive_markup is None:
        raise SystemExit(
            "PrimeFaces archive tab response did not update ocrForm:results"
        )
    archive_results_soup = BeautifulSoup(archive_markup, "html.parser")
    archive_export_anchor = find_export_anchor(archive_results_soup)
    archive_export_field, archive_export_value = mojarra_field(archive_export_anchor)
    archive_fields = multipart_fields(
        {
            "ocrForm": "ocrForm",
            archive_export_field: archive_export_value,
            "javax.faces.ViewState": archive_viewstate or viewstate,
        }
    )
    archive_csv_response = client.request(
        "POST",
        report_action,
        "hhs_05_archive_csv",
        ".csv",
        files=archive_fields,
        headers={"Referer": report_page.url},
    )
    return under_csv, archive_csv_response.content


def main():
    parser = argparse.ArgumentParser(description="Resolve HHS OCR breach counts")
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
        under_path = data_dir / "hhs_03_under_investigation_csv.csv"
        archive_path = data_dir / "hhs_05_archive_csv.csv"
        if not under_path.is_file() or not archive_path.is_file():
            raise SystemExit("offline HHS CSV snapshots are missing")
        under_raw, archive_raw = under_path.read_bytes(), archive_path.read_bytes()
    else:
        snapshots = []
        under_raw, archive_raw = download_lists(data_dir, snapshots)
        if not snapshots:
            raise SystemExit("HHS did not produce any snapshots")
        retrieved_at = snapshots[0]["retrieved_at"]

    under_rows, invalid_under = parse_csv(under_raw, "Under Investigation")
    archive_rows, invalid_archive = parse_csv(archive_raw, "Archive")
    if under_raw == archive_raw:
        raise SystemExit("HHS Archive CSV is identical to Under Investigation CSV")
    combined = under_rows + archive_rows
    seen = set()
    union_rows = []
    for row in combined:
        if row["raw"] in seen:
            continue
        seen.add(row["raw"])
        union_rows.append(row)
    checks = source_checks(
        under_rows, archive_rows, union_rows, invalid_under, invalid_archive
    )
    checks["archive_csv_differs_from_under_investigation"] = under_raw != archive_raw
    archive_dates = [row["day"] for row in archive_rows if row["day"] is not None]
    seeds = build_seeds(union_rows, checks)

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
    print("source_checks=" + json.dumps(checks, sort_keys=True))
    print(
        "archive_validation="
        + json.dumps(
            {
                "archive_csv_differs": under_raw != archive_raw,
                "archive_rows": len(archive_rows),
                "archive_earliest_submission_date": min(archive_dates).isoformat()
                if archive_dates
                else None,
                "archive_contains_2009_submission": any(
                    day.year == 2009 for day in archive_dates
                ),
                "archive_has_thousands_of_rows": len(archive_rows) >= 1000,
            },
            sort_keys=True,
        )
    )
    for seed_id, value in seeds.items():
        print(f"{seed_id}: {value['value']}")


if __name__ == "__main__":
    main()
