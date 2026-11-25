#!/usr/bin/env python3
"""Resolve SEC EDGAR filing counts from monthly full-text search pages.

Source: SEC EDGAR full-text search (EFTS), canonical URL
https://efts.sec.gov/LATEST/search-index. For every month from 2024-01 through
2026-10, search with forms=8-K and dateRange=custom using _source.file_date as
the date field. The two existing seeds combine q="Item 1.05" and q="Material
Cybersecurity Incidents"; include only accessions whose _source.items contains
1.05 and whose _source.form is exactly 8-K. The added
sec-8k-cybersecurity-incident-2026q3 seed uses the exact phrase query
q="\"cybersecurity incident\"", with no items filter, and counts distinct
accessions with form exactly 8-K in 2026-Q3; its history uses the same filter
for 2024-Q1 through 2026-Q2. forms=8-K selects the root form and can return
8-K/A hits, which are excluded from values but counted for comparison. A filing
with multiple document hits is counted once by accession. Snapshot rule: save
each exact response page as bytes, including redirects or transient HTTP error
bodies; paginate with from/size and reconcile retrieved hit counts with
hits.total. These are publication-dated records; late submissions, amendments,
and search-index revisions can change counts, and search indexing may lag.
"""

import argparse
import calendar
import hashlib
import json
import os
import re
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "sec_edgar"
SCRIPT_NAME = "sec_edgar.py"
SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
RETRY_DELAYS = (2, 4, 8, 16, 32)
PAGE_SIZE = 100
MAX_RESULT_BYTES = 300 * 1024
QUERIES = [
    ("item105", "Item 1.05"),
    ("material", "Material Cybersecurity Incidents"),
    ("cybersecurity_incident", '"cybersecurity incident"'),
]
SEEDS: list[dict[str, Any]] = [
    {
        "id": "sec-8k-item105-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"form": "8-K exactly", "items_contains": "1.05"},
        "history_granularity": "quarter",
    },
    {
        "id": "sec-8k-item105-2026-10",
        "role": "primary",
        "window_start": "2026-10-01",
        "window_end": "2026-10-31",
        "filter": {"form": "8-K exactly", "items_contains": "1.05"},
        "history_granularity": "month",
    },
    {
        "id": "sec-8k-cybersecurity-incident-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {
            "form": "8-K exactly",
            "exact_phrase": "cybersecurity incident",
            "items_filter": None,
        },
        "history_granularity": "quarter",
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


def sec_api_error(content):
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("errorType"):
        return None
    error_type = str(payload.get("errorType", ""))
    error_message = str(payload.get("errorMessage", ""))
    combined = f"{error_type} {error_message}".casefold()
    transient = any(
        token in combined
        for token in (
            "connectionerror",
            "timeout",
            "timed out",
            "getaddrinfo",
            "ebusy",
            "econnreset",
            "temporarily",
        )
    )
    return error_type, error_message, transient


def request_json(session, params, filename, data_dir, snapshots, request_clock):
    for attempt in range(len(RETRY_DELAYS) + 1):
        if request_clock[0] is not None:
            wait = 0.3 - (time.monotonic() - request_clock[0])
            if wait > 0:
                time.sleep(wait)
        request_clock[0] = time.monotonic()
        try:
            response = session.get(
                SEARCH_URL, params=params, timeout=120, allow_redirects=True
            )
        except requests.exceptions.ProxyError as exc:
            if is_connect_403(exc):
                raise SystemExit(
                    f"proxy refused HTTPS CONNECT to efts.sec.gov (HTTP 403): {exc}"
                ) from exc
            if attempt >= len(RETRY_DELAYS):
                raise SystemExit(f"SEC request failed after retries: {exc}") from exc
            time.sleep(RETRY_DELAYS[attempt])
            continue
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt >= len(RETRY_DELAYS):
                raise SystemExit(f"SEC request failed after retries: {exc}") from exc
            time.sleep(RETRY_DELAYS[attempt])
            continue

        application_error = (
            sec_api_error(response.content) if response.status_code < 400 else None
        )
        chain = [*list(response.history), response]
        for i, item in enumerate(chain):
            if item.status_code in (301, 302, 303, 307, 308):
                name = (
                    filename[:-5]
                    + f"_attempt{attempt}_redirect{i}_http{item.status_code}.bin"
                )
            elif item.status_code >= 400:
                name = filename[:-5] + f"_attempt{attempt}_http{item.status_code}.bin"
            elif item is response and application_error and application_error[2]:
                name = filename[:-5] + f"_attempt{attempt}_application_error.bin"
            else:
                name = (
                    filename
                    if i == len(chain) - 1
                    else filename[:-5]
                    + f"_attempt{attempt}_redirect{i}_http{item.status_code}.bin"
                )
            record_response(data_dir, snapshots, name, item.request.url, item.content)

        if response.status_code == 403:
            raise SystemExit(
                "efts.sec.gov returned HTTP 403; stopping without further requests"
            )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt >= len(RETRY_DELAYS):
                raise SystemExit(
                    f"SEC returned HTTP {response.status_code} after retries"
                )
            time.sleep(RETRY_DELAYS[attempt])
            continue
        if application_error:
            error_type, error_message, transient = application_error
            if transient:
                if attempt >= len(RETRY_DELAYS):
                    raise SystemExit(
                        f"SEC EFTS returned transient API error after retries: {error_type}: {error_message}"
                    )
                time.sleep(RETRY_DELAYS[attempt])
                continue
            raise SystemExit(
                f"SEC EFTS returned API error: {error_type}: {error_message}"
            )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise SystemExit(f"SEC request failed: {exc}") from exc
        return response
    raise SystemExit("SEC request retry loop ended unexpectedly")


def month_range():
    result = []
    year, month = 2024, 1
    while (year, month) <= (2026, 10):
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return result


def get_total(payload):
    hits = payload.get("hits")
    if not isinstance(hits, dict):
        raise SystemExit("SEC JSON response has no hits object")
    total = hits.get("total")
    if isinstance(total, dict):
        relation = total.get("relation")
        if relation not in (None, "eq"):
            raise SystemExit(
                f"SEC hits.total relation is {relation!r}; cannot verify complete pagination"
            )
        value = total.get("value")
    else:
        value = total
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"SEC hits.total is not an integer: {total!r}") from exc


def fetch_query(
    session, query_kind, query_text, month, data_dir, snapshots, request_clock
):
    year, month_number = map(int, month.split("-"))
    first_day = date(year, month_number, 1)
    last_day = date(year, month_number, calendar.monthrange(year, month_number)[1])
    offset = 0
    total = None
    all_hits = []
    page_lengths = []
    while True:
        filename = f"{query_kind}_{month}_from_{offset:06d}.json"
        params = {
            "q": query_text,
            # SEC EFTS maps this to root_forms=8-K, including amendments. The
            # comma-separated value in the source spec becomes an AND filter.
            "forms": "8-K",
            "dateRange": "custom",
            "startdt": first_day.isoformat(),
            "enddt": last_day.isoformat(),
            "from": offset,
            "size": PAGE_SIZE,
        }
        response = request_json(
            session, params, filename, data_dir, snapshots, request_clock
        )
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise SystemExit(f"SEC page {filename} was not valid JSON: {exc}") from exc
        page_total = get_total(payload)
        if total is None:
            total = page_total
        elif page_total != total:
            raise SystemExit(
                f"SEC hits.total changed within {query_kind} {month}: {total} -> {page_total}"
            )
        page_hits = payload.get("hits", {}).get("hits")
        if not isinstance(page_hits, list):
            raise SystemExit(f"SEC page {filename} has no hits list")
        if len(page_hits) > PAGE_SIZE:
            raise SystemExit(
                f"SEC page {filename} returned {len(page_hits)} hits, above requested page size {PAGE_SIZE}"
            )
        page_lengths.append(len(page_hits))
        all_hits.extend(page_hits)
        if len(all_hits) >= total:
            break
        if not page_hits:
            raise SystemExit(
                f"SEC pagination stopped early at {len(all_hits)} of {total} hits for {query_kind} {month}"
            )
        offset += len(page_hits)
    if len(all_hits) != total:
        raise SystemExit(
            f"SEC retrieved {len(all_hits)} hits but hits.total reported {total} for {query_kind} {month}"
        )
    hit_ids = [hit.get("_id") for hit in all_hits]
    if len(hit_ids) != len(set(hit_ids)):
        raise SystemExit(
            f"duplicate document _id across pages for {query_kind} {month}; pagination may have overlapped"
        )
    return all_hits, {
        "total": total,
        "retrieved": len(all_hits),
        "page_lengths": page_lengths,
    }


def has_item_105(items):
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        return False
    for item in items:
        text = str(item).strip()
        text = re.sub(r"^item\s+", "", text, flags=re.IGNORECASE)
        if re.search(r"(?<![\d.])1\.05(?!\d)", text):
            return True
    return False


def clean_company_name(value):
    text = re.sub(r"\s*\(CIK\s+\d+\)\s*$", "", str(value)).strip()
    return re.sub(r"\s+", " ", text)


def normalize_filings(all_hits_by_query, require_item_105=True):
    filings = {}
    item_hits = 0
    for hits in all_hits_by_query:
        for hit in hits:
            source = hit.get("_source") or {}
            items = source.get("items") or []
            if isinstance(items, str):
                items = [items]
            if require_item_105 and not has_item_105(items):
                continue
            if has_item_105(items):
                item_hits += 1
            adsh = str(
                source.get("adsh") or str(hit.get("_id", "")).split(":", 1)[0]
            ).strip()
            if not adsh:
                raise SystemExit(
                    f"SEC Item 1.05 hit has no accession number: {hit.get('_id')!r}"
                )
            form = str(source.get("form") or "").strip()
            file_date = str(source.get("file_date") or "").strip()
            try:
                parsed_date = date.fromisoformat(file_date)
            except ValueError as exc:
                raise SystemExit(
                    f"SEC filing {adsh} has invalid file_date {file_date!r}"
                ) from exc
            ciks = source.get("ciks") or []
            if isinstance(ciks, str):
                ciks = [ciks]
            display_names = source.get("display_names") or []
            if isinstance(display_names, str):
                display_names = [display_names]
            company_names = [
                clean_company_name(name) for name in display_names if str(name).strip()
            ]
            if adsh not in filings:
                filings[adsh] = {
                    "adsh": adsh,
                    "form": form,
                    "file_date": parsed_date.isoformat(),
                    "items": set(),
                    "ciks": set(),
                    "company_names": set(),
                }
            filing = filings[adsh]
            if filing["form"] != form or filing["file_date"] != parsed_date.isoformat():
                raise SystemExit(
                    f"SEC hits for accession {adsh} disagree on form or file_date"
                )
            filing["items"].update(str(item).strip() for item in items)
            filing["ciks"].update(str(cik).strip() for cik in ciks if str(cik).strip())
            filing["company_names"].update(company_names)
    normalized = []
    for filing in filings.values():
        normalized.append(
            {
                "adsh": filing["adsh"],
                "form": filing["form"],
                "file_date": filing["file_date"],
                "items": sorted(filing["items"]),
                "ciks": sorted(filing["ciks"]),
                "company_names": sorted(filing["company_names"]),
            }
        )
    normalized.sort(key=lambda row: (row["file_date"], row["adsh"]))
    return normalized, item_hits


def period_label(day, granularity):
    if granularity == "month":
        return day.strftime("%Y-%m")
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


def history_periods(granularity):
    if granularity == "month":
        labels = []
        year, month = 2024, 1
        while (year, month) <= (2026, 6):
            labels.append(f"{year:04d}-{month:02d}")
            month += 1
            if month == 13:
                year, month = year + 1, 1
        return labels
    return [
        f"{year}-Q{quarter}"
        for year in range(2024, 2027)
        for quarter in range(1, 5)
        if f"{year}-Q{quarter}" <= "2026-Q2"
    ]


def build_seeds(item105_filings, cybersecurity_filings, query_checks):
    output = {}
    for seed in SEEDS:
        start = date.fromisoformat(seed["window_start"])
        end = date.fromisoformat(seed["window_end"])
        granularity = seed["history_granularity"]
        is_cybersecurity_seed = seed["id"] == "sec-8k-cybersecurity-incident-2026q3"
        filings = cybersecurity_filings if is_cybersecurity_seed else item105_filings
        query_prefix = "cybersecurity_incident:" if is_cybersecurity_seed else None
        seed_query_checks = {
            key: value
            for key, value in query_checks.items()
            if (
                key.startswith(query_prefix)
                if query_prefix
                else not key.startswith("cybersecurity_incident:")
            )
        }
        qualifying = [
            filing
            for filing in filings
            if filing["form"] == "8-K"
            and start <= date.fromisoformat(filing["file_date"]) <= end
            and (is_cybersecurity_seed or has_item_105(filing["items"]))
        ]
        amendments = [
            filing
            for filing in filings
            if filing["form"] == "8-K/A"
            and start <= date.fromisoformat(filing["file_date"]) <= end
            and (is_cybersecurity_seed or has_item_105(filing["items"]))
        ]
        history = []
        for label in history_periods(granularity):
            count = sum(
                1
                for filing in filings
                if filing["form"] == "8-K"
                and (is_cybersecurity_seed or has_item_105(filing["items"]))
                and period_label(date.fromisoformat(filing["file_date"]), granularity)
                == label
                and date.fromisoformat(filing["file_date"]) < start
            )
            history.append({"period": label, "value": count})
        window_months = []
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            window_months.append(f"{y:04d}-{m:02d}")
            m += 1
            if m == 13:
                y, m = y + 1, 1
        detail = {
            "window_breakdown": {
                month: sum(
                    1 for filing in qualifying if filing["file_date"].startswith(month)
                )
                for month in window_months
            },
            "history_vintage": {
                "method": "publication_dated",
                "note": "Counts use SEC file_date; later amendments, late indexing, or EDGAR search-index revisions can change historical search results.",
            },
            "pagination_queries": len(seed_query_checks),
            "pagination_pages": sum(
                len(check["page_lengths"]) for check in seed_query_checks.values()
            ),
            "pagination_reported_hits": sum(
                check["total"] for check in seed_query_checks.values()
            ),
            "pagination_retrieved_hits": sum(
                check["retrieved"] for check in seed_query_checks.values()
            ),
        }
        if is_cybersecurity_seed:
            item_105_count = sum(
                1 for filing in qualifying if has_item_105(filing["items"])
            )
            detail.update(
                {
                    "exact_phrase": "cybersecurity incident",
                    "form_filter": "_source.form exactly 8-K",
                    "items_filter": None,
                    "eight_k_a_filings_matching_phrase": len(amendments),
                    "item_1_05_filings": item_105_count,
                    "item_1_05_share": round(item_105_count / len(qualifying), 6)
                    if qualifying
                    else None,
                    "item_1_05_share_percent": round(
                        100 * item_105_count / len(qualifying), 4
                    )
                    if qualifying
                    else None,
                    "item_1_05_share_denominator": "distinct original Form 8-K accessions in the Q3 exact-phrase count",
                }
            )
        else:
            detail.update(
                {
                    "eight_k_a_filings_with_item_1_05": len(amendments),
                    "filings": [
                        {
                            "adsh": filing["adsh"],
                            "form": filing["form"],
                            "file_date": filing["file_date"],
                            "items": filing["items"],
                            "ciks": filing["ciks"],
                            "company_names": filing["company_names"],
                        }
                        for filing in qualifying
                    ],
                }
            )
        output[seed["id"]] = {
            "role": seed["role"],
            "value": len(qualifying),
            "window_start": seed["window_start"],
            "window_end": seed["window_end"],
            "history_granularity": granularity,
            "history": history,
            "detail": detail,
            "notes": (
                "Count distinct accessions matching the exact cybersecurity incident phrase query; include only _source.form exactly 8-K and apply no _source.items filter."
                if is_cybersecurity_seed
                else "Count distinct accessions with _source.items containing 1.05; exclude 8-K/A from the value. SEC EFTS forms=8-K selects root form 8-K and includes amendments for comparison."
            ),
        }
    return output


def parse_saved_pages(data_dir, snapshots):
    grouped_hits = {query_kind: [] for query_kind, _query_text in QUERIES}
    query_checks = {}
    for query_kind, _query_text in QUERIES:
        for month in month_range():
            page_paths = []
            for item in snapshots:
                filename = item["file"]
                if not filename.endswith(".json"):
                    continue
                if Path(filename).name.startswith(f"{query_kind}_{month}_from_"):
                    match = re.search(r"_from_(\d+)\.json$", filename)
                    if match:
                        page_paths.append((int(match.group(1)), data_dir / filename))
            if not page_paths:
                raise SystemExit(f"no saved SEC pages for {query_kind} {month}")
            page_paths.sort(key=lambda pair: pair[0])
            hits_for_query = []
            reported_total = None
            expected_offset = 0
            lengths = []
            for offset, path in page_paths:
                if offset != expected_offset:
                    raise SystemExit(
                        f"SEC offline page gap for {query_kind} {month}: expected from={expected_offset}, got {offset}"
                    )
                try:
                    payload = json.loads(path.read_bytes().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SystemExit(f"invalid saved SEC page {path}: {exc}") from exc
                page_total = get_total(payload)
                if reported_total is None:
                    reported_total = page_total
                elif page_total != reported_total:
                    raise SystemExit(
                        f"saved SEC pages disagree on hits.total for {query_kind} {month}"
                    )
                page_hits = payload.get("hits", {}).get("hits")
                if not isinstance(page_hits, list):
                    raise SystemExit(f"saved SEC page {path} has no hits list")
                hits_for_query.extend(page_hits)
                lengths.append(len(page_hits))
                expected_offset += len(page_hits)
            if expected_offset != reported_total:
                raise SystemExit(
                    f"offline SEC retrieved {expected_offset} of {reported_total} hits for {query_kind} {month}"
                )
            grouped_hits[query_kind].append(hits_for_query)
            query_checks[f"{query_kind}:{month}"] = {
                "total": reported_total,
                "retrieved": expected_offset,
                "page_lengths": lengths,
            }
    return grouped_hits, query_checks


def main():
    parser = argparse.ArgumentParser(
        description="Resolve SEC Item 1.05 8-K filing counts"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "")
    if not user_agent.strip():
        raise SystemExit(
            "SEC_EDGAR_USER_AGENT must be set to a non-empty requester/contact User-Agent"
        )
    data_dir = args.data_dir / SOURCE_ID
    prepare_directory(data_dir, args.offline)
    previous = load_offline_manifest(args.out, data_dir) if args.offline else None

    if args.offline:
        assert previous is not None
        snapshots = previous["snapshots"]
        retrieved_at = previous["retrieved_at"]
    else:
        snapshots = []
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})
        request_clock = [None]
        for query_kind, query_text in QUERIES:
            for month in month_range():
                fetch_query(
                    session,
                    query_kind,
                    query_text,
                    month,
                    data_dir,
                    snapshots,
                    request_clock,
                )
        if not snapshots:
            raise SystemExit("SEC search produced no snapshots")
        retrieved_at = snapshots[0]["retrieved_at"]

    hits_by_query, query_checks = parse_saved_pages(data_dir, snapshots)
    item105_filings, item_hits = normalize_filings(
        hits_by_query["item105"] + hits_by_query["material"], require_item_105=True
    )
    cybersecurity_filings, _ = normalize_filings(
        hits_by_query["cybersecurity_incident"], require_item_105=False
    )
    seeds = build_seeds(item105_filings, cybersecurity_filings, query_checks)

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
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if len(rendered.encode("utf-8")) >= MAX_RESULT_BYTES:
        raise SystemExit(
            f"results file would exceed 300 KB ({len(rendered.encode('utf-8'))} bytes)"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    for item in sorted(snapshots, key=lambda item: item["file"]):
        print(f"{item['sha256']}  {item['file']}")
    print(f"snapshot_sha256  {snapshot_sha}")
    print(
        f"queries={len(query_checks)} pages={sum(len(check['page_lengths']) for check in query_checks.values())} "
        f"reported_hits={sum(check['total'] for check in query_checks.values())} "
        f"retrieved_hits={sum(check['retrieved'] for check in query_checks.values())} "
        f"item_1_05_document_hits={item_hits} unique_item_1_05_filings={len(item105_filings)} "
        f"unique_cybersecurity_phrase_filings={len(cybersecurity_filings)}"
    )
    for seed_id, value in seeds.items():
        if "item_1_05_share" in value["detail"]:
            print(
                f"{seed_id}: {value['value']} "
                f"(Item 1.05={value['detail']['item_1_05_filings']}/"
                f"{value['value']}, share={value['detail']['item_1_05_share_percent']}%)"
            )
        else:
            print(
                f"{seed_id}: {value['value']} "
                f"(8-K/A comparison={value['detail']['eight_k_a_filings_with_item_1_05']})"
            )
            print(
                f"{seed_id} filings="
                + json.dumps(
                    [
                        {
                            "file_date": filing["file_date"],
                            "company_names": filing["company_names"],
                        }
                        for filing in value["detail"]["filings"]
                    ],
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
