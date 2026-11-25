#!/usr/bin/env python3
"""Resolve Federal Register document-count seeds from API v1.

Canonical source: https://www.federalregister.gov/api/v1/documents.json .
The primary seed fedreg-ai-documents-2026q3 uses the exact search term
\"artificial intelligence\" (including its double quotes); the primary seed
fedreg-cybersecurity-documents-2026q3 uses the unquoted term cybersecurity.
The date field is publication_date. All document types are included, with no
other filters, over inclusive calendar months 2024-01 through 2026-09. Each
month/term request uses per_page=1 and the publication_date gte/lte conditions;
the count field is the monthly value. Raw API responses, including a separate
unquoted artificial-intelligence comparison response for 2026-07, are saved
byte-for-byte. Search-index counts can be revised, so values are as of retrieval.
"""

import argparse
import calendar
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "federal_register"
SCRIPT_NAME = "federal_register.py"
API_URL = "https://www.federalregister.gov/api/v1/documents.json"
USER_AGENT = "AIUC calibration-seed resolver (research)"
RETRY_DELAYS = (2, 4, 8, 16, 32)
START_MONTH = (2024, 1)
LAST_MONTH = (2026, 9)
WINDOW_MONTHS = ((2026, 7), (2026, 8), (2026, 9))
PHRASE_CHECK_MONTH = (2024, 6)

SEEDS: list[dict[str, Any]] = [
    {
        "id": "fedreg-ai-documents-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"term": '"artificial intelligence"'},
    },
    {
        "id": "fedreg-cybersecurity-documents-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"term": "cybersecurity"},
    },
]


def utc_now():
    return (
        dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def get_json(session, params):
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            response = session.get(API_URL, params=params, timeout=(15, 90))
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt == len(RETRY_DELAYS):
                raise RuntimeError(
                    f"Federal Register request failed after retries: {exc}"
                ) from exc
            time.sleep(RETRY_DELAYS[attempt])
            continue

        if response.status_code == 403:
            raise RuntimeError(
                f"Federal Register host refused the request with HTTP 403: {response.url}"
            )
        if response.status_code == 429 or 500 <= response.status_code <= 599:
            if attempt == len(RETRY_DELAYS):
                raise RuntimeError(
                    f"Federal Register request failed after retries with HTTP {response.status_code}: {response.url}"
                )
            response.close()
            time.sleep(RETRY_DELAYS[attempt])
            continue
        if not 200 <= response.status_code <= 299:
            raise RuntimeError(
                f"Federal Register request failed with HTTP {response.status_code}: {response.url}"
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
        # Reconstruct deterministic URL metadata when raw responses are available
        # but the result file has not yet been written. File mtimes give UTC seconds.
        snapshots = []
        filenames = [
            f"month__{seed['id']}__{year:04d}-{month:02d}.json"
            for seed in SEEDS
            for year, month in month_list()
        ]
        filenames.append(
            f"verify__ai-unquoted-{PHRASE_CHECK_MONTH[0]:04d}-{PHRASE_CHECK_MONTH[1]:02d}.json"
        )
        for filename in filenames:
            path = root / filename
            if not path.is_file():
                raise RuntimeError(f"offline snapshot is missing: {path}")
            if filename.startswith("verify__"):
                start, end = month_bounds(*PHRASE_CHECK_MONTH)
                term = "artificial intelligence"
            else:
                _, seed_id, period = filename[:-5].split("__", 2)
                year, month = map(int, period.split("-"))
                seed = next(item for item in SEEDS if item["id"] == seed_id)
                term = seed["filter"]["term"]
                start, end = month_bounds(year, month)
            params = {
                "per_page": 1,
                "conditions[term]": term,
                "conditions[publication_date][gte]": start,
                "conditions[publication_date][lte]": end,
            }
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
                    "file": filename,
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


def month_list():
    y, m = START_MONTH
    last_y, last_m = LAST_MONTH
    values = []
    while (y, m) <= (last_y, last_m):
        values.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return values


def month_bounds(year, month):
    start = dt.date(year, month, 1)
    end = dt.date(year, month, calendar.monthrange(year, month)[1])
    return start.isoformat(), end.isoformat()


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


def quarter_history(month_counts):
    values = []
    for year in range(2024, 2027):
        last_q = 2 if year == 2026 else 4
        for quarter in range(1, last_q + 1):
            months = (3 * quarter - 2, 3 * quarter - 1, 3 * quarter)
            value = sum(month_counts[(year, month)] for month in months)
            values.append({"period": f"{year:04d}-Q{quarter}", "value": value})
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Resolve Federal Register document-count seeds"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    root = args.data_dir / SOURCE_ID
    if args.offline:
        snapshots, payloads = load_offline_snapshots(root, args.out)
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"refusing to download into non-empty directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        snapshots = []
        payloads = {}
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        for seed in SEEDS:
            seed_id = seed["id"]
            term = seed["filter"]["term"]
            for year, month in month_list():
                start, end = month_bounds(year, month)
                params = {
                    "per_page": 1,
                    "conditions[term]": term,
                    "conditions[publication_date][gte]": start,
                    "conditions[publication_date][lte]": end,
                }
                body, url, retrieved_at = get_json(session, params)
                filename = f"month__{seed_id}__{year:04d}-{month:02d}.json"
                payloads[filename] = save_snapshot(
                    root, filename, url, body, retrieved_at, snapshots
                )

        compare_start, compare_end = month_bounds(*PHRASE_CHECK_MONTH)
        compare_params = {
            "per_page": 1,
            "conditions[term]": "artificial intelligence",
            "conditions[publication_date][gte]": compare_start,
            "conditions[publication_date][lte]": compare_end,
        }
        body, url, retrieved_at = get_json(session, compare_params)
        compare_name = f"verify__ai-unquoted-{PHRASE_CHECK_MONTH[0]:04d}-{PHRASE_CHECK_MONTH[1]:02d}.json"
        payloads[compare_name] = save_snapshot(
            root, compare_name, url, body, retrieved_at, snapshots
        )

    month_counts = {seed["id"]: {} for seed in SEEDS}
    for seed in SEEDS:
        for year, month in month_list():
            filename = f"month__{seed['id']}__{year:04d}-{month:02d}.json"
            if filename not in payloads:
                raise RuntimeError(
                    f"missing required Federal Register monthly response: {filename}"
                )
            payload = payloads[filename]
            try:
                count = int(payload["count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Federal Register response lacks integer count: {filename}"
                ) from exc
            if count < 0:
                raise RuntimeError(
                    f"Federal Register returned a negative count: {filename}"
                )
            month_counts[seed["id"]][(year, month)] = count

    ai_id = SEEDS[0]["id"]
    quoted_count = month_counts[ai_id][PHRASE_CHECK_MONTH]
    compare_name = f"verify__ai-unquoted-{PHRASE_CHECK_MONTH[0]:04d}-{PHRASE_CHECK_MONTH[1]:02d}.json"
    if compare_name not in payloads:
        raise RuntimeError(
            f"missing unquoted Federal Register comparison response: {compare_name}"
        )
    try:
        unquoted_count = int(payloads[compare_name]["count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Federal Register unquoted comparison response lacks an integer count"
        ) from exc
    phrase_comparison = {
        "month": f"{PHRASE_CHECK_MONTH[0]:04d}-{PHRASE_CHECK_MONTH[1]:02d}",
        "quoted_term": '"artificial intelligence"',
        "quoted_count": quoted_count,
        "unquoted_term": "artificial intelligence",
        "unquoted_count": unquoted_count,
        "counts_differ": quoted_count != unquoted_count,
        "interpretation": (
            "The quoted search returned fewer results than the unquoted search, consistent with double quotes requiring an exact phrase match."
            if quoted_count < unquoted_count
            else "The quoted and unquoted searches returned equal counts for this month; this comparison alone does not establish that quotes narrow to a phrase."
            if quoted_count == unquoted_count
            else "The quoted search returned more results than the unquoted search; this comparison does not support quotes narrowing to an exact phrase."
        ),
    }

    seed_results = {}
    for seed in SEEDS:
        seed_id = seed["id"]
        selected_months = month_counts[seed_id]
        history = quarter_history(selected_months)
        window_breakdown = {
            f"2026-{month:02d}": selected_months[(2026, month)]
            for _, month in WINDOW_MONTHS
        }
        detail = {
            "term": seed["filter"]["term"],
            "date_field": "publication_date",
            "document_types": "all",
            "window_breakdown": window_breakdown,
        }
        notes = "Counts use Federal Register publication_date and all document types. Search-index counts can be revised; values reflect retrieval time."
        if seed_id == ai_id:
            detail["quoted_vs_unquoted_comparison"] = phrase_comparison
            notes += " Double-quoted term behavior is documented with a same-month quoted/unquoted comparison in detail."
        seed_results[seed_id] = {
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
        "seeds": seed_results,
    }
    write_result(args.out, result)
    for item in sorted(snapshots, key=lambda entry: entry["file"]):
        print(f"{item['sha256']}  {item['file']}")
    print(f"snapshot_sha256  {result['snapshot_sha256']}")
    print(
        "seed_values  "
        + json.dumps(
            {seed_id: item["value"] for seed_id, item in seed_results.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"federal_register.py: {exc}", file=sys.stderr)
        raise
