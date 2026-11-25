#!/usr/bin/env python3
"""Resolve monthly NVD CVE publication totals from the NVD CVE API 2.0.

Canonical URL: https://services.nvd.nist.gov/rest/json/cves/2.0
Exact filters per seed: the all-CVE Q3, October and Q1 control seeds use
pubStartDate/pubEndDate and the valueless noRejected flag;
the prompt-injection Q3 seed adds keywordSearch=prompt injection and the
valueless keywordExactMatch flag. NVD searches the title and description text
according to its keywordSearch matching rules; prompt injection is constrained
to an exact phrase by keywordExactMatch. The date field is published
(pubStartDate and pubEndDate are inclusive UTC timestamps). Every month uses
one request with resultsPerPage=1 and reads totalResults; the response's sample
row count is checked against min(1, totalResults). Quarter values are sums of
their months. Rejected CVEs are excluded through noRejected. Histories and
windows use published dates, not last-modified dates; later rejection changes
can alter historical noRejected totals. The 2026-06 totals with and without
noRejected are recorded in detail. Requests are spaced by at least 6.5 seconds
without an API key.

This product uses data from the NVD API but is not endorsed or certified by
the NVD.
"""

import argparse
import calendar
import hashlib
import json
import os
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

SOURCE_ID = "nvd"
BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
USER_AGENT = "AIUC calibration-seed resolver (research)"
MANIFEST_NAME = "snapshot_manifest.json"
SEEDS: list[dict[str, Any]] = [
    {
        "id": "nvd-cves-published-2026q3",
        "role": "primary",
        "window": ("2026-07-01", "2026-09-30"),
        "filter": {"noRejected": True},
        "query_set": "all",
        "history_granularity": "quarter",
        "history_end": "2026-Q2",
    },
    {
        "id": "nvd-cves-published-2026-10",
        "role": "primary",
        "window": ("2026-10-01", "2026-10-31"),
        "filter": {"noRejected": True},
        "query_set": "all",
        "history_granularity": "month",
        "history_end": "2026-06",
    },
    {
        "id": "nvd-prompt-injection-cves-2026q3",
        "role": "primary",
        "window": ("2026-07-01", "2026-09-30"),
        "filter": {
            "keywordSearch": "prompt injection",
            "keywordExactMatch": True,
            "noRejected": True,
        },
        "query_set": "prompt_injection",
        "history_granularity": "quarter",
        "history_end": "2026-Q2",
    },
    {
        "id": "nvd-cves-published-2026q1",
        "role": "control",
        "window": ("2026-01-01", "2026-03-31"),
        "filter": {"noRejected": True},
        "query_set": "all",
        "history_granularity": "quarter",
        "history_end": "2025-Q4",
    },
]


def utc_now():
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def snapshot_hash(rows):
    manifest_text = "".join(
        f"{row['sha256']}  {row['file']}\n"
        for row in sorted(rows, key=lambda item: item["file"])
    )
    return sha256_bytes(manifest_text.encode("utf-8"))


def write_bytes(source_dir, relative_path, url, data, rows, retrieved_at=None):
    path = source_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    row = {
        "url": url,
        "file": relative_path.replace(os.sep, "/"),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "retrieved_at": retrieved_at or utc_now(),
    }
    rows.append(row)
    print(f"{row['sha256']}  {row['file']}", flush=True)
    return row


class RateLimiter:
    def __init__(self, minimum_interval):
        self.minimum_interval = minimum_interval
        self.last_request_end = None

    def wait(self):
        if self.last_request_end is None:
            return
        remaining = self.minimum_interval - (time.monotonic() - self.last_request_end)
        if remaining > 0:
            time.sleep(remaining)

    def mark_end(self):
        self.last_request_end = time.monotonic()


def request_snapshot(source_dir, session, url, relative_path, rows, limiter):
    backoffs = (2, 4, 8, 16, 32)
    for attempt in range(6):
        limiter.wait()
        try:
            response = session.get(url, timeout=(45, 180))
            retrieved_at = utc_now()
        except requests.RequestException as exc:
            limiter.mark_end()
            message = str(exc)
            if "403" in message and (
                "tunnel" in message.lower() or "connect" in message.lower()
            ):
                raise RuntimeError(
                    f"HTTPS CONNECT was refused for NVD: {message}"
                ) from exc
            if attempt == 5:
                raise RuntimeError(
                    f"request failed after retries for {url}: {exc}"
                ) from exc
            delay = backoffs[attempt]
            print(
                f"Transient connection failure; retrying in {delay}s: {exc}", flush=True
            )
            time.sleep(delay)
            continue
        limiter.mark_end()
        if response.status_code in (429, 500, 502, 503, 504):
            retry_path = (
                f"{relative_path}.attempt-{attempt + 1}.http-{response.status_code}"
            )
            write_bytes(
                source_dir, retry_path, url, response.content, rows, retrieved_at
            )
            if attempt == 5:
                raise RuntimeError(
                    f"HTTP {response.status_code} after retries for {url}"
                )
            delay = backoffs[attempt]
            print(f"HTTP {response.status_code}; retrying in {delay}s", flush=True)
            time.sleep(delay)
            continue
        final_path = relative_path
        if response.status_code != 200:
            final_path += f".http-{response.status_code}"
        write_bytes(source_dir, final_path, url, response.content, rows, retrieved_at)
        if response.status_code == 403:
            raise RuntimeError(f"HTTP 403 from NVD: {url}")
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} from NVD: {url}")
        return response.content, retrieved_at
    raise RuntimeError(f"request failed after retries: {url}")


def ensure_source_dir(data_dir, offline):
    source_dir = data_dir / SOURCE_ID
    if offline:
        if not source_dir.is_dir():
            raise RuntimeError(f"offline data directory does not exist: {source_dir}")
        return source_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.exists():
        if not source_dir.is_dir():
            raise RuntimeError(
                f"source path exists and is not a directory: {source_dir}"
            )
        if any(source_dir.iterdir()):
            raise RuntimeError(
                f"refusing to download into non-empty directory: {source_dir}"
            )
    else:
        source_dir.mkdir()
    return source_dir


def load_offline(source_dir):
    manifest_path = source_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"offline snapshot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["snapshots"]:
        data = (source_dir / row["file"]).read_bytes()
        if len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
            raise RuntimeError(f"offline snapshot hash mismatch: {row['file']}")
        print(f"{row['sha256']}  {row['file']}", flush=True)
    if snapshot_hash(manifest["snapshots"]) != manifest["snapshot_sha256"]:
        raise RuntimeError("offline snapshot manifest hash mismatch")
    return manifest


def month_labels(start_date, end_date):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    year, month = start.year, start.month
    labels = []
    while (year, month) <= (end.year, end.month):
        labels.append(f"{year:04d}-{month:02d}")
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return labels


def quarter_labels(end_label):
    end_year = int(end_label[:4])
    end_quarter = int(end_label[-1])
    labels = []
    year, quarter = 2024, 1
    while (year, quarter) <= (end_year, end_quarter):
        labels.append(f"{year:04d}-Q{quarter}")
        quarter += 1
        if quarter == 5:
            year += 1
            quarter = 1
    return labels


def months_for_quarter(label):
    year, quarter = int(label[:4]), int(label[-1])
    first_month = 3 * (quarter - 1) + 1
    return [f"{year:04d}-{month:02d}" for month in range(first_month, first_month + 3)]


def build_url(month, query_set, no_rejected=True):
    year, month_number = (int(part) for part in month.split("-"))
    last_day = calendar.monthrange(year, month_number)[1]
    params = [
        ("resultsPerPage", "1"),
        ("pubStartDate", f"{year:04d}-{month_number:02d}-01T00:00:00.000Z"),
        (
            "pubEndDate",
            f"{year:04d}-{month_number:02d}-{last_day:02d}T23:59:59.999Z",
        ),
    ]
    suffix = ""
    if query_set == "prompt_injection":
        params.append(("keywordSearch", "prompt injection"))
        suffix += "&keywordExactMatch"
    url = BASE_URL + "?" + urlencode(params) + suffix
    if no_rejected:
        url += "&noRejected"
    return url


def parse_total(data, month, query_set):
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid NVD JSON for {query_set} {month}: {exc}") from exc
    total = payload.get("totalResults")
    if not isinstance(total, int) or total < 0:
        raise RuntimeError(f"invalid totalResults for {query_set} {month}: {total!r}")
    rows = payload.get("vulnerabilities")
    if not isinstance(rows, list):
        raise RuntimeError(f"missing vulnerabilities list for {query_set} {month}")
    expected_sample_rows = min(1, total)
    if len(rows) != expected_sample_rows:
        raise RuntimeError(
            f"NVD totalResults/sample mismatch for {query_set} {month}: "
            f"totalResults={total}, sample rows={len(rows)}"
        )
    return total, payload


def ensure_month_data(
    month, query_set, source_dir, session, rows, limiter, totals, payloads, offline
):
    if month in totals[query_set]:
        return totals[query_set][month]
    relative = f"{query_set}/{month}.json"
    if offline:
        data = (source_dir / relative).read_bytes()
    else:
        url = build_url(month, query_set)
        data, _ = request_snapshot(source_dir, session, url, relative, rows, limiter)
    total, payload = parse_total(data, month, query_set)
    totals[query_set][month] = total
    payloads[query_set][month] = payload
    print(f"totalResults {query_set} {month}: {total}", flush=True)
    return total


def result_for_month(month, query_set, totals, payloads):
    return {
        "totalResults": totals[query_set][month],
        "sampleStatus": (
            payloads[query_set][month]["vulnerabilities"][0]["cve"].get("vulnStatus")
            if totals[query_set][month]
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Resolve NVD monthly CVE counts")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    source_dir = ensure_source_dir(data_dir, args.offline)

    if args.offline:
        stored = load_offline(source_dir)
        rows = stored["snapshots"]
        retrieved_at = stored["retrieved_at"]
        commit = stored.get("commit")
    else:
        stored = None
        rows = []
        retrieved_at = None
        commit = None
    session = requests.Session()
    session.trust_env = True
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
    )
    limiter = RateLimiter(6.5)
    totals = {"all": {}, "prompt_injection": {}}
    payloads = {"all": {}, "prompt_injection": {}}

    all_months = month_labels("2024-01-01", "2026-10-31")
    keyword_months = month_labels("2024-01-01", "2026-09-30")
    for month in all_months:
        ensure_month_data(
            month,
            "all",
            source_dir,
            session,
            rows,
            limiter,
            totals,
            payloads,
            args.offline,
        )
    for query_set in ("prompt_injection",):
        for month in keyword_months:
            ensure_month_data(
                month,
                query_set,
                source_dir,
                session,
                rows,
                limiter,
                totals,
                payloads,
                args.offline,
            )

    # The snapshot's retrieval time is its first response's.
    first_row = min(rows, key=lambda item: item["retrieved_at"])
    retrieved_at = retrieved_at or first_row["retrieved_at"]

    comparison = None
    if args.offline:
        assert stored is not None
        stored_comparison = stored.get("metadata", {}).get("no_rejected_comparison")
        if not stored_comparison:
            raise RuntimeError("offline snapshot lacks noRejected comparison metadata")
        comparison = stored_comparison
        if comparison.get("month") != "2026-06":
            raise RuntimeError("offline noRejected comparison must cover 2026-06")
        check_path = comparison["without_file"]
        check_data = (source_dir / check_path).read_bytes()
        unfiltered_total, unfiltered_payload = parse_total(
            check_data, comparison["month"], "all without noRejected"
        )
        filtered_total = totals["all"][comparison["month"]]
        if unfiltered_total != comparison["without_noRejected"]:
            raise RuntimeError(
                "offline noRejected comparison does not match snapshot metadata"
            )
        if filtered_total != comparison["with_noRejected"]:
            raise RuntimeError(
                "offline filtered count does not match comparison metadata"
            )
        comparison["without_sample_status"] = (
            unfiltered_payload["vulnerabilities"][0]["cve"].get("vulnStatus")
            if unfiltered_total
            else None
        )
    else:
        month = "2026-06"
        relative = f"comparison/{month}-without-noRejected.json"
        url = build_url(month, "all", no_rejected=False)
        data, _ = request_snapshot(source_dir, session, url, relative, rows, limiter)
        unfiltered_total, unfiltered_payload = parse_total(
            data, month, "all without noRejected"
        )
        filtered_total = totals["all"][month]
        without_status = (
            unfiltered_payload["vulnerabilities"][0]["cve"].get("vulnStatus")
            if unfiltered_total
            else None
        )
        comparison = {
            "month": month,
            "with_noRejected": filtered_total,
            "without_noRejected": unfiltered_total,
            "excluded_by_noRejected": unfiltered_total - filtered_total,
            "with_sample_status": result_for_month(month, "all", totals, payloads)[
                "sampleStatus"
            ],
            "without_sample_status": without_status,
            "without_file": relative,
        }
        print(
            "noRejected comparison "
            f"{comparison['month']}: with={comparison['with_noRejected']}, "
            f"without={comparison['without_noRejected']}",
            flush=True,
        )

    seeds_out = {}
    for seed in SEEDS:
        start, end = seed["window"]
        query_set = seed["query_set"]
        history = []
        if seed["history_granularity"] == "month":
            for month in month_labels("2024-01-01", seed["history_end"] + "-30"):
                history.append({"period": month, "value": totals["all"][month]})
        else:
            for quarter in quarter_labels(seed["history_end"]):
                history.append(
                    {
                        "period": quarter,
                        "value": sum(
                            totals[query_set][month]
                            for month in months_for_quarter(quarter)
                        ),
                    }
                )

        window_months = month_labels(start, end)
        detail = {
            "window_breakdown": {
                month: totals[query_set][month] for month in window_months
            },
            "filter": seed["filter"],
            "history_vintage": {
                "method": "publication_dated",
                "note": "Current noRejected data excludes CVEs rejected after 2026-06-30 even when their published date is in history; later rejection changes can revise these counts.",
            },
        }
        if seed["id"] == "nvd-cves-published-2026q3":
            detail["no_rejected_comparison"] = {
                key: value for key, value in comparison.items() if key != "without_file"
            }
        if query_set == "prompt_injection":
            detail["matching_note"] = (
                "NVD keywordSearch is applied to CVE title/description text; "
                "keywordExactMatch requests the exact phrase."
            )
        value = sum(totals[query_set][month] for month in window_months)
        notes = (
            "NVD totalResults supplies the monthly count. Each response requests "
            "resultsPerPage=1 as specified, and the returned sample row count was "
            "checked against min(1, totalResults). Counts use pubStartDate/pubEndDate "
            "and noRejected; later NVD revisions can change historical totals."
        )
        seeds_out[seed["id"]] = {
            "role": seed["role"],
            "value": value,
            "window_start": start,
            "window_end": end,
            "history_granularity": seed["history_granularity"],
            "history": history,
            "detail": detail,
            "notes": notes,
        }

    rows = sorted(rows, key=lambda item: item["file"])
    snapshot_sha = snapshot_hash(rows)
    if not args.offline:
        stored = {
            "retrieved_at": retrieved_at,
            "snapshots": rows,
            "commit": commit,
            "snapshot_sha256": snapshot_sha,
            "metadata": {"no_rejected_comparison": comparison},
        }
        (source_dir / MANIFEST_NAME).write_text(
            json.dumps(stored, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
    out = {
        "source_id": SOURCE_ID,
        "script": Path(__file__).name,
        "retrieved_at": retrieved_at,
        "snapshots": rows,
        "commit": commit,
        "snapshot_sha256": snapshot_sha,
        "seeds": seeds_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"{snapshot_sha}  snapshot_sha256", flush=True)
    for seed_id, seed in seeds_out.items():
        print(f"{seed_id}: {seed['value']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"nvd.py: {exc}", file=sys.stderr)
        sys.exit(1)
