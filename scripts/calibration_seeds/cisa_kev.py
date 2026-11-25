#!/usr/bin/env python3
"""Resolve CISA KEV seed windows and information-vintage histories.

The current window source is CISA's JSON feed:
https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
The history source is CISA's official mirror at https://github.com/cisagov/kev-data,
file known_exploited_vulnerabilities.json, read at the last default-branch commit
strictly before each seed's information cutoff. Each seed counts distinct cveID
values by dateAdded in its inclusive window. kev-microsoft-additions-2026q3 applies
the same rule after vendorProject is stripped and case-folded to Microsoft.
The source date field is dateAdded (YYYY-MM-DD); rows without string cveID or
dateAdded are ignored. Current-feed rows removed from the feed are absent, so
current-feed and Git-vintage histories can differ. Raw CISA feed bytes, the
default-branch lookup response, and canonical JSONL extracts of the Git fields
used are snapshotted. The CISA catalog can be revised.
"""

import argparse
import calendar
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "cisa_kev"
URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
GIT_REPOSITORY = "https://github.com/cisagov/kev-data"
GIT_FILE = "known_exploited_vulnerabilities.json"
USER_AGENT = "AIUC calibration-seed resolver (research)"
MANIFEST_NAME = "snapshot_manifest.json"
SNAPSHOT_NAME = "known_exploited_vulnerabilities.json"
REPO_NAME = "repo"
DEFAULT_BRANCH_FILE = "git-ls-remote-symref.txt"
VINTAGE_CUTOFFS = {
    "2026-07-01T00:00:00Z": "2026-06-21",
    "2026-01-01T00:00:00Z": "2025-12-22",
}
SEEDS: list[dict[str, Any]] = [
    {
        "id": "kev-additions-2026q3",
        "role": "primary",
        "window": ("2026-07-01", "2026-09-30"),
        "filter": {"distinct": "cveID"},
        "history_granularity": "quarter",
        "history_end": "2026-Q2",
        "information_cutoff": "2026-07-01T00:00:00Z",
    },
    {
        "id": "kev-microsoft-additions-2026q3",
        "role": "primary",
        "window": ("2026-07-01", "2026-09-30"),
        "filter": {
            "distinct": "cveID",
            "vendorProject": "Microsoft (strip and case-insensitive)",
        },
        "history_granularity": "quarter",
        "history_end": "2026-Q2",
        "information_cutoff": "2026-07-01T00:00:00Z",
    },
    {
        "id": "kev-additions-2026-10",
        "role": "primary",
        "window": ("2026-10-01", "2026-10-31"),
        "filter": {"distinct": "cveID"},
        "history_granularity": "month",
        "history_end": "2026-06",
        "information_cutoff": "2026-07-01T00:00:00Z",
    },
    {
        "id": "kev-additions-2026q1",
        "role": "control",
        "window": ("2026-01-01", "2026-03-31"),
        "filter": {"distinct": "cveID"},
        "history_granularity": "quarter",
        "history_end": "2025-Q4",
        "information_cutoff": "2026-01-01T00:00:00Z",
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


def online_get(source_dir, session, rows):
    backoffs = (2, 4, 8, 16, 32)
    for attempt in range(6):
        try:
            response = session.get(URL, timeout=(30, 120))
        except requests.RequestException as exc:
            message = str(exc)
            if "403" in message and (
                "tunnel" in message.lower() or "connect" in message.lower()
            ):
                raise RuntimeError(
                    f"HTTPS CONNECT was refused for cisa.gov: {message}"
                ) from exc
            if attempt == 5:
                raise RuntimeError(f"download failed after retries: {exc}") from exc
            delay = backoffs[attempt]
            print(
                f"Transient connection failure; retrying in {delay}s: {exc}", flush=True
            )
            time.sleep(delay)
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            relative = f"attempt-{attempt + 1}.http-{response.status_code}.response"
            write_bytes(source_dir, relative, URL, response.content, rows)
            response.close()
            if attempt == 5:
                raise RuntimeError(
                    f"HTTP {response.status_code} after retries for {URL}"
                )
            delay = backoffs[attempt]
            print(f"HTTP {response.status_code}; retrying in {delay}s", flush=True)
            time.sleep(delay)
            continue
        data = response.content
        response.close()
        relative = (
            SNAPSHOT_NAME
            if response.status_code == 200
            else f"{SNAPSHOT_NAME}.http-{response.status_code}"
        )
        write_bytes(source_dir, relative, URL, data, rows)
        if response.status_code == 403:
            raise RuntimeError(f"HTTP 403 from CISA KEV source: {URL}")
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code} from CISA KEV source: {URL}"
            )
        return data
    raise RuntimeError(f"download failed after retries: {URL}")


def is_proxy_403(message):
    lowered = message.lower()
    return "403" in lowered and any(
        token in lowered for token in ("connect", "tunnel", "proxy")
    )


def transient_git_failure(message):
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "could not resolve host",
            "connection reset",
            "connection refused",
            "connection timed out",
            "operation timed out",
            "failed to connect",
            "remote end hung up",
            "unexpected disconnect",
            "http/2 stream",
            "gnutls_recv_error",
            "curl 56",
            "curl 92",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "server returned 5",
        )
    )


def git_prefix():
    return [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "-c",
        f"http.userAgent={USER_AGENT}",
    ]


def run_git(args, retry_cleanup=None, binary=False, offline=False):
    waits = (2, 4, 8, 16, 32)
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    if offline:
        environment["GIT_NO_LAZY_FETCH"] = "1"
    for attempt in range(len(waits) + 1):
        completed = subprocess.run(
            args,
            cwd=Path.cwd(),
            env=environment,
            capture_output=True,
        )
        if completed.returncode == 0:
            return (
                completed.stdout if binary else completed.stdout.decode("utf-8").strip()
            )
        message = (completed.stdout + b"\n" + completed.stderr).decode(
            "utf-8", "replace"
        )
        if is_proxy_403(message):
            raise RuntimeError(
                f"GitHub host connection refused with HTTP 403 on CONNECT; stopping.\n{message.strip()}"
            )
        if attempt == len(waits) or not transient_git_failure(message):
            raise RuntimeError(
                f"git command failed ({completed.returncode}): {' '.join(map(str, args))}\n{message.strip()}"
            )
        if retry_cleanup is not None:
            retry_cleanup()
        delay = waits[attempt]
        print(
            f"Transient Git failure; retrying in {delay}s: {message.strip()}",
            flush=True,
        )
        time.sleep(delay)
    raise AssertionError("unreachable")


def discover_default_branch(source_dir, rows):
    raw = run_git(
        [*git_prefix(), "ls-remote", "--symref", GIT_REPOSITORY, "HEAD"], binary=True
    )
    match = re.search(rb"^ref: refs/heads/([^\t\r\n]+)\tHEAD$", raw, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(
            f"could not detect GitHub default branch from ls-remote --symref: {raw[:300]!r}"
        )
    branch = match.group(1).decode("utf-8")
    write_bytes(source_dir, DEFAULT_BRANCH_FILE, GIT_REPOSITORY, raw, rows)
    return branch


def clone_repository(source_dir, branch, offline=False):
    repo_dir = source_dir / REPO_NAME
    since_values = list(VINTAGE_CUTOFFS.values())
    shallow_since = min(since_values)
    command = [
        *git_prefix(),
        "clone",
        "--filter=blob:none",
        f"--shallow-since={shallow_since}",
        "--no-checkout",
        "--branch",
        branch,
        GIT_REPOSITORY,
        str(repo_dir),
    ]

    def cleanup():
        if repo_dir.exists():
            shutil.rmtree(repo_dir)

    run_git(command, retry_cleanup=cleanup, offline=offline)
    return repo_dir


def repo_git_prefix(repo_dir):
    return [*git_prefix(), f"--git-dir={repo_dir / '.git'}"]


def commit_time_utc(repo_dir, commit, offline=False):
    timestamp = int(
        run_git(
            [*repo_git_prefix(repo_dir), "show", "-s", "--format=%ct", commit],
            offline=offline,
        )
    )
    return (
        datetime.fromtimestamp(timestamp, UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_kev_extract(payload):
    catalog = payload.get("vulnerabilities")
    if not isinstance(catalog, list):
        raise RuntimeError("CISA KEV response has no vulnerabilities list")
    rows = []
    for entry in catalog:
        if not isinstance(entry, dict):
            continue
        cve_id = entry.get("cveID")
        added = entry.get("dateAdded")
        if isinstance(cve_id, str) and isinstance(added, str):
            rows.append(
                {
                    "cveID": cve_id,
                    "dateAdded": added,
                    "vendorProject": entry.get("vendorProject"),
                }
            )
    rows.sort(
        key=lambda row: (
            row["cveID"],
            row["dateAdded"],
            json.dumps(row, sort_keys=True, ensure_ascii=False),
        )
    )
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def load_vintage_payloads(source_dir, branch, repo_dir, metadata, snapshots, offline):
    git = repo_git_prefix(repo_dir)
    head = run_git([*git, "rev-parse", "HEAD"], offline=offline).lower()
    vintages = {}
    for cutoff, shallow_since in VINTAGE_CUTOFFS.items():
        commit = run_git(
            [*git, "rev-list", "-1", f"--before={cutoff}", branch],
            offline=offline,
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise RuntimeError(f"no commit found before {cutoff} on branch {branch}")
        recorded = metadata.get(cutoff) if metadata else None
        if recorded and recorded.get("commit") != commit:
            raise RuntimeError(
                f"local Git vintage commit for {cutoff} differs from saved metadata"
            )
        commit_time = commit_time_utc(repo_dir, commit, offline)
        raw = run_git(
            [*git, "show", f"{commit}:{GIT_FILE}"], binary=True, offline=offline
        )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid KEV mirror JSON at {commit}:{GIT_FILE}: {exc}"
            ) from exc
        extract = canonical_kev_extract(payload)
        file_name = f"vintage-{cutoff[:10]}.jsonl"
        if offline:
            path = source_dir / file_name
            if not path.is_file() or path.read_bytes() != extract:
                raise RuntimeError(
                    f"offline canonical Git extract differs from {commit}:{GIT_FILE}"
                )
            row = next((item for item in snapshots if item["file"] == file_name), None)
            if row is None or row["sha256"] != sha256_bytes(extract):
                raise RuntimeError(
                    f"offline vintage extract is missing from manifest: {file_name}"
                )
        else:
            write_bytes(
                source_dir,
                file_name,
                f"{GIT_REPOSITORY}/blob/{commit}/{GIT_FILE}",
                extract,
                snapshots,
            )
        vintages[cutoff] = {
            "payload": payload,
            "commit": commit,
            "commit_time_utc": commit_time,
            "shallow_since": shallow_since,
        }
    return head, vintages


def load_offline(source_dir):
    manifest_path = source_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"offline snapshot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["snapshots"]:
        path = source_dir / row["file"]
        data = path.read_bytes()
        if len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
            raise RuntimeError(f"offline snapshot hash mismatch: {row['file']}")
        print(f"{row['sha256']}  {row['file']}", flush=True)
    if snapshot_hash(manifest["snapshots"]) != manifest["snapshot_sha256"]:
        raise RuntimeError("offline snapshot manifest hash mismatch")
    return manifest


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


def period_key(day, granularity):
    if granularity == "month":
        return day[:7]
    parsed = date.fromisoformat(day)
    return f"{parsed.year:04d}-Q{(parsed.month - 1) // 3 + 1}"


def build_catalog_counts(payload):
    catalog = payload.get("vulnerabilities")
    if not isinstance(catalog, list):
        raise RuntimeError("CISA KEV response has no vulnerabilities list")
    general = {}
    microsoft = {}
    duplicate_rows = 0
    seen_rows = set()
    for entry in catalog:
        if not isinstance(entry, dict):
            continue
        cve_id = entry.get("cveID")
        added = entry.get("dateAdded")
        if not isinstance(cve_id, str) or not isinstance(added, str):
            continue
        try:
            date.fromisoformat(added)
        except ValueError as exc:
            raise RuntimeError(f"invalid dateAdded for {cve_id}: {added}") from exc
        key = (cve_id, added)
        if key in seen_rows:
            duplicate_rows += 1
        seen_rows.add(key)
        general.setdefault(added, set()).add(cve_id)
        vendor = entry.get("vendorProject")
        if isinstance(vendor, str) and vendor.strip().casefold() == "microsoft":
            microsoft.setdefault(added, set()).add(cve_id)
    return catalog, general, microsoft, duplicate_rows


def ids_in_period(counts, start, end):
    ids = set()
    for day, day_ids in counts.items():
        if start <= day <= end:
            ids.update(day_ids)
    return ids


def history_rows(counts, seed):
    if seed["history_granularity"] == "month":
        periods = month_labels("2024-01-01", "2026-06-30")
    else:
        periods = quarter_labels(seed["history_end"])
    history = []
    for period in periods:
        if seed["history_granularity"] == "month":
            year, month = (int(part) for part in period.split("-"))
            period_start = date(year, month, 1).isoformat()
            period_end = date(
                year, month, calendar.monthrange(year, month)[1]
            ).isoformat()
        else:
            year, quarter = int(period[:4]), int(period[-1])
            first_month = (quarter - 1) * 3 + 1
            period_start = date(year, first_month, 1).isoformat()
            last_month = first_month + 2
            period_end = date(
                year,
                last_month,
                calendar.monthrange(year, last_month)[1],
            ).isoformat()
        history.append(
            {
                "period": period,
                "value": len(ids_in_period(counts, period_start, period_end)),
            }
        )
    return history


def main():
    parser = argparse.ArgumentParser(description="Resolve CISA KEV seed counts")
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
        raw = (source_dir / SNAPSHOT_NAME).read_bytes()
        if sha256_bytes(raw) != next(
            row["sha256"] for row in rows if row["file"] == SNAPSHOT_NAME
        ):
            raise RuntimeError("offline KEV raw file does not match its snapshot row")
        retrieved_at = stored["retrieved_at"]
        commit = stored["commit"]
        metadata = stored["metadata"]
        branch = metadata["default_branch"]
    else:
        rows = []
        session = requests.Session()
        session.trust_env = True
        session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            }
        )
        raw = online_get(source_dir, session, rows)
        session.close()
        retrieved_at = rows[0]["retrieved_at"]
        branch = discover_default_branch(source_dir, rows)
        repo_dir = clone_repository(source_dir, branch)
        commit, vintages = load_vintage_payloads(
            source_dir, branch, repo_dir, {}, rows, False
        )
        metadata = {
            "default_branch": branch,
            "history_vintages": {
                cutoff: {
                    "commit": item["commit"],
                    "commit_time_utc": item["commit_time_utc"],
                }
                for cutoff, item in vintages.items()
            },
        }
        rows = sorted(rows, key=lambda item: item["file"])
        stored = {
            "retrieved_at": retrieved_at,
            "snapshots": rows,
            "commit": commit,
            "snapshot_sha256": snapshot_hash(rows),
            "metadata": metadata,
        }
        (source_dir / MANIFEST_NAME).write_text(
            json.dumps(stored, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    payload = json.loads(raw.decode("utf-8"))
    catalog, all_current_counts, current_microsoft_counts, duplicate_rows = (
        build_catalog_counts(payload)
    )
    catalog_version = payload.get("catalogVersion")
    date_released = payload.get("dateReleased")
    catalog_count = payload.get("count")
    if catalog_count is not None and catalog_count != len(catalog):
        raise RuntimeError(
            f"CISA catalog count field ({catalog_count}) does not match vulnerabilities rows ({len(catalog)})"
        )

    if args.offline:
        repo_dir = source_dir / REPO_NAME
        history_meta = metadata["history_vintages"]
        _, vintages = load_vintage_payloads(
            source_dir,
            branch,
            repo_dir,
            history_meta,
            rows,
            True,
        )
        if (
            stored.get("commit")
            != run_git(
                [*repo_git_prefix(repo_dir), "rev-parse", "HEAD"], offline=True
            ).lower()
        ):
            raise RuntimeError("offline mirror HEAD does not match result metadata")
    else:
        history_meta = metadata["history_vintages"]

    seeds_out = {}
    for seed in SEEDS:
        start, end = seed["window"]
        is_microsoft = seed["id"] == "kev-microsoft-additions-2026q3"
        window_counts = current_microsoft_counts if is_microsoft else all_current_counts
        vintage_info = vintages[seed["information_cutoff"]]
        _, vintage_counts, vintage_microsoft_counts, vintage_duplicate_rows = (
            build_catalog_counts(vintage_info["payload"])
        )
        history_counts = vintage_microsoft_counts if is_microsoft else vintage_counts
        current_history_counts = (
            current_microsoft_counts if is_microsoft else all_current_counts
        )
        history = history_rows(history_counts, seed)
        current_history = history_rows(current_history_counts, seed)
        comparison = [
            {
                "period": vintage_row["period"],
                "current_feed": current_row["value"],
                "vintage": vintage_row["value"],
                "difference": current_row["value"] - vintage_row["value"],
            }
            for current_row, vintage_row in zip(current_history, history)
        ]
        window_breakdown = {}
        for month in month_labels(start, end):
            month_start = date.fromisoformat(month + "-01")
            month_end = date(
                month_start.year,
                month_start.month,
                calendar.monthrange(month_start.year, month_start.month)[1],
            )
            window_breakdown[month] = len(
                ids_in_period(
                    window_counts, month_start.isoformat(), month_end.isoformat()
                )
            )
        value = len(ids_in_period(window_counts, start, end))
        detail = {
            "catalogVersion": catalog_version,
            "dateReleased": date_released,
            "count": catalog_count,
            "history_vintage": {
                "method": "git_commit",
                "commit": vintage_info["commit"],
                "commit_time_utc": vintage_info["commit_time_utc"],
            },
            "git_default_branch": branch,
            "current_feed_history_comparison": comparison,
            "window_breakdown": window_breakdown,
        }
        notes = (
            "Window value is computed from the current CISA feed as distinct CVE IDs by dateAdded. "
            "History is computed from the official CISA Git mirror at the last default-branch commit "
            f"strictly before {seed['information_cutoff']}. The comparison table records current-feed "
            "minus vintage history for every history period; CISA feed revisions can remove or alter rows."
        )
        if duplicate_rows:
            notes += (
                f" Current feed has {duplicate_rows} duplicate cveID/dateAdded row(s)."
            )
        if vintage_duplicate_rows:
            notes += f" Vintage mirror has {vintage_duplicate_rows} duplicate cveID/dateAdded row(s)."
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
        print(f"cisa_kev.py: {exc}", file=sys.stderr)
        sys.exit(1)
