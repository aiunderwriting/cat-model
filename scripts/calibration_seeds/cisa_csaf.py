#!/usr/bin/env python3
"""Resolve CISA ICS and ICS medical advisory counts from the CSAF Git source.

Canonical source: https://github.com/cisagov/CSAF
The current window is read from repository HEAD; history is read at the last
default-branch commit strictly before 2026-07-01T00:00:00Z. The repository uses
a blob-filtered shallow-since clone and sparse checkout of csaf_files/OT/white.
For cisa-ics-advisories-2026q3, include IDs matching
ICSA-YY-DDD-NN plus optional trailing letters, excluding ICSMA-, and count
distinct base IDs. For cisa-ics-medical-advisories-2026q3, include IDs matching
ICSMA-YY-DDD-NN plus optional trailing letters and count distinct base IDs.
For both seeds, the event date is date(2000 + YY, 1, 1) + (DDD - 1) days,
derived from document.tracking.id; the inclusive window is 2026-07-01 through
2026-09-30, and history uses the same ID-date rule. IDs that do not match the
pattern are excluded and counted. document.tracking.initial_release_date is
retained only for a comparison count. CISA republications keep the vendor's
original release dates, which can predate CISA publication by weeks or months.
current_release_date and status are preserved in the canonical extract but do
not filter counts. JSON files in the year folders are parsed; companion files
are not included in the extract. Current and vintage commit SHAs and sorted,
key-canonical JSONL extracts are snapshotted. CSAF advisories can be revised, so
results are tied to the recorded commits. A CISA
website listing request is made only as a completeness check and does not
determine which records are counted.
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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

SOURCE_ID = "cisa_csaf"
REPOSITORY_URL = "https://github.com/cisagov/CSAF.git"
TREE_BASE_URL = "https://github.com/cisagov/CSAF/tree"
LISTING_URL = (
    "https://www.cisa.gov/news-events/cybersecurity-advisories"
    "?f%5B0%5D=advisory_type%3A95"
)
USER_AGENT = "AIUC calibration-seed resolver (research)"
MANIFEST_NAME = "snapshot_manifest.json"
EXTRACT_NAME = "canonical_extract.jsonl"
VINTAGE_EXTRACT_NAME = "vintage_extract.jsonl"
REPO_NAME = "repo"
INFORMATION_CUTOFF = "2026-07-01T00:00:00Z"
SHALLOW_SINCE = "2026-06-21"
ADVISORY_ID_PATTERN = re.compile(
    r"(?P<prefix>ICSA|ICSMA)-(?P<year>\d{2})-(?P<ordinal>\d{3})-"
    r"(?P<number>\d{2})(?P<suffix>[A-Z]*)",
    re.IGNORECASE,
)
SEEDS: list[dict[str, Any]] = [
    {
        "id": "cisa-ics-advisories-2026q3",
        "role": "primary",
        "window": ("2026-07-01", "2026-09-30"),
        "filter": {
            "tracking_id_prefix": "ICSA-",
            "exclude_tracking_id_prefix": "ICSMA-",
            "pattern": "ICSA-YY-DDD-NN plus optional trailing letters",
            "date_field": "tracking.id (YY-DDD)",
            "distinct": "base_id",
        },
        "prefix": "ICSA-",
        "exclude_prefix": "ICSMA-",
        "history_end": "2026-Q2",
    },
    {
        "id": "cisa-ics-medical-advisories-2026q3",
        "role": "primary",
        "window": ("2026-07-01", "2026-09-30"),
        "filter": {
            "tracking_id_prefix": "ICSMA-",
            "pattern": "ICSMA-YY-DDD-NN plus optional trailing letters",
            "date_field": "tracking.id (YY-DDD)",
            "distinct": "base_id",
        },
        "prefix": "ICSMA-",
        "history_end": "2026-Q2",
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


def is_proxy_403_message(message):
    lowered = message.lower()
    return "403" in lowered and any(
        token in lowered for token in ("tunnel", "proxy", "connect")
    )


def transient_git_failure(message):
    lowered = message.lower()
    if is_proxy_403_message(message):
        return False
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


def run_git(command, cwd, retry_clone_dir=None, offline=False):
    backoffs = (2, 4, 8, 16, 32)
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    if offline:
        environment["GIT_NO_LAZY_FETCH"] = "1"
    for attempt in range(6):
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            if completed.stdout.strip():
                print(completed.stdout.strip(), flush=True)
            if completed.stderr.strip():
                print(completed.stderr.strip(), file=sys.stderr, flush=True)
            return completed.stdout.strip()
        error = (completed.stderr or completed.stdout).strip()
        if is_proxy_403_message(error):
            raise RuntimeError(f"HTTPS CONNECT was refused for GitHub: {error}")
        if attempt == 5 or not transient_git_failure(error):
            raise RuntimeError(f"git command failed after retries: {error}")
        if retry_clone_dir is not None and retry_clone_dir.exists():
            shutil.rmtree(retry_clone_dir)
        delay = backoffs[attempt]
        print(f"Transient Git failure; retrying in {delay}s: {error}", flush=True)
        time.sleep(delay)
    raise RuntimeError("git command failed after retries")


def clone_sparse(source_dir):
    repo_dir = source_dir / REPO_NAME
    cwd = Path.cwd().resolve()
    if cwd == source_dir.resolve() or source_dir.resolve() in cwd.parents:
        raise RuntimeError("refusing to run git from inside the data directory")
    clone_command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "-c",
        f"http.userAgent={USER_AGENT}",
        "clone",
        "--filter=blob:none",
        f"--shallow-since={SHALLOW_SINCE}",
        "--sparse",
        REPOSITORY_URL,
        str(repo_dir),
    ]
    run_git(clone_command, cwd, retry_clone_dir=repo_dir)
    git_args = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "-c",
        f"http.userAgent={USER_AGENT}",
        f"--git-dir={repo_dir / '.git'}",
        f"--work-tree={repo_dir}",
    ]
    run_git([*git_args, "sparse-checkout", "set", "csaf_files/OT/white"], cwd)
    commit = run_git([*git_args, "rev-parse", "HEAD"], cwd)
    remote_head = run_git(
        [*git_args, "symbolic-ref", "--short", "refs/remotes/origin/HEAD"], cwd
    )
    if not remote_head.startswith("origin/"):
        raise RuntimeError(f"unexpected origin/HEAD symbolic ref: {remote_head!r}")
    return repo_dir, commit, remote_head.removeprefix("origin/")


def is_connect_refused(exc):
    message = str(exc)
    return "403" in message and (
        "tunnel" in message.lower() or "connect" in message.lower()
    )


def check_listing(source_dir, session, rows):
    backoffs = (2, 4, 8, 16, 32)
    for attempt in range(6):
        try:
            response = session.get(LISTING_URL, timeout=(30, 90))
            retrieved_at = utc_now()
        except requests.RequestException as exc:
            if is_connect_refused(exc):
                raise RuntimeError(
                    f"HTTPS CONNECT was refused for CISA listing host: {exc}"
                ) from exc
            transient = isinstance(
                exc,
                (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ContentDecodingError,
                ),
            )
            if not transient or attempt == 5:
                raise RuntimeError(
                    f"CISA listing request failed after retries: {exc}"
                ) from exc
            delay = backoffs[attempt]
            print(
                f"Listing check connection failure; retrying in {delay}s: {exc}",
                flush=True,
            )
            time.sleep(delay)
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            path = f"listing-attempt-{attempt + 1}.http-{response.status_code}"
            response_body = response.content
            response.close()
            write_bytes(
                source_dir, path, LISTING_URL, response_body, rows, retrieved_at
            )
            if attempt == 5:
                raise RuntimeError(
                    f"CISA listing returned HTTP {response.status_code} after retries"
                )
            delay = backoffs[attempt]
            print(
                f"Listing check HTTP {response.status_code}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
            continue
        path = "listing.html"
        if response.status_code != 200:
            path += f".http-{response.status_code}"
        response_body = response.content
        response.close()
        write_bytes(source_dir, path, LISTING_URL, response_body, rows, retrieved_at)
        return {
            "url": LISTING_URL,
            "reachable": bool(response.ok),
            "status": f"http_{response.status_code}",
            "final_url": response.url,
            "note": (
                "Reachable for the optional completeness check."
                if response.ok
                else "Listing did not return a successful HTTP status."
            ),
        }
    return {
        "url": LISTING_URL,
        "reachable": False,
        "status": "request_failed",
        "note": "Listing check failed after retries.",
    }


def utc_date(value, field_name, file_path):
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"missing {field_name} in {file_path}")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"invalid {field_name} in {file_path}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).date().isoformat()


def parse_advisory_id(advisory_id):
    if not isinstance(advisory_id, str):
        return None, None
    match = ADVISORY_ID_PATTERN.fullmatch(advisory_id)
    if not match:
        return None, None
    year = int(match.group("year"))
    ordinal = int(match.group("ordinal"))
    if not 1 <= ordinal <= 366:
        return None, None
    id_date = date(2000 + year, 1, 1) + timedelta(days=ordinal - 1)
    base_id = (
        f"{match.group('prefix').upper()}-{year:02d}-{ordinal:03d}-"
        f"{match.group('number')}"
    )
    return base_id, id_date.isoformat()


def read_advisories(repo_dir):
    base = repo_dir / "csaf_files" / "OT" / "white"
    if not base.is_dir():
        raise RuntimeError(f"sparse checkout is missing {base}")
    json_files = [
        path
        for path in sorted(base.rglob("*"))
        if path.is_file()
        if path.suffix.lower() == ".json"
        and path.parent.name.isdigit()
        and len(path.parent.name) == 4
    ]
    if not json_files:
        raise RuntimeError(f"no CSAF JSON files found under {base}")
    records = []
    for path in json_files:
        relative_repo_path = path.relative_to(repo_dir).as_posix()
        try:
            advisory = json.loads(path.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid CSAF JSON {relative_repo_path}: {exc}"
            ) from exc
        tracking = advisory.get("document", {}).get("tracking", {})
        advisory_id = tracking.get("id")
        base_id, id_date = parse_advisory_id(advisory_id)
        initial_date = utc_date(
            tracking.get("initial_release_date"),
            "document.tracking.initial_release_date",
            relative_repo_path,
        )
        records.append(
            {
                "id": advisory_id,
                "base_id": base_id,
                "id_date": id_date,
                "initial_release_date": initial_date,
                "current_release_date": tracking.get("current_release_date"),
                "status": tracking.get("status"),
                "path": relative_repo_path,
            }
        )
    return records


def canonical_extract(records):
    ordered = sorted(records, key=lambda row: (str(row["id"] or ""), row["path"]))
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in ordered
    )
    return content.encode("utf-8")


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


def quarter_date_range(label):
    year, quarter = int(label[:4]), int(label[-1])
    first_month = 3 * (quarter - 1) + 1
    last_month = first_month + 2
    return (
        date(year, first_month, 1),
        date(year, last_month, calendar.monthrange(year, last_month)[1]),
    )


def get_row_bytes(source_dir, row):
    data = (source_dir / row["file"]).read_bytes()
    if len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
        raise RuntimeError(f"snapshot hash mismatch: {row['file']}")
    return data


def repo_git_prefix(repo_dir):
    return [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        f"--git-dir={repo_dir / '.git'}",
        f"--work-tree={repo_dir}",
    ]


def commit_time_utc(repo_dir, commit, offline):
    timestamp = int(
        run_git(
            [*repo_git_prefix(repo_dir), "show", "-s", "--format=%ct", commit],
            Path.cwd(),
            offline=offline,
        )
    )
    return (
        datetime.fromtimestamp(timestamp, UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_or_check_extract(
    source_dir, file_name, url, content, rows, retrieved_at, offline
):
    path = source_dir / file_name
    if offline:
        row = next((item for item in rows if item["file"] == file_name), None)
        if row is None or path.read_bytes() != content:
            raise RuntimeError(
                f"offline canonical extract differs from repository commit: {file_name}"
            )
        if row["sha256"] != sha256_bytes(content) or row["bytes"] != len(content):
            raise RuntimeError(
                f"offline canonical extract metadata mismatch: {file_name}"
            )
        return
    path.write_bytes(content)
    rows.append(
        {
            "url": url,
            "file": file_name,
            "sha256": sha256_bytes(content),
            "bytes": len(content),
            "retrieved_at": retrieved_at,
        }
    )
    print(f"{sha256_bytes(content)}  {file_name}", flush=True)


def matching_records(records, seed):
    return [
        row
        for row in records
        if isinstance(row["id"], str)
        and row["id"].upper().startswith(seed["prefix"])
        and not (
            seed.get("exclude_prefix")
            and row["id"].upper().startswith(seed["exclude_prefix"])
        )
    ]


def history_values(records, seed, periods):
    matching = matching_records(records, seed)
    parseable = [row for row in matching if row["base_id"] is not None]
    values = {}
    for period in periods:
        period_start, period_end = quarter_date_range(period)
        values[period] = len(
            {
                row["base_id"]
                for row in parseable
                if period_start <= date.fromisoformat(row["id_date"]) <= period_end
            }
        )
    return values


def main():
    parser = argparse.ArgumentParser(description="Resolve CSAF ICS advisory counts")
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
        commit = stored["commit"]
        metadata = stored["metadata"]
        branch = metadata["default_branch"]
        vintage_commit = metadata["history_vintage"]["commit"]
        site_check = metadata.get("site_check")
        if not site_check:
            raise RuntimeError("offline snapshot lacks listing reachability metadata")
        repo_dir = source_dir / REPO_NAME
        git = repo_git_prefix(repo_dir)
        actual_head = run_git([*git, "rev-parse", "HEAD"], Path.cwd(), offline=True)
        if actual_head.lower() != commit.lower():
            raise RuntimeError(
                "offline repository HEAD does not match snapshot metadata"
            )
        expected_vintage = run_git(
            [
                *git,
                "rev-list",
                "-1",
                f"--before={INFORMATION_CUTOFF}",
                f"origin/{branch}",
            ],
            Path.cwd(),
            offline=True,
        )
        if expected_vintage.lower() != vintage_commit.lower():
            raise RuntimeError(
                "offline history vintage commit does not match saved metadata"
            )
        vintage_time = commit_time_utc(repo_dir, vintage_commit, True)
        if vintage_time != metadata["history_vintage"]["commit_time_utc"]:
            raise RuntimeError(
                "offline history vintage time does not match saved metadata"
            )
    else:
        rows = []
        retrieved_at = utc_now()
        repo_dir, commit, branch = clone_sparse(source_dir)
        git = repo_git_prefix(repo_dir)
        vintage_commit = run_git(
            [
                *git,
                "rev-list",
                "-1",
                f"--before={INFORMATION_CUTOFF}",
                f"origin/{branch}",
            ],
            Path.cwd(),
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{40,64}", vintage_commit):
            raise RuntimeError(
                f"no default-branch commit found before {INFORMATION_CUTOFF}"
            )
        vintage_time = commit_time_utc(repo_dir, vintage_commit, False)
        metadata = {
            "default_branch": branch,
            "history_vintage": {
                "commit": vintage_commit,
                "commit_time_utc": vintage_time,
            },
        }

    git = repo_git_prefix(repo_dir)
    run_git(
        [*git, "checkout", "--force", "--detach", commit],
        Path.cwd(),
        offline=args.offline,
    )
    current_records = read_advisories(repo_dir)
    write_or_check_extract(
        source_dir,
        EXTRACT_NAME,
        f"{TREE_BASE_URL}/{commit}/csaf_files/OT/white",
        canonical_extract(current_records),
        rows,
        retrieved_at,
        args.offline,
    )
    run_git(
        [*git, "checkout", "--force", "--detach", vintage_commit],
        Path.cwd(),
        offline=args.offline,
    )
    vintage_records = read_advisories(repo_dir)
    write_or_check_extract(
        source_dir,
        VINTAGE_EXTRACT_NAME,
        f"{TREE_BASE_URL}/{vintage_commit}/csaf_files/OT/white",
        canonical_extract(vintage_records),
        rows,
        retrieved_at,
        args.offline,
    )
    run_git(
        [*git, "checkout", "--force", "--detach", commit],
        Path.cwd(),
        offline=args.offline,
    )

    if not args.offline:
        session = requests.Session()
        session.trust_env = True
        session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
                "Accept-Encoding": "identity",
            }
        )
        site_check = check_listing(source_dir, session, rows)
        session.close()
        metadata["site_check"] = site_check

    rows = sorted(rows, key=lambda item: item["file"])
    snapshot_sha = snapshot_hash(rows)
    if args.offline:
        if snapshot_sha != stored["snapshot_sha256"]:
            raise RuntimeError("offline snapshot manifest hash mismatch")
    else:
        stored = {
            "retrieved_at": retrieved_at,
            "snapshots": rows,
            "commit": commit,
            "snapshot_sha256": snapshot_sha,
            "metadata": metadata,
        }
        (source_dir / MANIFEST_NAME).write_text(
            json.dumps(stored, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

    seeds_out = {}
    for seed in SEEDS:
        current_matching = matching_records(current_records, seed)
        vintage_matching = matching_records(vintage_records, seed)
        unparseable = [row for row in current_matching if row["base_id"] is None]
        parseable = [row for row in current_matching if row["base_id"] is not None]
        start, end = seed["window"]
        window_start = date.fromisoformat(start)
        window_end = date.fromisoformat(end)
        window_ids = {
            row["base_id"]
            for row in parseable
            if window_start <= date.fromisoformat(row["id_date"]) <= window_end
        }
        initial_date_window_ids = {
            row["base_id"]
            for row in parseable
            if window_start
            <= date.fromisoformat(row["initial_release_date"])
            <= window_end
        }
        breakdown = {
            month: len(
                {
                    row["base_id"]
                    for row in parseable
                    if row["id_date"].startswith(month + "-")
                }
            )
            for month in month_labels(start, end)
        }
        periods = quarter_labels(seed["history_end"])
        current_history = history_values(current_records, seed, periods)
        vintage_history = history_values(vintage_records, seed, periods)
        history = [
            {"period": period, "value": vintage_history[period]} for period in periods
        ]
        history_comparison = [
            {
                "period": period,
                "current_head": current_history[period],
                "vintage": vintage_history[period],
                "difference": current_history[period] - vintage_history[period],
            }
            for period in periods
        ]

        site_message = f"CISA website listing check: {site_check['status']} ({site_check['note']})."
        notes = (
            "Counts distinct base IDs by the CISA release date encoded in tracking.id. The window uses current HEAD; "
            f"history uses the same ID-date rule at the last default-branch commit strictly before {INFORMATION_CUTOFF}. "
            "CISA republications keep vendor release dates, so initial_release_date is only a comparison count and may "
            f"predate CISA publication. Trailing update letters are stripped from IDs. {site_message}"
        )
        unparseable_examples = [
            {"id": row["id"], "path": row["path"]}
            for row in sorted(
                unparseable, key=lambda item: (str(item["id"] or ""), item["path"])
            )[:10]
        ]
        seeds_out[seed["id"]] = {
            "role": seed["role"],
            "value": len(window_ids),
            "window_start": start,
            "window_end": end,
            "history_granularity": "quarter",
            "history": history,
            "detail": {
                "window_breakdown": breakdown,
                "initial_release_date_window_value": len(initial_date_window_ids),
                "unparseable_ids": len(unparseable),
                "unparseable_id_examples": unparseable_examples,
                "history_vintage_unparseable_ids": sum(
                    1 for row in vintage_matching if row["base_id"] is None
                ),
                "history_vintage": {
                    "method": "git_commit",
                    "commit": vintage_commit,
                    "commit_time_utc": vintage_time,
                },
                "current_head_history_comparison": history_comparison,
            },
            "notes": notes,
        }

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
        detail = seed["detail"]
        print(
            f"{seed_id}: id_date={seed['value']}, "
            f"initial_release_date={detail['initial_release_date_window_value']}, "
            f"unparseable_ids={detail['unparseable_ids']}",
            flush=True,
        )
    print(f"website listing check: {site_check['status']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"cisa_csaf.py: {exc}", file=sys.stderr)
        sys.exit(1)
