#!/usr/bin/env python3
r"""Resolve Q3 2026 DeFiHackLabs incident counts with a Git history vintage.

Source: https://github.com/SunWeb3Sec/DeFiHackLabs. For the primary seed, the
window is 2026-07-01 through 2026-09-30 inclusive and uses README.md and
past/<year>/README.md files at current HEAD. History uses those same paths at
the last default-branch commit strictly before 2026-07-01T00:00:00Z. The clone
uses --filter=blob:none --no-checkout --shallow-since=2026-06-15; file contents
are read with git show at the two commits, without checking out the repository.
Each line matching ^###\s*(\d{8})\s+(\S.*)$ with a valid calendar date is one
incident entry, counted once per line. There is no (date, first-word)
deduplication; identical full heading lines in the same file still count
separately and are reported. The event date is the YYYYMMDD heading date.
Raw README blobs and sorted canonical JSONL extracts are snapshotted. Community
incident listings can be backfilled or edited; the heading date is used as
written, with no assumed reporting lag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

SOURCE_ID = "defihacklabs"
REPOSITORY = "https://github.com/SunWeb3Sec/DeFiHackLabs"
USER_AGENT = "AIUC calibration-seed resolver (research)"
MANIFEST_NAME = "snapshot_manifest.json"
EXTRACT_FILE = "canonical_extract.jsonl"
VINTAGE_EXTRACT_FILE = "vintage_extract.jsonl"
REPO_NAME = "repo"
INFORMATION_CUTOFF = "2026-07-01T00:00:00Z"
SHALLOW_SINCE = "2026-06-15"
HEADING_RE = re.compile(r"^###\s*(\d{8})\s+(\S.*)$")
YEAR_FILES = (
    ("README.md", False),
    ("past/2024/README.md", False),
    ("past/2025/README.md", False),
    ("past/2026/README.md", True),
)
SEEDS: list[dict[str, Any]] = [
    {
        "id": "defihacklabs-incidents-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {
            "heading_pattern": "^###\\s*(\\d{8})\\s+(\\S.*)$",
            "valid_calendar_date": True,
            "count_each_matching_line": True,
            "deduplicate": False,
        },
        "date_field": "heading YYYYMMDD",
    }
]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_hash(snapshots: list[dict[str, Any]]) -> str:
    lines = [
        f"{item['sha256']}  {item['file']}"
        for item in sorted(snapshots, key=lambda entry: entry["file"])
    ]
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


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


def git_prefix() -> list[str]:
    return [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "-c",
        f"http.userAgent={USER_AGENT}",
    ]


def repo_git_prefix(repo_dir: Path) -> list[str]:
    return [*git_prefix(), f"--git-dir={repo_dir / '.git'}", f"--work-tree={repo_dir}"]


def proxy_403(message: str) -> bool:
    lowered = message.lower()
    return "403" in lowered and any(
        token in lowered for token in ("connect", "tunnel", "proxy")
    )


def transient_git_failure(message: str) -> bool:
    lowered = message.lower()
    if proxy_403(message):
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


def run_git(
    command: list[str],
    *,
    binary: bool = False,
    offline: bool = False,
    retry_cleanup: Any = None,
) -> str | bytes:
    waits = (2, 4, 8, 16, 32)
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    if offline:
        environment["GIT_NO_LAZY_FETCH"] = "1"
    for attempt in range(len(waits) + 1):
        completed = subprocess.run(
            command,
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
        if proxy_403(message):
            raise RuntimeError(
                f"GitHub host connection refused with HTTP 403 on CONNECT; stopping.\n{message.strip()}"
            )
        if attempt == len(waits) or not transient_git_failure(message):
            raise RuntimeError(
                f"git command failed ({completed.returncode}): {' '.join(map(str, command))}\n{message.strip()}"
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


def clone_repository(source_root: Path) -> tuple[Path, str, str]:
    repo_dir = source_root / REPO_NAME
    command = [
        *git_prefix(),
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        f"--shallow-since={SHALLOW_SINCE}",
        REPOSITORY,
        str(repo_dir),
    ]

    def cleanup() -> None:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)

    run_git(command, retry_cleanup=cleanup)
    git = repo_git_prefix(repo_dir)
    commit = str(run_git([*git, "rev-parse", "HEAD"])).lower()
    remote_head = str(
        run_git([*git, "symbolic-ref", "--short", "refs/remotes/origin/HEAD"])
    )
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise RuntimeError(f"unexpected repository HEAD: {commit!r}")
    if not remote_head.startswith("origin/"):
        raise RuntimeError(f"unexpected origin/HEAD symbolic ref: {remote_head!r}")
    return repo_dir, commit, remote_head.removeprefix("origin/")


def vintage_commit(repo_dir: Path, branch: str, offline: bool) -> tuple[str, str]:
    git = repo_git_prefix(repo_dir)
    commit = str(
        run_git(
            [
                *git,
                "rev-list",
                "-1",
                f"--before={INFORMATION_CUTOFF}",
                f"origin/{branch}",
            ],
            offline=offline,
        )
    ).lower()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise RuntimeError(
            f"no default-branch commit found before {INFORMATION_CUTOFF}"
        )
    timestamp = int(
        str(run_git([*git, "show", "-s", "--format=%ct", commit], offline=offline))
    )
    committed_at = (
        datetime.fromtimestamp(timestamp, UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return commit, committed_at


def read_file_at_commit(
    repo_dir: Path,
    commit: str,
    relative_path: str,
    optional: bool,
    offline: bool,
) -> bytes | None:
    try:
        data = run_git(
            [*repo_git_prefix(repo_dir), "show", f"{commit}:{relative_path}"],
            binary=True,
            offline=offline,
        )
        assert isinstance(data, bytes)
        return data
    except RuntimeError as exc:
        if optional and "does not exist in" in str(exc):
            return None
        raise


def record_snapshot(
    source_root: Path,
    snapshots: list[dict[str, Any]],
    file_name: str,
    url: str,
    data: bytes,
    retrieved_at: str,
    offline: bool,
) -> None:
    path = source_root / file_name
    digest = sha256_bytes(data)
    if offline:
        row = next((item for item in snapshots if item["file"] == file_name), None)
        if row is None or not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f"offline file differs from its Git commit: {file_name}")
        if row["sha256"] != digest or row["bytes"] != len(data):
            raise RuntimeError(f"offline snapshot metadata mismatch: {file_name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    snapshots.append(
        {
            "url": url,
            "file": file_name,
            "sha256": digest,
            "bytes": len(data),
            "retrieved_at": retrieved_at,
        }
    )
    print(f"{digest}  {file_name}", flush=True)


def collect_readmes(
    source_root: Path,
    repo_dir: Path,
    commit: str,
    version: str,
    snapshots: list[dict[str, Any]],
    retrieved_at: str,
    offline: bool,
) -> tuple[dict[str, bytes], list[str]]:
    readmes: dict[str, bytes] = {}
    missing_years: list[str] = []
    for relative, optional in YEAR_FILES:
        raw = read_file_at_commit(repo_dir, commit, relative, optional, offline)
        if raw is None:
            missing_years.append(relative)
            continue
        readmes[relative] = raw
        record_snapshot(
            source_root,
            snapshots,
            f"{version}/{relative}",
            f"{REPOSITORY}/blob/{commit}/{relative}",
            raw,
            retrieved_at,
            offline,
        )
    required = {"README.md", "past/2024/README.md", "past/2025/README.md"}
    absent = sorted(required - readmes.keys())
    if absent:
        raise RuntimeError(
            f"required README file(s) are absent at {commit}: {', '.join(absent)}"
        )
    return readmes, missing_years


def valid_heading_date(value: str) -> date | None:
    if not re.fullmatch(r"\d{8}", value):
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def read_occurrences(
    readmes: dict[str, bytes],
) -> tuple[list[dict[str, Any]], int, int]:
    occurrences: list[dict[str, Any]] = []
    matching_headings = 0
    invalid_date_headings = 0
    for relative in sorted(readmes):
        try:
            text = readmes[relative].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"README is not UTF-8: {relative}: {exc}") from exc
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = HEADING_RE.match(line)
            if not match:
                continue
            matching_headings += 1
            raw_date, _name = match.groups()
            parsed_date = valid_heading_date(raw_date)
            if parsed_date is None:
                invalid_date_headings += 1
                continue
            occurrences.append(
                {
                    "id": f"{parsed_date.isoformat()}:{relative}:{line_number:08d}",
                    "date": parsed_date.isoformat(),
                    "file": relative,
                    "line": line_number,
                    "line_text": line,
                }
            )
    occurrences.sort(key=lambda item: item["id"])
    return occurrences, matching_headings, invalid_date_headings


def canonical_extract(occurrences: list[dict[str, Any]]) -> bytes:
    text = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for item in sorted(occurrences, key=lambda item: item["id"])
    )
    return text.encode("utf-8")


def exact_duplicate_heading_lines(
    occurrences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item in occurrences:
        grouped[(item["file"], item["line_text"])].append(item["line"])
    duplicates = []
    for (relative, line_text), line_numbers in sorted(grouped.items()):
        if len(line_numbers) > 1:
            duplicates.append(
                {
                    "file": relative,
                    "line_text": line_text,
                    "line_numbers": sorted(line_numbers),
                    "occurrences": len(line_numbers),
                }
            )
    return duplicates


def month_labels() -> list[str]:
    labels: list[str] = []
    year, month = 2024, 1
    while (year, month) <= (2026, 10):
        labels.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return labels


def quarter_labels() -> list[str]:
    labels: list[str] = []
    year, quarter = 2024, 1
    while (year, quarter) <= (2026, 2):
        labels.append(f"{year:04d}-Q{quarter}")
        quarter += 1
        if quarter == 5:
            quarter = 1
            year += 1
    return labels


def quarter_for_day(day: date) -> str:
    return f"{day.year:04d}-Q{(day.month - 1) // 3 + 1}"


def history_counts(occurrences: list[dict[str, Any]]) -> dict[str, int]:
    counts = {quarter: 0 for quarter in quarter_labels()}
    for item in occurrences:
        period = quarter_for_day(date.fromisoformat(item["date"]))
        if period in counts:
            counts[period] += 1
    return counts


def result_seed(
    current_occurrences: list[dict[str, Any]],
    vintage_occurrences: list[dict[str, Any]],
    current_matching: int,
    vintage_matching: int,
    current_invalid: int,
    vintage_invalid: int,
    current_missing: list[str],
    vintage_missing: list[str],
    vintage: dict[str, str],
) -> dict[str, Any]:
    month_counts = {month: 0 for month in month_labels()}
    current_quarters = history_counts(current_occurrences)
    vintage_quarters = history_counts(vintage_occurrences)
    q3_months = {"2026-07": 0, "2026-08": 0, "2026-09": 0}
    q3_count = 0
    for item in current_occurrences:
        day = date.fromisoformat(item["date"])
        month = f"{day.year:04d}-{day.month:02d}"
        if month in month_counts:
            month_counts[month] += 1
        if date(2026, 7, 1) <= day <= date(2026, 9, 30):
            q3_count += 1
            q3_months[month] += 1
    zero_months = [month for month, count in month_counts.items() if count == 0]
    current_duplicates = exact_duplicate_heading_lines(current_occurrences)
    vintage_duplicates = exact_duplicate_heading_lines(vintage_occurrences)
    periods = quarter_labels()
    comparison = [
        {
            "period": period,
            "current_head": current_quarters[period],
            "vintage": vintage_quarters[period],
            "difference": current_quarters[period] - vintage_quarters[period],
        }
        for period in periods
    ]
    detail: dict[str, Any] = {
        "window_breakdown": q3_months,
        "monthly_counts_2024-01_through_2026-10": month_counts,
        "zero_months_2024-01_through_2026-10": zero_months,
        "matching_heading_lines_current_head": current_matching,
        "invalid_calendar_date_heading_count_current_head": current_invalid,
        "matching_heading_lines_history_vintage": vintage_matching,
        "invalid_calendar_date_heading_count_history_vintage": vintage_invalid,
        "optional_files_missing_current_head": current_missing,
        "optional_files_missing_history_vintage": vintage_missing,
        "current_exact_duplicate_heading_line_groups": current_duplicates,
        "current_exact_duplicate_extra_occurrences": sum(
            item["occurrences"] - 1 for item in current_duplicates
        ),
        "history_vintage_exact_duplicate_heading_line_groups": vintage_duplicates,
        "history_vintage_exact_duplicate_extra_occurrences": sum(
            item["occurrences"] - 1 for item in vintage_duplicates
        ),
        "history_vintage": {
            "method": "git_commit",
            "commit": vintage["commit"],
            "commit_time_utc": vintage["commit_time_utc"],
        },
        "current_head_history_comparison": comparison,
    }
    notes = (
        "Every valid dated heading line counts once; no date/first-word deduplication is applied, and identical "
        "full lines in the same file count separately. The window uses current HEAD; history uses the same rule "
        f"at the last default-branch commit strictly before {INFORMATION_CUTOFF}. "
        f"Ignored {current_invalid} current-HEAD matching heading line(s) with invalid calendar dates."
    )
    if current_duplicates:
        notes += f" Current HEAD has {sum(item['occurrences'] - 1 for item in current_duplicates)} extra occurrence(s) from exact duplicate heading lines."
    if vintage_duplicates:
        notes += f" The history vintage has {sum(item['occurrences'] - 1 for item in vintage_duplicates)} extra occurrence(s) from exact duplicate heading lines."
    if current_missing:
        notes += (
            " Optional current-HEAD README file(s) absent: "
            + ", ".join(current_missing)
            + "."
        )
    if vintage_missing:
        notes += (
            " Optional history-vintage README file(s) absent: "
            + ", ".join(vintage_missing)
            + "."
        )
    if zero_months:
        notes += f" Months with zero current-HEAD entries in the monthly detail: {', '.join(zero_months)}."
    return {
        SEEDS[0]["id"]: {
            "role": "primary",
            "value": q3_count,
            "window_start": SEEDS[0]["window_start"],
            "window_end": SEEDS[0]["window_end"],
            "history_granularity": "quarter",
            "history": [
                {"period": period, "value": vintage_quarters[period]}
                for period in periods
            ],
            "detail": detail,
            "notes": notes,
        }
    }


def load_offline_manifest(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"offline snapshot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["snapshots"]:
        path = source_root / row["file"]
        data = path.read_bytes()
        if len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
            raise RuntimeError(f"offline snapshot hash mismatch: {row['file']}")
        print(f"{row['sha256']}  {row['file']}", flush=True)
    if snapshot_hash(manifest["snapshots"]) != manifest["snapshot_sha256"]:
        raise RuntimeError("offline snapshot manifest hash mismatch")
    return manifest


def write_or_check_extract(
    source_root: Path,
    file_name: str,
    url: str,
    content: bytes,
    snapshots: list[dict[str, Any]],
    retrieved_at: str,
    offline: bool,
) -> None:
    path = source_root / file_name
    if offline:
        row = next((entry for entry in snapshots if entry["file"] == file_name), None)
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
    snapshots.append(
        {
            "url": url,
            "file": file_name,
            "sha256": sha256_bytes(content),
            "bytes": len(content),
            "retrieved_at": retrieved_at,
        }
    )
    print(f"{sha256_bytes(content)}  {file_name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve DeFiHackLabs incident seed counts."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    data_root = args.data_dir.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    source_root = (
        data_root / SOURCE_ID if args.offline else prepare_online_dir(data_root)
    )
    if args.offline and not source_root.is_dir():
        raise RuntimeError(f"offline source directory does not exist: {source_root}")

    if args.offline:
        stored = load_offline_manifest(source_root)
        snapshots = stored["snapshots"]
        retrieved_at = stored["retrieved_at"]
        commit = stored["commit"]
        metadata = stored["metadata"]
        branch = metadata["default_branch"]
        vintage = metadata["history_vintage"]
        repo_dir = source_root / REPO_NAME
        git = repo_git_prefix(repo_dir)
        actual_head = str(run_git([*git, "rev-parse", "HEAD"], offline=True)).lower()
        if actual_head != commit:
            raise RuntimeError(
                "offline repository HEAD does not match snapshot metadata"
            )
        calculated_vintage, calculated_time = vintage_commit(repo_dir, branch, True)
        if (
            calculated_vintage != vintage["commit"]
            or calculated_time != vintage["commit_time_utc"]
        ):
            raise RuntimeError("offline history vintage does not match saved metadata")
    else:
        snapshots = []
        retrieved_at = utc_now()
        repo_dir, commit, branch = clone_repository(source_root)
        vintage_commit_sha, vintage_time = vintage_commit(repo_dir, branch, False)
        vintage = {"commit": vintage_commit_sha, "commit_time_utc": vintage_time}
        metadata = {
            "default_branch": branch,
            "history_vintage": vintage,
        }

    current_readmes, current_missing = collect_readmes(
        source_root, repo_dir, commit, "current", snapshots, retrieved_at, args.offline
    )
    vintage_readmes, vintage_missing = collect_readmes(
        source_root,
        repo_dir,
        vintage["commit"],
        "vintage",
        snapshots,
        retrieved_at,
        args.offline,
    )
    current_occurrences, current_matching, current_invalid = read_occurrences(
        current_readmes
    )
    vintage_occurrences, vintage_matching, vintage_invalid = read_occurrences(
        vintage_readmes
    )
    write_or_check_extract(
        source_root,
        EXTRACT_FILE,
        f"{REPOSITORY}/tree/{commit}/{EXTRACT_FILE}",
        canonical_extract(current_occurrences),
        snapshots,
        retrieved_at,
        args.offline,
    )
    write_or_check_extract(
        source_root,
        VINTAGE_EXTRACT_FILE,
        f"{REPOSITORY}/tree/{vintage['commit']}/{VINTAGE_EXTRACT_FILE}",
        canonical_extract(vintage_occurrences),
        snapshots,
        retrieved_at,
        args.offline,
    )

    snapshots.sort(key=lambda entry: entry["file"])
    digest = snapshot_hash(snapshots)
    if args.offline:
        if digest != stored["snapshot_sha256"]:
            raise RuntimeError("offline snapshot hash differs from manifest")
    else:
        stored = {
            "retrieved_at": retrieved_at,
            "snapshots": snapshots,
            "commit": commit,
            "snapshot_sha256": digest,
            "metadata": metadata,
        }
        (source_root / MANIFEST_NAME).write_text(
            json.dumps(stored, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    print(f"snapshot_sha256  {digest}")

    output = {
        "source_id": SOURCE_ID,
        "script": Path(__file__).name,
        "retrieved_at": retrieved_at,
        "snapshots": snapshots,
        "commit": commit,
        "snapshot_sha256": digest,
        "seeds": result_seed(
            current_occurrences,
            vintage_occurrences,
            current_matching,
            vintage_matching,
            current_invalid,
            vintage_invalid,
            current_missing,
            vintage_missing,
            vintage,
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"defihacklabs.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
