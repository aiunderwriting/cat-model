#!/usr/bin/env python3
"""Resolve Q3 2026 GitHub-reviewed advisories and vintage history.

Source: https://github.com/github/advisory-database, tree
advisories/github-reviewed. The window uses current HEAD; history uses the last
default-branch commit strictly before 2026-07-01T00:00:00Z. The sole seed,
ghsa-reviewed-2026q3, includes advisories whose database_specific.github_reviewed_at
UTC date is 2026-07-01 through 2026-09-30 inclusive and whose withdrawn field is
null. History uses that same date field and withdrawn rule in each quarter from
2024-Q1 through 2026-Q2. Published-date and folder-month counts are comparison
details using the same current-HEAD withdrawal filter. Snapshots contain sorted,
canonical JSONL extracts of the exact fields used. Advisory records and review
timestamps can be revised in later commits; the history vintage records the
commit that fixes the pre-information-date view.
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
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

SOURCE_ID = "ghsa"
REPOSITORY = "https://github.com/github/advisory-database"
REVIEWED_TREE = "advisories/github-reviewed"
USER_AGENT = "AIUC calibration-seed resolver (research)"
EXTRACT_FILE = "canonical_extract.jsonl"
VINTAGE_EXTRACT_FILE = "vintage_extract.jsonl"
INFORMATION_CUTOFF = "2026-07-01T00:00:00Z"
SHALLOW_SINCE = "2026-06-21"

# Each seed declares its role, inclusive calendar window, and readable filter.
SEEDS: list[dict[str, Any]] = [
    {
        "id": "ghsa-reviewed-2026q3",
        "role": "primary",
        "window_start": "2026-07-01",
        "window_end": "2026-09-30",
        "filter": {"withdrawn": False},
        "date_field": "database_specific.github_reviewed_at",
    },
]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_utc_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        if len(raw) == 10:
            return date.fromisoformat(raw)
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).date()
    except (TypeError, ValueError):
        return None


def as_period(day: date) -> str:
    quarter = (day.month - 1) // 3 + 1
    return f"{day.year:04d}-Q{quarter}"


def quarter_periods(
    start_year: int, start_quarter: int, end_year: int, end_quarter: int
) -> list[str]:
    periods: list[str] = []
    year, quarter = start_year, start_quarter
    while (year, quarter) <= (end_year, end_quarter):
        periods.append(f"{year:04d}-Q{quarter}")
        quarter += 1
        if quarter == 5:
            quarter = 1
            year += 1
    return periods


def month_start_end(label: str) -> tuple[date, date]:
    year, month = (int(part) for part in label.split("-"))
    after = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return date(year, month, 1), date.fromordinal(after.toordinal() - 1)


def folder_month(relative_path: str) -> str | None:
    """Return a YYYY-MM folder pair if the repository path contains one."""
    parts = PurePosixPath(relative_path).parts[:-1]
    for part in parts:
        match = re.fullmatch(r"(\d{4})[-_](0[1-9]|1[0-2])", part)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
    for index in range(len(parts) - 1):
        year_part, month_part = parts[index], parts[index + 1]
        if re.fullmatch(r"\d{4}", year_part) and re.fullmatch(
            r"0?[1-9]|1[0-2]", month_part
        ):
            return f"{year_part}-{int(month_part):02d}"
    return None


def load_prior_result(path: Path) -> dict[str, Any] | None:
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        candidate.get("source_id") != SOURCE_ID
        or candidate.get("script") != Path(__file__).name
    ):
        return None
    return candidate


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
    args: list[str],
    retry_cleanup: Any = None,
    offline: bool = False,
) -> subprocess.CompletedProcess[str]:
    waits = (2, 4, 8, 16, 32)
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    if offline:
        environment["GIT_NO_LAZY_FETCH"] = "1"
    for attempt in range(len(waits) + 1):
        completed = subprocess.run(
            args,
            text=True,
            env=environment,
            capture_output=True,
        )
        if completed.returncode == 0:
            return completed
        message = (completed.stdout or "") + "\n" + (completed.stderr or "")
        if proxy_403(message):
            raise RuntimeError(
                f"Git host connection refused with HTTP 403 on CONNECT; stopping.\n{message.strip()}"
            )
        if attempt == len(waits) or not transient_git_failure(message):
            raise RuntimeError(
                f"git command failed ({completed.returncode}): {' '.join(map(str, args))}\n{message.strip()}"
            )
        if retry_cleanup is not None:
            retry_cleanup()
        time.sleep(waits[attempt])
    raise AssertionError("unreachable")


def git_prefix(source_root: Path) -> list[str]:
    return [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "-c",
        f"http.userAgent={USER_AGENT}",
        "--git-dir",
        str(source_root / ".git"),
        "--work-tree",
        str(source_root),
    ]


def clone_repository(source_root: Path) -> tuple[str, str]:
    clone_args = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "clone",
        "--filter=blob:none",
        f"--shallow-since={SHALLOW_SINCE}",
        "--sparse",
        REPOSITORY,
        str(source_root),
    ]

    def remove_partial_clone() -> None:
        if source_root.exists():
            shutil.rmtree(source_root)
        source_root.mkdir(parents=True, exist_ok=True)

    run_git(clone_args, retry_cleanup=remove_partial_clone)
    git = git_prefix(source_root)
    run_git([*git, "sparse-checkout", "set", REVIEWED_TREE])
    commit = run_git([*git, "rev-parse", "HEAD"]).stdout.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        raise RuntimeError(f"unexpected git commit SHA: {commit!r}")
    remote_head = run_git(
        [*git, "symbolic-ref", "--short", "refs/remotes/origin/HEAD"]
    ).stdout.strip()
    if not remote_head.startswith("origin/"):
        raise RuntimeError(f"unexpected origin/HEAD symbolic ref: {remote_head!r}")
    return commit.lower(), remote_head.removeprefix("origin/")


def read_advisories(
    source_root: Path,
) -> tuple[list[dict[str, Any]], list[Path], bytes]:
    root = source_root / REVIEWED_TREE
    if not root.is_dir():
        raise RuntimeError(f"missing sparse-checkout directory: {root}")
    paths = sorted(path for path in root.rglob("GHSA-*.json") if path.is_file())
    if not paths:
        raise RuntimeError(f"no GHSA JSON files found under {root}")
    advisories: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            raw = path.read_bytes()
            item = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"could not parse advisory file {path}: {error}"
            ) from error
        if not isinstance(item, dict):
            raise RuntimeError(f"advisory JSON is not an object: {path}")
        database_specific = item.get("database_specific") or {}
        if not isinstance(database_specific, dict):
            database_specific = {}
        withdrawn_value = item.get("withdrawn")
        relative = path.relative_to(source_root).as_posix()
        advisory = {
            "id": item.get("id"),
            "published": item.get("published"),
            "withdrawn_flag": withdrawn_value is not None,
            "github_reviewed_at": database_specific.get("github_reviewed_at"),
            "review_date": parse_utc_date(database_specific.get("github_reviewed_at")),
            "published_date": parse_utc_date(item.get("published")),
            "folder_month": folder_month(relative),
            "path": relative,
        }
        advisories.append(advisory)
        canonical_rows.append(
            {
                "github_reviewed_at": advisory["github_reviewed_at"],
                "id": advisory["id"],
                "published": advisory["published"],
                "path": relative,
                "withdrawn": advisory["withdrawn_flag"],
            }
        )
    canonical_rows.sort(key=lambda row: (str(row.get("id") or ""), row["path"]))
    canonical_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in canonical_rows
    )
    extract_bytes = canonical_text.encode("utf-8")
    return advisories, paths, extract_bytes


def comparison_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if not row["withdrawn_flag"]]
    q3_start, q3_end = date(2026, 7, 1), date(2026, 9, 30)
    published_count = sum(
        1
        for row in candidates
        if row["published_date"] is not None
        and q3_start <= row["published_date"] <= q3_end
    )
    folder_months: dict[str, int] = {}
    for row in candidates:
        month = row["folder_month"]
        if month in {"2026-07", "2026-08", "2026-09"}:
            folder_months[month] = folder_months.get(month, 0) + 1
    missing_folder = sum(1 for row in candidates if row["folder_month"] is None)
    recognized_date_folders = sum(
        1 for row in candidates if row["folder_month"] is not None
    )
    return {
        "published_date_q3_count": published_count,
        "folder_month_q3_count": sum(folder_months.values())
        if recognized_date_folders
        else None,
        "folder_month_q3_breakdown": {
            month: folder_months.get(month, 0)
            for month in ("2026-07", "2026-08", "2026-09")
            if folder_months.get(month, 0)
        },
        "advisories_without_parseable_folder_month": missing_folder,
    }


def build_seeds(
    current_rows: list[dict[str, Any]],
    vintage_rows: list[dict[str, Any]],
    current_files_parsed: int,
    vintage_files_parsed: int,
    vintage: dict[str, str],
) -> dict[str, Any]:
    history_periods = quarter_periods(2024, 1, 2026, 2)
    result: dict[str, Any] = {}
    missing_reviewed = sum(1 for row in current_rows if row["review_date"] is None)
    missing_reviewed_invalid = sum(
        1
        for row in current_rows
        if row["github_reviewed_at"] not in (None, "") and row["review_date"] is None
    )
    for spec in SEEDS:
        candidates = [row for row in current_rows if not row["withdrawn_flag"]]
        vintage_candidates = [row for row in vintage_rows if not row["withdrawn_flag"]]
        current_counts: dict[str, int] = {period: 0 for period in history_periods}
        vintage_counts: dict[str, int] = {period: 0 for period in history_periods}
        window_months = {"2026-07": 0, "2026-08": 0, "2026-09": 0}
        window_count = 0
        for row in candidates:
            day = row["review_date"]
            if day is None:
                continue
            period = as_period(day)
            if period in current_counts:
                current_counts[period] += 1
            if date(2026, 7, 1) <= day <= date(2026, 9, 30):
                window_count += 1
                label = f"{day.year:04d}-{day.month:02d}"
                window_months[label] += 1
        for row in vintage_candidates:
            day = row["review_date"]
            if day is None:
                continue
            period = as_period(day)
            if period in vintage_counts:
                vintage_counts[period] += 1
        comparison = comparison_counts(current_rows)
        history_comparison = [
            {
                "period": period,
                "current_head": current_counts[period],
                "vintage": vintage_counts[period],
                "difference": current_counts[period] - vintage_counts[period],
            }
            for period in history_periods
        ]
        result[spec["id"]] = {
            "role": spec["role"],
            "value": window_count,
            "window_start": spec["window_start"],
            "window_end": spec["window_end"],
            "history_granularity": "quarter",
            "history": [
                {"period": period, "value": vintage_counts[period]}
                for period in history_periods
            ],
            "detail": {
                "files_parsed_current_head": current_files_parsed,
                "files_parsed_history_vintage": vintage_files_parsed,
                "window_breakdown": window_months,
                "published_date_q3_count_for_comparison": comparison[
                    "published_date_q3_count"
                ],
                "folder_month_q3_count_for_comparison": comparison[
                    "folder_month_q3_count"
                ],
                "folder_month_q3_breakdown_for_comparison": comparison[
                    "folder_month_q3_breakdown"
                ],
                "advisories_without_parseable_folder_month": comparison[
                    "advisories_without_parseable_folder_month"
                ],
                "advisories_missing_github_reviewed_at": missing_reviewed,
                "advisories_with_invalid_github_reviewed_at": missing_reviewed_invalid,
                "comparison_filters": "exclude advisories with a non-null withdrawn field; alternate Q3 date source",
                "history_vintage": {
                    "method": "git_commit",
                    "commit": vintage["commit"],
                    "commit_time_utc": vintage["commit_time_utc"],
                },
                "current_head_history_comparison": history_comparison,
            },
            "notes": (
                "The window uses current HEAD. History uses the UTC date of database_specific.github_reviewed_at "
                "and the withdrawn status at the last default-branch commit strictly before "
                f"{INFORMATION_CUTOFF}. The comparison table records current-HEAD minus vintage history. "
                f"{missing_reviewed} current-HEAD advisory file(s) lacked a parseable github_reviewed_at."
            ),
        }
    return result


def commit_time_utc(source_root: Path, commit: str, offline: bool) -> str:
    timestamp = int(
        run_git(
            [*git_prefix(source_root), "show", "-s", "--format=%ct", commit],
            offline=offline,
        ).stdout.strip()
    )
    return (
        datetime.fromtimestamp(timestamp, UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_offline_manifest(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"offline snapshot manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["snapshots"]:
        path = source_root / row["file"]
        data = path.read_bytes()
        if len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
            raise RuntimeError(f"offline snapshot hash mismatch: {row['file']}")
        print(f"{row['sha256']}  {row['file']}")
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
    print(f"{sha256_bytes(content)}  {file_name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve GitHub Security Advisory seed counts."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    out_path = args.out.expanduser().resolve()
    data_root = args.data_dir.expanduser().resolve()

    if args.offline:
        source_root = data_root / SOURCE_ID
        if not source_root.is_dir():
            raise RuntimeError(
                f"offline source directory does not exist: {source_root}"
            )
        stored = load_offline_manifest(source_root)
        snapshots = stored["snapshots"]
        retrieved_at = stored["retrieved_at"]
        commit = stored["commit"]
        metadata = stored["metadata"]
        branch = metadata["default_branch"]
        vintage_commit = metadata["history_vintage"]["commit"]
        git = git_prefix(source_root)
        actual_head = (
            run_git([*git, "rev-parse", "HEAD"], offline=True).stdout.strip().lower()
        )
        if actual_head != commit:
            raise RuntimeError(
                "offline repository HEAD does not match snapshot metadata"
            )
        expected_vintage = (
            run_git(
                [
                    *git,
                    "rev-list",
                    "-1",
                    f"--before={INFORMATION_CUTOFF}",
                    f"origin/{branch}",
                ],
                offline=True,
            )
            .stdout.strip()
            .lower()
        )
        if expected_vintage != vintage_commit:
            raise RuntimeError(
                "offline history vintage commit does not match saved metadata"
            )
        calculated_vintage_time = commit_time_utc(source_root, vintage_commit, True)
        if calculated_vintage_time != metadata["history_vintage"]["commit_time_utc"]:
            raise RuntimeError(
                "offline history vintage commit time does not match saved metadata"
            )
    else:
        source_root = prepare_online_dir(data_root)
        retrieved_at = utc_now()
        snapshots = []
        commit, branch = clone_repository(source_root)
        git = git_prefix(source_root)
        vintage_commit = (
            run_git(
                [
                    *git,
                    "rev-list",
                    "-1",
                    f"--before={INFORMATION_CUTOFF}",
                    f"origin/{branch}",
                ]
            )
            .stdout.strip()
            .lower()
        )
        if not re.fullmatch(r"[0-9a-f]{40,64}", vintage_commit):
            raise RuntimeError(
                f"no default-branch commit found before {INFORMATION_CUTOFF}"
            )
        vintage_time = commit_time_utc(source_root, vintage_commit, False)
        metadata = {
            "default_branch": branch,
            "history_vintage": {
                "commit": vintage_commit,
                "commit_time_utc": vintage_time,
            },
        }

    git = git_prefix(source_root)
    run_git([*git, "checkout", "--force", "--detach", commit], offline=args.offline)
    current_rows, current_paths, current_extract = read_advisories(source_root)
    write_or_check_extract(
        source_root,
        EXTRACT_FILE,
        f"{REPOSITORY}/tree/{commit}/{REVIEWED_TREE}",
        current_extract,
        snapshots,
        retrieved_at,
        args.offline,
    )
    run_git(
        [*git, "checkout", "--force", "--detach", vintage_commit], offline=args.offline
    )
    vintage_rows, vintage_paths, vintage_extract = read_advisories(source_root)
    write_or_check_extract(
        source_root,
        VINTAGE_EXTRACT_FILE,
        f"{REPOSITORY}/tree/{vintage_commit}/{REVIEWED_TREE}",
        vintage_extract,
        snapshots,
        retrieved_at,
        args.offline,
    )
    run_git([*git, "checkout", "--force", "--detach", commit], offline=args.offline)

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
        (source_root / "snapshot_manifest.json").write_text(
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
        "seeds": build_seeds(
            current_rows,
            vintage_rows,
            len(current_paths),
            len(vintage_paths),
            cast(dict[str, str], metadata["history_vintage"]),
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
        print(f"ghsa.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
