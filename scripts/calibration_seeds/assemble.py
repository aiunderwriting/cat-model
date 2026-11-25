"""Assemble the respondent spread seed files from the catalogue and resolver outputs.

Usage:
    python -I scripts/calibration_seeds/assemble.py --results-dir <results>

Reads scripts/calibration_seeds/catalogue.yaml and <results>/<source_id>.json
(written by the per-source scripts) and writes into
data/calibration/respondent_spread/:
  seeds.yaml        instrument-visible seed definitions
  resolutions.yaml  resolved values and provenance, read only by the fit
  sources.yaml      source URLs, terms and snapshot identities

It stops at an inconsistency, such as a seed without a result, a role other
than primary, control or secondary, a window that the quantity text does not
state, a primary window outside 2026-07-01..2026-10-31, a primary or secondary
window starting by the information date, history ending after it, a seed
without a value, a primary or control seed retrieved on another date than the
catalogue's, more than three primary seeds in a block, fewer than 40 primary
seeds or 15 blocks, or a role count outside its bounds (controls 2-4,
secondary 3-6).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from catmodel.yaml_io import safe_load

PRIMARY_WINDOW = (dt.date(2026, 7, 1), dt.date(2026, 10, 31))
MAX_PRIMARY_PER_BLOCK = 3
MIN_PRIMARY, MIN_BLOCKS = 40, 15
ROLE_BOUNDS = {"control": (2, 4), "secondary": (3, 6)}
MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def period_end(label: str) -> dt.date:
    """Last day of a period labelled YYYY, YYYY-Qn or YYYY-MM."""
    if re.fullmatch(r"\d{4}", label):
        return dt.date(int(label), 12, 31)
    m = re.fullmatch(r"(\d{4})-Q([1-4])", label)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        month = 3 * q
    else:
        m = re.fullmatch(r"(\d{4})-(\d{2})", label)
        if not m:
            raise ValueError(f"unrecognised period label {label!r}")
        year, month = int(m.group(1)), int(m.group(2))
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return nxt - dt.timedelta(days=1)


def window_tokens(start: dt.date, end: dt.date) -> list[list[str]]:
    """Alternative spellings of a window, as token lists; the quantity text must
    contain every token of one of them."""
    iso = [start.isoformat(), end.isoformat()]
    compact = [start.strftime("%Y%m%d"), end.strftime("%Y%m%d")]
    months = []
    d = dt.date(start.year, start.month, 1)
    while d <= end:
        months.append(d)
        d = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    sgo = [m.strftime("%b").upper() + f"-{m.year}" for m in months]
    names = [MONTH_NAMES[m.month - 1] for m in months]
    return [iso, compact, sgo, [*names, str(end.year)]]


def check_window_text(seed_id: str, text: str, start: dt.date, end: dt.date) -> None:
    for alt in window_tokens(start, end):
        if all(tok in text for tok in alt):
            return
    raise SystemExit(
        f"{seed_id}: quantity text does not state the window {start}..{end}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--results-dir", required=True, type=Path)
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    catalogue_path = repo_root / "scripts" / "calibration_seeds" / "catalogue.yaml"
    output_dir = repo_root / "data" / "calibration" / "respondent_spread"
    results_dir = args.results_dir.expanduser().resolve()
    cat = safe_load(catalogue_path.read_text(encoding="utf-8"))
    retrieval_date = dt.date.fromisoformat(cat["retrieval_date"])
    latest_training_cutoff = dt.date.fromisoformat(cat["latest_training_cutoff"])
    sources = cat["sources"]

    results: dict[str, dict] = {}
    for sid in sources:
        path = results_dir / f"{sid}.json"
        if not path.exists():
            raise SystemExit(f"missing results for source {sid}: {path}")
        results[sid] = json.loads(path.read_text(encoding="utf-8"))

    seeds_out, res_out = [], []
    roles: Counter[str] = Counter()
    primary_blocks: dict[str, list[str]] = defaultdict(list)
    used_sources: set[str] = set()
    seen: set[str] = set()
    for s in cat["seeds"]:
        sid = s["seed_id"]
        if sid in seen:
            raise SystemExit(f"duplicate seed {sid}")
        seen.add(sid)
        for field in ("quantity", "evidence"):
            if "PLACEHOLDER" in str(s[field]):
                raise SystemExit(f"{sid}: {field} still has a placeholder")
        role, src_id = s["role"], s["source"]
        if role != "primary" and role not in ROLE_BOUNDS:
            raise SystemExit(f"{sid}: unknown role {role}")
        used_sources.add(src_id)
        roles[role] += 1
        if role == "primary":
            primary_blocks[s["block_id"]].append(sid)
        src = sources[src_id]
        src_res = results[src_id]
        r = src_res.get("seeds", {}).get(sid)
        if r is None:
            raise SystemExit(f"{sid}: no result in {src_id}.json")
        start = dt.date.fromisoformat(r["window_start"])
        end = dt.date.fromisoformat(r["window_end"])
        if r["role"] != role:
            raise SystemExit(
                f"{sid}: role {role} in catalogue but {r['role']} in results"
            )
        history, value, notes = r["history"], r["value"], r.get("notes", "")
        if role == "primary" and not (
            PRIMARY_WINDOW[0] <= start <= end <= PRIMARY_WINDOW[1]
        ):
            raise SystemExit(
                f"{sid}: primary window {start}..{end} outside {PRIMARY_WINDOW}"
            )
        check_window_text(sid, s["quantity"], start, end)
        info_date = (
            start - dt.timedelta(days=1)
            if role == "control"
            else latest_training_cutoff
        )
        # Primary and secondary windows start after the information date.
        if role in ("primary", "secondary") and start <= info_date:
            raise SystemExit(f"{sid}: window starts before the information date")
        if not history:
            raise SystemExit(f"{sid}: empty history")
        for h in history:
            if period_end(str(h["period"])) > info_date:
                raise SystemExit(
                    f"{sid}: history period {h['period']} ends after {info_date}"
                )
            if not isinstance(h["value"], int) or h["value"] < 0:
                raise SystemExit(f"{sid}: bad history value {h}")
        if role in ("primary", "control"):
            if not isinstance(value, int):
                raise SystemExit(f"{sid}: {role} seed has no value")
            got = dt.datetime.fromisoformat(
                src_res["retrieved_at"].replace("Z", "+00:00")
            ).date()
            if got != retrieval_date:
                raise SystemExit(
                    f"{sid}: retrieved {got}, catalogue says {retrieval_date}"
                )
        elif not isinstance(value, int | float):
            raise SystemExit(f"{sid}: {role} seed has no value")
        script_path = f"scripts/calibration_seeds/{src['script']}"
        seeds_out.append(
            {
                "seed_id": sid,
                "role": role,
                "block_id": s["block_id"],
                "quantity": " ".join(s["quantity"].split()),
                "unit": s["unit"],
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "information_date": info_date.isoformat(),
                "source": {"name": src["name"], "url": src["url"]},
                "evidence": {
                    "text": " ".join(s["evidence"].split()),
                    "history": [
                        {
                            "period": str(h["period"]),
                            "period_end": period_end(str(h["period"])).isoformat(),
                            "value": h["value"],
                        }
                        for h in history
                    ],
                },
                "resolution_script": script_path,
                "can_resolve_to_zero": bool(s["can_resolve_to_zero"]),
            }
        )
        res_out.append(
            {
                "seed_id": sid,
                "value": value,
                "retrieved_at": src_res.get("retrieved_at"),
                "snapshot_url": src["data_url"],
                "snapshot_sha256": src_res.get("snapshot_sha256"),
                "script": script_path,
                "notes": " ".join(str(notes).split()),
            }
        )

    over = {
        b: ids for b, ids in primary_blocks.items() if len(ids) > MAX_PRIMARY_PER_BLOCK
    }
    if over:
        raise SystemExit(
            f"blocks with more than {MAX_PRIMARY_PER_BLOCK} primary seeds: {over}"
        )
    if roles["primary"] < MIN_PRIMARY or len(primary_blocks) < MIN_BLOCKS:
        raise SystemExit(
            f"need >= {MIN_PRIMARY} primary seeds in >= {MIN_BLOCKS} blocks; "
            f"have {roles['primary']} in {len(primary_blocks)}"
        )
    for role, (lo, hi) in ROLE_BOUNDS.items():
        if not lo <= roles[role] <= hi:
            raise SystemExit(f"{roles[role]} {role} seeds; need {lo}-{hi}")

    src_out = []
    for sid in sorted(used_sources):
        src, r = sources[sid], results.get(sid, {})
        src_out.append(
            {
                "source_id": sid,
                "name": src["name"],
                "url": src["url"],
                "data_url": src["data_url"],
                "script": f"scripts/calibration_seeds/{src['script']}",
                "license_terms": " ".join(src["terms"].split()),
                "terms_url": src["terms_url"],
                "automated_access": " ".join(src["automated_access"].split()),
                "retrieved_at": r.get("retrieved_at"),
                "commit": r.get("commit"),
                "snapshot_sha256": r.get("snapshot_sha256"),
            }
        )

    script_paths = sorted(
        {f"scripts/calibration_seeds/{sources[sid]['script']}" for sid in used_sources}
    )
    script_digests = {}
    for script_path in script_paths:
        script_file = repo_root / script_path
        if script_file.is_symlink() or not script_file.is_file():
            raise SystemExit(
                f"resolution script is missing or not a regular file: {script_path}"
            )
        script_digests[script_path] = hashlib.sha256(
            script_file.read_bytes()
        ).hexdigest()

    resolution_protocol = {
        "scripts": script_digests,
        "snapshot_rule": " ".join(cat["snapshot_rule"].split()),
        "replacement_rule": " ".join(cat["replacement_rule"].split()),
    }
    seed_document = {
        "seeds": seeds_out,
        "latest_training_cutoff": latest_training_cutoff.isoformat(),
        "resolution_protocol": resolution_protocol,
        "super_blocks": {
            name: list(blocks) for name, blocks in cat.get("super_blocks", {}).items()
        },
        "curation_sensitive_seed_ids": list(cat.get("curation_sensitive_seed_ids", [])),
    }
    unknown = sorted(
        block
        for blocks in seed_document["super_blocks"].values()
        for block in blocks
        if block not in primary_blocks
    )
    if unknown:
        raise SystemExit(f"super-blocks name unknown primary blocks: {unknown}")
    primary_ids = {row["seed_id"] for row in seeds_out if row["role"] == "primary"}
    stray = sorted(set(seed_document["curation_sensitive_seed_ids"]) - primary_ids)
    if stray:
        raise SystemExit(f"curation-sensitive ids are not primary seeds: {stray}")
    dump = dict(
        sort_keys=False, allow_unicode=True, width=110, default_flow_style=False
    )
    header = "# Generated by scripts/calibration_seeds/assemble.py; do not edit.\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seeds.yaml").write_text(
        header
        + "# Instrument-visible seed definitions. Contains no resolved values.\n"
        + yaml.safe_dump(seed_document, **dump),
        encoding="utf-8",
    )
    (output_dir / "resolutions.yaml").write_text(
        header
        + "# Resolved values and provenance. Read only by the fit; never by the instrument.\n"
        + yaml.safe_dump({"resolutions": res_out}, **dump),
        encoding="utf-8",
    )
    (output_dir / "sources.yaml").write_text(
        header + yaml.safe_dump({"sources": src_out}, **dump), encoding="utf-8"
    )

    blocks = sorted(primary_blocks)
    print(f"seeds by role: {dict(roles)}")
    print(
        f"primary blocks ({len(blocks)}): "
        + ", ".join(f"{b}={len(primary_blocks[b])}" for b in blocks)
    )
    zero = sum(
        1 for s in seeds_out if s["role"] == "primary" and s["can_resolve_to_zero"]
    )
    print(f"primary seeds that can resolve to zero: {zero}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
