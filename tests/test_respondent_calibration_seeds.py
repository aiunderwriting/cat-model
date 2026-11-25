from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from catmodel.collection.respondent_calibration import build_calibration_instrument
from catmodel.contracts.elicitation import Respondent
from catmodel.contracts.respondent_calibration import (
    load_calibration_seeds,
    load_seed_resolutions,
    resolution_script_digests,
    resolved_outcomes,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEED_PATH = _REPO_ROOT / "data/calibration/respondent_spread/seeds.yaml"
_RESOLUTION_PATH = _REPO_ROOT / "data/calibration/respondent_spread/resolutions.yaml"
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


def test_committed_seed_package_loads_and_resolves() -> None:
    seed_set = load_calibration_seeds(_SEED_PATH)
    resolutions = load_seed_resolutions(_RESOLUTION_PATH)

    outcomes = resolved_outcomes(seed_set, resolutions)

    primary = tuple(seed for seed in seed_set.seeds if seed.role == "primary")
    primary_blocks = {seed.block_id for seed in primary}
    assert len(primary) >= 40
    assert len(primary_blocks) >= 15
    assert len(outcomes) == len(seed_set.seeds)

    assert resolution_script_digests(seed_set, _REPO_ROOT) == {}

    respondents = (
        Respondent(respondent_id="gpt-6-luna:0", model_id="gpt-6-luna", sample_index=0),
    )
    instrument = build_calibration_instrument(_SEED_PATH, respondents)
    assert tuple(seed.seed_id for seed in instrument.seeds) == tuple(
        seed.seed_id for seed in seed_set.seeds
    )
    assert instrument.latest_training_cutoff == date(2026, 6, 30)

    assert all(
        row.period_end < seed.window_start
        for seed in seed_set.seeds
        for row in seed.evidence.history
    )


def test_frozen_resolvers_contain_no_email_addresses() -> None:
    script_root = _REPO_ROOT / "scripts/calibration_seeds"
    email_files = [
        path.relative_to(_REPO_ROOT)
        for path in script_root.rglob("*")
        if path.is_file()
        and _EMAIL_PATTERN.search(path.read_text(encoding="utf-8", errors="replace"))
    ]

    assert email_files == []


def test_committed_robustness_groups_cover_the_primary_blocks() -> None:
    """Every primary block sits in one super-block, and the curation-sensitive
    seeds are the primary DeFiHackLabs, Exploit Database and OSV MAL- seeds."""
    seed_set = load_calibration_seeds(_SEED_PATH)
    primary = [seed for seed in seed_set.seeds if seed.role == "primary"]
    grouped = [block for blocks in seed_set.super_blocks.values() for block in blocks]
    assert sorted(grouped) == sorted({seed.block_id for seed in primary})
    assert set(seed_set.curation_sensitive_seed_ids) == {
        seed.seed_id
        for seed in primary
        if seed.seed_id.startswith(("defihacklabs-", "exploitdb-", "osv-mal-"))
    }
