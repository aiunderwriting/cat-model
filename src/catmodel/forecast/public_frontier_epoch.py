"""Public-frontier releases from a captured Epoch AI ECI model-score table.

The table is Epoch AI's ``eci_scores.csv`` (CC BY 4.0): one row per model with
its ECI score, 90% bootstrap interval, release date, organization and model
accessibility. Scores are on the current retrospective fit; release dates are
proxies for public availability. No score, date or attribution is imputed.
"""

import csv
import io
import re
from collections.abc import Iterable
from datetime import UTC, datetime

from catmodel.contracts.governance import SourceScaleAnchor
from catmodel.contracts.public_frontier import PublicFrontierRelease

EPOCH_ECI_SCORES_PATH = "evidence/epoch_eci_scores.csv"
# CC BY 4.0 credit carried by every forecast built from the table.
EPOCH_ATTRIBUTION = (
    "ECI data: Epoch AI, 'Capabilities & benchmarking', published online at "
    "epoch.ai, retrieved from https://epoch.ai/benchmarks; licensed CC BY 4.0."
)
EPOCH_ECI_COLUMNS = frozenset(
    {
        "Model",
        "Display name",
        "eci",
        "eci_ci_low",
        "eci_ci_high",
        "date",
        "Organization",
        "Model accessibility",
    }
)
# Attribution convention: Epoch organization labels that belong to each
# forecast lab. Google research and Google DeepMind releases are one lab.
EPOCH_LAB_ORGANIZATIONS: dict[str, frozenset[str]] = {
    "openai": frozenset({"OpenAI"}),
    "anthropic": frozenset({"Anthropic"}),
    "google": frozenset({"Google DeepMind", "Google", "Google DeepMind,Google"}),
    "meta": frozenset({"Meta AI"}),
    "xai": frozenset({"xAI"}),
}
# Accessibility labels that make a model part of the public frontier. Unknown,
# limited and unreleased models are excluded rather than assumed public.
EPOCH_PUBLIC_ACCESSIBILITY = frozenset(
    {
        "API access",
        "Hosted access (no API)",
        "Open weights (unrestricted)",
        "Open weights (restricted use)",
        "Open weights (non-commercial)",
    }
)
# Epoch reports 90% bootstrap intervals; half their width over this normal
# quantile is the benchmark-error SD.
EPOCH_INTERVAL_Z = 1.6448536269514722


def _release_id(model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    if not slug:
        raise ValueError(f"Epoch model name {model!r} has no identifier characters")
    return slug


def epoch_eci_releases(
    text: str,
    *,
    evidence_ref: str,
    labs: Iterable[str],
    as_of: datetime,
    scale_anchors: Iterable[SourceScaleAnchor],
) -> tuple[PublicFrontierRelease, ...]:
    """Eligible public releases of the named labs available by ``as_of``.

    A row is selected when its organization maps to a requested lab, its
    accessibility is public and its release date is on or before as-of. A
    selected row must carry a score and date. A missing interval is allowed
    only for one of the source's scale anchors, the models to which Epoch
    assigns fixed scores to define the ECI scale (authored in the table's
    provenance), matched exactly by model name, organization and score; it
    gives zero benchmark error.
    """
    labs = tuple(labs)
    anchors = {
        (anchor.model, anchor.organization, anchor.value) for anchor in scale_anchors
    }
    unknown = sorted(set(labs) - set(EPOCH_LAB_ORGANIZATIONS))
    if unknown:
        raise ValueError(f"no Epoch organization mapping for labs {unknown}")
    organizations = {
        organization: lab
        for lab in labs
        for organization in EPOCH_LAB_ORGANIZATIONS[lab]
    }
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    missing = sorted(EPOCH_ECI_COLUMNS - set(reader.fieldnames or ()))
    if missing:
        raise ValueError(f"Epoch ECI table lacks columns {missing}")
    releases = []
    for row in reader:
        lab = organizations.get(row["Organization"])
        if lab is None or row["Model accessibility"] not in EPOCH_PUBLIC_ACCESSIBILITY:
            continue
        if not row["eci"] or not row["date"]:
            raise ValueError(f"Epoch row {row['Model']!r} lacks a score or date")
        available_at = datetime.fromisoformat(row["date"]).replace(tzinfo=UTC)
        if available_at > as_of:
            continue
        if bool(row["eci_ci_low"]) != bool(row["eci_ci_high"]):
            raise ValueError(f"Epoch row {row['Model']!r} has a one-sided interval")
        low, high = row["eci_ci_low"], row["eci_ci_high"]
        if not low and (
            (row["Model"], row["Organization"], float(row["eci"])) not in anchors
        ):
            raise ValueError(
                f"Epoch row {row['Model']!r} ({row['Organization']}, ECI {row['eci']}) "
                "lacks a 90% interval but is not one of the scale anchors in the "
                f"table's provenance {sorted(anchors)}; a missing interval cannot be "
                "read as zero benchmark error"
            )
        benchmark_sd = (
            (float(high) - float(low)) / (2 * EPOCH_INTERVAL_Z) if low else 0.0
        )
        interval = (
            f"90% bootstrap interval [{low}, {high}]"
            if low
            else "fixed scale anchor without interval"
        )
        releases.append(
            PublicFrontierRelease(
                release_id=_release_id(row["Model"]),
                lab_id=lab,
                available_at=available_at,
                value=float(row["eci"]),
                benchmark_error_sd=benchmark_sd,
                evidence_refs=(evidence_ref,),
                availability_basis="release_date_proxy",
                rationale=(
                    f"Epoch ECI {row['eci']} for {row['Display name']} "
                    f"({row['Organization']}; {row['Model accessibility']}); "
                    f"{interval}; Epoch release date {row['date']}. "
                    "Source: Epoch AI, CC BY 4.0."
                ),
            )
        )
    return tuple(releases)
