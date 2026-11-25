"""Synthetic general-ECI lab forecasts for tests, and the collection of some
of a public study's trajectory nodes."""

from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from catmodel.contracts.public_frontier import PublicFrontierForecast


def collect_study_nodes(study: Path, draw_ids: Collection[str], **options: Any):
    """Freeze and collect only the contexts of the trajectory nodes
    ``draw_ids`` of a public study's plan, leaving the plan's other nodes
    uncollected. ``options`` are those of ``run_collections``, which
    ``run_study_collection`` gives every context of the plan."""
    from catmodel.collection.contextual_collection import run_collections
    from catmodel.contracts.contextual_risk import ContextualRiskInstrument
    from catmodel.json_io import strict_json_loads
    from catmodel.setup.public_study import public_study_collections

    return run_collections(
        [
            (
                ContextualRiskInstrument.model_validate(
                    strict_json_loads(instrument.read_bytes())
                ),
                capture,
            )
            for instrument, capture in public_study_collections(study, draw_ids)
        ],
        **options,
    )


def synthetic_release(lab, release, date, value, *, sd=1.0, basis=None):
    return {
        "lab_id": lab,
        "release_id": release,
        "available_at": date,
        "value": value,
        "benchmark_error_sd": sd,
        "availability_basis": basis or "documented_public_release",
        "evidence_refs": [f"urn:fixture:{release}"],
        "rationale": "Synthetic availability evidence for software verification.",
    }


# Each lab's last record is its observed as-of frontier: lab_a 100, lab_b 90.
SYNTHETIC_RELEASES = (
    synthetic_release("lab_a", "a-prewindow", "2025-11-01T00:00:00Z", 88),
    synthetic_release("lab_a", "a-1", "2026-02-01T00:00:00Z", 93),
    synthetic_release("lab_a", "a-small", "2026-05-01T00:00:00Z", 91),
    synthetic_release("lab_a", "a-2", "2026-08-15T00:00:00Z", 100),
    synthetic_release("lab_b", "b-1", "2026-03-01T00:00:00Z", 85),
    synthetic_release("lab_b", "b-2", "2026-09-01T00:00:00Z", 90),
)


def synthetic_forecast(*, uncertain=True, releases=SYNTHETIC_RELEASES):
    return PublicFrontierForecast.model_validate(
        {
            "forecast_id": "synthetic-public-frontier",
            "as_of": datetime(2026, 10, 1, tzinfo=UTC),
            "frontier_definition": "Synthetic public lab frontiers on one ECI scale; verified availability is required.",
            "coordinates": [
                {
                    "lab_id": lab,
                    "domain_id": "general",
                    "scale_id": "eci",
                    "units": "ECI points",
                    "evidence_refs": ["urn:fixture:synthetic-public-source"],
                    "rationale": "Synthetic lab frontier for software verification.",
                    "limitations": ["No real measurements."],
                }
                for lab in ("lab_a", "lab_b")
            ],
            "trend_window_start": datetime(2026, 1, 1, tzinfo=UTC),
            "trend_discrepancy_sd": 2,
            "rate_prior": {
                "mean": 10,
                "sd": 2 if uncertain else 0,
                "rationale": "Synthetic rate prior, not an empirical estimate.",
            },
            "releases": list(releases),
            "pace_shape": 3,
            "pace_rationale": "Synthetic log-logistic shape, not a production default.",
            "pooling_rationale": "Synthetic labs share one trend rate; each lab draws its own future pace.",
            "likelihood_rationale": "Synthetic Gaussian discrepancy on each frontier-raising release.",
            "limitations": ["Synthetic software fixture only."],
        }
    )
