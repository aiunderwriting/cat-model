"""Package replay-validated contextual judgments as a native run setup."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from catmodel.contracts.incident import (
    DEFAULT_MAX_REFERENCE_STATE_YEARS,
    DEFAULT_WORK_ALLOWANCES,
    DEFAULT_YEARS_PER_STATE,
    Evidence,
)
from catmodel.contracts.run_setup import EvidenceSet, RunSetup
from catmodel.elicitation.contextual_risk import (
    contextual_tail_diagnostic_limitations,
)
from catmodel.setup.contextual_inputs import (
    load_contextual_collection,
)
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.excluded_samples import MAX_EXCLUDED_SAMPLE_SHARE
from catmodel.setup.store import build_setup
from catmodel.stored_files import StoredFiles


def import_contextual_setup(
    *,
    source: Path,
    destination: Path,
    setup_id: str,
    source_bundle: Path | None = None,
    years_per_state: int = DEFAULT_YEARS_PER_STATE,
    max_reference_state_years: int | None = DEFAULT_MAX_REFERENCE_STATE_YEARS,
    work_allowances: Mapping[str, int] = DEFAULT_WORK_ALLOWANCES,
    seed: int = 42,
    max_excluded_sample_share: float = MAX_EXCLUDED_SAMPLE_SHARE,
    code_commit: str | None = None,
) -> RunSetup:
    """Replay original calls and generate the setup without new collection.

    All captured prompts, attempts, answers and upstream source bytes remain in
    the package. A model whose respondents lacking one leaf exceed
    ``max_excluded_sample_share`` does not answer that leaf, and its
    respondents read other models' answers to it
    (``POOL-UNANSWERED-SCENARIOS``); any other respondent with an unanswered
    leaf is excluded, within the same share (``POOL-EXCLUDED-SAMPLES``).
    """
    instrument, artifact, files = load_contextual_collection(source, source_bundle)
    setup = compile_contextual_setup(
        instrument,
        artifact,
        setup_id=setup_id,
        years_per_state=years_per_state,
        max_reference_state_years=max_reference_state_years,
        work_allowances=work_allowances,
        seed=seed,
        max_excluded_sample_share=max_excluded_sample_share,
        files=files,
    )
    digest = files.sha256("contextual-risk.json")
    original_attempts = tuple(
        files.sha256(name) for name in sorted(files) if "/attempt-" in name
    )
    evidence = Evidence(
        evidence_id="captured_contextual_risk",
        origin="elicitation",
        original_sha256=digest,
        selected_sha256=digest,
        raw_attempt_sha256=original_attempts,
        target="Entity-conditioned expected incident counts and paired societal/lab economic loss marks.",
        conditioning="Exact named lab, deployed model, population, period, fixed controls/exposure/adoption and constant capability path from the captured instrument.",
        units="Original native monetary units and qualified incident counts; no insurance-payment interpretation.",
        transport="Captured answers are replay-validated. The declared linear parameter interpolation fills integer constant-path queries; no future-path distribution is inferred.",
        acquisition={
            "instrument_sha256": instrument.digest(),
            "source_bundle_manifest_sha256": instrument.source_bundle_manifest_sha256,
            "context_id": instrument.context_id,
        },
        review=artifact.evidence_status,
        reliance="Numerical admission does not establish empirical validity. Original judgments and any unavailable node or observable remain explicit.",
        rights="Original source and acquisition rights remain attached to the captured bytes; no external sharing is performed.",
    )
    setup = replace(
        setup,
        evidence=EvidenceSet(
            records=(evidence, *setup.evidence.records),
            files=StoredFiles.merged(files.prefixed("captured/"), setup.evidence.files),
        ),
        limitations=(
            *setup.limitations,
            "Respondent spread is uncalibrated: stated parameter quantiles are "
            "read at face value (k = 1).",
            *contextual_tail_diagnostic_limitations(((artifact, instrument),)),
        ),
    )
    return build_setup(
        destination=destination,
        setup=setup,
        code_commit=code_commit,
    )
