"""Offline public-forecast planning and replayed conditional-risk assembly."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np

from catmodel.artifacts import freeze_json
from catmodel.contracts.capability_paths import JointCapabilityCoordinate
from catmodel.contracts.contextual_risk import (
    ContextualRiskArtifact,
    ContextualRiskInstrument,
    render_exposure,
)
from catmodel.contracts.elicitation import Respondent
from catmodel.contracts.incident import (
    DEFAULT_MAX_COMPARISON_STATE_YEARS,
    DEFAULT_MAX_REFERENCE_STATE_YEARS,
    DEFAULT_WORK_ALLOWANCES,
    DEFAULT_YEARS_PER_STATE,
    Evidence,
)
from catmodel.contracts.public_frontier import (
    NO_PROGRESS_NODE_ID,
    PublicFrontierBacktest,
    PublicFrontierForecast,
    PublicFrontierQuadrature,
    PublicTrendPosterior,
    progress_elapsed_years,
)
from catmodel.contracts.public_frontier_condition import (
    ObservedFrontier,
    PublicTrajectoryCondition,
)
from catmodel.contracts.run_setup import EvidenceSet, RunSetup
from catmodel.elicitation.contextual_risk import (
    contextual_tail_diagnostic_limitations,
)
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.forecast.public_frontier import (
    compare_public_frontier,
    observed_frontier,
    one_year_before,
)
from catmodel.forecast.public_frontier_backtest import backtest_public_frontier
from catmodel.forecast.public_frontier_epoch import (
    EPOCH_ATTRIBUTION,
    EPOCH_ECI_SCORES_PATH,
    epoch_eci_releases,
)
from catmodel.forecast.public_frontier_quadrature import (
    DEFAULT_TRAJECTORY_NODES,
    public_frontier_quadrature,
)
from catmodel.frozen_calls import (
    CaptureFiles,
)
from catmodel.json_io import strict_json_loads
from catmodel.output_paths import require_new_evidence_output
from catmodel.resources import scenarios_path
from catmodel.scenarios.loader import load_scenarios
from catmodel.setup.contextual_hierarchy import HierarchyOutcome, hierarchy_outcomes
from catmodel.setup.contextual_inputs import (
    build_contextual_instrument,
    load_contextual_collection,
    require_scenario_nodes,
)
from catmodel.setup.contextual_node import (
    UnitLaw,
    hypothetical_entity_limitations,
    keep_unit_laws,
    unit_laws_of,
)
from catmodel.setup.excluded_samples import MAX_EXCLUDED_SAMPLE_SHARE, RespondentKey
from catmodel.setup.public_frontier import (
    MapTasks,
    PublicFrontierRiskInput,
    context_variant,
    map_in_order,
    public_frontier_setup_fields,
    public_state_node,
)
from catmodel.setup.public_frontier_comparisons import (
    CONTEXT_VARIANT_FAMILIES,
    AuthoredCapStresses,
    PublicForecastAlternative,
)
from catmodel.setup.sensitivity_catalog import (
    EXPOSURE_SCALE_VARIANTS,
    SAFEGUARD_LEVELS,
)
from catmodel.setup.store import build_setup
from catmodel.stored_files import StoredFiles
from catmodel.worker_pool import WorkerPool, memory_bounded_workers

_RATIONALES = (
    "exposure_rationale",
    "residual_independence_rationale",
    "parameter_coupling_rationale",
)


def _rationales(value: object) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or set(value) != set(_RATIONALES)
        or any(not isinstance(item, str) or not item.strip() for item in value.values())
    ):
        raise ValueError("public study requires three explicit assembly rationales")
    return cast(dict[str, str], value)


def _read(path: Path, contract):
    return contract.model_validate(strict_json_loads(path.read_bytes()))


def _read_saved(path: Path, contract):
    """Read a file this package wrote, which states its schema version."""
    return contract.from_saved(strict_json_loads(path.read_bytes()))


def prepare_public_forecast_file(
    *, source_bundle: Path, template: Path, output: Path
) -> PublicFrontierForecast:
    """Fill a forecast template's releases from the bundle's Epoch ECI table.

    The template is a complete forecast without ``releases``. The captured
    table must carry its provenance in the bundle manifest, including the
    scale anchors that alone may lack an interval; each release cites the table
    by path and digest, and the forecast carries the scale anchors, which
    questions quote as the scale's reference, and the scale correspondence
    the provenance records, if any, which they quote to say what a point means.
    """
    require_new_evidence_output(output)
    forecast = _prepared_forecast(source_bundle, template)
    output.parent.mkdir(parents=True, exist_ok=True)
    freeze_json(output, forecast.model_dump(mode="json"))
    return forecast


def _prepared_forecast(source_bundle: Path, template: Path) -> PublicFrontierForecast:
    """The forecast ``prepare_public_forecast_file`` writes."""
    snapshot = capture_source_bundle_snapshot(source_bundle)
    if not snapshot.exists(EPOCH_ECI_SCORES_PATH):
        raise ValueError(f"source bundle does not capture {EPOCH_ECI_SCORES_PATH}")
    provenance = [
        source
        for source in snapshot.manifest.external_sources
        if EPOCH_ECI_SCORES_PATH in source.captured_paths
    ]
    if not any(
        source.uri
        and source.pulled_at
        and source.license_access_basis
        and "CC BY" in source.license_access_basis
        for source in provenance
    ):
        raise ValueError(
            f"source bundle must record the provenance of {EPOCH_ECI_SCORES_PATH}: "
            "its URI, retrieval time and CC BY license basis"
        )
    payload = strict_json_loads(template.read_bytes())
    if not isinstance(payload, dict) or "releases" in payload:
        raise ValueError("the forecast template must be an object without releases")
    labs = [row.get("lab_id") for row in payload.get("coordinates", ())]
    anchors = [anchor for source in provenance for anchor in source.scale_anchors or ()]
    correspondences = [
        source.scale_correspondence
        for source in provenance
        if source.scale_correspondence is not None
    ]
    if len(correspondences) > 1:
        raise ValueError(
            f"source bundle records more than one scale correspondence for {EPOCH_ECI_SCORES_PATH}"
        )
    releases = epoch_eci_releases(
        snapshot.read_text(EPOCH_ECI_SCORES_PATH),
        evidence_ref=f"evidence_file:{EPOCH_ECI_SCORES_PATH}#sha256={snapshot.sha256(EPOCH_ECI_SCORES_PATH)}",
        labs=labs,
        as_of=datetime.fromisoformat(payload["as_of"]).astimezone(UTC),
        scale_anchors=anchors,
    )
    return PublicFrontierForecast.model_validate(
        {
            **payload,
            "releases": [row.model_dump(mode="json") for row in releases],
            "scale_anchors": [anchor.model_dump(mode="json") for anchor in anchors],
            **(
                {"scale_correspondence": correspondences[0].model_dump(mode="json")}
                if correspondences
                else {}
            ),
            "limitations": list(
                dict.fromkeys((*payload.get("limitations", ()), EPOCH_ATTRIBUTION))
            ),
        }
    )


def backtest_public_frontier_file(source: Path, output: Path) -> PublicFrontierBacktest:
    """Score pace-dispersion candidates on the forecast's own releases offline,
    at ``backtest_public_frontier``'s default horizons, shapes, draws and seed."""
    require_new_evidence_output(output)
    result = backtest_public_frontier(_read(source, PublicFrontierForecast))
    output.parent.mkdir(parents=True, exist_ok=True)
    freeze_json(output, result.model_dump(mode="json"))
    return result


FORECAST_COMPARISON_DRAWS = 4096
"""Sampled paths per variant in the offline forecast comparison (numerical)."""


def _condition(forecast, quadrature, node, status):
    """One node's trajectory as a risk condition, with what gives its scale a
    reference: the models whose fixed scores define it, each lab's as-of
    score, from which its path starts, the roster's highest score one
    year earlier and, where the forecast records one, what its publisher
    reports a gain on the scale to have corresponded to."""
    earlier = one_year_before(quadrature.as_of)
    earlier_value = observed_frontier(forecast, earlier)
    return PublicTrajectoryCondition(
        trajectory_sha256=quadrature.trajectory_sha256(node),
        draw_id=node.draw_id,
        observed_at=quadrature.as_of,
        observed_values=tuple(
            quadrature.posterior.labs[row.lab_id].as_of_value
            for row in quadrature.coordinates
        ),
        earlier_frontier=(
            None
            if earlier_value is None
            else ObservedFrontier(at=earlier, value=earlier_value)
        ),
        scale_anchors=forecast.scale_anchors,
        scale_correspondence=forecast.scale_correspondence,
        times=quadrature.times,
        interpolation=quadrature.interpolation,
        coordinates=tuple(
            JointCapabilityCoordinate(
                coordinate_id=f"public_{row.lab_id}",
                lab_id=row.lab_id,
                domain_id="general",
                access="public",
                scale_id=row.scale_id,
                definition=f"Publicly available general ECI frontier for {row.lab_id}, in {row.units}.",
            )
            for row in quadrature.coordinates
        ),
        values=node.values,
        target_lab="public_frontier",
        source_model_status=status,
        source_references=quadrature.source_references,
        rationale="The no-progress counterfactual: every lab stays at its observed as-of frontier through the period. This node carries weight only under the no-progress forecast alternative; no separate internal capability forecast."
        if node.draw_id == NO_PROGRESS_NODE_ID
        else "One full public trajectory node, a joint draw of every lab's path, conditions the annual risk judgment; no separate internal capability forecast.",
        limitations=tuple(
            dict.fromkeys(
                (
                    *quadrature.assumptions,
                    *(
                        item
                        for row in quadrature.coordinates
                        for item in row.limitations
                    ),
                )
            )
        ),
    )


def write_public_frontier_study(
    *,
    forecast_path: Path,
    public_instrument: Path,
    not_yet_public_instrument: Path,
    assumptions_path: Path,
    output: Path,
    nodes: int | None = None,
    seed: int = 0,
) -> dict:
    """Freeze trajectory nodes and paired instruments without making provider calls.

    The forecast period is the templates' common policy period. ``nodes`` is
    the odd number of interior quadrature nodes; a no-progress node is added.
    By default a headline study has ``DEFAULT_TRAJECTORY_NODES``, and a variant
    study, whose templates carry an exposure-scale variant or a safeguard
    level, the ``(DEFAULT_TRAJECTORY_NODES - 1) // 2`` of that rule's embedded
    rule, as import requires of the variants of a default headline. ``seed``
    drives only the offline forecast comparison's sampled summaries.

    Every instrument states the capability fields of a trajectory condition
    (``_trajectory_capability``), whatever scale and anchors its template has.
    """
    require_new_evidence_output(output)
    forecast = _read(forecast_path, PublicFrontierForecast)
    rationales = _rationales(strict_json_loads(assumptions_path.read_bytes()))
    templates = {
        "public": _read_saved(public_instrument, ContextualRiskInstrument),
        "not_yet_public": _read_saved(
            not_yet_public_instrument, ContextualRiskInstrument
        ),
    }
    return _plan_study(forecast, templates, rationales, output, nodes, seed)


def _trajectory_capability(forecast: PublicFrontierForecast) -> dict[str, object]:
    """What a public study's instruments state in place of a scale reference
    and capability anchors.

    An instrument's condition is its node's complete trajectory, so its scale
    is the forecast's general-ECI scale without a scalar reference, and its
    answers are stored under a single zero selector.
    """
    coordinate = forecast.coordinates[0]
    return {
        "capability_scale": {
            "scale_id": coordinate.scale_id,
            "quantity_definition": "Public general ECI context",
            "unit": coordinate.units,
            "reference_definition": "The complete public trajectory is the risk condition; no scalar reference is used.",
            "measurement_protocol": "Authored predictive public trajectory, not an observed source-model capability.",
        },
        "capability_domain_id": "general",
        "reference_eci": 0,
        "anchor_scores": (0,),
        "integer_min": 0,
        "integer_max": 0,
    }


def public_study_templates(
    context: Mapping[str, Any],
    *,
    forecast: PublicFrontierForecast,
    source_bundle: Path,
    respondents: Sequence[Respondent],
    scenario_ids: Sequence[str] = (),
    variant: str | None = None,
) -> dict[str, ContextualRiskInstrument]:
    """The public and the not-yet-public template of a public study's context.

    Each is built for ``respondents`` and ``scenario_ids``
    (``build_contextual_instrument``). The context need not state a
    capability scale or anchors: the templates take the fields of
    ``CAPABILITY_ANCHOR_FIELDS`` from ``forecast``
    (``_trajectory_capability``), in place of any the context states.
    ``variant`` builds a variant study's templates: an ``exposure_scale``
    variant, whose factor multiplies every path-linked deployment scale, or a
    ``safeguard_level``, whose authored texts replace the stipulated
    safeguards.
    """
    variants = (*EXPOSURE_SCALE_VARIANTS, *SAFEGUARD_LEVELS)
    if variant is not None and variant not in variants:
        raise ValueError(f"variant must be one of {list(variants)}")
    return {
        status: build_contextual_instrument(
            {**context, **_trajectory_capability(forecast)},
            source_bundle=source_bundle,
            respondents=respondents,
            scenario_ids=scenario_ids,
            source_status=status,
            exposure_variant=variant if variant in EXPOSURE_SCALE_VARIANTS else None,
            safeguard_level=variant if variant in SAFEGUARD_LEVELS else None,
        )
        for status in ("public", "not_yet_public")
    }


def _plan_study(
    forecast: PublicFrontierForecast,
    templates: Mapping[str, ContextualRiskInstrument],
    rationales: Mapping[str, str],
    output: Path,
    nodes: int | None,
    seed: int,
) -> dict:
    """Plan the study ``write_public_frontier_study`` describes from its
    forecast, its public and not-yet-public templates and its rationales."""
    if any(row.public_trajectory is not None for row in templates.values()):
        raise ValueError("supply unconditioned templates, not an existing public study")
    if nodes is None:
        variants = {context_variant(template) for template in templates.values()}
        if len(variants) != 1:
            raise ValueError(
                "public and not-yet-public templates must carry the same context "
                "variant"
            )
        nodes = (
            DEFAULT_TRAJECTORY_NODES
            if variants == {None}
            else (DEFAULT_TRAJECTORY_NODES - 1) // 2
        )
    first, second = templates.values()
    if first.period != second.period:
        raise ValueError(
            "public and not-yet-public templates must share one policy period"
        )
    if forecast.as_of > min(first.period.forecast_as_of, first.period.data_cutoff_utc):
        raise ValueError("public forecast exceeds the instrument information cutoff")
    period_start = first.period.policy_period_start_utc
    period_end = first.period.policy_period_end_utc
    quadrature = public_frontier_quadrature(
        forecast, period_start=period_start, period_end=period_end, nodes=nodes
    )
    _condition(forecast, quadrature, quadrature.nodes[0], "public")
    if (
        first.lab.model_dump(exclude={"related_entity_ids"})
        != second.lab.model_dump(exclude={"related_entity_ids"})
        or first.period != second.period
        or first.respondents != second.respondents
        or first.source_bundle_manifest_sha256 != second.source_bundle_manifest_sha256
    ):
        raise ValueError(
            "public and not-yet-public templates must share lab, period, source bundle and respondents"
        )
    if any(
        getattr(first, name) != getattr(second, name)
        for name in (
            "lab_compensation_definition",
            "lab_annual_business_loss_definition",
            "lab_loss_scope",
            "lab_annual_business_loss_dependence",
        )
    ):
        raise ValueError("source-status templates must share the same lab-loss scope")
    if (
        first.path_linked_exposure != second.path_linked_exposure
        or first.fixed_exposure != second.fixed_exposure
    ):
        raise ValueError("source-status templates must share one stipulated exposure")
    median = quadrature.median_progress
    scenarios = {row.id: row for row in load_scenarios(scenarios_path())}
    for template in templates.values():
        require_scenario_nodes(template, scenarios)
    # A chain's exposure driver is chosen per capability access, so it may
    # differ between the source statuses; it is not an incident definition.
    definitions = [
        [
            {
                **node.model_dump(
                    mode="json", exclude={"frequency_chain": {"exposure"}}
                ),
                "population": node.population.model_dump(
                    mode="json", exclude={"deployment_ids"}
                ),
            }
            for node in sorted(template.nodes, key=lambda node: node.node_id)
        ]
        for template in templates.values()
    ]
    if definitions[0] != definitions[1]:
        raise ValueError(
            "source-status templates must share the same incident definitions"
        )
    if (
        first.model.entity_id == second.model.entity_id
        or first.deployment.entity_id == second.deployment.entity_id
    ):
        raise ValueError(
            "source-status templates need distinct model and deployment identities"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with TemporaryDirectory(prefix=".public-study-", dir=output.parent) as temporary:
        stage = Path(temporary) / "study"
        (stage / "instruments").mkdir(parents=True)
        freeze_json(stage / "forecast.json", forecast.model_dump(mode="json"))
        freeze_json(stage / "public-frontier.json", quadrature.model_dump(mode="json"))
        freeze_json(
            stage / "trend-posterior.json",
            quadrature.posterior.model_dump(mode="json"),
        )
        freeze_json(
            stage / "forecast-comparison.json",
            compare_public_frontier(
                forecast,
                period_start=period_start,
                period_end=period_end,
                n_draws=FORECAST_COMPARISON_DRAWS,
                seed=seed,
            ),
        )
        for node in quadrature.nodes:
            for status, template in templates.items():
                identifier = f"{node.draw_id}-{status}"
                instrument = ContextualRiskInstrument.model_validate(
                    {
                        **template.model_dump(mode="json"),
                        **_rendered_exposure(template, node.progress / median),
                        "context_id": identifier,
                        "coordinate_kind": "public_trajectory",
                        "public_trajectory": _condition(
                            forecast, quadrature, node, status
                        ),
                        **_trajectory_capability(forecast),
                    }
                )
                filename = f"instruments/{identifier}.json"
                freeze_json(stage / filename, instrument.model_dump(mode="json"))
                rows.append(
                    {
                        "draw_id": node.draw_id,
                        "source_model_status": status,
                        "instrument": filename,
                        "capture": f"collections/{identifier}",
                    }
                )
        plan = {
            "schema_version": 1,
            "forecast": "forecast.json",
            "quadrature": "public-frontier.json",
            "posterior": "trend-posterior.json",
            "target_lab": "public_frontier",
            "contexts": rows,
            **rationales,
        }
        freeze_json(stage / "public-study.json", plan)
        require_new_evidence_output(output)
        stage.rename(output)
    return {
        "trajectory_nodes": len(quadrature.nodes),
        "conditional_collections": len(rows),
        "target_lab": "public_frontier",
        "plan": str(output / "public-study.json"),
    }


def plan_public_frontier_study(
    *,
    source_bundle: Path,
    forecast_path: Path,
    context_path: Path,
    respondents: Sequence[Respondent],
    assumptions_path: Path,
    output: Path,
    scenario_ids: Sequence[str] = (),
    variant: str | None = None,
    nodes: int | None = None,
    seed: int = 0,
) -> dict:
    """Plan a public study from a source bundle, a forecast and one context.

    The forecast at ``forecast_path`` either states its ``releases`` and is
    used as it is, or has none and takes them from the bundle's Epoch ECI
    table (``prepare_public_forecast_file``). The context at ``context_path``
    describes both source statuses (``source_statuses``): its public and its
    not-yet-public template are built for ``respondents``, ``scenario_ids``
    and ``variant`` (``public_study_templates``), which takes their capability
    scale from the forecast.

    The study is the one ``write_public_frontier_study`` plans from that
    forecast and those templates: it gives ``nodes`` and ``seed`` their
    meaning and keeps the forecast as the study's ``forecast.json``.
    """
    require_new_evidence_output(output)
    context = strict_json_loads(context_path.read_bytes())
    if not isinstance(context, dict) or "source_statuses" not in context:
        raise ValueError(
            "a public study's context is a JSON object whose source_statuses "
            "describe its public and not-yet-public source models"
        )
    payload = strict_json_loads(forecast_path.read_bytes())
    forecast = (
        _prepared_forecast(source_bundle, forecast_path)
        if isinstance(payload, dict) and "releases" not in payload
        else PublicFrontierForecast.model_validate(payload)
    )
    templates = public_study_templates(
        context,
        forecast=forecast,
        source_bundle=source_bundle,
        respondents=respondents,
        scenario_ids=scenario_ids,
        variant=variant,
    )
    rationales = _rationales(strict_json_loads(assumptions_path.read_bytes()))
    return _plan_study(forecast, templates, rationales, output, nodes, seed)


def _rendered_exposure(
    template: ContextualRiskInstrument, progress_ratio: float
) -> dict[str, object]:
    """A node instrument's path-linked scales at its progress ratio."""
    exposure = template.path_linked_exposure
    if exposure is None:
        return {}
    text = template.deployment_binding.use_population
    rendered = exposure.model_copy(
        update={"progress_ratio": progress_ratio, "use_population_template": text}
    )
    return {
        "path_linked_exposure": rendered.model_dump(mode="json"),
        "deployment_binding": {
            **template.deployment_binding.model_dump(mode="json"),
            "use_population": render_exposure(text, rendered),
        },
    }


def _path(root: Path, value: object) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise ValueError(
            "public study paths must be relative and stay within the study"
        )
    return root / value


def _same_quadrature(
    left: PublicFrontierQuadrature, right: PublicFrontierQuadrature
) -> bool:
    """Check a recomputed quadrature against the frozen one up to roundoff.

    Evidence identities, labels, authored pace shapes and caps and the rule
    are compared exactly. Every value the fit or the rule computes may differ by
    floating-point roundoff across platforms and library versions and is
    compared within a roundoff tolerance: node paths, the posterior's rate
    location and variance, the estimated trend discrepancy and its interval,
    and each law's discrepancy and weights.
    """
    numeric = {"nodes", "base", "alternatives", "posterior", "median_progress"}
    if left.model_dump(exclude=numeric) != right.model_dump(exclude=numeric):
        return False
    a, b = left.posterior, right.posterior
    fitted = {
        "rate_location",
        "rate_variance",
        "trend_discrepancy_sd",
        "trend_discrepancy_interval",
    }
    if a.model_dump(exclude=fitted) != b.model_dump(exclude=fitted):
        return False
    if (a.trend_discrepancy_interval is None) != (b.trend_discrepancy_interval is None):
        return False
    if not _close(
        (
            a.rate_location,
            a.rate_variance,
            a.trend_discrepancy_sd,
            *(a.trend_discrepancy_interval or ()),
        ),
        (
            b.rate_location,
            b.rate_variance,
            b.trend_discrepancy_sd,
            *(b.trend_discrepancy_interval or ()),
        ),
    ):
        return False
    if [node.draw_id for node in left.nodes] != [node.draw_id for node in right.nodes]:
        return False
    # Each node's rates and progress are tied to its path by the contract's own
    # validator, so comparing paths and progress compares the whole draw.
    if not np.allclose(
        [node.values for node in left.nodes],
        [node.values for node in right.nodes],
        rtol=1e-8,
        atol=1e-7,
    ) or not _close(
        (left.median_progress, *(node.progress for node in left.nodes)),
        (right.median_progress, *(node.progress for node in right.nodes)),
    ):
        return False
    laws = [(left.base, right.base), *zip(left.alternatives, right.alternatives)]
    computed = {"weights", "embedded_weights", "trend_discrepancy_sd"}
    return len(left.alternatives) == len(right.alternatives) and all(
        x.model_dump(exclude=computed) == y.model_dump(exclude=computed)
        and _close((x.trend_discrepancy_sd,), (y.trend_discrepancy_sd,))
        and _close(x.weights, y.weights)
        and _close(x.embedded_weights, y.embedded_weights)
        for x, y in laws
    )


def _close(left: Sequence[float], right: Sequence[float]) -> bool:
    """Equal length and equal up to cross-platform roundoff (relative 1e-8)."""
    return len(left) == len(right) and bool(
        np.allclose(left, right, rtol=1e-8, atol=1e-10)
    )


def _forecast_alternatives(
    quadrature: PublicFrontierQuadrature,
) -> dict[str, PublicForecastAlternative]:
    """Each alternative law's exact weights on the shared trajectory nodes."""
    ids = [node.draw_id for node in quadrature.nodes]
    return {
        **{
            law.label: PublicForecastAlternative(
                path_weights=dict(zip(ids, law.weights, strict=True)),
                provenance={
                    "variant": law.variant,
                    "pace_shape": law.pace_shape,
                    "pace_cap": law.pace_cap,
                    "trend_discrepancy_sd": law.trend_discrepancy_sd,
                    "trajectory_rule": quadrature.rule,
                },
            )
            for law in quadrature.alternatives
        },
        **{
            label: PublicForecastAlternative(None, {}, reason)
            for label, reason in quadrature.unavailable_alternatives.items()
        },
    }


def _quadrature_record(
    quadrature: PublicFrontierQuadrature,
    fields: Mapping[str, Any],
    inputs: tuple[PublicFrontierRiskInput, ...],
) -> dict[str, object]:
    """Base-law parameters, node paths and weights needed by replay exhibits.

    ``state_draws`` maps the headline states to their nodes; exposure-scale variant
    states belong to other studies' nodes and are not listed.
    """
    ids = [node.draw_id for node in quadrature.nodes]
    arms = {arm.alternative_id for arm in fields["alternatives"]}
    instrument = inputs[0].instrument
    path_linked = instrument.path_linked_exposure
    max_growth_multiple = (
        None if path_linked is None else path_linked.max_growth_multiple
    )

    if any(
        (
            None
            if row.instrument.path_linked_exposure is None
            else row.instrument.path_linked_exposure.max_growth_multiple
        )
        != max_growth_multiple
        for row in inputs
    ):
        raise ValueError(
            "public study inputs must share one path-linked growth ceiling"
        )

    def weights(law):
        return {
            "variant": law.variant,
            "full": dict(zip(ids, law.weights, strict=True)),
            "embedded": dict(zip(ids, law.embedded_weights, strict=True)),
        }

    return {
        "rule": quadrature.rule,
        "interior_nodes": quadrature.interior_nodes,
        "embedded_interior_nodes": (quadrature.interior_nodes - 1) // 2,
        "node_progress": {node.draw_id: node.progress for node in quadrature.nodes},
        "base_law": {
            "pace_shape": quadrature.base.pace_shape,
            "pace_cap": quadrature.base.pace_cap,
            "posterior": quadrature.posterior.model_dump(mode="json"),
            "progress_elapsed_years": progress_elapsed_years(
                quadrature.as_of, quadrature.times
            ),
            "median_progress": quadrature.median_progress,
        },
        "node_eci": {
            node.draw_id: {
                "period_start": {
                    row.lab_id: node.values[0][index]
                    for index, row in enumerate(quadrature.coordinates)
                },
                "period_end": {
                    row.lab_id: node.values[-1][index]
                    for index, row in enumerate(quadrature.coordinates)
                },
            }
            for node in quadrature.nodes
        },
        "max_growth_multiple": max_growth_multiple,
        "draw_weights": {
            fields["execution"].reference_arm: weights(quadrature.base),
            **{
                f"public_forecast::{law.label}": weights(law)
                for law in quadrature.alternatives
                if f"public_forecast::{law.label}" in arms
            },
        },
        "state_draws": {
            state.state_id: public_state_node(state.state_id)
            for state in fields["joint_states"]
            if public_state_node(state.state_id) in ids
        },
    }


def _cap_stresses(
    instrument: ContextualRiskInstrument,
) -> dict[str, AuthoredCapStresses | str]:
    """Each node's authored repository cap stresses, or why they do not apply:
    they apply where the frozen societal cap and tail shape match the
    repository's severity governance.

    The lab-loss cap is not compared: a stress moves it only while it equals the
    tail-class endpoint the stress replaces.
    """
    scenarios = {row.id: row for row in load_scenarios(scenarios_path())}
    result: dict[str, AuthoredCapStresses | str] = {}
    for node in instrument.nodes:
        scenario_id = node.population.scenario_id
        scenario = scenarios.get(scenario_id)
        governance = None if scenario is None else scenario.severity_governance
        if governance is None:
            result[scenario_id] = (
                "No repository scenario authors severity governance for this node."
            )
            continue
        total = node.total
        if (None if total is None else total.cap) != governance.severity_cap or (
            total is not None and total.gpd_shape != governance.gpd_shape
        ):
            result[scenario_id] = (
                "The repository severity governance differs from the frozen node's societal cap or tail shape, so its authored stresses do not apply."
            )
            continue
        result[scenario_id] = AuthoredCapStresses(
            cap_values=governance.effective_severity_cap_sensitivity_values,
            uncapped_alternative=governance.severity_cap_uncapped_stress_alternative,
            endpoint_stress=governance.uncapped_endpoint_stress,
            endpoint_from_tail_class=governance.endpoint_from_tail_class,
        )
    return result


@dataclass(frozen=True)
class _ReplayedStudy:
    """One public study's frozen forecast, quadrature and replayed captures."""

    forecast: PublicFrontierForecast
    quadrature: PublicFrontierQuadrature
    rationales: dict[str, str]
    inputs: tuple[PublicFrontierRiskInput, ...]
    files: StoredFiles
    evidence: tuple[Evidence, ...]


def _study_plan(raw: bytes) -> dict:
    plan = strict_json_loads(raw)
    if (
        not isinstance(plan, dict)
        or set(plan)
        != {
            "schema_version",
            "forecast",
            "quadrature",
            "posterior",
            "target_lab",
            "contexts",
            *_RATIONALES,
        }
        or plan["schema_version"] != 1
    ):
        raise ValueError("invalid public study plan")
    return plan


def _study_rows(plan: dict) -> list[dict[str, str]]:
    if not isinstance(plan["contexts"], list):
        raise ValueError("public study contexts must be a list")
    for row in plan["contexts"]:
        if not isinstance(row, dict) or set(row) != {
            "draw_id",
            "source_model_status",
            "instrument",
            "capture",
        }:
            raise ValueError("invalid public study context")
        if any(not isinstance(value, str) or not value for value in row.values()):
            raise ValueError("public study context fields must be nonempty strings")
    return cast(list[dict[str, str]], plan["contexts"])


def public_study_collections(
    source: Path, draw_ids: Collection[str] = ()
) -> list[tuple[Path, Path]]:
    """The instrument and capture directory of each context the plan lists,
    in plan order; with ``draw_ids``, only those trajectory nodes'."""
    rows = _study_rows(_study_plan(source.read_bytes()))
    unknown = set(draw_ids) - {row["draw_id"] for row in rows}
    if unknown:
        raise ValueError(
            "public study has no trajectory node " + ", ".join(sorted(unknown))
        )
    return [
        (_path(source.parent, row["instrument"]), _path(source.parent, row["capture"]))
        for row in rows
        if not draw_ids or row["draw_id"] in draw_ids
    ]


@dataclass(frozen=True)
class _CollectionTask:
    """One capture to read, so it can be sent to a worker process."""

    capture: Path
    source_bundle: Path | None


@dataclass(frozen=True)
class _ReadCollection:
    """One capture read and checked, with what the check worked out that the
    rest of the import reads: its taxonomy-parent outcomes and its cases'
    unit laws."""

    instrument: ContextualRiskInstrument
    artifact: ContextualRiskArtifact
    files: CaptureFiles
    outcomes: Mapping[tuple[float, RespondentKey], HierarchyOutcome]
    unit_laws: Mapping[Any, UnitLaw | None]


def _read_collection(task: _CollectionTask) -> _ReadCollection:
    """One capture read and checked (``_ReadCollection``).

    Checking a collection compiles every case's law at unit count, the larger
    part of the work, and a process keeps those laws only while it lives. The
    taxonomy-parent outcomes need the same laws, so they are worked out here,
    beside the check, and the draw's selections are later made from them. The
    unit laws go back too, for the disclosures the import builds from them.
    """
    instrument, artifact, captured = load_contextual_collection(
        task.capture, task.source_bundle
    )
    outcomes = hierarchy_outcomes(instrument, artifact)
    return _ReadCollection(
        instrument, artifact, captured, outcomes, unit_laws_of(instrument)
    )


def _replay_study(
    source: Path,
    source_bundle: Path | None,
    *,
    path_weights: Callable[[PublicFrontierQuadrature], dict[str, float]],
    evidence_prefix: str,
    map_tasks: MapTasks = map_in_order,
) -> _ReplayedStudy:
    """Recompute a study's quadrature and replay the captures of its weighted nodes.

    ``path_weights`` gives each node's weight; nodes it omits are not read, so
    their rows need not be collected. Files are keyed by their study paths and
    evidence IDs start with ``evidence_prefix``. Captures are read and checked
    independently; ``map_tasks`` runs the reads, in worker processes where
    the caller supplies them, and returns them in the study's order.
    """
    raw = source.read_bytes()
    plan = _study_plan(raw)
    rationales = _rationales({key: plan[key] for key in _RATIONALES})
    if plan["target_lab"] != "public_frontier":
        raise ValueError("public study target lab must be public_frontier")
    root = source.parent
    forecast = _read(_path(root, plan["forecast"]), PublicFrontierForecast)
    quadrature = _read_saved(_path(root, plan["quadrature"]), PublicFrontierQuadrature)
    posterior = _read(_path(root, plan["posterior"]), PublicTrendPosterior)
    if posterior != quadrature.posterior:
        raise ValueError("retained posterior differs from the trajectory quadrature")
    reproduced = public_frontier_quadrature(
        forecast,
        period_start=quadrature.times[0],
        period_end=quadrature.times[-1],
        nodes=quadrature.interior_nodes,
    )
    if not _same_quadrature(reproduced, quadrature):
        raise ValueError(
            "trajectory quadrature differs from its frozen forecast and rule"
        )
    _condition(forecast, quadrature, quadrature.nodes[0], "public")
    weights = path_weights(quadrature)
    expected = {
        (draw_id, status)
        for draw_id in weights
        for status in ("public", "not_yet_public")
    }
    seen, inputs, evidence = set(), [], []
    # Each capture's files stay on disk; only the study's own four are held.
    files: list[Mapping[str, bytes]] = [
        {
            "public-study.json": raw,
            "forecast.json": _path(root, plan["forecast"]).read_bytes(),
            "public-frontier.json": _path(root, plan["quadrature"]).read_bytes(),
            "trend-posterior.json": _path(root, plan["posterior"]).read_bytes(),
        }
    ]
    by_id = {node.draw_id: node for node in quadrature.nodes}
    # Questions name only their node's trajectory; the whole rule, weights
    # included, travels beside each input.
    rule_sha256 = quadrature.digest()
    median = quadrature.median_progress
    rows = []
    for index, row in enumerate(_study_rows(plan)):
        key = row["draw_id"], row["source_model_status"]
        if key[0] not in by_id or key[1] not in ("public", "not_yet_public"):
            raise ValueError("public study has duplicate or unknown draw/status pair")
        if key not in expected:
            continue
        if key in seen:
            raise ValueError("public study has duplicate or unknown draw/status pair")
        seen.add(key)
        rows.append((index, key, row))
    collections = map_tasks(
        _read_collection,
        [
            _CollectionTask(_path(root, row["capture"]), source_bundle)
            for _, _, row in rows
        ],
    )
    for (index, key, row), read in zip(rows, collections, strict=True):
        instrument, artifact, captured = read.instrument, read.artifact, read.files
        keep_unit_laws(read.unit_laws)
        original = _read_saved(_path(root, row["instrument"]), ContextualRiskInstrument)
        capture = _path(root, row["capture"])
        if (
            _read_saved(
                capture / "contextual-instrument.json", ContextualRiskInstrument
            )
            != original
        ):
            raise ValueError("captured question differs from the planned instrument")
        if forecast.as_of > min(
            instrument.period.forecast_as_of, instrument.period.data_cutoff_utc
        ):
            raise ValueError(
                "public forecast exceeds the instrument information cutoff"
            )
        if (
            instrument.period.policy_period_start_utc,
            instrument.period.policy_period_end_utc,
        ) != (quadrature.times[0], quadrature.times[-1]):
            raise ValueError(
                "captured instrument period differs from the public forecast period"
            )
        if instrument.public_trajectory != _condition(
            forecast, quadrature, by_id[key[0]], key[1]
        ):
            raise ValueError("captured trajectory differs from its forecast node")
        exposure = instrument.path_linked_exposure
        if exposure is not None and not math.isclose(
            cast(float, exposure.progress_ratio),
            by_id[key[0]].progress / median,
            rel_tol=1e-8,
            abs_tol=1e-12,
        ):
            raise ValueError("captured path-linked exposure differs from its node")
        inputs.append(
            PublicFrontierRiskInput(
                instrument=instrument,
                artifact=artifact,
                path_weight=weights[key[0]],
                quadrature_sha256=rule_sha256,
                files=captured,
                outcomes=read.outcomes,
            )
        )
        files.append(captured.prefixed(f"contexts/{index}/"))
        artifact_digest = captured.sha256("contextual-risk.json")
        evidence.append(
            Evidence(
                evidence_id=f"{evidence_prefix}_{index}",
                origin="elicitation",
                original_sha256=artifact_digest,
                selected_sha256=artifact_digest,
                raw_attempt_sha256=tuple(
                    captured.sha256(name)
                    for name in sorted(captured)
                    if "/attempt-" in name
                ),
                target="Annual conditional count and joint loss law for one public trajectory and source status",
                conditioning=f"{key[0]} / {key[1]}; joint public lab general-ECI trajectories, target public_frontier; trajectory node of the {quadrature.rule} rule",
                units="Original annual count and monetary units",
                transport="Whole conditional law retained in its path-weighted mixture",
                review=artifact.evidence_status,
                reliance="Replay validates provenance, not predictive accuracy",
                rights="Original captured source and acquisition rights retained",
            )
        )
    if seen != expected:
        raise ValueError("public study must retain every node and both source statuses")
    return _ReplayedStudy(
        forecast=forecast,
        quadrature=quadrature,
        rationales=rationales,
        inputs=tuple(inputs),
        files=StoredFiles.merged(*files),
        evidence=tuple(evidence),
    )


def _base_weights(quadrature: PublicFrontierQuadrature) -> dict[str, float]:
    """Every node at its base forecast weight, the no-progress node included."""
    return {
        node.draw_id: weight
        for node, weight in zip(quadrature.nodes, quadrature.base.weights, strict=True)
    }


def _context_variant(
    source: Path,
    source_bundle: Path | None,
    headline: _ReplayedStudy,
    map_tasks: MapTasks = map_in_order,
) -> tuple[str, str, _ReplayedStudy]:
    """Replay one context-variant study and check it against the headline;
    returns its family, the one variant every instrument of it carries, its
    variant ID and the replayed study.

    Every node of the variant's plan is read: the embedded rule's interior nodes
    and the no-progress node, which carries no weight in any arm and gives
    each variant current-frontier states for capability-conditioned reporting.
    ``map_tasks`` runs the reads of its captures (``_replay_study``).
    """
    uncollected = [
        capture
        for _, capture in public_study_collections(source, (NO_PROGRESS_NODE_ID,))
        if not (capture / "contextual-risk.json").is_file()
    ]
    if uncollected:
        raise ValueError(
            f"variant study {source} must collect its no-progress node "
            f"{NO_PROGRESS_NODE_ID} for both source statuses; uncollected: "
            + ", ".join(str(path) for path in uncollected)
        )
    variant = _replay_study(
        source,
        source_bundle,
        path_weights=_base_weights,
        evidence_prefix="risk",
        map_tasks=map_tasks,
    )
    variants = {context_variant(row.instrument) for row in variant.inputs}
    found = next(iter(variants))
    if len(variants) != 1 or found is None:
        raise ValueError(
            f"{source} is not a variant study: every instrument of one carries "
            "the same exposure-scale variant or the same safeguard level"
        )
    family, variant_id = found
    kind = CONTEXT_VARIANT_FAMILIES[family]
    a, b = headline.quadrature, variant.quadrature
    if (
        a.forecast_sha256 != b.forecast_sha256
        or a.times != b.times
        or b.interior_nodes != (a.interior_nodes - 1) // 2
    ):
        raise ValueError(
            f"{kind.noun} variant study needs the headline forecast, period and "
            "the embedded rule's node count"
        )
    if family == "exposure_scale":
        exposure = variant.inputs[0].instrument.path_linked_exposure
        assert exposure is not None and exposure.variant is not None
        hidden = f"every path-linked scale times {exposure.variant.factor:g}"
    else:
        hidden = f"the study config's {variant_id} safeguard texts"
    return (
        family,
        variant_id,
        replace(
            variant,
            files=variant.files.prefixed(f"{kind.state_prefix}-{variant_id}/"),
            evidence=tuple(
                row.model_copy(
                    update={
                        "evidence_id": f"{family}_{variant_id}_{row.evidence_id}",
                        "conditioning": f"{row.conditioning}; {family} variant "
                        f"{variant_id}, {hidden} (not shown to respondents)",
                    }
                )
                for row in variant.evidence
            ),
        ),
    )


# Before a task has run in a worker: a bound on a worker's peak resident
# memory while it reads one capture of a public study or compiles one
# trajectory draw, its imports included. Later maps size workers at the last
# measured peak.
DRAW_WORKER_BYTES = 1_000_000_000


def _draw_map(pool: WorkerPool) -> MapTasks:
    """``MapTasks`` on ``pool``: as many tasks at once as available memory
    holds at the last measured worker peak (before any, ``DRAW_WORKER_BYTES``),
    returned in task order. Captures and draws are each mapped by their own,
    so one's peak does not size the other's workers."""
    measured: list[int] = []

    def map_tasks(function, tasks):
        per_worker = measured[-1] if measured else DRAW_WORKER_BYTES
        processes = min(len(tasks), memory_bounded_workers(pool.workers, per_worker))
        results = pool.map(function, tasks, concurrency=processes)
        if pool.peak_task_bytes is not None:
            measured.append(pool.peak_task_bytes)
        return results

    return map_tasks


def import_public_frontier_study(
    *,
    source: Path,
    destination: Path,
    setup_id: str,
    source_bundle: Path | None = None,
    years_per_state: int = DEFAULT_YEARS_PER_STATE,
    max_reference_state_years: int | None = DEFAULT_MAX_REFERENCE_STATE_YEARS,
    max_comparison_state_years: int | None = DEFAULT_MAX_COMPARISON_STATE_YEARS,
    work_allowances: Mapping[str, int] = DEFAULT_WORK_ALLOWANCES,
    seed: int = 42,
    comparison_families: tuple[str, ...] | None = None,
    variant_studies: tuple[Path, ...] = (),
    max_excluded_sample_share: float = MAX_EXCLUDED_SAMPLE_SHARE,
    code_commit: str | None = None,
    workers: int = 1,
) -> RunSetup:
    """Replay every paired capture before forming one weighted loss mixture.

    The forecast is integrated over the study's trajectory nodes with their
    frozen base weights. ``comparison_families=None`` selects the compact
    context-bound default (``DEFAULT_CONTEXT_BOUND_FAMILIES``). Forecast
    alternatives reweight the same nodes by their exact quadrature weights; cap
    stresses come from the repository's severity governance only where it
    matches the frozen node. Each of ``variant_studies`` is a public study of
    the same forecast on the embedded rule's nodes and the no-progress node,
    and the variant its instruments carry gives its family: an
    ``exposure_scale`` variant's deployment scales carry one variant factor,
    and a ``safeguard_level`` variant's stipulated safeguards are one authored
    level. The setup holds the variants by family, ``exposure_scale`` first,
    in the order given within a family, whatever the order of the families.
    A variant's no-progress node must match the headline's and enters no arm.
    Every count law is read as stated (``POOL-SPREAD``). In each study a model
    whose samples lacking one leaf exceed ``max_excluded_sample_share`` does
    not answer that leaf, and its states read other models' answers to it; a
    variant study also leaves a model out of each leaf it does not answer in
    the headline (``POOL-UNANSWERED-SCENARIOS``). Any other respondent with an
    unanswered leaf at a node has no state there, within the same share
    (``POOL-EXCLUDED-SAMPLES``). Up to ``workers`` processes compile the
    trajectory draws of the reference and each comparison arm at once
    (``_draw_map``); the setup does not depend on it. Workers are spawned, so
    a Python script that calls this with more than one worker must guard its
    entry point with ``if __name__ == "__main__":``.
    """
    require_new_evidence_output(destination)
    # These workers read and check the captures and are stopped before the
    # draws compile: the laws each compiled for its checks are not needed
    # again, and the draws' workers start without them.
    with WorkerPool(workers) as pool:
        read_captures = _draw_map(pool)
        headline = _replay_study(
            source,
            source_bundle,
            path_weights=_base_weights,
            evidence_prefix="public_path_risk",
            map_tasks=read_captures,
        )
        forecast, quadrature = headline.forecast, headline.quadrature
        inputs, rationales = headline.inputs, headline.rationales
        carried = {
            found[0]
            for row in inputs
            if (found := context_variant(row.instrument)) is not None
        }
        if carried:
            family = min(carried)
            raise ValueError(
                "the headline study cannot carry "
                + (
                    "an exposure variant"
                    if family == "exposure_scale"
                    else "a safeguard level"
                )
                + f"; import it with {CONTEXT_VARIANT_FAMILIES[family].study_option}"
            )
        variants: dict[str, dict[str, _ReplayedStudy]] = {
            family: {} for family in CONTEXT_VARIANT_FAMILIES
        }
        for path in variant_studies:
            family, variant_id, variant = _context_variant(
                path, source_bundle, headline, read_captures
            )
            if variant_id in variants[family]:
                raise ValueError(f"{family} variant {variant_id!r} is imported twice")
            variants[family][variant_id] = variant
        variants = {family: found for family, found in variants.items() if found}
    captured_files = [headline.files]
    captured_evidence = list(headline.evidence)
    for by_variant in variants.values():
        for variant in by_variant.values():
            captured_files.append(variant.files)
            captured_evidence.extend(variant.evidence)
    files = StoredFiles.merged(*captured_files)
    # Each simulated arm's law overrides are written beside the destination as
    # the arm compiles, and copied into the package when it is built, so the
    # import holds one arm at a time.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".setup-arms-", dir=destination.parent) as arms:
        # The workers live only while draws compile, so the end of the import,
        # its largest allocation, does not share memory with them.
        with WorkerPool(workers) as pool:
            fields = public_frontier_setup_fields(
                tuple(inputs),
                setup_id=setup_id,
                years_per_state=years_per_state,
                max_reference_state_years=max_reference_state_years,
                max_comparison_state_years=max_comparison_state_years,
                work_allowances=work_allowances,
                seed=seed,
                epistemic_tail_scale=1.0,
                comparison_families=comparison_families,
                forecast_alternatives=_forecast_alternatives(quadrature)
                if comparison_families is None
                or "public_forecast" in comparison_families
                else {},
                cap_stresses=_cap_stresses(inputs[0].instrument),
                context_variants={
                    family: {
                        variant_id: variant.inputs
                        for variant_id, variant in by_variant.items()
                    }
                    for family, by_variant in variants.items()
                },
                embedded_path_weights=dict(
                    zip(
                        (node.draw_id for node in quadrature.nodes),
                        quadrature.base.embedded_weights,
                        strict=True,
                    )
                ),
                max_excluded_sample_share=max_excluded_sample_share,
                map_tasks=_draw_map(pool),
                override_directory=Path(arms),
                **rationales,
            )
        evidence = Evidence(
            evidence_id="public_frontier_conditional_risk",
            origin="fit"
            if any(row.observation_count for row in quadrature.posterior.labs.values())
            else "stipulation",
            original_sha256=files.sha256("forecast.json"),
            selected_sha256=files.sha256("public-frontier.json"),
            target=f"Joint public trajectory from {quadrature.times[0].isoformat()} to {quadrature.times[-1].isoformat()} and paired source-status loss laws",
            conditioning="Annual loss laws conditional on complete joint lab general-ECI trajectories anchored at observed as-of frontiers, each lab at its own rate B·M_l; the forecast's frontier progress integrated over fixed quadrature nodes, each carrying one representative joint path",
            units="General ECI points, annual counts and ultimate loss",
            transport="; ".join(rationales.values()),
            review="Conditional Bayesian trend and authored forward rate assumptions; retained risk judgments remain unvalidated",
            acquisition={
                "trend_method": quadrature.posterior.method,
                "trend_observation_counts": {
                    lab_id: row.observation_count
                    for lab_id, row in quadrature.posterior.labs.items()
                },
                "as_of_frontier": {
                    lab_id: row.as_of_value
                    for lab_id, row in quadrature.posterior.labs.items()
                },
                "pace_parameters": "authored",
                "pace_shape": forecast.pace_shape,
                "pace_cap": forecast.pace_cap,
                "target_lab": "public_frontier",
                "trajectory_quadrature": _quadrature_record(quadrature, fields, inputs),
            },
            reliance="Deterministic trajectory quadrature with an embedded-rule error indicator; no empirical covariance or complete residual-dependence claim",
            rights="Original captured source rights retained",
        )
        # The compiled setup opens with its hypothetical-entity disclosure.
        hypothetical = len(hypothetical_entity_limitations(fields["scope"].entities))
        limitations = fields["limitations"]
        # Constructed once, so validated once, with this import's evidence and
        # limitations in place.
        composed: dict[str, Any] = {
            **fields,
            "evidence": EvidenceSet(
                records=(
                    evidence,
                    *captured_evidence,
                    *fields["evidence"].records,
                ),
                files=StoredFiles.merged(
                    files.prefixed("captured/"), fields["evidence"].files
                ),
            ),
            "limitations": (
                *limitations[:hypothetical],
                # POOL-SPREAD: no setup reads a respondent fit.
                "Respondent spread is uncalibrated: stated parameter quantiles "
                "are read at face value (k = 1).",
                *limitations[hypothetical:],
                *contextual_tail_diagnostic_limitations(
                    (row.artifact, row.instrument) for row in inputs
                ),
            ),
        }
        del fields
        built = [RunSetup(**composed)]
        del composed
        # Passed inline so that build_setup holds the only reference and can
        # release the composed setup before decoding its saved copy.
        return build_setup(
            destination=destination, setup=built.pop(), code_commit=code_commit
        )
