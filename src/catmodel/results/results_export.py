"""Portable presentation index over validated native run setups and annual samples."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import numbers
import statistics
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import numpy as np

from catmodel.artifacts import freeze_json, frozen_json
from catmodel.contracts.contextual_risk import (
    ContextualRiskArtifact,
    ContextualRiskInstrument,
)
from catmodel.contracts.incident import counts_lab_loss
from catmodel.contracts.output import (
    ArtifactReference,
    ExceedancePoint,
    OutputSeries,
    ResultsExport,
)
from catmodel.contracts.public_frontier import (
    PublicFrontierQuadrature,
    PublicQuadratureNode,
)
from catmodel.contracts.results import ContextualChainSummary, ResultsDiagnostics
from catmodel.contracts.run_setup import RunSetup
from catmodel.elicitation.contextual_frequency import driving_chain_answer
from catmodel.elicitation.eal_crosscheck import final_answer, pooled_answer
from catmodel.hashing import canonical_json
from catmodel.json_io import require_marker, strict_json_loads
from catmodel.output_paths import require_new_evidence_output
from catmodel.resources import scenarios_path
from catmodel.results.conditional import ConditionalCapture
from catmodel.results.execution import (
    CAPTURE_MANIFEST,
    StateLayout,
    arm_weights,
    load_run_capture,
    reduce_capture,
    validate_captured_laws,
)
from catmodel.results.reduction import ReductionSpec
from catmodel.results.results_file import (
    diagnostics_path,
    diagnostics_payload,
    load_results,
    read_diagnostics_bytes,
    results_payload,
)
from catmodel.scenarios.loader import load_scenarios
from catmodel.setup.contextual_node import (
    leaf_query,
    unit_law,
)
from catmodel.setup.excluded_samples import EXCLUDED_SAMPLES_PATH, kept_cases
from catmodel.setup.store import (
    MANIFEST,
    OMITTED_EVIDENCE,
    captured_answer_files,
    omitted_evidence_record,
    omitted_files,
    setup_code_provenance,
)
from catmodel.simulation.capability_mixture import effective_sample_size
from catmodel.simulation.incident import source_status_view
from catmodel.simulation.incident_runtime import state_key
from catmodel.stored_files import StoredFiles
from catmodel.worker_pool import WorkerPool


def _pointer(*parts: str) -> str:
    return "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in parts)


CAPABILITY_SCENARIOS_CSV_COLUMNS = (
    "scenario",
    "label",
    "scenario_status",
    "forecast_probability",
    "percentile",
    "progress",
    "progress_cdf",
    "eci_period_start",
    "eci_period_end",
    "lower_node",
    "upper_node",
    "interpolation_weight",
    "view",
    "measure",
    "source_status",
    "quantity",
    "non_exceedance_probability",
    "return_period_years",
    "loss_threshold",
    "value",
    "status",
    "reason",
    "sampling_standard_error",
    "sampling_reason",
    "half_width",
    "relative_half_width",
    "minimum_batch_exceedances",
    "years_per_state",
)
CAPABILITY_NODES_CSV_COLUMNS = (
    "node_id",
    "progress",
    "progress_cdf",
    "eci_period_start",
    "eci_period_end",
    "base_weight",
    "arm",
    "samples",
    "view",
    "measure",
    "source_status",
    "expected_annual_loss",
    "status",
    "reason",
    "sampling_standard_error",
    "sampling_reason",
)


def _csv_text(columns, rows) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_csv_cell(row.get(column)) for column in columns])
    return output.getvalue()


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return repr(float(value))
    return str(value)


def _scenario_csv_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if record.get("status") != "available":
        return []
    rows = []
    for scenario in record.get("scenarios", ()):
        interpolation = scenario.get("interpolation") or {}
        lower = interpolation.get("lower_node") or {}
        upper = interpolation.get("upper_node") or {}
        common = {
            "scenario": scenario.get("scenario"),
            "label": scenario.get("label"),
            "scenario_status": scenario.get("status"),
            "forecast_probability": scenario.get("forecast_probability"),
            "percentile": scenario.get("percentile"),
            "progress": scenario.get("progress"),
            "progress_cdf": scenario.get("progress_cdf"),
            "eci_period_start": scenario.get("eci_period_start"),
            "eci_period_end": scenario.get("eci_period_end"),
            "lower_node": lower.get("node_id"),
            "upper_node": upper.get("node_id"),
            "interpolation_weight": interpolation.get("weight"),
        }
        for view in sorted(scenario.get("by_view") or {}):
            data = scenario["by_view"][view]
            shared = {
                **common,
                "view": view,
                "measure": data.get("measure"),
                "source_status": data.get("source_status"),
                "sampling_standard_error": data.get("sampling_standard_error"),
                "sampling_reason": data.get("sampling_reason"),
                "years_per_state": data.get("years_per_state"),
            }
            rows.append(
                {
                    **shared,
                    "quantity": "expected_annual_loss",
                    "value": data.get("expected_annual_loss"),
                    "status": data.get("status", scenario.get("status")),
                    "reason": data.get("reason", scenario.get("reason")),
                }
            )
            for point in data.get("quantiles", ()):
                rows.append(
                    {
                        **shared,
                        "quantity": "annual_loss_quantile",
                        "non_exceedance_probability": point.get(
                            "non_exceedance_probability"
                        ),
                        "return_period_years": point.get("return_period_years"),
                        "value": point.get("annual_loss"),
                        "status": point.get("status"),
                        "reason": point.get("reason"),
                        "half_width": point.get("half_width"),
                        "relative_half_width": point.get("relative_half_width"),
                        "minimum_batch_exceedances": point.get(
                            "minimum_batch_exceedances"
                        ),
                    }
                )
            for point in data.get("exceedance", ()):
                rows.append(
                    {
                        **shared,
                        "quantity": "exceedance_probability",
                        "loss_threshold": point.get("loss"),
                        "value": point.get("probability"),
                        "status": point.get("status"),
                        "reason": point.get("reason"),
                        "half_width": point.get("half_width"),
                        "relative_half_width": point.get("relative_half_width"),
                        "minimum_batch_exceedances": point.get(
                            "minimum_batch_exceedances"
                        ),
                    }
                )
    return rows


def _node_csv_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if record.get("status") != "available":
        return []
    rows = []
    for node in record.get("nodes", ()):
        for view in sorted(node.get("by_view") or {}):
            data = node["by_view"][view]
            rows.append(
                {
                    "node_id": node.get("node_id"),
                    "progress": node.get("progress"),
                    "progress_cdf": node.get("progress_cdf"),
                    "eci_period_start": node.get("eci_period_start"),
                    "eci_period_end": node.get("eci_period_end"),
                    "base_weight": node.get("base_weight"),
                    "arm": node.get("arm"),
                    "samples": node.get("samples"),
                    "view": view,
                    "measure": data.get("measure"),
                    "source_status": data.get("source_status"),
                    "expected_annual_loss": data.get("expected_annual_loss"),
                    "status": data.get("status"),
                    "reason": data.get("reason"),
                    "sampling_standard_error": data.get("sampling_standard_error"),
                    "sampling_reason": data.get("sampling_reason"),
                }
            )
    return rows


def _capability_artifacts(record: Mapping[str, Any]) -> dict[str, bytes]:
    """Render every capability artifact once for export and replay verification."""
    return {
        "capability_scenarios.json": frozen_json(record),
        "capability_scenarios.csv": _csv_text(
            CAPABILITY_SCENARIOS_CSV_COLUMNS, _scenario_csv_rows(record)
        ).encode("utf-8"),
        "capability_nodes.csv": _csv_text(
            CAPABILITY_NODES_CSV_COLUMNS, _node_csv_rows(record)
        ).encode("utf-8"),
    }


def _no_view_diagnostics() -> ResultsDiagnostics:
    """The ``diagnostics`` section of exported results. An export carries
    what its reproduction check covers, the results' headline and exhibits;
    the view diagnostics are not compared, so its diagnostics file holds
    none, and ``results replay`` rederives them from the exported capture."""
    return {"arms": {}}


def _validated_report(
    package: RunSetup,
    draws,
    capture,
    results_path,
    code_commit,
    pool=None,
    *,
    exported: bool = False,
    layout: StateLayout | None = None,
):
    """The results to export, their headline and exhibits without view
    diagnostics: reduced from the capture, or read from ``results_path`` with
    the diagnostics file beside it (``load_results``) and checked to reproduce
    from the capture. ``exported`` results are an export's own, whose
    diagnostics file must hold no view diagnostics. ``layout`` is the setup's
    compiled states (``validate_captured_laws``)."""
    if results_path is None:
        targets = arm_weights(package, None)
        band_reason = validate_captured_laws(
            package, draws, capture, targets, pool=pool, layout=layout
        )
        report = reduce_capture(
            package,
            draws,
            capture,
            None,
            setup_code_provenance(code_commit),
            ReductionSpec(),
            band_reason=band_reason,
            pool=pool,
        )
        report["producer_provenance"] = capture.producer_provenance
        report["diagnostics"] = _no_view_diagnostics()
        return report
    report = load_results(results_path, diagnostics=True)
    if exported and report.get("diagnostics") != _no_view_diagnostics():
        raise ValueError(
            "an export's diagnostics file holds its results' exhibits and no "
            "view diagnostics"
        )
    report["diagnostics"] = _no_view_diagnostics()
    try:
        identity = report["analysis_spec"]
        state_ids = {
            state_key(state.state_id): state.state_id for state in package.joint_states
        }
        weights = {
            arm: {state_ids[key]: value for key, value in rows.items()}
            for arm, rows in identity["state_weights_by_arm"].items()
        }
        spec = ReductionSpec(**identity["reduction_spec"])
        code = identity["reducer_provenance"]
        targets = arm_weights(package, None, weights)
        band_reason = validate_captured_laws(
            package, draws, capture, targets, pool=pool, layout=layout
        )
        replay = reduce_capture(
            package,
            draws,
            capture,
            None,
            code,
            spec,
            weights_by_arm=weights,
            band_reason=band_reason,
            pool=pool,
        )
        replay["producer_provenance"] = capture.producer_provenance
        # The view diagnostics are not compared.
        replay["diagnostics"] = _no_view_diagnostics()
        # How the results were computed is audit context, outside the results;
        # when present (null included) it may only say how many worker
        # processes were requested.
        if "execution_context" in report and not _worker_count_context(
            report["execution_context"]
        ):
            raise ValueError(
                "export results execution context must be exactly a worker count"
            )
        reported = {
            key: value for key, value in report.items() if key != "execution_context"
        }
        # As JSON, so keys and sequences compare as the report stores them.
        if not _reproduces(reported, strict_json_loads(canonical_json(replay))):
            raise ValueError(
                "export results do not reproduce from the selected native sources"
            )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("export results lack a valid native analysis record") from exc
    return report


# A reported number reproduces when its replay is within this relative
# tolerance, as replay's closed-form native means are
# (``execution.CLOSED_FORM_REPLAY_RELATIVE_TOLERANCE``): the same reduction on
# another CPU uses other BLAS kernels and SIMD paths, which change the last bits
# of long sums.
REPRODUCTION_RELATIVE_TOLERANCE = 1e-9

# The chain summary reads the interior trajectory node nearest the base
# forecast's median frontier progress; its path-linked exposure has progress
# ratio node progress over median progress. The chain summary records each
# row's actual ratio, so this check only guards against reading another node.
CENTER_RATIO_TOLERANCE = 1e-4


# A difference's last bits are those of its operands, named beside it in its
# record: an arm comparison's difference and its half-width, a tornado row's,
# the respondent-sampling exhibit's and the trajectory quadrature's
# differences, a quadrature node's difference contribution, and the
# reconciliation of the EAL with the empirical mean. A respondent-sampling
# standard error is computed from the weighted state EALs whose sums are its
# record's EALs, so its last bits are theirs: a change's two EALs, or the
# level's one.
_SAMPLING_OPERANDS = (
    ("control_eal", "alternative_eal"),
    ("expected_annual_loss", "embedded_rule_expected_annual_loss"),
    ("expected_annual_loss",),
)
_OPERANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "absolute_difference": (
        ("reference", "alternative"),
        *_SAMPLING_OPERANDS[:2],
    ),
    "half_width": (("reference", "alternative"),),
    "eal_reconciliation_difference": (
        ("expected_annual_loss", "normalized_empirical_mean"),
    ),
    "difference_contribution": (("full_contribution", "embedded_contribution"),),
    "sampling_standard_error": _SAMPLING_OPERANDS,
}
# A standard error's last bits are its estimate's, as are a quadrature node's
# standard deviation's those of its mean.
_ESTIMATES = {
    "tvar_numerical_standard_error": "tvar",
    "empirical_mean_standard_error": "expected_annual_loss",
    "simulated_mean_standard_error": "simulated_mean",
    "state_eal_sd": "state_eal_mean",
}
# A z's difference over its standard error: the z of the respondent-sampling
# exhibits and of an EAL reconciliation (``tail_gate.eal_reconciliation``).
_Z_ERRORS = {
    "sampling_z": "sampling_standard_error",
    "z": "simulated_mean_standard_error",
}


def _magnitude(value) -> float:
    """``|value|`` for a finite number, else 0."""
    if (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return abs(float(value))
    return 0.0


def _scale_of(key: str, record: Mapping) -> float:
    """The magnitude, beside its own, that the number under ``key`` is
    compared relative to, from the replayed ``record`` that holds it, so a
    reported record cannot widen its own tolerance.

    A difference, a comparison's half-width and a respondent-sampling
    standard error are relative to their operands, and a z (a sampling z or
    an EAL reconciliation's) to its operands over its standard error, times
    one plus its magnitude; a status record's half-width and a standard
    error to their estimate (a half-width over its relative half-width); a
    numerical interval's endpoints to the larger endpoint, the estimate plus
    its half-width; and a ``relative_*`` ratio to one. Every other number,
    integers and squared quantities included, scales no other, and is
    compared relative to itself alone; that includes a control-variate TVaR,
    whose term ``b (exact - limited) / (1 - p)`` amplifies a difference that
    is checked against the TVaR's own magnitude."""
    if key.startswith("relative_"):
        return 1.0
    if key in _Z_ERRORS:
        # The difference over its standard error: both carry their operands'
        # last bits, amplified by one over the standard error.
        name = _Z_ERRORS[key]
        error = _magnitude(record.get(name))
        if error > 0:
            return (1 + _magnitude(record[key])) * _scale_of(name, record) / error
        return 0.0
    for operands in _OPERANDS.get(key, ()):
        if all(name in record for name in operands):
            return max(_magnitude(record[name]) for name in operands)
    if key == "half_width":
        ratio = _magnitude(record.get("relative_half_width"))
        if ratio > 0:
            return _magnitude(_magnitude(record[key]) / ratio)
        return 0.0
    if key in _ESTIMATES:
        return _magnitude(record.get(_ESTIMATES[key]))
    if key.endswith("_numerical_interval_95") and isinstance(record[key], list):
        return max(map(_magnitude, record[key]), default=0.0)
    return 0.0


def _reproduces(reported, replayed, scale: float = 0.0) -> bool:
    """Whether reported results reproduce their replay: equal structure, keys,
    strings, statuses, booleans, integers and digests, and floats within
    ``REPRODUCTION_RELATIVE_TOLERANCE`` of the larger of the two and their
    ``scale`` (``_scale_of``)."""
    if isinstance(reported, bool) or isinstance(replayed, bool):
        return type(reported) is type(replayed) and reported == replayed
    if isinstance(reported, float) or isinstance(replayed, float):
        if not (
            isinstance(reported, numbers.Real) and isinstance(replayed, numbers.Real)
        ):
            return False
        a, b = float(reported), float(replayed)
        if a == b:
            return True
        if not (math.isfinite(a) and math.isfinite(b)):
            return False
        return abs(a - b) <= REPRODUCTION_RELATIVE_TOLERANCE * max(
            abs(a), abs(b), scale
        )
    if isinstance(reported, Mapping) and isinstance(replayed, Mapping):
        return set(reported) == set(replayed) and all(
            _reproduces(reported[key], replayed[key], _scale_of(str(key), replayed))
            for key in reported
        )
    if isinstance(reported, list | tuple) and isinstance(replayed, list | tuple):
        return len(reported) == len(replayed) and all(
            _reproduces(a, b, scale) for a, b in zip(reported, replayed, strict=True)
        )
    return type(reported) is type(replayed) and reported == replayed


def _worker_count_context(context) -> bool:
    """Whether a report's execution context is exactly ``{"workers": n}`` with
    n a positive integer."""
    return (
        isinstance(context, dict)
        and set(context) == {"workers"}
        and type(context["workers"]) is int
        and context["workers"] >= 1
    )


def _curve(draws, weights, view, thresholds):
    rows = [(draws[key], mass) for key, mass in sorted(weights.items()) if mass > 0]
    values = np.concatenate([draw.annual_losses[view] for draw, _ in rows])
    masses = np.concatenate([mass * draw.normalized_weights() for draw, mass in rows])
    positive = masses > 0
    values, masses = values[positive], masses[positive]
    if thresholds is None:
        thresholds = np.unique(
            np.concatenate(([0.0], np.quantile(values, np.linspace(0, 1, 101))))
        )
    # Each draw weighs its years alike, so the masses take few distinct
    # values: a tail's exact sum is then its count of each value times that
    # value, rounded once, which is what ``math.fsum`` returns.
    distinct, index = np.unique(masses, return_inverse=True)
    grouped = len(distinct) <= max(1, len(masses) // 64)
    if grouped:
        ratios = [float(value).as_integer_ratio() for value in distinct]
        scale = max(denominator for _, denominator in ratios)
        numerators = [
            numerator * (scale // denominator) for numerator, denominator in ratios
        ]
        ordered = [np.sort(values[index == row]) for row in range(len(distinct))]
    result = []
    for threshold in thresholds:
        tail = masses[values > threshold]
        if grouped:
            total = sum(
                (len(row) - int(np.searchsorted(row, threshold, side="right")))
                * numerator
                for row, numerator in zip(ordered, numerators, strict=True)
            )
            probability = min(1.0, total / scale)
        else:
            probability = min(1.0, math.fsum(tail.tolist()))
        result.append(
            ExceedancePoint(
                loss=float(threshold),
                probability=probability,
                useful_exceedances=len(tail),
                tail_weight_ess=0.0 if len(tail) == 0 else effective_sample_size(tail),
            )
        )
    return tuple(result)


def _view_state_weights(report: Mapping[str, Any], arm: str, view: str):
    """The arm's state weights in ``view``, as its results give them: for a
    view that counts the lab's loss, a lab-loss view or a gross view, those
    of the arm's ``lab_loss_state_weights`` where a state is left out of
    such views, and otherwise the arm's."""
    weights = report["analysis_spec"]["state_weights_by_arm"][arm]
    if counts_lab_loss(view):
        return report["arms"][arm].get("lab_loss_state_weights", weights)
    return weights


def _series(package: RunSetup, draws, report: dict[str, Any], thresholds):
    labs = [
        entity.entity_id for entity in package.scope.entities if entity.kind == "lab"
    ]
    peers = (
        ["welfare", *(f"lab_loss::{lab}" for lab in labs)]
        if labs
        else ["welfare", "lab_loss"]
    )
    views = tuple(dict.fromkeys([*peers, *package.execution.requested_views]))
    result = []
    for arm in package.execution.required_arms:
        arm_result = report["arms"][arm]
        for view in views:
            metric: dict[str, Any] | None = arm_result.get("metrics", {}).get(view)
            if metric is None:
                metric = {
                    "status": "unavailable",
                    "reason": arm_result.get(
                        "reason",
                        "This observable was not requested in the frozen execution plan.",
                    ),
                }
                if view == "lab_loss":
                    metric["reason"] = (
                        "No named lab/group boundary is authored in this study."
                    )
            available = metric["status"] == "available"
            source_scope = source_status_view(view)
            measure_view = view if source_scope is None else source_scope[1]
            lab = (
                measure_view.removeprefix("lab_loss::")
                if measure_view.startswith("lab_loss::")
                else None
            )
            result.append(
                OutputSeries(
                    id=f"{arm}/{view}",
                    arm=arm,
                    view=view,
                    measure="lab_economic_loss"
                    if measure_view.startswith("lab_loss")
                    else "societal_harm"
                    if measure_view == "welfare"
                    else "supporting_diagnostic",
                    lab_id=lab,
                    quantity="annual_occurrence_maximum"
                    if view.startswith("oep_")
                    else "annual_aggregate",
                    status="available" if available else "unavailable",
                    reason=None if available else metric["reason"],
                    basis=metric.get("basis"),
                    metrics_pointer=_pointer("arms", arm, "metrics", view)
                    if available
                    else None,
                    curve=_curve(
                        draws[arm],
                        _view_state_weights(report, arm, view),
                        view,
                        thresholds,
                    )
                    if available
                    else (),
                )
            )
    return tuple(result)


def _conditioning(package: RunSetup, capture_directory):
    if not package.conditional_contexts:
        return {
            "status": "unavailable",
            "reason": "No lab/deployment integer ECI kernels were authored in this study.",
            "contexts": [],
        }
    conditional = ConditionalCapture(package, capture_directory)
    contexts = []
    for context in package.conditional_contexts:
        failures = []
        # Kernels that hold every view but those that count the lab's loss.
        without_lab_loss = []
        for kernel in context.kernels:
            try:
                conditional.kernel_at(context.context_id, kernel.eci, kernel.arm_id)
            except ValueError as exc:
                failures.append(
                    {"eci": kernel.eci, "arm": kernel.arm_id, "reason": str(exc)}
                )
                continue
            reason = conditional.lab_loss_unavailable_reason(
                context.context_id, kernel.eci, kernel.arm_id
            )
            if reason is not None:
                without_lab_loss.append(
                    {"eci": kernel.eci, "arm": kernel.arm_id, "reason": reason}
                )
        contexts.append(
            {
                **context.model_dump(mode="json"),
                "status": "unavailable" if failures else "supported",
                "failures": failures,
                **(
                    {"lab_loss_unavailable": without_lab_loss}
                    if without_lab_loss
                    else {}
                ),
            }
        )
    supported = sum(row["status"] == "supported" for row in contexts)
    return {
        "status": "available"
        if supported == len(contexts)
        else "partial"
        if supported
        else "unavailable",
        "contexts": contexts,
        "semantics": "Point kernels use authored conditional component weights, independently of forecast mass. Event conditioning uses a separately supplied discrete forecast with positive event mass.",
    }


def _median(values):
    finite = [float(value) for value in values if value is not None]
    return float(statistics.median(finite)) if finite else None


def _chain_summary_for_context(
    instrument: ContextualRiskInstrument,
    kept: ContextualRiskArtifact,
    source_status: Literal["public", "not_yet_public"],
    scenarios,
) -> tuple[ContextualChainSummary, ...]:
    """Summarize each chain at one context, using per-case medians across samples.

    ``kept`` holds the cases the run keeps of the context's capture, each
    stored case once (``excluded_samples.kept_cases``): no case of an
    excluded sample, and none of a model's answers to a leaf it does not
    answer. Each numeric response field is the median of that field's
    per-case p50 over those cases, one for each kept respondent sample. A
    public trajectory contributes only its center quadrature node; a
    constant context uses its center anchor score.
    """
    if instrument.public_trajectory is not None:
        anchor_score = instrument.anchor_scores[0]
    else:
        scores = sorted(instrument.anchor_scores)
        anchor_score = scores[len(scores) // 2]
    cases_by_node = {}
    for case in kept.cases:
        if case.anchor_score == anchor_score:
            cases_by_node.setdefault(case.node_id, []).append(case)

    rows: list[ContextualChainSummary] = []
    for node in instrument.nodes:
        scenario_id = node.population.scenario_id
        scenario = scenarios.get(scenario_id)
        chain = node.frequency_chain
        exposure = None if chain is None else instrument.chain_exposure(chain)
        chain_cases = cases_by_node.get(node.node_id, ())
        rates = []
        opportunity_counts = []
        stage_values: dict[str, list[float]] = {}
        chain_counts = []
        direct_counts = []
        pooled_counts = []
        severity_medians = []
        gross_means = []
        for case in chain_cases:
            if chain is None:
                continue
            assert exposure is not None
            chain_answer = driving_chain_answer(case)
            chain_count = case.chain_count
            direct = final_answer(case)
            pooled = pooled_answer(case)
            if chain_answer is not None and chain_answer.status == "available":
                assert chain_answer.opportunity_rate is not None
                rates.append(chain_answer.opportunity_rate.p50)
                opportunity_counts.append(
                    exposure.count * chain_answer.opportunity_rate.p50
                )
                for stage in chain_answer.stages:
                    stage_values.setdefault(stage.stage_id, []).append(
                        stage.probability.p50
                    )
            if chain_count is not None:
                chain_counts.append(chain_count.p50)
            if direct.status == "available":
                if direct.rate is not None:
                    direct_counts.append(direct.rate.p50)
                if direct.total_loss is not None:
                    severity_medians.append(direct.total_loss.p50)
            if pooled.status == "available":
                if pooled.rate is not None:
                    pooled_counts.append(pooled.rate.p50)
                try:
                    compiled = unit_law(
                        leaf_query(
                            instrument,
                            node.node_id,
                            pooled,
                            case.anchor_score,
                        )
                    )
                except ValueError:
                    compiled = None
                if compiled is not None and compiled.gross is not None:
                    gross_means.append(compiled.gross)

        stages = () if chain is None else chain.stages
        terminal_id = None if not stages else stages[-1].stage_id
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario_id if scenario is None else scenario.name,
                "source_status": source_status,
                "context_id": instrument.context_id,
                "trajectory_draw_id": (
                    None
                    if instrument.public_trajectory is None
                    else instrument.public_trajectory.draw_id
                ),
                "progress_ratio": (
                    None
                    if instrument.path_linked_exposure is None
                    else instrument.path_linked_exposure.progress_ratio
                ),
                "anchor_score": anchor_score,
                "authoring_status": None if chain is None else chain.authoring_status,
                "base_unit": None if exposure is None else exposure.unit,
                "base_count": None if exposure is None else exposure.count,
                "base_derivation": None if exposure is None else exposure.derivation,
                "scale_id": None if exposure is None else exposure.scale_id,
                "scale_level": None if exposure is None else exposure.scale_level,
                "factor": None if exposure is None else exposure.factor,
                "opportunity_rate_p50": _median(rates),
                "opportunities_per_year_p50": _median(opportunity_counts),
                "stages_p50": [
                    {
                        "stage_id": stage.stage_id,
                        "probability_p50": _median(
                            stage_values.get(stage.stage_id, ())
                        ),
                    }
                    for stage in stages
                ],
                "terminal_stage_p50": (
                    None
                    if terminal_id is None
                    else _median(stage_values.get(terminal_id, ()))
                ),
                "implied_incidents_per_year_p50": _median(chain_counts),
                "direct_count_p50": _median(direct_counts),
                "pooled_count_p50": _median(pooled_counts),
                "severity_median": _median(severity_medians),
                "mean_gross_loss_per_incident": _median(gross_means),
                "case_count": len(chain_cases),
            }
        )
    return tuple(rows)


def _center_node(quadrature: PublicFrontierQuadrature) -> PublicQuadratureNode:
    """The interior node nearest the base forecast's median frontier progress."""
    median = quadrature.median_progress
    return min(
        quadrature.nodes[1:],
        key=lambda node: (abs(math.log(node.progress / median)), node.draw_id),
    )


def _require_center_ratio(progress_ratio: float | None, expected: float) -> None:
    """Refuse a center context whose exposure is not at its node's progress ratio."""
    if progress_ratio is None or not math.isclose(
        progress_ratio, expected, rel_tol=CENTER_RATIO_TOLERANCE
    ):
        raise ValueError("center trajectory exposure differs from its node's ratio")


def _kept_capture(
    files: Mapping[str, bytes],
    node: str,
    instrument: ContextualRiskInstrument,
    artifact: ContextualRiskArtifact,
) -> ContextualRiskArtifact:
    """The cases the run keeps of one capture of the study's ``node``, by the
    setup's ``excluded_samples`` exhibit (``excluded_samples.kept_cases``)."""
    artifact.validate_against(instrument)
    record = files.get(EXCLUDED_SAMPLES_PATH)
    if record is None:
        raise ValueError("contextual evidence lacks its excluded-sample exhibit")
    return kept_cases(strict_json_loads(record), node, artifact)


def _chain_summary(package: RunSetup) -> tuple[ContextualChainSummary, ...]:
    """Build chain summaries from captured contextual evidence in the setup,
    over the cases the run keeps."""
    return _chain_panel(package.evidence.files)


def _chain_panel(files: Mapping[str, bytes]) -> tuple[ContextualChainSummary, ...]:
    """The chain summaries of the captured contextual evidence among a setup's
    evidence ``files`` (``_chain_summary``)."""
    summary: list[ContextualChainSummary] = []
    plan_bytes = files.get("captured/public-study.json")
    quadrature_bytes = files.get("captured/public-frontier.json")
    if plan_bytes is not None and quadrature_bytes is not None:
        plan = strict_json_loads(plan_bytes)
        quadrature = PublicFrontierQuadrature.from_saved(
            strict_json_loads(quadrature_bytes)
        )
        scenarios = {row.id: row for row in load_scenarios(scenarios_path())}
        center = _center_node(quadrature)
        contexts = plan.get("contexts")
        if not isinstance(contexts, list):
            raise ValueError("public study evidence lacks its context roster")
        found = set()
        for index, raw_row in enumerate(contexts):
            if not isinstance(raw_row, dict):
                continue
            row = cast(dict[str, object], raw_row)
            if row.get("draw_id") != center.draw_id:
                continue
            status = row.get("source_model_status")
            if (
                not isinstance(status, str)
                or status not in {"public", "not_yet_public"}
                or status in found
            ):
                raise ValueError("public study center-node statuses are invalid")
            found.add(status)
            prefix = f"captured/contexts/{index}/"
            instrument_bytes = files.get(prefix + "contextual-instrument.json")
            artifact_bytes = files.get(prefix + "contextual-risk.json")
            if instrument_bytes is None or artifact_bytes is None:
                raise ValueError("public study center context lacks captured risk")
            instrument = ContextualRiskInstrument.from_saved(
                strict_json_loads(instrument_bytes)
            )
            artifact = ContextualRiskArtifact.from_saved(
                strict_json_loads(artifact_bytes)
            )
            if instrument.public_trajectory is None or (
                instrument.public_trajectory.draw_id != center.draw_id
                or instrument.public_trajectory.source_model_status != status
            ):
                raise ValueError("public study center capture differs from its roster")
            if instrument.path_linked_exposure is not None:
                _require_center_ratio(
                    instrument.path_linked_exposure.progress_ratio,
                    center.progress / quadrature.median_progress,
                )
            summary.extend(
                _chain_summary_for_context(
                    instrument,
                    _kept_capture(files, center.draw_id, instrument, artifact),
                    status,
                    scenarios,
                )
            )
        if found != {"public", "not_yet_public"}:
            raise ValueError("public study lacks both center-node source statuses")
        return tuple(summary)

    instrument_paths = sorted(
        name
        for name in files
        if name.endswith("contextual-instrument.json")
        and name.startswith("captured/")
        and "/contexts/" not in name
    )
    scenarios = (
        {row.id: row for row in load_scenarios(scenarios_path())}
        if instrument_paths
        else {}
    )
    for path in instrument_paths:
        artifact_path = (
            path.removesuffix("contextual-instrument.json") + "contextual-risk.json"
        )
        if artifact_path not in files:
            continue
        instrument = ContextualRiskInstrument.from_saved(strict_json_loads(files[path]))
        artifact = ContextualRiskArtifact.from_saved(
            strict_json_loads(files[artifact_path])
        )
        status = (
            instrument.public_trajectory.source_model_status
            if instrument.public_trajectory is not None
            else "public"
            if instrument.capability_access == "public"
            else "not_yet_public"
        )
        summary.extend(
            _chain_summary_for_context(
                instrument,
                # A fixed-anchor setup is one node of samples: its context.
                _kept_capture(files, instrument.context_id, instrument, artifact),
                status,
                scenarios,
            )
        )
    return tuple(summary)


_SOURCE = (
    "Native artifacts are authoritative and copied without redaction. This "
    "export does not confer permission to redistribute selected evidence."
)
_SOURCE_WITHOUT_ANSWERS = (
    "Native artifacts are authoritative. The setup's copy leaves out the "
    f"captured prompts, responses and parsed answers, which its {OMITTED_EVIDENCE} "
    "lists by path and SHA-256; every other file is copied without redaction. "
    "This export does not confer permission to redistribute selected evidence."
)
_CHAIN_SUMMARY = (
    "Each chain response field is the median of its per-case p50 across the "
    "respondents and samples whose cases the run keeps, each stored case once: "
    "no case of an excluded sample and none of a model's answers to a scenario "
    "it does not answer. Public studies report the center trajectory node, "
    "whose progress ratio is one up to quadrature rounding and is recorded per "
    "row; constant contexts use their center anchor score."
)
_CHAIN_SUMMARY_WITHOUT_ANSWERS = (
    " The answers it is computed from are left out of this export: the panel "
    "is as computed when the export was made, and it is checked only against "
    "those files, by their recorded SHA-256."
)


def _index(
    package: RunSetup,
    draws,
    capture,
    report,
    references,
    outputs,
    thresholds,
    capture_directory,
    chain_summary: tuple[ContextualChainSummary, ...],
    *,
    answers_omitted: bool,
):
    """The export's index. ``chain_summary`` is the panel of the setup's
    captured answers (``_chain_summary``), and ``answers_omitted`` says that
    the export's copy of the setup leaves those answers out."""
    sample_sets = {
        arm: {
            "components": [
                {
                    "state_id": capture.states[state].state_id,
                    "weight": weight,
                    "record_pointer": _pointer("arms", arm, state),
                }
                for state, weight in sorted(
                    report["analysis_spec"]["state_weights_by_arm"][arm].items()
                )
                if state in capture.arms.get(arm, {})
            ],
            "status": report["arms"][arm]["status"],
        }
        for arm in package.execution.required_arms
    }
    scope = package.scope
    return ResultsExport(
        source=references,
        outputs=outputs,
        scope={
            "title": scope.title,
            "period_start": scope.period_start.isoformat(),
            "period_end": scope.period_end.isoformat(),
            "forecast_as_of": scope.forecast_as_of.isoformat(),
            "data_cutoff": scope.data_cutoff.isoformat(),
            "entities": [entity.model_dump(mode="json") for entity in scope.entities],
            "deployments": [row.model_dump(mode="json") for row in scope.deployments],
            "populations": [row.model_dump(mode="json") for row in scope.populations],
            "limitations": list(package.limitations),
            "coverage": "Only the explicitly modeled populations and deployments; no claim of complete real-world all-risk coverage.",
        },
        series=_series(package, draws, report, thresholds),
        sample_sets=sample_sets,
        conditioning=_conditioning(package, capture_directory),
        chain_summary=chain_summary,
        conventions={
            "exceedance": "P(loss > threshold); strict inequality, in the native monetary unit.",
            "distribution": "Native weighted joint annual samples approximate the model law; the display curve is not its complete representation.",
            "numerical_precision": "Native results and draw records retain numerical precision and moment applicability. Curve support counts and weight ESS are diagnostics, not confidence intervals or epistemic uncertainty.",
            "pairing": "Within a native arm/state, all observable arrays share rows. Never align row indices across different states or arms.",
            "missing": "Unavailable means unmodeled or unretained, never zero. A zero empirical tail does not establish an impossible loss.",
            "lab_economic_loss": "Incident-linked ultimate lab/group costs and liabilities before recoveries, counted once. Enterprise/franchise value effects are unquantified unless explicitly modeled; insurance payouts are a different observable.",
            "source": _SOURCE_WITHOUT_ANSWERS if answers_omitted else _SOURCE,
            "chain_summary": _CHAIN_SUMMARY
            + (_CHAIN_SUMMARY_WITHOUT_ANSWERS if answers_omitted else ""),
        },
    )


def export_results(
    *,
    setup_path: Path,
    capture_directory: Path,
    output: Path,
    results_path: Path | None = None,
    thresholds: tuple[float, ...] | None = None,
    code_commit: str | None = None,
    workers: int = 1,
    include_answers: bool = False,
) -> ResultsExport:
    """Validate, then copy native artifacts and write one versioned export.json.

    The export's results are ``results.json``, as supplied at ``results_path``
    or as reduced from the capture, and a diagnostics file that holds their
    exhibits and no view diagnostics. ``workers`` processes check the captured
    laws and reduce the views, as in replay; the export does not depend on it.

    The export's copy of the setup leaves out every file that holds a captured
    prompt, a response or parsed answers (``store.holds_captured_answers``) and
    lists each by path and SHA-256 in ``omitted-evidence.json`` beside the
    copy's manifest; a setup without such files is copied whole, with no
    record. ``include_answers`` copies every setup whole. The chain summary is
    computed from the whole setup either way, so the setup at ``setup_path``
    must hold its answers."""
    require_new_evidence_output(output)
    if (setup_path / OMITTED_EVIDENCE).exists():
        raise ValueError(
            "the setup leaves out its captured answers, which the export's chain "
            "summary is computed from; export from the setup it was copied from"
        )
    if thresholds is not None and (
        not thresholds
        or tuple(sorted(set(thresholds))) != thresholds
        or any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in thresholds)
    ):
        raise ValueError(
            "thresholds must be distinct increasing finite nonnegative numbers"
        )
    results_bytes = None if results_path is None else results_path.read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(prefix=".results-export-", dir=output.parent) as tmp,
        # The states the load compiles as its check are the ones replayed.
        StateLayout.loading(setup_path, Path(tmp)) as layout,
    ):
        package = layout.package
        stage = Path(tmp) / "export"
        left_out = {} if include_answers else captured_answer_files(package.manifest)
        for name in package.files:
            if name in left_out:
                continue
            path = stage / "native/setup" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            package.files.write(name, path)
        if left_out:
            freeze_json(
                stage / "native/setup" / OMITTED_EVIDENCE,
                omitted_evidence_record(package.files.sha256(MANIFEST), left_out),
            )
        for path in capture_directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("run capture rejects symlinks")
            if path.is_file():
                target = stage / "native/capture" / path.relative_to(capture_directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        result_path = stage / "native/results.json"
        draws, capture = load_run_capture(stage / "native/capture")
        with WorkerPool(workers) as pool:
            if results_bytes is None or results_path is None:
                report = _validated_report(
                    package, draws, capture, None, code_commit, pool, layout=layout
                )
                freeze_json(result_path, results_payload(report))
            else:
                # Both files are read where the export holds them, so what is
                # checked is what is exported. A results file that would be
                # refused is refused before a missing diagnostics file.
                result_path.write_bytes(results_bytes)
                load_results(result_path)
                staged = diagnostics_path(result_path)
                staged.write_bytes(read_diagnostics_bytes(results_path))
                report = _validated_report(
                    package,
                    draws,
                    capture,
                    result_path,
                    code_commit,
                    pool,
                    layout=layout,
                )
                staged.unlink()
        # The reported exhibits, without the view diagnostics.
        freeze_json(diagnostics_path(result_path), diagnostics_payload(report))
        paths = {
            "setup": stage / "native/setup" / MANIFEST,
            "capture": stage / "native/capture" / CAPTURE_MANIFEST,
            "results": result_path,
            "diagnostics": diagnostics_path(result_path),
        }
        references = {
            name: ArtifactReference(
                path=path.relative_to(stage).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for name, path in paths.items()
        }
        capability_artifacts = _capability_artifacts(report["capability_scenarios"])
        for name, data in capability_artifacts.items():
            (stage / name).write_bytes(data)
        outputs = {
            name: ArtifactReference(
                path=name,
                sha256=hashlib.sha256(data).hexdigest(),
            )
            for name, data in capability_artifacts.items()
        }
        index = _index(
            package,
            draws,
            capture,
            report,
            references,
            outputs,
            thresholds,
            stage / "native/capture",
            # From the whole setup, whose answers the copy may leave out.
            _chain_summary(package),
            answers_omitted=bool(left_out),
        )
        freeze_json(stage / "export.json", index.model_dump(mode="json"))
        stage.rename(output)
    return index


def _restored_evidence(
    package: RunSetup, omitted: Mapping[str, str], answers: Path
) -> StoredFiles:
    """The evidence of a setup copy with the files it leaves out read from
    ``answers``, a directory that holds them at their paths in the setup. Each
    must have the SHA-256 the copy records for it."""
    answers = answers.expanduser().resolve()
    restored = StoredFiles.under(answers, omitted, omitted)
    for name in omitted:
        if not (answers / name).is_file():
            raise ValueError(f"the answers directory lacks an omitted file: {name}")
    restored.check()
    return StoredFiles.merged(package.evidence.files, restored.unprefixed("evidence/"))


def load_results_export(
    directory: Path, *, answers: Path | None = None
) -> ResultsExport:
    """Verify portable source closure and rederive display values from native data.

    An export whose setup copy leaves its captured answers out
    (``omitted-evidence.json``) is checked as a whole one is, except for its
    chain summary, the one value computed from those answers: it is taken as
    the index states it. ``answers`` is a directory that holds the omitted
    files at their paths in the setup, such as the setup the export was made
    from. With it, every omitted file must have its recorded SHA-256 and the
    chain summary is rederived from them and compared, as for a whole export."""
    directory = directory.resolve()
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("export rejects symlinks")
    payload = strict_json_loads((directory / "export.json").read_bytes())
    require_marker(
        payload,
        "format",
        "cat.results/1",
        "export.json format",
        "export the results again with the current software",
    )
    index = ResultsExport.model_validate(payload)
    for reference in index.source.values():
        path = directory / reference.path
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference.sha256:
            raise ValueError("export source checksum mismatch")
    for reference in index.outputs.values():
        path = directory / reference.path
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference.sha256:
            raise ValueError("export output checksum mismatch")
    results = directory / index.source["results"].path
    if directory / index.source["diagnostics"].path != diagnostics_path(results):
        raise ValueError(
            "export diagnostics reference must name the file beside its results"
        )
    # The states the load compiles as its check are the ones replayed.
    with StateLayout.loading((directory / index.source["setup"].path).parent) as layout:
        package = layout.package
        draws, capture = load_run_capture(
            (directory / index.source["capture"].path).parent
        )
        report = _validated_report(
            package, draws, capture, results, None, exported=True, layout=layout
        )
    omitted = omitted_files(package)
    if answers is not None and not omitted:
        raise ValueError(
            "the export leaves no file out, so it is checked without an answers "
            "directory"
        )
    if not omitted:
        chain_summary = _chain_summary(package)
    elif answers is not None:
        chain_summary = _chain_panel(_restored_evidence(package, omitted, answers))
    else:
        # Computed from files the export does not hold.
        chain_summary = index.chain_summary
    capability_record = report["capability_scenarios"]
    for name, expected_bytes in _capability_artifacts(capability_record).items():
        path = directory / index.outputs[name].path
        if path.read_bytes() != expected_bytes:
            raise ValueError(f"export {name} differs from its native results")
    expected = _index(
        package,
        draws,
        capture,
        report,
        index.source,
        index.outputs,
        None,
        (directory / index.source["capture"].path).parent,
        chain_summary,
        answers_omitted=bool(omitted),
    )
    series = []
    for row in expected.series:
        supplied = next((value for value in index.series if value.id == row.id), None)
        if supplied is not None and row.status == "available":
            row = row.model_copy(
                update={
                    "curve": _curve(
                        draws[row.arm],
                        _view_state_weights(report, row.arm, row.view),
                        row.view,
                        tuple(point.loss for point in supplied.curve),
                    )
                }
            )
        series.append(row)
    expected = expected.model_copy(update={"series": tuple(series)})
    if index != expected:
        raise ValueError("export index differs from its native source artifacts")
    return index
