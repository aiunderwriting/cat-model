"""Frozen physical incident execution and annual-capture reduction."""

from __future__ import annotations

import contextlib
import math
import pickle
import secrets
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self, cast

import structlog
from pydantic import BaseModel, ConfigDict, Field

from catmodel.artifacts import freeze_json
from catmodel.contracts.incident import ExecutionPlan
from catmodel.contracts.primitives import Sha256Digest
from catmodel.contracts.results import (
    RESULTS_SCHEMA_VERSION,
    CapabilityScenarios,
    CapturedCodeProvenance,
    EalParameterBand,
    NativeResults,
    ScenarioContributions,
    ScenarioGroups,
    SensitivitySummary,
    SmallEventAggregation,
    TailGateResults,
    Workload,
)
from catmodel.contracts.run_setup import RunSetup
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import require_marker, strict_json_loads
from catmodel.output_paths import require_new_evidence_output
from catmodel.probability.numerics import valid_probability_weights
from catmodel.results.capability_scenarios import capability_scenarios
from catmodel.results.capability_storage import (
    StoredDraw,
    decode_conditional_draw,
    encode_conditional_draw,
)
from catmodel.results.eal_parameter_band import (
    BAND_MISMATCH_REASON,
    band_replays,
    borrowed_band_reason,
    compute_parameter_band,
    parameter_band_method,
    reported_parameter_band,
    withheld_parameter_band,
)
from catmodel.results.reduction import (
    DEFAULT_TAIL_LEVELS,
    ReductionSpec,
    reduce_captures,
)
from catmodel.results.respondent_sampling import (
    ViewWeights,
    respondent_sampling,
    sample_groups,
)
from catmodel.results.result_disclosures import (
    result_limitations,
    scenario_coverage,
    study_scope,
)
from catmodel.results.results_file import diagnostics_path, write_results
from catmodel.results.scenario_contributions import scenario_contributions
from catmodel.results.scenario_groups import library_classification, scenario_groups
from catmodel.results.sensitivity_results import sensitivity_results
from catmodel.results.sensitivity_summary import tornado_svg
from catmodel.setup.store import (
    load_setup,
    required_state_ids,
    setup_code_provenance,
)
from catmodel.simulation.capability_mixture import (
    DETERMINISTIC_EAL_PRECISION_BASIS,
    ConditionalCapabilityDraws,
    NativeMean,
)
from catmodel.simulation.execution_budget import (
    WorkLimitExceeded,
    execution_budget,
    reference_followers,
    simulation_identity,
    work_allowances,
    work_limits,
)
from catmodel.simulation.incident import (
    CompiledIncidentState,
    compile_incident_state,
    observable_bases,
)
from catmodel.simulation.incident_moments import (
    CERTIFIED_INTEGRATION_RELATIVE_ERROR,
    analytic_known_zero_views,
    analytic_linear_eals,
)
from catmodel.simulation.incident_runtime import (
    POSITIVE_MEAN_UNDERFLOW_REASON,
    SimulatedState,
    simulate_incident_state,
    small_event_audit,
    state_key,
    state_stream_seed,
    stream_design,
    view_admission,
)
from catmodel.simulation.small_events import SmallEventControls, small_event_results
from catmodel.simulation.tail_gate import (
    TailGateControls,
    doubling_limits,
    gated_years,
    round_arms,
    tail_gate_results,
    unconverged_drivers,
    year_prefix,
)
from catmodel.worker_pool import (
    WORKER_BASE_BYTES,
    WorkerPool,
    memory_bounded_workers,
    simulation_worker_bytes,
)

CAPTURE_MANIFEST = "run-capture.json"

# A reduction worker's memory per state-year of the (arm, view) it reduces,
# beside WORKER_BASE_BYTES: the view's losses, masses, their sorted and batched
# copies and the per-level work, measured at 109 to 117 bytes.
REDUCTION_BYTES_PER_STATE_YEAR = 130

logger = structlog.get_logger(__name__)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CapturedState(_Record):
    state_id: str
    epistemic_group_id: str


class CapturedSupport(_Record):
    unavailable_views: dict[str, str]
    unavailable_functionals: dict[str, dict[str, str]]
    view_bases: dict[str, dict[str, str]]


class CapturedConditional(_Record):
    conditional_key: Sha256Digest
    seed: int = Field(strict=True, ge=0)
    support_sha256: Sha256Digest
    draw: StoredDraw
    execution: dict[str, object]


RUN_CAPTURE_SCHEMA_VERSION = 1


class RunCapture(_Record):
    schema_version: Literal[1] = RUN_CAPTURE_SCHEMA_VERSION
    grain: Literal["annual"] = "annual"
    physical_joint_semantics: Literal["physical_joint_samples"] = (
        "physical_joint_samples"
    )
    originating_setup_sha256: Sha256Digest
    originating_scientific_sha256: Sha256Digest
    states: dict[Sha256Digest, CapturedState]
    supports: dict[Sha256Digest, CapturedSupport]
    arms: dict[str, dict[Sha256Digest, CapturedConditional]]
    execution_plan: dict[str, object]
    state_weights_by_arm: dict[str, dict[Sha256Digest, float]]
    producer_provenance: dict[str, object]
    evidence_reliance: dict[str, object]
    tail_gate_rounds: list[dict[str, object]]
    # The reference arm's EAL parameter band at the run's weights; replay
    # recomputes it from the laws and the reduction reads it.
    eal_parameter_band: dict[str, object]
    content_sha256: Sha256Digest

    @cached_property
    def plan(self) -> ExecutionPlan:
        return ExecutionPlan.model_validate(self.execution_plan)

    @cached_property
    def execution_plan_sha256(self) -> str:
        return canonical_json_sha256(self.execution_plan)


class StateLayout:
    """Every required arm's states: each state's conditional key, and its
    compiled form when asked.

    A compiled state is megabytes and a study has one for every arm and state,
    so they are not held together. Inside a ``with`` block each is written to
    a temporary directory as it is compiled, and ``compiled`` reads it back; a
    state the plan does not require of the arm is compiled when asked. The
    directory is made in ``scratch`` or, where that does not exist yet, in
    the nearest directory above it that does; with no ``scratch``, in the
    system's temporary directory. ``keys`` gives each state's conditional
    key, by arm and state key.

    ``StateLayout.loading(setup_path, scratch)`` loads the setup on entry and
    keeps the states the load compiles as its check (``store.validate_setup``),
    so a command reads each arm's laws once; ``package`` is the loaded setup.
    ``StateLayout(package, scratch)`` compiles the states of a setup already
    loaded, arm by arm.
    """

    def __init__(
        self,
        package: RunSetup | None,
        scratch: Path | None = None,
        *,
        setup_path: Path | None = None,
    ) -> None:
        if (package is None) == (setup_path is None):
            raise ValueError("a layout takes a loaded setup or a setup's path")
        self._package = package
        self._setup_path = setup_path
        self._scratch = scratch
        self._directory: TemporaryDirectory | None = None
        self._paths: dict[tuple[str, str], Path] = {}
        self._collected: dict[str, dict[str, str]] = {}
        self._state_ids: dict[str, str] = {}
        self.keys: dict[str, dict[str, str]] = {}

    @classmethod
    def loading(cls, setup_path: Path, scratch: Path | None = None) -> Self:
        """The layout of the setup saved at ``setup_path``, loaded on entry."""
        return cls(None, scratch, setup_path=setup_path)

    @classmethod
    def keys_of(cls, package: RunSetup) -> dict[str, dict[str, str]]:
        """Each required arm's conditional keys, by state key."""
        layout = cls(package)
        layout._compile_all()
        return layout.keys

    @property
    def package(self) -> RunSetup:
        if self._package is None:
            raise RuntimeError("a layout's setup is loaded inside its with block")
        return self._package

    def _keep(self, arm: str, state_id: str, compiled: CompiledIncidentState) -> None:
        key = state_key(state_id)
        self._collected.setdefault(arm, {})[key] = compiled.conditional_key
        if self._directory is not None:
            path = Path(self._directory.name) / f"{len(self._paths)}.pickle"
            with path.open("xb") as handle:
                pickle.dump(compiled, handle, pickle.HIGHEST_PROTOCOL)
            self._paths[arm, key] = path

    def _settle(self) -> None:
        """Order the keys by the plan's arms and the setup's states, however
        the states were compiled."""
        package = self.package
        self._state_ids = {
            state_key(state.state_id): state.state_id for state in package.joint_states
        }
        self.keys = {
            arm: {
                key: self._collected[arm][key]
                for key in self._state_ids
                if key in self._collected.get(arm, {})
            }
            for arm in package.execution.required_arms
        }

    def _load(self) -> None:
        assert self._setup_path is not None
        self._package = load_setup(self._setup_path, compiled=self._keep)
        self._settle()

    def _compile_all(self) -> None:
        package = self.package
        identifiers = tuple(state.state_id for state in package.joint_states)
        for arm in package.execution.required_arms:
            selected = set(required_state_ids(package.execution, arm, identifiers))
            for state_id in identifiers:
                if state_id in selected:
                    self._keep(
                        arm, state_id, compile_incident_state(package, state_id, arm)
                    )
        # The last arm's laws are not kept once its states are compiled.
        package.overrides.release()
        self._settle()

    def __enter__(self) -> Self:
        # A directory the command has not made yet is not made here, so a
        # command that then refuses its inputs leaves nothing behind.
        scratch = self._scratch
        while scratch is not None and not scratch.is_dir():
            scratch = None if scratch.parent == scratch else scratch.parent
        self._directory = TemporaryDirectory(prefix=".compiled-", dir=scratch)
        try:
            if self._setup_path is None:
                self._compile_all()
            else:
                self._load()
        except BaseException:
            self._paths.clear()
            self._directory.cleanup()
            self._directory = None
            raise
        return self

    def __exit__(self, *_: object) -> None:
        assert self._directory is not None
        self._paths.clear()
        self._directory.cleanup()
        self._directory = None

    def compiled(self, arm: str, key: str) -> CompiledIncidentState:
        """The compiled state of ``arm`` at state ``key``."""
        path = self._paths.get((arm, key))
        if path is None:
            return compile_incident_state(self.package, self._state_ids[key], arm)
        with path.open("rb") as handle:
            return pickle.load(handle)


def _weights(package: RunSetup, weights: Mapping[str, float] | None):
    states = {state.state_id: state for state in package.joint_states}
    if weights is not None and not isinstance(weights, Mapping):
        raise ValueError("target state weights must be a JSON object keyed by state ID")
    raw = (
        {key: state.weight for key, state in states.items()}
        if weights is None
        else dict(weights)
    )
    if set(raw) != set(states):
        raise ValueError("target weights must name exactly the declared joint states")
    if not valid_probability_weights(raw.values(), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "target state weights must be finite, nonnegative and sum to one"
        )
    # Stratum order follows the declared states, not a JSON object's key order.
    return {state_key(key): raw[key] for key in states}


def arm_weights(package, weights, weights_by_arm=None):
    """Each required arm's target weights by state key: ``weights_by_arm``
    when given; otherwise ``weights`` (the setup's state weights when None)
    for every arm but one that authors its own state weights, which keeps
    them. One vector that differs from the setup's cannot say what such an
    arm's targets become, so it is refused."""
    reference = _weights(package, weights)
    required = package.execution.required_arms
    if weights_by_arm is not None:
        if not isinstance(weights_by_arm, Mapping) or set(weights_by_arm) != set(
            required
        ):
            raise ValueError(
                "arm target weights must cover exactly the required arms: "
                + ", ".join(required)
            )
        resolved = {
            arm: _weights(package, vector) for arm, vector in weights_by_arm.items()
        }
        if (
            weights is not None
            and resolved[package.execution.reference_arm] != reference
        ):
            raise ValueError(
                "reference target weights conflict with the explicit arm targets"
            )
        return resolved
    authored = {
        row.alternative_id: row.state_weights
        for row in package.alternatives
        if row.alternative_id in required and row.state_weights is not None
    }
    if weights is not None and authored and reference != _weights(package, None):
        raise ValueError(
            "one target vector cannot reweight a setup whose arms author their "
            f"own state weights ({', '.join(authored)}): give every required arm "
            f"({', '.join(required)}) its own target weights, keyed by arm ID"
        )
    return {
        arm: _weights(package, authored[arm]) if arm in authored else reference
        for arm in required
    }


@dataclass(frozen=True)
class _Claims:
    """What replay re-derives from a captured draw's law: each view's moment
    availability (mean, variance) and native mean, and the years. It leaves
    the arrays behind, so a worker process checks a law without them."""

    views: dict[str, tuple[bool, bool, NativeMean | None]]
    years: int

    @classmethod
    def of(cls, draw: ConditionalCapabilityDraws) -> _Claims:
        return cls(
            {
                view: (row.mean_exists, row.variance_exists, row.native_mean)
                for view, row in draw.views.items()
            },
            draw.n_simulations,
        )


def _validate_captured_support(compiled, plan, claims, support, known_zero, analytic):
    """Re-derive semantic claims from the law, not the capture's own checksum.
    ``known_zero`` and ``analytic`` are the law's known-zero views and
    analytic EALs of the captured views."""
    reasons, finite_variance, functional = view_admission(
        compiled, plan.requested_views
    )
    available = {view for view in plan.requested_views if view not in reasons}
    if set(claims.views) != available:
        raise ValueError("captured views differ from law-derived observable admission")
    if support.unavailable_views != reasons or support.view_bases != observable_bases(
        compiled, plan.requested_views
    ):
        raise ValueError("captured support differs from law-derived admission or bases")
    mean_exists = {view: view not in functional for view in available}
    for view, value in analytic.values.items():
        if value == 0 and view not in known_zero and view not in functional:
            functional.setdefault(view, {})["expected_annual_loss"] = (
                POSITIVE_MEAN_UNDERFLOW_REASON
            )
            finite_variance[view] = False
    if support.unavailable_functionals != functional:
        raise ValueError("captured functional support differs from the conditional law")
    if {view: row[0] for view, row in claims.views.items()} != mean_exists or {
        view: row[1] for view, row in claims.views.items()
    } != {view: finite_variance[view] for view in available}:
        raise ValueError(
            "captured moment availability differs from the conditional law"
        )


# Closed-form native means replay to this relative tolerance rather than to a
# few ulps: the same formulas evaluated on another platform (other NumPy SIMD,
# BLAS or libm paths) can differ in the last bits of each term, and a mean
# accumulated over many states, components and quantile segments carries that
# difference, so a 1e-12 tolerance would refuse faithful cross-machine replays.
CLOSED_FORM_REPLAY_RELATIVE_TOLERANCE = 1e-9


def _native_mean_replays(value, bound, expected, expected_bound) -> bool:
    """Whether a captured native mean agrees with its recomputed benchmark.

    A closed-form mean (bound zero) must match to
    ``CLOSED_FORM_REPLAY_RELATIVE_TOLERANCE``. A numerically integrated mean
    replays when the captured and recomputed certified intervals
    ``[value - bound, value + bound]`` overlap, each bound within the
    quadrature tolerance of its value: both intervals contain the exact mean,
    and a last-bit change in a distribution function can move the refinement,
    and hence the bound, far more than the value.
    """
    if not isinstance(bound, int | float) or not math.isfinite(bound) or bound < 0:
        return False
    if bound == 0 or expected_bound == 0:
        return bound == expected_bound and math.isclose(
            value, expected, rel_tol=CLOSED_FORM_REPLAY_RELATIVE_TOLERANCE, abs_tol=0
        )
    return (
        bound <= CERTIFIED_INTEGRATION_RELATIVE_ERROR * value
        and expected_bound <= CERTIFIED_INTEGRATION_RELATIVE_ERROR * expected
        and abs(value - expected) <= bound + expected_bound
    )


def _validate_deterministic_mean(native, estimator, value, bound, known_zero, label):
    if native.estimator != estimator:
        raise ValueError(
            f"captured {label} estimator {native.estimator!r} has no supported deterministic validation"
        )
    precision = native.precision or {}
    if (
        precision.get("precision_basis") != DETERMINISTIC_EAL_PRECISION_BASIS
        or precision.get("numerical_standard_error") != 0
        or precision.get("standard_error_upper_bound") != 0
        or "integration_error_bound" not in precision
    ):
        raise ValueError(
            f"captured {label} lacks its deterministic integration provenance"
        )
    if precision.get("known_zero_law") is not known_zero:
        raise ValueError(
            f"captured {label} zero-law certificate differs from the complete conditional law"
        )
    if not _native_mean_replays(
        native.value, precision["integration_error_bound"], value, bound
    ):
        raise ValueError(
            f"captured {label} or its integration error bound differs from its law-derived analytic benchmark"
        )


def _validate_native_analytic_eals(compiled, claims, known_zero=None, analytic=None):
    """Rederive each captured native EAL, a mean over every parameter rank, from
    the law; ``known_zero`` and ``analytic`` are computed when not given."""
    views = tuple(claims.views)
    if known_zero is None:
        known_zero = analytic_known_zero_views(compiled, views)
    if analytic is None:
        analytic = analytic_linear_eals(compiled, views)
    expected = {
        view: value
        for view, value in analytic.values.items()
        if value > 0 or view in known_zero
    }
    captured = {
        view: row[2] for view, row in claims.views.items() if row[2] is not None
    }
    if set(captured) != set(expected):
        raise ValueError(
            "captured native EAL views differ from the law-derived analytic benchmark support"
        )
    for view, native in captured.items():
        _validate_deterministic_mean(
            native,
            "native_analytic_mean",
            expected[view],
            analytic.error_bounds[view],
            view in known_zero,
            "native EAL",
        )


def _positive(weights: Mapping[str, float]) -> dict[str, float]:
    return {key: weight for key, weight in weights.items() if weight > 0}


def _view_weights(
    package: RunSetup, weights_by_arm: Mapping[str, Mapping[str, float]]
) -> dict[str, ViewWeights]:
    """Each arm's weights in each of its views, by state key: the arm's own,
    and in the views that count the lab's loss those of the states that are
    not left out of them (``SampleGroups.view_weights``)."""
    groups = sample_groups(package.joint_states, state_key)
    return {
        arm: groups.view_weights(weights) for arm, weights in weights_by_arm.items()
    }


def _band_arguments(package, plan, layout, weights_by_arm, draws) -> dict:
    """The reference arm's band inputs under the producing ``plan`` at
    ``weights_by_arm``'s weights: the compiled positive-weight states, the
    weights of the views that count the lab's loss where a state is left out
    of them, and the views that every state with weight in them captures."""
    reference = plan.reference_arm
    by_view = _view_weights(package, weights_by_arm)[reference]
    weights = _positive(by_view.arm)
    states = {state_key(state.state_id): state for state in package.joint_states}
    arm_plan = plan.for_arm(reference)
    controls = TailGateControls.from_controls(
        arm_plan.numerical_controls.get("tail_gate")
    )
    return {
        "compiled": {key: layout.compiled(reference, key) for key in weights},
        "weights": weights,
        "lab_loss_weights": _positive(by_view.counting_lab_loss)
        if by_view.differ and by_view.counting_lab_loss is not None
        else None,
        "epistemic_groups": {key: states[key].epistemic_group_id for key in weights},
        "views": [
            view
            for view in arm_plan.requested_views
            if (of := by_view.of(view)) is not None
            and all(
                view in draws[reference][key].annual_losses
                for key, weight in of.items()
                if weight > 0
            )
        ],
        "seed": arm_plan.seed,
        "batches": controls.batches,
        "tolerance": controls.tolerance,
    }


def _captured_band_reason(package, layout, capture, draws) -> str | None:
    """None when the captured band replays from the laws, else why results
    withhold it. A band captured with other band constants or by other band
    code, or whose numbers lie outside the replay tolerance, is withheld; the
    rest of the capture still replays."""
    arguments = _band_arguments(
        package, capture.plan, layout, capture.state_weights_by_arm, draws
    )
    captured = capture.eal_parameter_band
    if captured.get("method") != parameter_band_method(
        seed=arguments["seed"],
        batches=arguments["batches"],
        tolerance=arguments["tolerance"],
    ):
        return BAND_MISMATCH_REASON
    if not band_replays(captured, compute_parameter_band(**arguments)):
        return BAND_MISMATCH_REASON
    return None


def _reference_law_draws(package, draws, capture, targets):
    """Each positive state's draw under the reference arm's conditional law.

    The reference arm simulates only its positive states; a state another arm
    alone weights in ``targets``, such as the no-progress trajectory node of the
    no-progress forecast arm, is taken from the arm that simulated it under
    the same law, compiling only those few states under the reference law.
    """
    reference = package.execution.reference_arm
    result = dict(draws[reference])
    positive = {
        key for weights in targets.values() for key, weight in weights.items() if weight
    }
    keys = {}
    for arm, rows in capture.arms.items():
        for key, row in rows.items():
            if key in result or key not in positive:
                continue
            if key not in keys:
                keys[key] = compile_incident_state(
                    package, capture.states[key].state_id, reference
                ).conditional_key
            if row.conditional_key == keys[key]:
                result[key] = draws[arm][key]
    return result


def _view_map(pool: WorkerPool | None, draws):
    """The reduction's map over (arm, view) tasks: ``pool``'s, as many at once
    as memory holds at a task's bound for the largest arm (a base plus
    ``REDUCTION_BYTES_PER_STATE_YEAR`` per state-year of one view), or one at a
    time in this process."""
    if pool is None:
        return WorkerPool(1).map
    state_years = max(
        (sum(draw.n_simulations for draw in rows.values()) for rows in draws.values()),
        default=0,
    )
    return partial(
        pool.map,
        concurrency=memory_bounded_workers(
            pool.workers,
            WORKER_BASE_BYTES + REDUCTION_BYTES_PER_STATE_YEAR * state_years,
        ),
    )


def reduce_capture(
    package,
    draws,
    capture,
    weights,
    code,
    spec,
    weights_by_arm=None,
    band_reason=None,
    pool: WorkerPool | None = None,
) -> NativeResults:
    """Reduce validated captures; ``band_reason`` withholds the captured
    parameter band when validation found it does not replay. ``pool``
    reduces each arm's views in worker processes."""
    targets = arm_weights(package, weights, weights_by_arm)
    groups = sample_groups(package.joint_states, state_key)
    provenance = {
        "content_sha256": capture.content_sha256,
        "originating_setup_sha256": capture.originating_setup_sha256,
        "originating_scientific_sha256": capture.originating_scientific_sha256,
        "producer_provenance": capture.producer_provenance,
        "producer_execution_plan": capture.execution_plan,
        "selected_setup_sha256": package.manifest_sha256,
        "selected_setup_plan_sha256": package.execution_plan_sha256,
        "grain": capture.grain,
        "physical_joint_semantics": capture.physical_joint_semantics,
        "support_records": {
            key: record.model_dump(mode="json")
            for key, record in capture.supports.items()
        },
        "support_by_arm_state": {
            arm: {state: row.support_sha256 for state, row in rows.items()}
            for arm, rows in capture.arms.items()
        },
    }
    if package.contracts:
        provenance["financial_view_semantics"] = {
            "aggregation": "SUM_UNCOORDINATED_CONTRACT_EXPOSURE",
            "interpretation": "Financial totals sum authored contract exposures and may exceed unique gross harm when contracts cover the same component.",
            "outside_limit_expenses": "Declared outside-limit expenses bypass the occurrence retention and occurrence limit; annual transforms follow the authored contract.",
        }
    current_assessment = {
        "evidence": [
            evidence.model_dump(mode="json") for evidence in package.evidence.records
        ],
        "limitations": package.limitations,
        "assessor_provenance": code,
    }
    provenance["current_assessment_sha256"] = canonical_json_sha256(current_assessment)
    provenance["current_assessment"] = current_assessment
    # The capture's own evidence assessment, which a later setup can revise:
    # always its digest, and its content only where it differs from the current.
    provenance["producer_assessment_sha256"] = canonical_json_sha256(
        capture.evidence_reliance
    )
    producer = {
        key: capture.evidence_reliance.get(key) for key in ("evidence", "limitations")
    }
    if canonical_json_sha256(producer) != canonical_json_sha256(
        {key: current_assessment[key] for key in ("evidence", "limitations")}
    ):
        provenance["producer_assessment"] = producer
    # The reduction's report becomes a run's: the sections below complete
    # it, and the caller adds the producer provenance and execution context.
    report = cast(
        NativeResults,
        reduce_captures(
            scientific_sha256=package.scientific_sha256,
            plan_sha256=capture.execution_plan_sha256,
            draws_by_arm=draws,
            state_weights=targets[package.execution.reference_arm],
            state_weights_by_arm=targets,
            comparison_requirements=package.execution.comparison_requirements,
            required_arms=package.execution.required_arms,
            reference_arm=package.execution.reference_arm,
            matched_controls={
                arm.alternative_id: arm.matched_control_id
                for arm in package.alternatives
                if arm.alternative_id in package.execution.required_arms
                and arm.matched_control_id is not None
            },
            views=package.execution.requested_views,
            reducer_provenance=code,
            spec=spec,
            capture_provenance=provenance,
            epistemic_groups={
                state_key(state.state_id): state.epistemic_group_id
                for state in package.joint_states
            },
            sample_groups=groups,
            view_map=_view_map(pool, draws),
        ),
    )
    report["schema_version"] = RESULTS_SCHEMA_VERSION
    report["study"] = study_scope(package)
    # Each exhibit producer builds its record dynamically; the results
    # schema tests check the records against their contracts.
    report["tail_gate"] = cast(
        TailGateResults,
        tail_gate_results(
            package.execution,
            capture.tail_gate_rounds,
            report,
            _arm_states(capture),
            _followers(capture),
        ),
    )
    reference_weights = targets[package.execution.reference_arm]
    report["small_event_aggregation"] = cast(
        SmallEventAggregation,
        small_event_results(
            {
                arm: {state: row.execution for state, row in rows.items()}
                for arm, rows in capture.arms.items()
            },
            targets,
            package.execution.reference_arm,
            spec.levels,
            view_weights={
                arm: groups.view_weights(weights).of for arm, weights in targets.items()
            },
        ),
    )
    report["coverage"]["scenarios"] = scenario_coverage(
        package,
        report,
        {
            state.state_id
            for state in package.joint_states
            if reference_weights[state_key(state.state_id)] > 0
        },
    )
    report["limitations"] = result_limitations(package, report)
    report["scenario_contributions"] = cast(
        ScenarioContributions,
        scenario_contributions(
            package,
            report,
            draws[package.execution.reference_arm],
            reference_weights,
            view_weights=groups.view_weights(reference_weights),
        ),
    )
    report["scenario_groups"] = cast(
        ScenarioGroups,
        scenario_groups(
            report["scenario_contributions"],
            library_classification(
                population.scenario_id for population in package.scope.populations
            ),
        ),
    )
    reweighted = targets != arm_weights(package, None)
    report["sensitivity_summary"] = cast(
        SensitivitySummary,
        sensitivity_results(
            package,
            report,
            _reference_law_draws(package, draws, capture, targets),
            reference_weights,
            levels=spec.levels,
            reweighted=reweighted,
            view_weights=groups.view_weights(reference_weights),
            sampling=respondent_sampling(
                report,
                draws,
                targets,
                groups,
                reference_arm=package.execution.reference_arm,
            ),
        ),
    )
    reference = package.execution.reference_arm
    report["sensitivity_summary"]["eal_parameter_band"] = cast(
        EalParameterBand,
        withheld_parameter_band(band_reason)
        if band_reason is not None
        else reported_parameter_band(
            capture.eal_parameter_band,
            report["arms"][reference].get("metrics", {}),
            reweighted=_positive(reference_weights)
            != _positive(capture.state_weights_by_arm[reference]),
            borrowed=partial(
                borrowed_band_reason, groups, groups.view_weights(reference_weights)
            ),
        ),
    )
    report["capability_scenarios"] = cast(
        CapabilityScenarios,
        capability_scenarios(
            package,
            draws,
            capture,
            report,
            targets,
            reweighted=reweighted,
        ),
    )
    report["workload"] = cast(
        Workload,
        execution_budget(
            capture.plan,
            {
                arm: {state: row.conditional_key for state, row in rows.items()}
                for arm, rows in capture.arms.items()
            },
        ),
    )
    return report


def setup_workload(package: RunSetup) -> dict:
    """Count the frozen setup's simulation work, including reused draws."""
    return execution_budget(package.execution, StateLayout.keys_of(package))


def load_with_workload(setup_path: Path) -> tuple[RunSetup, dict]:
    """Load the setup saved at ``setup_path`` and count its simulation work
    (``setup_workload``) from the states the load compiles as its check, so
    each arm is read once."""
    layout = StateLayout.loading(setup_path)
    layout._load()
    return layout.package, execution_budget(layout.package.execution, layout.keys)


def _gate_controls(package, followers) -> dict[str, str | None]:
    """Each deciding comparison arm's control, the arm its effects are measured
    against: its matched control or else the reference arm, which also stands
    in for a control that follows it. None for the reference arm and for an
    arm that is its own control."""
    plan = package.execution
    matched = {
        row.alternative_id: row.matched_control_id
        for row in package.alternatives
        if row.matched_control_id is not None
    }
    result: dict[str, str | None] = {}
    for arm in plan.required_arms:
        control = matched.get(arm, plan.reference_arm)
        if control in followers:
            control = plan.reference_arm
        result[arm] = None if control == arm else control
    return result


def _doubles(arm, failing, limits, gate_controls, doubling) -> bool:
    """Whether a deciding arm doubles after a round: it can still double within
    its cap, and its drivers or its control's failed in the round, so its
    effects are measured on years where its control's tails are still being
    refined."""
    control = gate_controls[arm]
    return doubling < limits[arm] and bool(
        failing[arm] or (control is not None and failing.get(control))
    )


@dataclass(frozen=True)
class _Extension:
    """One state's simulation to extend in a round, for the first arm (in arm
    order) that requires it: everything a worker process needs."""

    arm: str
    plan: ExecutionPlan
    compiled: CompiledIncidentState
    response: str
    years: int
    previous: SimulatedState | None
    split_cache: dict


def _extend(task: _Extension) -> tuple[SimulatedState, dict]:
    """Extend one state's simulation to the task's years; returns it and its
    small-event split cache. Runs in a worker process or in this one."""
    try:
        simulated = simulate_incident_state(
            task.plan,
            task.compiled,
            response_sha256=task.response,
            years=task.years,
            previous=task.previous,
            split_cache=task.split_cache,
        )
    except WorkLimitExceeded as error:
        raise error.in_arm(task.arm) from None
    return simulated, task.split_cache


def _round_tasks(extensions, layout, plans, response, years, simulated, split_caches):
    """A round's extensions as tasks, built on demand. Each takes its compiled
    state from ``layout`` and moves its state's earlier years and split cache
    out of ``simulated`` and ``split_caches``, so a compiled state is held
    only while its task is, and the arrays once, by the task, until its
    result replaces them. The pool builds every task as it submits them to
    worker processes, and each as it starts when they run here."""
    for identity, (arm, key) in extensions.items():
        yield _Extension(
            arm,
            plans[arm],
            layout.compiled(arm, key),
            response,
            years[arm],
            simulated.pop(identity, None),
            split_caches.pop(identity, {}),
        )


def _simulate_rounds(
    layout,
    plans,
    controls,
    weights,
    response,
    limits,
    followers,
    gate_controls,
    pool: WorkerPool,
):
    """Simulate every arm's states, doubling each arm's years while its drivers fail.

    Each deciding arm doubles after every round in which its drivers, or its
    control's (``gate_controls``), fail and it can still double within its cap
    (``limits``), and then keeps its draws, even where another arm extends a
    simulation they shared. An arm that follows the reference arm
    (``followers``) is not simulated or evaluated: it takes the reference
    arm's draws. A year's draws do not depend on how many years are
    simulated, so a doubling extends each simulation with the new years rather
    than redrawing the earlier ones (``simulate_incident_state``). Equal
    simulation identities give equal draws, so arms that leave a state's
    conditional law and numerical controls unchanged share that state's
    simulation while they double together. ``weights`` holds each arm's
    weights in each of its views (``ViewWeights``): a driver is gated at its
    own.

    A round's simulations are independent, so ``pool`` runs them in worker
    processes, as many at once as memory holds beside the arrays this process
    keeps for the round (``simulation_worker_bytes``). They are assembled in
    arm and state order, so the draws and every refusal are those of running
    them one by one. Returns each arm's draws, withheld views and simulation
    audits at its final years, and each round's unconverged drivers of the arms it
    evaluated.
    """
    simulated: dict[str, SimulatedState] = {}
    split_caches: dict[str, dict] = {}
    rounds: list[dict[str, object]] = []
    draws, unavailable, audits = {}, {}, {}
    active = [arm for arm in layout.keys if arm not in followers]
    doubling = 0
    # The last measured peak of a round's tasks, with that round's years.
    measured: tuple[int, int] | None = None
    while active:
        # Active arms release their draws first, so only ``simulated`` keeps
        # the arrays this round extends.
        for arm in active:
            for results in (draws, unavailable, audits):
                results.pop(arm, None)
        identities = {
            arm: {
                key: simulation_identity(plans[arm], arm, key, conditional_key)
                for key, conditional_key in layout.keys[arm].items()
            }
            for arm in active
        }
        # The first arm, in arm order, that requires each simulation, and its
        # state.
        extensions: dict[str, tuple[str, str]] = {}
        for arm in active:
            for key, identity in identities[arm].items():
                extensions.setdefault(identity, (arm, key))
        years = {arm: plans[arm].simulations * 2**doubling for arm in active}
        # This process keeps each extended state's arrays: a float per year
        # and requested view.
        arrays = sum(
            8 * years[arm] * len(plans[arm].requested_views)
            for arm, _ in extensions.values()
        )
        processes = min(
            len(extensions),
            memory_bounded_workers(
                pool.workers,
                simulation_worker_bytes(max(years.values()), measured),
                reserved=arrays,
            ),
        )
        logger.info(
            "simulating tail gate round",
            doubling=doubling,
            arms=len(active),
            simulations=len(extensions),
            years_per_state=max(years.values()),
            processes=processes,
        )
        extended = pool.map(
            _extend,
            _round_tasks(
                extensions, layout, plans, response, years, simulated, split_caches
            ),
            concurrency=processes,
        )
        if pool.peak_task_bytes is not None:
            measured = (pool.peak_task_bytes, max(years.values()))
        for identity, (state, cache) in zip(extensions, extended, strict=True):
            simulated[identity], split_caches[identity] = state, cache
        del extended
        failing = {}
        for arm in active:
            draws[arm], unavailable[arm], audits[arm] = {}, {}, {}
            for key, identity in identities[arm].items():
                (
                    draws[arm][key],
                    unavailable[arm][key],
                    audits[arm][key],
                ) = simulated[identity]
            failing[arm] = unconverged_drivers(
                draws[arm],
                weights[arm].arm,
                controls[arm],
                plans[arm].requested_views,
                DEFAULT_TAIL_LEVELS,
                view_weights=weights[arm].of,
            )
        rounds.append({"doubling": doubling, "unconverged_views": failing})
        active = [
            arm
            for arm in active
            if _doubles(arm, failing, limits, gate_controls, doubling)
        ]
        doubling += 1
    reference = plans[next(iter(layout.keys))].reference_arm
    for arm in followers:
        for results in (draws, unavailable, audits):
            results[arm] = {key: results[reference][key] for key in layout.keys[arm]}
    return (
        {arm: draws[arm] for arm in layout.keys},
        {arm: unavailable[arm] for arm in layout.keys},
        {arm: audits[arm] for arm in layout.keys},
        rounds,
    )


class IncompleteRun(RuntimeError):
    """A run whose simulation is saved but whose results are not at its output:
    they were not derived, or the finished run could not be moved there. Its
    message names the saved capture and the command that derives results from
    it."""


# The note an incomplete run leaves at its output's root, beside ``capture/``.
INCOMPLETE_RUN_NOTE = "INCOMPLETE.txt"
_RUN_RESULTS = "results.json"
_RUN_RESULT_FILES = (
    _RUN_RESULTS,
    diagnostics_path(Path(_RUN_RESULTS)).name,
    "tornado.svg",
)


def _cause(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


@contextlib.contextmanager
def _staging(output: Path) -> Generator[Path]:
    """The directory in which one run is staged: ``run`` inside a temporary
    directory beside ``output``.

    Leaving removes the temporary directory, with whatever a failed run left
    in it, unless the staged run is still there with a complete capture, one
    whose manifest, frozen last, exists. That is the capture's only copy, so
    it stays: nothing that fails after the simulation deletes its capture.
    """
    directory = TemporaryDirectory(prefix=".run-", dir=output.parent, delete=False)
    stage = Path(directory.name) / "run"
    try:
        yield stage
    finally:
        if not (stage / "capture" / CAPTURE_MANIFEST).exists():
            directory.cleanup()


def _move_out(stage: Path, output: Path) -> tuple[Path, OSError | None]:
    """Move a staged run to ``output``; returns where the run then is and,
    when that is elsewhere, why ``output`` did not take it.

    A path created since the run began, or one the run may not be moved to,
    does not take it. The run then goes to a new directory beside ``output``,
    ``<output>.unpublished-<random>``, and when it cannot be moved there
    either it stays where it is staged.
    """
    try:
        stage.rename(output)
    except OSError as error:
        aside = output.with_name(f"{output.name}.unpublished-{secrets.token_hex(4)}")
        try:
            stage.rename(aside)
        except OSError:
            return stage, error
        return aside, error
    return output, None


def _replay_command(setup_path: Path, run: Path) -> str:
    return (
        f"catmodel results replay --setup {setup_path} "
        f"--capture {run / 'capture'} --output <new directory>"
    )


def _keep_capture(
    stage: Path, output: Path, setup_path: Path, error: BaseException
) -> tuple[Path, str]:
    """Move a run whose results failed or were interrupted to ``output``
    (``_move_out``) with its complete capture, without partial results and
    with a note saying how to derive them; returns where the run is and that
    command. Hours of simulation are kept rather than removed with the
    temporary directory."""
    try:
        for name in _RUN_RESULT_FILES:
            (stage / name).unlink(missing_ok=True)
    except OSError:
        # Partial results that cannot be removed are not moved to an output.
        saved = stage
    else:
        saved, _ = _move_out(stage, output)
    command = _replay_command(setup_path, saved)
    # On a full disk the note may fail; the raised error still says how to
    # recover.
    with contextlib.suppress(OSError):
        (saved / INCOMPLETE_RUN_NOTE).write_text(
            "This run is incomplete. Its simulation finished and its capture is "
            f"saved in capture/, but its results were not derived "
            f"({_cause(error)}).\nDerive them from the capture with:\n\n"
            f"    {command}\n",
            encoding="utf-8",
        )
    logger.error(
        "the run's results were not derived; its capture is saved",
        capture=str(saved / "capture"),
        derive_with=command,
    )
    return saved, command


def run_setup(
    *,
    setup_path: Path,
    output: Path,
    code_commit: str | None = None,
    workers: int = 1,
) -> NativeResults:
    """Execute the frozen setup and retain immutable annual draws.

    Up to ``workers`` processes simulate each tail gate round's states
    (``_simulate_rounds``) and reduce each arm's views; on the same platform
    and dependency versions the results do not depend on it, and the report
    records it as execution context. Workers are spawned, so a Python script
    that calls this with more than one worker must guard its entry point with
    ``if __name__ == "__main__":``.

    Once the simulation's capture is complete, nothing that fails deletes it:
    a run whose results are not derived, or which ``output`` cannot take,
    raises ``IncompleteRun`` naming where its capture is saved
    (``_keep_capture``, ``_move_out``).
    """
    pool = WorkerPool(workers)
    require_new_evidence_output(output)
    # The states the load compiles as its check are the run's.
    with StateLayout.loading(setup_path, output.parent) as layout, pool:
        package = layout.package
        code = setup_code_provenance(code_commit)
        weights = arm_weights(package, None)
        conditional_keys = layout.keys
        response = canonical_json_sha256(conditional_keys)
        plans = {arm: package.execution.for_arm(arm) for arm in conditional_keys}
        controls = {
            arm: TailGateControls.from_controls(
                plan.numerical_controls.get("tail_gate")
            )
            for arm, plan in plans.items()
        }
        followers = reference_followers(package.execution, conditional_keys)
        limits = doubling_limits(
            package.execution,
            {arm: len(states) for arm, states in conditional_keys.items()},
            followers,
        )
        draws, unavailable, audits, rounds = _simulate_rounds(
            layout,
            plans,
            controls,
            _view_weights(package, weights),
            response,
            limits,
            followers,
            _gate_controls(package, followers),
            pool,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with _staging(output) as stage:
            capture_root = stage / "capture"
            capture_root.mkdir(parents=True)
            supports, arms = {}, {}
            for arm, arm_draws in draws.items():
                arms[arm] = {}
                for state, draw in arm_draws.items():
                    audit = audits[arm][state]
                    support = CapturedSupport(
                        unavailable_views=unavailable[arm][state],
                        unavailable_functionals=audit.functional_unavailable_by_view,
                        view_bases=audit.view_bases,
                    )
                    support_key = canonical_json_sha256(support.model_dump(mode="json"))
                    supports[support_key] = support
                    record, data = encode_conditional_draw(draw)
                    path = capture_root / record.array_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        if path.read_bytes() != data:
                            raise ValueError("numeric content identity collision")
                    else:
                        path.write_bytes(data)
                    arms[arm][state] = CapturedConditional(
                        conditional_key=conditional_keys[arm][state],
                        seed=audit.stream_seed,
                        support_sha256=support_key,
                        draw=record,
                        execution=audit.execution_record(),
                    )
            provisional = RunCapture(
                originating_setup_sha256=package.manifest_sha256,
                originating_scientific_sha256=package.scientific_sha256,
                states={
                    state_key(state.state_id): CapturedState(
                        state_id=state.state_id,
                        epistemic_group_id=state.epistemic_group_id,
                    )
                    for state in package.joint_states
                },
                supports=supports,
                arms=arms,
                execution_plan=strict_json_loads(package.files["execution-plan.json"]),
                state_weights_by_arm=weights,
                tail_gate_rounds=rounds,
                eal_parameter_band=compute_parameter_band(
                    **_band_arguments(
                        package, package.execution, layout, weights, draws
                    )
                ),
                producer_provenance=code,
                evidence_reliance={
                    "status": "author_assessments",
                    "evidence": [
                        evidence.model_dump(mode="json")
                        for evidence in package.evidence.records
                    ],
                    "limitations": package.limitations,
                },
                content_sha256="0" * 64,
            )
            capture = provisional.model_copy(
                update={
                    "content_sha256": canonical_json_sha256(
                        provisional.model_dump(mode="json", exclude={"content_sha256"})
                    )
                }
            )
            freeze_json(
                capture_root / CAPTURE_MANIFEST, capture.model_dump(mode="json")
            )
            # Released first, so the reloaded capture's arrays are held only once.
            del draws, unavailable, audits, arm_draws, draw
            try:
                draws, capture = load_run_capture(capture_root)
                report = reduce_capture(
                    package,
                    draws,
                    capture,
                    None,
                    code,
                    ReductionSpec(),
                    pool=pool,
                )
                report["producer_provenance"] = cast(CapturedCodeProvenance, code)
                report["execution_context"] = {"workers": workers}
                write_results(stage / _RUN_RESULTS, report)
                with (stage / "tornado.svg").open("x", encoding="utf-8") as handle:
                    handle.write(tornado_svg(report["sensitivity_summary"]))
            except BaseException as error:
                saved, command = _keep_capture(stage, output, setup_path, error)
                if isinstance(error, Exception):
                    raise IncompleteRun(
                        f"the run's results were not derived ({_cause(error)}); "
                        f"its simulation is saved in {saved / 'capture'}, from "
                        f"which `{command}` derives them"
                    ) from error
                raise
            saved, refusal = _move_out(stage, output)
            if refusal is not None:
                raise IncompleteRun(
                    f"the run finished but could not be moved to {output} "
                    f"({_cause(refusal)}); it is saved whole in {saved}, results "
                    f"included, and its simulation in {saved / 'capture'}, from "
                    f"which `{_replay_command(setup_path, saved)}` derives them again"
                ) from refusal
    return report


def load_run_capture(directory: Path):
    directory = directory.resolve()
    path = directory / CAPTURE_MANIFEST
    if path.is_symlink():
        raise ValueError("run capture rejects symlinks")
    payload = strict_json_loads(path.read_bytes())
    if not isinstance(payload, dict) or payload.get(
        "content_sha256"
    ) != canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    ):
        raise ValueError("run capture metadata checksum mismatch")
    require_marker(
        payload,
        "schema_version",
        RUN_CAPTURE_SCHEMA_VERSION,
        "run capture schema version",
        "recapture it by executing the frozen plan with the current software, "
        "or replay it with its pinned producer software",
    )
    capture = RunCapture.model_validate(payload)
    if set(capture.state_weights_by_arm) != set(capture.plan.required_arms) or any(
        set(weights) != set(capture.states)
        or any(not math.isfinite(value) or value < 0 for value in weights.values())
        or not math.isclose(math.fsum(weights.values()), 1, abs_tol=1e-12)
        for weights in capture.state_weights_by_arm.values()
    ):
        raise ValueError("capture arm target weights differ from its frozen support")
    if set(capture.arms) != set(capture.plan.required_arms):
        raise ValueError(
            "run capture comparison arms differ from its frozen requirements"
        )
    if not capture.states or any(
        state_key(row.state_id) != key for key, row in capture.states.items()
    ):
        raise ValueError(
            "run capture state identity differs from its declared identifier"
        )
    if any(
        set(rows)
        != {
            state_key(identifier)
            for identifier in required_state_ids(
                capture.plan, arm, (state.state_id for state in capture.states.values())
            )
        }
        or any(
            weight > 0 and key not in rows
            for key, weight in capture.state_weights_by_arm[arm].items()
        )
        for arm, rows in capture.arms.items()
    ):
        raise ValueError(
            "run capture conditional support differs from frozen state requirements"
        )
    referenced = {
        row.support_sha256 for rows in capture.arms.values() for row in rows.values()
    }
    if referenced != set(capture.supports) or any(
        canonical_json_sha256(row.model_dump(mode="json")) != key
        for key, row in capture.supports.items()
    ):
        raise ValueError(
            "run capture support metadata differs from its content identity or closure"
        )
    payloads = {
        row.draw.array_path for rows in capture.arms.values() for row in rows.values()
    }
    if {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    } != {CAPTURE_MANIFEST, *payloads}:
        raise ValueError("run capture contains missing or untracked files")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("run capture rejects symlinks")
    response = canonical_json_sha256(
        {
            arm: {state: row.conditional_key for state, row in rows.items()}
            for arm, rows in capture.arms.items()
        }
    )
    years = _gated_years(capture)
    reference = capture.arms[capture.plan.reference_arm]
    if any(
        row != reference[state]
        for arm in _followers(capture)
        for state, row in capture.arms[arm].items()
    ):
        raise ValueError("a following arm's capture differs from the reference arm's")
    draws = {}
    # Arms that share a state's draw store one array file and record; each is
    # decoded once and the arms hold the same arrays.
    decoded = {}
    for arm, rows in capture.arms.items():
        draws[arm] = {}
        for state, row in rows.items():
            if row.seed != state_stream_seed(
                capture.plan.seed, capture.states[state].state_id
            ):
                raise ValueError(
                    "capture stream seed differs from its frozen plan and state"
                )
            identity = (state, row.seed, row.draw.model_dump_json())
            if identity not in decoded:
                decoded[identity] = decode_conditional_draw(
                    row.draw,
                    (directory / row.draw.array_path).read_bytes(),
                    stratum_sha256=state,
                    response_sha256=response,
                    seed=row.seed,
                )
            draw = decoded[identity]
            if draw.n_simulations != years[arm]:
                raise ValueError(
                    "capture arrays differ from their frozen numerical controls"
                )
            draws[arm][state] = draw

    return draws, capture


def _arm_states(capture: RunCapture) -> dict[str, int]:
    """Each arm's count of simulated states, which sets its cap on years per state."""
    return {arm: len(rows) for arm, rows in capture.arms.items()}


def _followers(capture: RunCapture) -> frozenset[str]:
    """The arms that follow the capture's reference arm (``reference_followers``)."""
    return reference_followers(
        capture.plan,
        {
            arm: {state: row.conditional_key for state, row in rows.items()}
            for arm, rows in capture.arms.items()
        },
    )


def _gated_years(capture: RunCapture) -> dict[str, int]:
    """Each arm's simulated years per state, re-derived from the frozen controls
    and the recorded rounds (``gated_years``)."""
    return gated_years(
        capture.plan,
        capture.tail_gate_rounds,
        _arm_states(capture),
        _followers(capture),
    )[0]


# Realized counts of a summed origin; every other field is re-derived.
_REALIZED_SMALL_EVENT_FIELDS = frozenset(
    {
        "large_events",
        "individually_drawn_events",
        "summed_events",
        "summed_years",
        "fallback_years",
        "event_by_event_years",
    }
)


def _matches(captured, expected) -> bool:
    """Equal structure, with floats equal to a relative 1e-9 across platforms."""
    if isinstance(expected, float) and isinstance(captured, int | float):
        return math.isclose(captured, expected, rel_tol=1e-9, abs_tol=0.0)
    if isinstance(expected, Mapping) and isinstance(captured, Mapping):
        return set(captured) == set(expected) and all(
            _matches(captured[key], expected[key]) for key in expected
        )
    if isinstance(expected, list | tuple) and isinstance(captured, list | tuple):
        return len(captured) == len(expected) and all(
            _matches(a, b) for a, b in zip(captured, expected, strict=True)
        )
    return captured == expected


def _validate_small_events(compiled, plan, claims, execution, support):
    """Re-derive the state's split points, moments and variances from its law.

    Only realized event counts come from the capture; a split point, moment,
    small-event variance or envelope status that the law, the views and the
    frozen controls do not reproduce is refused.
    """
    record = execution.get("small_events")
    if not isinstance(record, Mapping) or not isinstance(
        record.get("by_origin"), Mapping
    ):
        raise ValueError("capture lacks its small-event record")
    views = tuple(
        view for view in plan.requested_views if view not in support.unavailable_views
    )
    _, expected = small_event_audit(
        compiled,
        views,
        SmallEventControls.from_controls(plan.numerical_controls.get("small_events")),
    )
    captured = {
        **record,
        "by_origin": {
            origin: {
                key: value
                for key, value in row.items()
                if key not in _REALIZED_SMALL_EVENT_FIELDS
            }
            for origin, row in record["by_origin"].items()
        },
    }
    if not _matches(captured, expected):
        raise ValueError(
            "capture small-event split points, moments or variances differ from the law and its frozen controls"
        )
    _validate_realized_small_events(record, execution, claims.years)


def _validate_realized_small_events(record, execution, years: int) -> None:
    """Check the captured realized small-event counts against each other.

    Draws are not replayed, so the counts cannot be re-derived. Each summed
    origin's large events must not exceed its individually drawn events, which
    with its summed events must add to its initiating count, and its summed,
    fallback and event-by-event years must not together exceed the simulated
    years; the summed events of all origins must equal the state's summed
    initiators.
    """
    counts = execution.get("initiating_counts", {})
    summed = 0
    for origin, row in record["by_origin"].items():
        if row.get("status") != "summed_small_events":
            continue
        values = [row.get(field) for field in sorted(_REALIZED_SMALL_EVENT_FIELDS)]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("capture lacks an origin's realized small-event counts")
        summed += row["summed_events"]
        if (
            row["large_events"] > row["individually_drawn_events"]
            or row["individually_drawn_events"] + row["summed_events"]
            != counts.get(origin)
            or row["summed_years"] + row["fallback_years"] + row["event_by_event_years"]
            > years
        ):
            raise ValueError(
                "capture realized small-event counts are inconsistent with the origin's initiating count or years"
            )
    if summed != execution.get("summed_small_initiators"):
        raise ValueError(
            "capture summed small events differ from the state's summed initiators"
        )


def _validate_work(plan, years: int, execution) -> None:
    """Re-derive a captured state's work limits, the plan's allowances per year
    times its captured ``years``, and check its recorded limits and counts."""
    limits = work_limits(work_allowances(plan.numerical_controls), years)
    if execution.get("work_limits") != limits or any(
        isinstance(count := execution.get(name), bool)
        or not isinstance(count, int)
        or not 0 <= count <= limit
        for name, limit in limits.items()
    ):
        raise ValueError(
            "capture work counts or limits differ from the plan's work allowances "
            "per year times the captured years"
        )


def _validate_gate_rounds(
    capture: RunCapture,
    draws,
    gate_controls,
    view_weights: Mapping[str, ViewWeights] | None = None,
) -> None:
    """Recompute every round's unconverged drivers on the final draws' prefixes.

    A year's draws do not depend on how many years are simulated, so each
    earlier round's draws are a prefix of the final ones: every round must
    reproduce the drivers of the arms it evaluated, and an arm must have
    doubled after exactly the rounds in which its drivers, or its control's
    (``gate_controls``), failed below its cap. ``view_weights`` holds each
    arm's captured weights in each of its views, at which the run gated its
    drivers; without it every view has the arm's captured weights.
    """
    if view_weights is None:
        view_weights = {
            arm: ViewWeights.same(weights)
            for arm, weights in capture.state_weights_by_arm.items()
        }
    plans = {name: capture.plan.for_arm(name) for name in draws}
    controls = {
        name: TailGateControls.from_controls(plan.numerical_controls.get("tail_gate"))
        for name, plan in plans.items()
    }
    limits = doubling_limits(capture.plan, _arm_states(capture), _followers(capture))
    last = round_arms(capture.tail_gate_rounds)
    for row in capture.tail_gate_rounds:
        doubling = cast(int, row["doubling"])
        recorded = cast(Mapping[str, list[str]], row["unconverged_views"])
        for arm in recorded:
            years = plans[arm].simulations * 2**doubling
            failing = unconverged_drivers(
                {key: year_prefix(draw, years) for key, draw in draws[arm].items()},
                view_weights[arm].arm,
                controls[arm],
                plans[arm].requested_views,
                DEFAULT_TAIL_LEVELS,
                view_weights=view_weights[arm].of,
            )
            if failing != recorded[arm]:
                raise ValueError(
                    "capture tail gate rounds are not reproduced by the captured years"
                )
        # Every evaluated arm's drivers are reproduced, so each arm's doubling
        # follows from them.
        for arm in recorded:
            doubles = _doubles(arm, recorded, limits, gate_controls, doubling)
            if doubles and doubling == last[arm]:
                raise ValueError(
                    "capture tail gate stopped an arm whose drivers failed and whose cap allowed another doubling"
                )
            if not doubles and doubling < last[arm]:
                raise ValueError(
                    "capture tail gate doubled an arm whose drivers had converged or reached its cap"
                )


@dataclass(frozen=True)
class _LawCheck:
    """One captured conditional to re-derive from its law: everything a worker
    process needs."""

    compiled: CompiledIncidentState
    plan: ExecutionPlan
    claims: _Claims
    execution: dict[str, object]
    support: CapturedSupport


def _check_law(task: _LawCheck) -> None:
    """Refuse a captured conditional whose claims its law does not reproduce:
    support and moment availability, native EALs, how its draws were
    addressed (``stream_design``), work and small events. Runs in a worker
    process or in this one."""
    views = tuple(task.claims.views)
    known_zero = analytic_known_zero_views(task.compiled, views)
    analytic = analytic_linear_eals(task.compiled, views)
    _validate_captured_support(
        task.compiled, task.plan, task.claims, task.support, known_zero, analytic
    )
    _validate_native_analytic_eals(task.compiled, task.claims, known_zero, analytic)
    if task.execution.get("stream_design") != stream_design(
        task.compiled, task.plan.seed
    ):
        raise ValueError(
            "capture stream design differs from this engine's: its draws were "
            "addressed differently, so recapture it by executing the frozen setup"
        )
    _validate_work(task.plan, task.claims.years, task.execution)
    _validate_small_events(
        task.compiled, task.plan, task.claims, task.execution, task.support
    )


def validate_captured_laws(
    package,
    draws,
    capture,
    targets,
    *,
    require_target_coverage=True,
    validate_band=True,
    pool: WorkerPool | None = None,
    layout: StateLayout | None = None,
) -> str | None:
    """Rederive every captured claim from the laws; refuse the capture on a
    mismatch. Returns why results must withhold the captured parameter band,
    which never refuses a capture: None when it replays, is not validated
    (``validate_band``) or will be withheld for reweighting anyway.

    Every record's conditional law and years are checked first, in arm and
    state order. Then each distinct captured conditional (its arm's plan and
    its whole record) is re-derived from its law once: an arm that follows
    the reference arm holds the reference arm's records, which
    ``load_run_capture`` has checked equal. ``pool`` re-derives them in worker
    processes; a refusal is the first failing distinct record in arm and state
    order, as re-deriving them one by one gives. ``layout`` is the setup's
    states as its load compiled them (``StateLayout.loading``); without one
    they are compiled here, into the system's temporary directory.
    """
    states = {state_key(state.state_id) for state in package.joint_states}
    with contextlib.ExitStack() as stack:
        if layout is None:
            layout = stack.enter_context(StateLayout(package))
        years = _gated_years(capture)
        checks: dict[
            str, tuple[str, str, ExecutionPlan, _Claims, dict, CapturedSupport]
        ] = {}
        for arm, arm_draws in draws.items():
            if arm not in layout.keys:
                continue
            for key, draw in arm_draws.items():
                conditional_key = layout.keys[arm].get(key)
                if conditional_key is None and key in states:
                    conditional_key = layout.compiled(arm, key).conditional_key
                if (
                    conditional_key is None
                    or capture.arms[arm][key].conditional_key != conditional_key
                ):
                    raise ValueError(
                        "capture conditional law, ownership or ordered transforms differ from the incident specification"
                    )
                if draw.n_simulations != years[arm]:
                    raise ValueError(
                        "capture arrays differ from their frozen numerical controls"
                    )
                row = capture.arms[arm][key]
                plan = capture.plan.for_arm(arm)
                identity = canonical_json_sha256(
                    {
                        "plan": plan.model_dump(mode="json"),
                        "row": row.model_dump(mode="json"),
                    }
                )
                if identity not in checks:
                    checks[identity] = (
                        arm,
                        key,
                        plan,
                        _Claims.of(draw),
                        row.execution,
                        capture.supports[row.support_sha256],
                    )
        pool = pool or WorkerPool(1)
        pool.map(
            _check_law,
            # Built as the pool takes them, so a compiled state is held only
            # while its check is.
            (
                _LawCheck(layout.compiled(arm, key), plan, claims, execution, support)
                for arm, key, plan, claims, execution, support in checks.values()
            ),
            concurrency=memory_bounded_workers(pool.workers, WORKER_BASE_BYTES),
        )
        _validate_gate_rounds(
            capture,
            draws,
            _gate_controls(package, _followers(capture)),
            _view_weights(package, capture.state_weights_by_arm),
        )
        for arm, arm_draws in draws.items():
            if arm not in layout.keys:
                continue
            if require_target_coverage and any(
                weight > 0 and key not in arm_draws
                for key, weight in targets[arm].items()
            ):
                raise ValueError(
                    "newly positive state has no retained conditional capture"
                )
        reference = capture.plan.reference_arm
        if not validate_band or _positive(targets[reference]) != _positive(
            capture.state_weights_by_arm[reference]
        ):
            return None
        return _captured_band_reason(package, layout, capture, draws)


def replay_capture(
    *,
    setup_path: Path,
    capture_directory: Path,
    output: Path,
    state_weights: Mapping[str, float] | None = None,
    reduction_spec: ReductionSpec | None = None,
    code_commit: str | None = None,
    state_weights_by_arm: Mapping[str, Mapping[str, float]] | None = None,
    workers: int = 1,
) -> NativeResults:
    """Derive results from an immutable capture without simulating.

    Without target weights each arm keeps the setup's. ``state_weights`` gives
    every arm one target vector; where an arm authors its own state weights it
    is refused unless it equals the setup's. ``state_weights_by_arm`` supplies
    explicit targets for every required arm, which such a setup needs to
    reweight.
    ``workers`` processes check the captured laws and reduce each arm's views;
    the results do not depend on it, and the report records it as execution
    context.
    """
    pool = WorkerPool(workers)
    require_new_evidence_output(output)
    require_new_evidence_output(diagnostics_path(output))
    if output.resolve().is_relative_to(capture_directory.resolve()):
        raise ValueError("reduction cannot modify its originating capture")
    # The states the load compiles as its check are the ones replayed.
    with StateLayout.loading(setup_path, output.parent) as layout, pool:
        package = layout.package
        draws, capture = load_run_capture(capture_directory)
        targets = arm_weights(package, state_weights, state_weights_by_arm)
        band_reason = validate_captured_laws(
            package, draws, capture, targets, pool=pool, layout=layout
        )
        report = reduce_capture(
            package,
            draws,
            capture,
            state_weights,
            setup_code_provenance(code_commit),
            reduction_spec or ReductionSpec(),
            weights_by_arm=state_weights_by_arm,
            band_reason=band_reason,
            pool=pool,
        )
    report["producer_provenance"] = cast(
        CapturedCodeProvenance, capture.producer_provenance
    )
    report["execution_context"] = {"workers": workers}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_results(output, report)
    return report
