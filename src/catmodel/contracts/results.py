"""Native reduction results, before any consumer-specific projection.

A ``NativeResults`` report is stored in two files
(``catmodel.results.results_file``). ``results.json`` holds its headline, the
sections of ``HeadlineResults``: the study, the analysis identity, coverage,
limitations, each arm's statistics, the comparisons and each scenario's
contribution. The diagnostics file beside it holds the rest: the exhibits, the
sections of ``ResultsExhibits``, and the per-arm view diagnostics of its
``diagnostics`` section. In each file every reason is replaced by its code
(``intern_reasons``) and the codes are resolved in one ``reasons`` table.

Every field either file holds is declared here with its type, whether it is
always present and the values it takes; ``catmodel.contracts.results_schema``
generates the files' JSON Schema documents from these types. A ``dict`` is a
map whose keys are open (views, arms, states, scenarios, levels and the like)
and whose values are typed; a TypedDict is a record with exactly its fields.
"""

import hashlib
import typing
from collections.abc import Callable, Mapping
from typing import Any, Literal, NotRequired, TypedDict

from catmodel.contracts.contextual_risk import ExhaustedCallKind
from catmodel.worktree import CodeTreeState

# The layout of results.json and of its diagnostics file. It increases with
# every change to either file's fields, their types or their meaning
# (docs/setup-run-results.md#results-file-versions).
RESULTS_SCHEMA_VERSION = 1

# Hexadecimal digits of the SHA-256 of a reason's text that form its code.
REASON_CODE_LENGTH = 12

type SourceStatus = Literal["public", "not_yet_public"]
type Availability = Literal["available", "unavailable"]


class ContextualChainStageSummary(TypedDict):
    stage_id: str
    probability_p50: float | None


class ContextualChainSummary(TypedDict):
    scenario_id: str
    scenario_name: str
    source_status: SourceStatus
    context_id: str
    trajectory_draw_id: str | None
    progress_ratio: float | None
    anchor_score: float | None
    authoring_status: str | None
    base_unit: str | None
    base_count: float | None
    base_derivation: str | None
    scale_id: str | None
    scale_level: float | None
    factor: float | None
    opportunity_rate_p50: float | None
    opportunities_per_year_p50: float | None
    stages_p50: list[ContextualChainStageSummary]
    terminal_stage_p50: float | None
    implied_incidents_per_year_p50: float | None
    direct_count_p50: float | None
    pooled_count_p50: float | None
    severity_median: float | None
    mean_gross_loss_per_incident: float | None
    case_count: int


# --- Provenance and the analysis identity ---------------------------------


class NumericalEnvironment(TypedDict):
    python: str
    platform: str
    # Each numerical package's installed version, by distribution name.
    packages: dict[str, str]


class _CodeIdentity(TypedDict):
    commit: str | None
    auto_resolved: bool
    tree_state: CodeTreeState
    diff_sha256: str | None
    numerical_environment: NumericalEnvironment


class CodeProvenance(_CodeIdentity):
    """The running code's identity and numerical environment
    (``catmodel.setup.store.setup_code_provenance``)."""

    version: str | None


class CapturedCodeProvenance(_CodeIdentity):
    """The code provenance a run capture recorded, as replay copies it; a
    capture written before the installed version was recorded lacks
    ``version``."""

    version: NotRequired[str | None]


type RequirementStatus = Literal["required", "unavailable", "inapplicable"]


class _RequirementScope(TypedDict):
    family_id: str
    scenario_ids: list[str]
    held_fixed: list[str]


class ComparisonRequirementRecord(_RequirementScope):
    """A frozen comparison requirement
    (``catmodel.contracts.incident.ComparisonRequirement``)."""

    arm_id: str | None
    affected_views: list[str] | None
    status: RequirementStatus
    reason: str | None


class AuthoredComparisonRequirement(_RequirementScope):
    """A comparison requirement as a plan file gives it; an omitted field
    takes its default."""

    arm_id: NotRequired[str | None]
    affected_views: NotRequired[list[str] | None]
    status: NotRequired[RequirementStatus]
    reason: NotRequired[str | None]


class ExecutionRulesRecord(TypedDict):
    work_limit_exceedance: NotRequired[Literal["abort_without_capture"]]


class TailGateSettings(TypedDict):
    """A plan's ``tail_gate`` control as authored or frozen
    (``catmodel.simulation.tail_gate.TailGateControls``); an omitted setting
    takes its default."""

    tolerance: NotRequired[float]
    batches: NotRequired[int]
    max_reference_state_years: NotRequired[int | None]
    max_comparison_state_years: NotRequired[int | None]
    views: NotRequired[list[str] | None]
    levels: NotRequired[list[float] | None]


class SmallEventSettings(TypedDict):
    """A plan's ``small_events`` control
    (``catmodel.simulation.small_events.SmallEventControls``); an omitted
    setting takes its default."""

    target_large_events_per_year: NotRequired[float]
    max_body_skewness: NotRequired[float]
    body: NotRequired[Literal["normal", "individual"]]
    bucket_rate_ratio: NotRequired[float]
    top_rank_score: NotRequired[float]
    min_summed_count: NotRequired[int]


# A plan's numerical controls by name (``NUMERICAL_CONTROLS`` of
# ``catmodel.contracts.incident``): ``tail_gate`` and ``small_events`` are
# records, which null leaves at their defaults, and every other control, such
# as a work allowance, a positive integer.
type NumericalControls = dict[str, int | TailGateSettings | SmallEventSettings | None]


class ExecutionPlanRecord(TypedDict):
    """A setup's execution plan as its ``execution-plan.json`` holds it
    (``catmodel.contracts.incident.ExecutionPlan``); an omitted field takes
    its default."""

    plan_id: str
    required_arms: list[str]
    reference_arm: NotRequired[str]
    simulations: int
    seed: int
    requested_views: list[str]
    capture_grain: NotRequired[Literal["annual_joint"]]
    method: NotRequired[Literal["physical_marked_poisson"]]
    staged_rules: NotRequired[ExecutionRulesRecord]
    numerical_controls: NotRequired[NumericalControls]
    simulations_by_arm: NotRequired[dict[str, int]]
    numerical_controls_by_arm: NotRequired[dict[str, NumericalControls]]
    comparison_requirements: NotRequired[list[AuthoredComparisonRequirement]]
    requested_comparison_families: NotRequired[list[str]]
    required_state_ids_by_arm: NotRequired[dict[str, list[str]]]


class EvidenceRecord(TypedDict):
    """One evidence record of a setup (``catmodel.contracts.incident.Evidence``).

    ``acquisition`` is the record's acquisition detail, which the evidence
    contract leaves open; the sensitivity summary reports the exhibits some of
    them carry.
    """

    evidence_id: str
    origin: Literal["observation", "fit", "elicitation", "stipulation"]
    original_sha256: str
    selected_sha256: str
    target: str
    conditioning: str
    units: str
    transport: str
    acquisition: dict[str, object] | None
    raw_attempt_sha256: list[str]
    review: str
    reliance: str
    rights: str


class EvidenceAssessment(TypedDict):
    """The setup's evidence records and limitations as the reducer assessed
    them."""

    evidence: list[EvidenceRecord]
    limitations: list[str]
    assessor_provenance: CodeProvenance


class ProducerAssessment(TypedDict):
    """The capture's own evidence records and limitations, where they differ
    from the current setup's."""

    evidence: list[EvidenceRecord]
    limitations: list[str]


class ViewBasis(TypedDict):
    """A view's currency, monetary and accounting basis, with the severity
    unit of a per-scenario view and the source status of a source-status
    view."""

    currency: str
    monetary_basis: str
    accounting_standing: str
    severity_unit: NotRequired[str]
    source_public_status: NotRequired[SourceStatus]


class SupportRecord(TypedDict):
    """What one captured conditional state retains
    (``catmodel.results.execution.CapturedSupport``): each unretained view
    and each unidentified functional, with why, and each view's basis."""

    unavailable_views: dict[str, str]
    unavailable_functionals: dict[str, dict[str, str]]
    view_bases: dict[str, ViewBasis]


class FinancialViewSemantics(TypedDict):
    aggregation: Literal["SUM_UNCOORDINATED_CONTRACT_EXPOSURE"]
    interpretation: str
    outside_limit_expenses: str


class CaptureProvenance(TypedDict):
    """The capture a result reduces and the setup it was reduced under
    (``catmodel.results.execution.reduce_capture``)."""

    content_sha256: str
    originating_setup_sha256: str
    originating_scientific_sha256: str
    producer_provenance: CapturedCodeProvenance
    producer_execution_plan: ExecutionPlanRecord
    selected_setup_sha256: str
    selected_setup_plan_sha256: str
    grain: Literal["annual"]
    physical_joint_semantics: Literal["physical_joint_samples"]
    # Each captured state's support record, by its digest, and the digest of
    # each arm's states' records.
    support_records: dict[str, SupportRecord]
    support_by_arm_state: dict[str, dict[str, str]]
    # Present when the setup has contracts.
    financial_view_semantics: NotRequired[FinancialViewSemantics]
    current_assessment_sha256: str
    current_assessment: EvidenceAssessment
    producer_assessment_sha256: str
    # Present when the capture's assessment differs from the current one.
    producer_assessment: NotRequired[ProducerAssessment]


class ReductionSpecRecord(TypedDict):
    """The reduction's options (``catmodel.results.reduction.ReductionSpec``)."""

    levels: tuple[float, ...]
    atom_convention: Literal["left_inverse_fractional_atom"]
    required_grain: Literal["annual"]
    # Each joint threshold's view and loss.
    joint_thresholds: tuple[tuple[str, float], ...]


class AnalysisSpec(TypedDict):
    scientific_sha256: str
    plan_sha256: str
    capture_provenance: CaptureProvenance
    # Each arm's captured draws' digests, by state.
    captures: dict[str, dict[str, str]]
    state_weights: dict[str, float]
    state_weights_by_arm: dict[str, dict[str, float]]
    required_arms: tuple[str, ...]
    reference_arm: str
    matched_controls: dict[str, str]
    comparison_requirements: list[ComparisonRequirementRecord]
    views: tuple[str, ...]
    reduction_spec: ReductionSpecRecord
    reducer_provenance: CodeProvenance
    epistemic_groups: dict[str, str] | None


# --- Statistics of one view ------------------------------------------------


class StatisticStatus(TypedDict):
    """Whether one reported number is available and, if not, why.

    ``unavailable``: the law gives it no finite value, and the number is null.
    ``withheld``: it exists, but this run cannot represent or support it. A
    withheld VaR or TVaR keeps its point value for reference and comparisons;
    it must not be quoted as an estimate.
    """

    status: Literal["available", "unavailable", "withheld"]
    reason: str | None


type TvarCoefficientBasis = Literal["regression", "infinite_variance", "no_exact_mean"]


class TvarEstimator(TypedDict):
    """The control-variate TVaR's coefficient and why it has that value.

    The TVaR is the sample tail mean plus ``coefficient`` times the analytic
    EAL less the simulated years' mean, over one minus the level, and at
    least the VaR. ``coefficient_basis`` is ``regression`` (estimated), or by
    law ``infinite_variance`` (one) or ``no_exact_mean`` (zero: the view has
    no analytic EAL, closed-form or certified).
    """

    method: Literal["control_variate"]
    coefficient: float
    coefficient_basis: TvarCoefficientBasis


class TailPoint(TypedDict):
    """One level's VaR and TVaR, with the years beyond the VaR."""

    var: float | None
    tvar: float | None
    tvar_estimator: TvarEstimator | None
    exceedance_count: int


class TailMetrics(TailPoint):
    # Batch-means 95% intervals of simulation error, a TVaR's widened by its
    # integration margin (``GateStatus``); null when unavailable, and the
    # diagnostics' tail record says why.
    var_numerical_interval_95: list[float] | None
    tvar_numerical_interval_95: list[float] | None


class GateStatus(TypedDict):
    """A statistic's batch-means 95% half-width and whether it is within the
    tail gate's tolerance (``catmodel.simulation.tail_gate``). A TVaR's
    half-width adds its integration margin: its coefficient times the
    certified integration error bound of its EAL, over one minus the level."""

    half_width: float | None
    relative_half_width: float | None
    status: Literal["converged", "unconverged"]
    # Why it is unconverged, where not by its half-width or support.
    reason: NotRequired[str]


class SupportedGateStatus(GateStatus):
    # The sparsest batch's supporting years; null when none are needed.
    minimum_batch_exceedances: int | None


class GateLevel(TypedDict):
    """The tail gate's verdict on one level's VaR and TVaR."""

    var: GateStatus
    tvar: GateStatus | None
    minimum_batch_exceedances: int | None


class EalPrecision(TypedDict):
    """The precision of a reported EAL, aggregated over its states."""

    numerical_standard_error: NotRequired[float | None]
    standard_error_upper_bound: NotRequired[float | None]
    integration_error_bound: NotRequired[float]
    precision_basis: Literal[
        "deterministic_integration",
        "sum_of_weighted_standard_error_bounds_no_independence_assumed",
        "unavailable_component_estimator_precision",
        "sampling_design_aware_empirical_mean_precision_required",
        "unavailable_functional",
    ]
    precision_suppression_reason: str | None


class StratumEalPrecision(TypedDict):
    """One state's native EAL estimator precision, as its native mean
    records it, or why it has none."""

    numerical_standard_error: float | None
    standard_error_upper_bound: NotRequired[float]
    precision_basis: NotRequired[Literal["deterministic_integration"]]
    integration_error_bound: NotRequired[float]
    known_zero_law: NotRequired[bool]
    precision_suppression_reason: NotRequired[str]


class EalTailCoherence(TypedDict):
    """Whether the analytic EAL fits the simulated years beside it."""

    status: Literal["coherent", "incoherent"]
    reason: str | None
    exceeds_sampled_maximum: NotRequired[bool]
    exceeds_tvar_levels: NotRequired[list[str]]


class EalReconciliation(TypedDict):
    """The tail gate's reconciliation of the simulated years with the
    analytic EAL (``catmodel.simulation.tail_gate.eal_reconciliation``): its
    statistic, z, standard error and status, ``inconsistent`` when it
    withholds the VaR and TVaR."""

    statistic: Literal["mean", "limited_mean"]
    limit_level: str | None
    limit: float | None
    simulated_mean: float
    simulated_mean_standard_error: float
    expected_annual_loss: float
    integration_error_bound: float
    z: float | None
    status: Literal["consistent", "inconsistent"]


class UnavailableViewResult(TypedDict):
    status: Literal["unavailable"]
    reason: str
    # Why each positive-weight state lacks the view.
    state_reasons: dict[str, str]


class AvailableViewResult(TypedDict):
    status: Literal["available"]
    basis: ViewBasis | None
    expected_annual_loss: float | None
    # ``normalized_empirical_mean``, or the native mean estimator of every
    # positive-weight state (``native_analytic_mean``); each state's is in the
    # view's diagnostics.
    eal_estimator: str
    # The EAL's precision; each state's is in the view's diagnostics.
    eal_precision: EalPrecision
    # Whether the analytic EAL exceeds the largest simulated annual loss or a
    # reported TVaR, which the simulated years then understate; present for a
    # native EAL.
    eal_tail_coherence: NotRequired[EalTailCoherence]
    # Null without an analytic EAL or an available tail gate.
    eal_reconciliation: EalReconciliation | None
    # Empirical mean minus the EAL.
    eal_reconciliation_difference: float | None
    normalized_empirical_mean: float
    annual_loss_variance: float | None
    annual_loss_variance_estimator: Literal["weighted_empirical_variance"]
    normalized_empirical_variance: float | None
    annual_probability_positive: float
    annual_loss_median: float
    # Keyed by level.
    tail_metrics: dict[str, TailMetrics]
    # Keyed by statistic: expected_annual_loss, annual_probability_positive,
    # annual_loss_median, annual_loss_variance, normalized_empirical_variance,
    # var::<level> and tvar::<level>.
    statistics: dict[str, StatisticStatus]
    sampling_method: Literal["stratified_conditional_capability_mixture"]


type ViewResult = AvailableViewResult | UnavailableViewResult


# --- Diagnostics -------------------------------------------------------------


class MeanDiagnostics(TypedDict):
    """The empirical mean's batch-means standard error
    (``catmodel.results.reduction.mean_diagnostics``)."""

    empirical_mean_standard_error: float | None
    empirical_mean_standard_error_unavailable_reason: str | None
    standard_error_basis: Literal["batch_means"]
    empirical_mean_numerical_interval_95: list[float] | None
    interval_unavailable_reason: str | None


class TailSupport(TypedDict):
    """One state's descriptive support beyond a level's VaR."""

    independent_unit_count: int
    tail_bearing_independent_unit_count: int
    tail_mass_effective_units: float
    tail_excess_effective_units: float


class TailDiagnostic(TypedDict):
    """One level's tail support and the method and reasons of its intervals
    (``catmodel.results.reduction.tail_diagnostics``)."""

    # The coefficient-zero TVaR, so the control term can be read.
    tvar_sample_tail_mean: float | None
    strict_exceedance_count: int
    positive_year_count: int
    simulated_year_count: int
    strict_exceedance_probability: float
    tail_weight_effective_sample_size: float
    tail_mass_effective_units: float | None
    tail_support_by_stratum: dict[str, TailSupport]
    convergence: GateLevel | None
    var_interval_method: Literal["known_zero_law", "batch_means"] | None
    tvar_numerical_standard_error: float | None
    var_interval_unavailable_reason: str | None
    tvar_interval_unavailable_reason: str | None


class TailConvergence(TypedDict):
    batches: int
    years_per_state: int
    # Why the tail gate is unavailable for the view.
    unavailable_reason: str | None
    tolerance: float
    probability_positive: SupportedGateStatus | None


class TailDiagnostics(TypedDict):
    tails: dict[str, TailDiagnostic]
    convergence: TailConvergence


class StratumDiagnostics(TypedDict):
    probability: float
    sample_mass: float
    n_simulations: int
    positive_loss_samples: int
    seed: int


class ViewDiagnostics(TypedDict):
    """One available view's precision and support diagnostics, kept out of
    its headline record (``catmodel.results.reduction.tail_diagnostics`` and
    ``mean_diagnostics`` give their fields)."""

    # Each positive-weight state's EAL estimator and precision, by stratum
    # digest; the estimators are empty when the EAL is the empirical mean.
    eal_estimator_by_stratum: dict[str, str]
    eal_precision_by_stratum: dict[str, StratumEalPrecision]
    # The empirical mean's batch-means standard error.
    mean_diagnostics: MeanDiagnostics
    # The tail gate's view record and, by level, tail support, each state's
    # tail support, convergence and the intervals' method and reasons.
    tail_diagnostics: TailDiagnostics
    stratum_diagnostics: dict[str, StratumDiagnostics]
    unsimulated_zero_mass_strata: list[str]


class DiagnosticLimitations(TypedDict):
    """What the mean and tail diagnostics of every view of an arm cover."""

    mean_diagnostics: str
    tail_diagnostics: str


class ArmDiagnostics(TypedDict):
    limitations: DiagnosticLimitations
    views: dict[str, ViewDiagnostics]


class ResultsDiagnostics(TypedDict):
    """Each reduced arm's view diagnostics, by arm; an arm with no available
    view has none."""

    arms: dict[str, ArmDiagnostics]


# --- Arms, comparisons and rankings -----------------------------------------


class UnavailableArmResult(TypedDict):
    status: Literal["unavailable"]
    reason: str


class ReducedArmResult(TypedDict):
    status: Literal["partial", "reduced"]
    missing_views: list[str]
    metrics: dict[str, ViewResult]
    # The probability that every joint threshold is exceeded in one year,
    # with its tail-gate convergence and status, when the reduction asks for
    # joint thresholds. Where the status is unavailable there is no
    # probability and the convergence is null: a threshold's view is
    # unavailable in the arm, or the thresholds' views weigh its states
    # differently.
    joint_exceedance_probability: NotRequired[float]
    joint_exceedance_convergence: NotRequired[SupportedGateStatus | None]
    joint_exceedance_status: NotRequired[StatisticStatus]
    # Each epistemic group's EAL, its states mixed by weight and every count
    # integrated over its parameter rank: in a public study the groups are
    # respondents, so this is the spread between respondents and none of the
    # parameter uncertainty within a respondent's count laws, which the
    # sensitivity summary's EAL parameter band shows. By view, then group.
    conditional_eal_by_epistemic_group: NotRequired[dict[str, dict[str, float | None]]]
    conditional_eal_unavailable_reasons_by_epistemic_group: NotRequired[
        dict[str, dict[str, str]]
    ]
    # The arm's state weights in the views that count the lab's own loss,
    # its lab-loss views and its gross views, where a state with weight is
    # left out of them (POOL-EXCLUDED-SAMPLES): that state weighs zero and
    # its sampling group's other states share its weight. Every other view,
    # and every view of an arm without this field, uses the arm's weights in
    # analysis_spec.state_weights_by_arm.
    lab_loss_state_weights: NotRequired[dict[str, float]]


type ArmResult = UnavailableArmResult | ReducedArmResult


class ComparisonResult(TypedDict):
    reference: float
    alternative: float
    absolute_difference: float | None
    absolute_difference_unavailable_reason: str | None
    relative_change: float | None
    relative_change_unavailable_reason: str | None


class ComparedValue(ComparisonResult):
    """One statistic's effect; the comparison tables name its two arms."""

    # The years per state both arms are compared on, the shorter arm's.
    years_per_state: int | None
    # A TVaR effect's control-variate coefficient, applied to both arms; null
    # when none suits both and each keeps its own.
    control_variate_coefficient: NotRequired[float | None]
    # The effect's paired batch-means 95% half-width, with both arms'
    # integration margins for a TVaR; absent for an EAL unless both arms'
    # EALs are the simulated years' means.
    half_width: NotRequired[float]
    # Present when the tail gate withholds the effect, with why.
    status: NotRequired[Literal["withheld"]]
    reason: NotRequired[str]


class UnavailableEffect(TypedDict):
    """A statistic with no effect because one arm has no value for it; it is
    withheld instead when either arm's statistic is withheld."""

    status: Literal["unavailable", "withheld"]
    reason: str
    years_per_state: int | None
    control_variate_coefficient: NotRequired[None]


class UnavailableComparison(TypedDict):
    status: Literal["unavailable"]
    reason: str


type ViewComparison = (
    dict[str, ComparedValue | UnavailableEffect] | UnavailableComparison
)
type ComparisonTable = dict[str, dict[str, ViewComparison]]


class RankedComparison(TypedDict):
    """One ranked arm; its full effect is in ``comparisons`` or
    ``reference_comparisons``."""

    rank: int
    arm: str
    absolute_difference: float


class AvailableRanking(TypedDict):
    status: Literal["available"]
    basis: Literal["absolute_point_effect_without_precision_ordering"]
    reference_arm: str
    arms: list[RankedComparison]
    # Each arm left out because the tail gate withheld its effect, with why.
    withheld_arms: dict[str, str]


type RankingStatus = Literal[
    "available", "unavailable", "withheld", "not_requested", "inapplicable"
]


class RankingDisposition(TypedDict):
    status: Literal["unavailable", "withheld", "not_requested", "inapplicable"]
    reason: str


# A view's ranking by statistic, or one disposition when every statistic of
# the view has the same one.
type ViewRanking = dict[str, AvailableRanking | RankingDisposition] | RankingDisposition


class ViewOutcome(TypedDict):
    status: Availability


class AssessedRequirement(ComparisonRequirementRecord):
    """A frozen requirement with its outcome and each requested view's
    comparison status; the comparison itself is in ``comparisons``."""

    scope_basis: Literal["authored_perturbation_scope"]
    requested_views: list[str]
    view_outcomes: dict[str, ViewOutcome]
    outcome_status: Literal[
        "available", "partial", "missing", "unassessed", "unavailable", "inapplicable"
    ]
    outcome_reason: NotRequired[str]


class ComparisonFamily(TypedDict):
    """One comparison family's requirements, status and rankings."""

    # Indices into ``comparison_requirements``.
    requirement_indices: list[int]
    scenario_ids: list[str]
    required_arms: list[str]
    status: Literal["complete", "partial", "inapplicable"]
    rankings: dict[str, ViewRanking]


# --- Coverage, scope and limitations ------------------------------------------


class ComparisonCoverage(TypedDict):
    status: Literal["complete", "partial", "not_requested", "inapplicable"]
    required_alternative_arms: list[str]
    missing_arms: list[str]
    missing_views: dict[str, list[str]]
    unavailable_comparison_views: dict[str, dict[str, str]]
    # Every authored comparison requirement resolved as available or inapplicable.
    requirements_complete: bool


class ViewCoverage(TypedDict):
    view: str
    status: Literal["available", "unavailable", "not_requested"]
    reason: str | None


# A catalog family's kind (``catmodel.setup.sensitivity_catalog``), or
# ``register_challenge`` for a frozen requirement outside the catalog.
type FamilyKind = Literal[
    "assumption", "numerical", "diagnostic", "exhibit", "register_challenge"
]


class FamilyCoverage(TypedDict):
    family_id: str
    kind: FamilyKind
    status: Literal[
        "complete", "partial", "unavailable", "inapplicable", "not_requested"
    ]
    reason: str | None


class ScenarioCoverage(TypedDict):
    scenario_id: str
    status: Literal["included", "structural_zero", "unavailable"]
    reason: str | None
    unavailable_views: dict[str, str]


class ReducedCoverage(TypedDict):
    """What the reduction includes and excludes, each exclusion with a reason."""

    comparisons: ComparisonCoverage
    views: list[ViewCoverage]
    sensitivity_families: list[FamilyCoverage]


class ResultCoverage(ReducedCoverage):
    """What the result includes and excludes, each exclusion with a reason."""

    # From the setup's populations, which a run or replay adds.
    scenarios: list[ScenarioCoverage]


class StudyEntity(TypedDict):
    entity_id: str
    kind: Literal["model", "lab", "operator", "deployment", "population"]
    display_name: str
    revision: str | None
    status: Literal["actual", "hypothetical"]


class StudyDeployment(TypedDict):
    deployment_id: str
    model_id: str


class StudyMonetaryBasis(TypedDict):
    currency: str
    monetary_basis: str


class StudyScope(TypedDict):
    """What a result describes, from its setup's scope: the setup, the period,
    the named entities and every population's currency and monetary basis."""

    setup_id: str
    scope_id: str
    title: str
    period_start: str
    period_end: str
    forecast_as_of: str
    data_cutoff: str
    availability_cutoff: str
    information_date: str
    scenario_ids: list[str]
    # Each distinct currency and monetary basis of the populations, in order.
    monetary_bases: list[StudyMonetaryBasis]
    entities: list[StudyEntity]
    deployments: list[StudyDeployment]


class ResultLimitation(TypedDict):
    source: Literal["setup", "run", "model"]
    text: str
    reference: NotRequired[str]


# --- Tail gate, small events and workload --------------------------------------


class TailGateControlsRecord(TypedDict):
    """An arm's resolved tail gate (``TailGateControls.record``)."""

    tolerance: float
    batches: int
    max_reference_state_years: int | None
    max_comparison_state_years: int | None
    views: list[str] | None
    levels: list[float] | None
    minimum_batch_exceedances: int


class TailGateRound(TypedDict):
    doubling: int
    # Each arm the round evaluated, with its unconverged driver views.
    unconverged_views: dict[str, list[str]]


class ArmTailGate(TypedDict):
    controls: TailGateControlsRecord
    driver_views: list[str]
    # The reference arm, for an arm that takes its years.
    follows_arm: str | None
    states: int
    years_per_state: int
    max_years_per_state: int
    # Null for an arm that follows the reference arm.
    drivers_converged: bool | None
    # ``<view>::<statistic>`` of each statistic the gate withheld.
    withheld_statistics: list[str]


class TailGateResults(TypedDict):
    """Each arm's tail gate controls, doubling rounds and withheld tail
    statistics (``catmodel.simulation.tail_gate.tail_gate_results``)."""

    method: Literal["batch_means_sectioning"]
    doublings: int
    rounds: list[TailGateRound]
    by_arm: dict[str, ArmTailGate]


class SmallEventControlsRecord(TypedDict):
    """The resolved small-event controls (``SmallEventControls.record``)."""

    target_large_events_per_year: float
    max_body_skewness: float
    body: Literal["normal", "individual"]
    bucket_rate_ratio: float
    top_rank_score: float
    min_summed_count: int


class SmallEventCounts(TypedDict):
    individually_drawn: int
    summed: int
    fallback_years: int


class SmallEventSplitRange(TypedDict):
    states: int
    min_usd: float
    max_usd: float


class SmallEventViewEnvelope(TypedDict):
    body_standard_deviation: float
    # By level; null when the arm's envelope is unavailable.
    tvar_bound: dict[str, float | None]


class SmallEventArmEnvelope(TypedDict):
    reason: str | None
    views: dict[str, SmallEventViewEnvelope]


class SmallEventAggregation(TypedDict):
    """Split points, controls and each view's TVaR envelope of the summed
    small events (``catmodel.simulation.small_events.small_event_results``)."""

    method: str
    # Null when no state records small-event controls.
    controls: SmallEventControlsRecord | None
    events: SmallEventCounts
    # Keyed ``<scenario>::<source status>``.
    split_points_by_origin: dict[str, SmallEventSplitRange]
    error_envelope: dict[str, SmallEventArmEnvelope]
    envelope_basis: str


class ArmWorkload(TypedDict):
    conditional_states: int
    years_per_state: int
    conditional_years: int
    simulated_conditional_years: int
    follows_arm: str | None
    max_years_per_state: int
    max_conditional_years: int
    # Each work limit at the most years, keyed by the count it bounds.
    max_work_per_state: dict[str, int]


class Workload(TypedDict):
    """Requested and simulated conditional years by arm
    (``catmodel.simulation.execution_budget.execution_budget``)."""

    arms: int
    by_arm: dict[str, ArmWorkload]
    simulated_conditional_years: int
    reused_conditional_years: int
    reference_conditional_years: int
    reference_work_multiple: float
    cost_basis: str


class ExecutionContext(TypedDict):
    workers: int


# --- Scenario contributions ---------------------------------------------------


class ScenarioContribution(TypedDict):
    scenario_id: str
    view: str
    status: Literal["available", "unavailable", "not_requested"]
    reason: str | None
    expected_annual_loss: float | None
    share_of_total: float | None


class UnallocatedLoss(TypedDict):
    """Pooled EAL not carried by any scenario, and whether years reconcile."""

    expected_annual_loss: float | None
    share_of_total: NotRequired[float | None]
    reason: str | None
    annual_losses_reconcile: bool | None


class UnavailableScenarioContributions(TypedDict):
    status: Literal["unavailable"]
    reason: str


class AvailableScenarioContributions(TypedDict):
    """Each scenario's EAL and share of the pooled EAL in the headline view,
    and the pooled EAL no scenario carries
    (``catmodel.results.scenario_contributions``)."""

    status: Literal["available"]
    measure: Literal["welfare", "gross", "net_resource_cost"]
    expected_annual_loss: float | None
    total_reason: str | None
    scenarios: list[ScenarioContribution]
    unallocated: UnallocatedLoss
    limitations: str


type ScenarioContributions = (
    AvailableScenarioContributions | UnavailableScenarioContributions
)


class ScenarioGroup(TypedDict):
    """The scenarios with one value on a grouping axis, and their summed EAL.
    A ``lower_bound`` leaves out the group's scenarios without an EAL."""

    value: str
    scenario_ids: list[str]
    status: Literal["available", "lower_bound", "unavailable"]
    reason: str | None
    expected_annual_loss: float | None
    share_of_total: float | None


class ScenarioGroupAxis(TypedDict):
    """One grouping axis. Each classified scenario is in one of its groups."""

    axis: str
    groups: list[ScenarioGroup]


class UnavailableScenarioGroups(TypedDict):
    """No groups: the scenario contributions are unavailable, or, when
    ``inapplicable``, the setup has no scenario of the library."""

    status: Literal["unavailable", "inapplicable"]
    reason: str


class AvailableScenarioGroups(TypedDict):
    """The scenario EALs of ``scenario_contributions`` grouped by reporting
    domain and by facet (``catmodel.results.scenario_groups``)."""

    status: Literal["available"]
    measure: Literal["welfare", "gross", "net_resource_cost"]
    expected_annual_loss: float | None
    axes: list[ScenarioGroupAxis]
    unclassified_scenarios: list[str]
    limitations: str


type ScenarioGroups = AvailableScenarioGroups | UnavailableScenarioGroups


# --- Sensitivity summary ---------------------------------------------------------


class Unavailable(TypedDict):
    status: Literal["unavailable"]
    reason: str


class WithheldExhibit(TypedDict):
    """An exhibit computed from the setup's own weights, not reported for a
    reweighted result."""

    status: Literal["unavailable"]
    reason: str


class ViewEal(TypedDict):
    status: Availability
    expected_annual_loss: float | None
    basis: ViewBasis | None
    reason: str | None


class ComparisonGap(TypedDict):
    """A frozen requirement that did not resolve as available or
    inapplicable."""

    family_id: str
    arm_id: str | None
    scenario_ids: list[str]
    status: RequirementStatus
    outcome_status: Literal[
        "available", "partial", "missing", "unassessed", "unavailable", "inapplicable"
    ]
    reason: str | None
    outcome_reason: str | None


class TornadoRow(TypedDict):
    arm: str
    control_arm: str
    control_eal: float
    alternative_eal: float
    absolute_difference: float
    relative_change: float | None
    mean_changing: bool
    half_width: float | None
    # The change's respondent-sampling standard error and z
    # (``catmodel.results.respondent_sampling.row_sampling``).
    sampling_standard_error: float | None
    sampling_z: float | None
    sampling_reason: str | None


class ResultsRanking(TypedDict):
    status: RankingStatus
    reason: str | None


class Tornado(TypedDict):
    """Executed EAL effects of assumption challenges in the headline view
    (``catmodel.results.sensitivity_summary.sensitivity_summary``)."""

    view: str | None
    basis: ViewBasis | None
    eal: ViewEal | None
    rows: list[TornadoRow]
    extrapolation_note: str | None
    unavailable_arms: dict[str, str]
    excluded_arms: dict[str, str]
    not_assessed_families: list[str]
    results_ranking: ResultsRanking
    interpretation: str


class SamplingLevel(TypedDict):
    status: Literal["available"]
    expected_annual_loss: float
    sampling_standard_error: float
    relative_sampling_standard_error: float | None


class SamplingComparison(TypedDict):
    status: Literal["available"]
    control_arm: str
    control_eal: float
    alternative_eal: float
    absolute_difference: float
    sampling_standard_error: float
    sampling_z: float | None


class UnavailableSamplingComparison(TypedDict):
    status: Literal["unavailable"]
    control_arm: str
    reason: str


class AvailableRespondentSampling(TypedDict):
    """Each sampling view's reference EAL with its respondent-sampling
    standard error, and each arm's change against its control with the
    change's standard error and z
    (``catmodel.results.respondent_sampling.respondent_sampling``)."""

    register_id: Literal["NUM-RESPONDENT-SAMPLING"]
    status: Literal["available"]
    groups: int
    samples: int
    smallest_group: int
    noise_z: float
    interpretation: str
    views: dict[str, SamplingLevel | Unavailable]
    # By arm, then view.
    comparisons: dict[
        str, dict[str, SamplingComparison | UnavailableSamplingComparison]
    ]


class UnavailableRespondentSampling(TypedDict):
    register_id: Literal["NUM-RESPONDENT-SAMPLING"]
    status: Literal["unavailable"]
    reason: str


type RespondentSampling = AvailableRespondentSampling | UnavailableRespondentSampling


class QuadratureNode(TypedDict):
    """One trajectory node's weights, kept samples and EAL contributions."""

    progress: float
    full_weight: float
    embedded_weight: float
    samples: int
    state_eal_mean: float
    # Unreported, with the reason, at a node of one sample and at a node where
    # a sample reads a donor's answers (POOL-UNANSWERED-SCENARIOS).
    state_eal_sd: float | None
    state_eal_sd_reason: str | None
    full_contribution: float
    embedded_contribution: float
    difference_contribution: float


class QuadratureArm(TypedDict):
    expected_annual_loss: float
    embedded_rule_expected_annual_loss: float
    absolute_difference: float
    relative_difference: float | None
    sampling_standard_error: float | None
    sampling_z: float | None
    sampling_reason: str | None
    by_node: dict[str, QuadratureNode]
    indicator: Literal["exact", "informative", "uninformative"]
    indicator_reason: str | None


class _QuadratureRule(TypedDict):
    rule: str
    interior_nodes: int
    embedded_interior_nodes: int


class AvailableTrajectoryQuadrature(_QuadratureRule):
    """Each integrated arm's EAL under the full and the embedded trajectory
    rule (``catmodel.results.sensitivity_results``)."""

    view: str
    interpretation: str
    status: Literal["available"]
    estimator: Literal["native_analytic_mean", "empirical_mean"]
    by_arm: dict[str, QuadratureArm]


class UnavailableTrajectoryQuadrature(_QuadratureRule):
    # Absent from the record of a reweighted result.
    view: NotRequired[str | None]
    interpretation: NotRequired[str]
    status: Literal["unavailable"]
    reason: str


type TrajectoryQuadrature = (
    AvailableTrajectoryQuadrature | UnavailableTrajectoryQuadrature
)


class FrequencyArmEstimate(TypedDict):
    status: Literal["available", "unavailable", "inapplicable"]
    expected_annual_loss: NotRequired[float]
    reason: NotRequired[str]


class FollowUpEffect(TypedDict):
    """One follow-up call's EAL effect in a view: the EAL of the arm that
    undoes it and of the arm it is compared with
    (``catmodel.setup.contextual_compilation.follow_up_effects``)."""

    status: Literal["available", "unavailable", "inapplicable"]
    before_arm: str
    after_arm: str
    expected_annual_loss_before: NotRequired[float]
    expected_annual_loss_after: NotRequired[float]
    # After over before, less one; null where the EAL before is zero.
    relative_change: NotRequired[float | None]
    reason: NotRequired[str]


class FollowUpCounts(TypedDict):
    """The kept cases a follow-up call was asked of and those whose answer it
    replaced."""

    cases_asked: int
    answers_replaced: int


class FrequencyViewEstimate(TypedDict):
    by_arm: dict[str, FrequencyArmEstimate]
    # By scenario, then arm; null where an arm has no rescaled mean.
    by_scenario: dict[str, dict[str, float | None]]
    # By follow-up call: severity_retry, chain_guardrail, eal_crosscheck.
    follow_up_effects: dict[str, FollowUpEffect]
    run_expected_annual_loss: float | None
    # Each arm but ``final``, relative to ``final``.
    relative_effect: dict[str, float | None]


class FrequencyEstimate(TypedDict):
    """Reference EAL under each count arm and the dual collection's
    validation metrics, from the setup's evidence
    (``catmodel.setup.contextual_compilation.frequency_estimate_record``),
    with each view's EAL in this run.

    ``frequency_collection``, ``eal_crosscheck`` and ``hierarchy`` are the
    collection's validation summaries, which setup import computes and the
    setup's evidence carries; their layout is the collection's
    (``catmodel.calibration``), not the results'.
    """

    method: Literal["analytic_frequency_rescaling"]
    interpretation: str
    tail_statistics: str
    chain_pool_weight: float
    views: dict[str, FrequencyViewEstimate]
    # By follow-up call, like each view's ``follow_up_effects``.
    follow_ups: dict[str, FollowUpCounts]
    frequency_collection: dict[str, object]
    eal_crosscheck: dict[str, object]
    hierarchy: dict[str, object]


class LargeLossAlternative(TypedDict):
    status: Availability
    expected_annual_loss: NotRequired[float]
    relative_effect: NotRequired[float]


class LargeLossViewEal(TypedDict):
    """A view's EAL under the run's laws and each analytic anchor
    alternative: the anchor ignored, its conditional odds scaled, and the
    two-anchor tail through the stated very-large-loss share."""

    expected_annual_loss: float | None
    anchor_ignored: LargeLossAlternative
    anchor_lower: LargeLossAlternative
    anchor_higher: LargeLossAlternative
    two_anchor: LargeLossAlternative


class LargeLossTailAlternative(TypedDict):
    """The modelled counts at both levels and the gross EAL share from
    incidents at or above the large-loss level under an alternative tail."""

    very_large_count: float | None
    large_count: float | None
    gross_eal_share_at_or_above_large: float | None


class LargeLossShapeAlternatives(TypedDict):
    shape_lower: LargeLossTailAlternative
    shape_higher: LargeLossTailAlternative


class _LargeLossMeasures(TypedDict):
    status: Literal["available"]
    stated_large_count: float | None
    pooled_large_count: float | None
    extrapolated_large_count: float | None
    modelled_large_count: float | None
    stated_very_large_count: float | None
    pooled_very_large_count: float | None
    modelled_very_large_count: float | None
    probability_at_least_one_large: float | None
    probability_withheld_reasons: list[str]
    gross_eal_share_at_or_above_large: float | None
    gross_eal_share_withheld_reasons: list[str]
    severity_unresolved_gross_eal_share: float | None
    tail_shape_alternatives: LargeLossShapeAlternatives
    # Under the two-anchor tail, which reproduces the pooled very-large count
    # wherever the stated share shapes the law.
    two_anchor_tail: LargeLossTailAlternative
    eal_alternatives: dict[str, LargeLossViewEal]


class _LargeLossUnavailable(TypedDict):
    status: Literal["unavailable"]
    reasons: list[str]


class _LargeLossCases(TypedDict):
    # Each case counted once: by status, by how its anchor enters its law and
    # by how its very-large-loss share enters its two-anchor law.
    case_statuses: dict[str, int]
    law_roles: dict[str, int]
    very_large_law_roles: dict[str, int]


class AvailableLargeLossScenario(_LargeLossMeasures, _LargeLossCases):
    pass


class UnavailableLargeLossScenario(_LargeLossUnavailable, _LargeLossCases):
    pass


class AvailableLargeLossTotal(_LargeLossMeasures, _LargeLossCases):
    severity_unresolved_cases: int


class UnavailableLargeLossTotal(_LargeLossUnavailable):
    pass


class InapplicableLargeLossTotal(TypedDict):
    status: Literal["inapplicable"]


class LargeLossLevels(TypedDict):
    large_loss: float
    very_large_loss: float | None


class LargeLossAnchor(TypedDict):
    """Expected annual numbers of incidents at or above the large-loss levels
    and their EAL effects, from the setup's evidence
    (``catmodel.setup.contextual_compilation.large_loss_record``)."""

    method: Literal["analytic_tail_measures"]
    levels: LargeLossLevels | None
    interpretation: str
    by_scenario: dict[str, AvailableLargeLossScenario | UnavailableLargeLossScenario]
    total: (
        AvailableLargeLossTotal | UnavailableLargeLossTotal | InapplicableLargeLossTotal
    )


class ModelSamples(TypedDict):
    samples: int
    excluded: int
    excluded_share: float


class UnansweredLeaf(TypedDict):
    """A leaf an excluded sample lacks: its direct question exhausted its
    validation attempts in this context."""

    context_id: str
    node_id: str


class ExcludedSample(TypedDict):
    node: str
    # The respondent's ID, model and sample index.
    respondent: tuple[str, str, int]
    unanswered: list[UnansweredLeaf]


class RenormalizedWeight(TypedDict):
    samples: int
    kept: int
    model_weight: float
    weight_per_sample: float
    weight_per_kept_sample: float


class ExhaustedSubjectCalls(TypedDict):
    calls: int
    attempts: int
    by_cause: dict[str, int]


class ExhaustedCalls(TypedDict):
    parse_attempts: int
    calls: int
    by_kind: dict[ExhaustedCallKind, int]
    by_cause: dict[str, int]
    # By leaf or parent, then call kind.
    by_subject: dict[str, dict[ExhaustedCallKind, ExhaustedSubjectCalls]]


type LabLossPart = Literal["lab_compensation", "lab_annual_business_loss"]
"""A part of the lab's own loss that a direct answer gives or declares
unavailable: its compensation share or its annual business loss."""


class LabLossUnavailableCases(TypedDict):
    # A case that declares both parts counts once here and once under each.
    cases: int
    by_part: dict[LabLossPart, int]


class LabLossCaseRead(TypedDict):
    """A case whose direct answer declares the lab's loss unavailable, as a
    left-out sample's states read it."""

    context_id: str
    node_id: str
    # The respondent that gave the answer: the sample itself, or the donor
    # whose case of the leaf it borrows. Its ID, model and sample index.
    respondent: tuple[str, str, int]


class LabLossLeftOutSample(TypedDict):
    node: str
    # The respondent's ID, model and sample index.
    respondent: tuple[str, str, int]
    reads: list[LabLossCaseRead]


class LabLossUnavailable(LabLossUnavailableCases):
    """The cases whose direct answer declares a part of the lab's loss
    unavailable, over every capture, and what follows for the views that
    count the lab's own loss, the lab-loss views and the gross views
    (``POOL-EXCLUDED-SAMPLES``). No such case excludes a sample or changes a
    weight in any other view."""

    by_subject: dict[str, LabLossUnavailableCases]
    # none: no such case. samples_left_out: each kept sample whose states
    # read such a case is left out of those views, where it weighs zero and
    # its sampling group's other samples share its weight. views_withheld:
    # leaving them out would breach an import guard, so none is left out and
    # those views are unavailable.
    disposition: Literal["none", "samples_left_out", "views_withheld"]
    # Under samples_left_out only: the left-out samples, their number, and
    # their number by respondent model.
    left_out_samples: NotRequired[int]
    left_out_by_model: NotRequired[dict[str, int]]
    left_out: NotRequired[list[LabLossLeftOutSample]]
    # The guard, under views_withheld only.
    withheld_because: NotRequired[str]


class UnansweredPair(TypedDict):
    """A leaf that a model does not answer: what makes it one, the model's
    samples over every node of the study, those lacking the leaf and the
    exhausted direct calls behind them."""

    node_id: str
    scenario_id: str
    model_id: str
    # The study's captures (own); for a variant study, those and its headline
    # study (own_and_headline) or the headline study alone (headline).
    basis: Literal["own", "own_and_headline", "headline"]
    samples: int
    lacking_samples: int
    lacking_share: float
    exhausted_direct_calls: ExhaustedSubjectCalls


class AnsweringModelShare(TypedDict):
    # The model's share of the leaf's pooled weight over the study at the
    # nodes' reference weights, beside one over the number of answering
    # models: its share if they kept equally many samples at every node and
    # every answering sample lent equally often.
    realized_share: float
    equal_share: float


class UnansweredLeafPool(TypedDict):
    scenario_id: str
    answering_models: dict[str, AnsweringModelShare]


class UnansweredMatchedControl(TypedDict):
    """How a variant study's unanswered pairs and donors compare with its
    headline's, whose captures its matched control reweights."""

    # The variant's pairs that the headline answers.
    pairs_not_in_headline: int
    # Over the pairs of both studies, each borrowing sample at each node, and
    # those whose donor is not the sample's donor in the headline there.
    donors: int
    donors_not_headline: int
    differs: bool


class UnansweredScenarios(TypedDict):
    """Each leaf that a model does not answer, the models that answer it with
    their shares of its pool, the leaves one model answers and, for a variant
    study, the comparison with its matched control
    (``POOL-UNANSWERED-SCENARIOS``)."""

    register_id: Literal["POOL-UNANSWERED-SCENARIOS"]
    pairs: list[UnansweredPair]
    by_leaf: dict[str, UnansweredLeafPool]
    leaves_on_one_model: int
    matched_control: NotRequired[UnansweredMatchedControl]
    interpretation: str


class ExcludedSamplesRecord(TypedDict):
    """The excluded samples, their shares and the weights they renormalized,
    every exhausted call, every lab loss declared unavailable and, for a
    study with one, the leaves a model does not answer
    (``catmodel.setup.excluded_samples.excluded_samples_record``)."""

    register_id: Literal["POOL-EXCLUDED-SAMPLES"]
    max_excluded_sample_share: float
    samples: int
    excluded_samples: int
    excluded_share: float
    excluded_share_weighted: float
    by_model: dict[str, ModelSamples]
    excluded: list[ExcludedSample]
    # By node, then model.
    renormalized_weights: dict[str, dict[str, RenormalizedWeight]]
    exhausted_calls: ExhaustedCalls
    lab_loss_unavailable: LabLossUnavailable
    interpretation: str
    unanswered_scenarios: NotRequired[UnansweredScenarios]


class ExcludedSamples(ExcludedSamplesRecord):
    """The headline's excluded-sample disclosure, with each context variant
    study's by variant."""

    exposure_scale_variants: NotRequired[dict[str, ExcludedSamplesRecord]]
    safeguard_level_variants: NotRequired[dict[str, ExcludedSamplesRecord]]


class WorldOutputRecord(TypedDict):
    measure: str
    year: int
    currency: str
    value: float
    basis: str
    status: Literal["OK", "INITIAL_ESTIMATE", "PLACEHOLDER"]
    citation: str


class PlausibilityFlagsRecord(TypedDict):
    leaf_share: float
    total_share: float
    status: Literal["OK", "INITIAL_ESTIMATE", "PLACEHOLDER"]


class PlausibilityRow(TypedDict):
    status: Literal["available"]
    expected_annual_loss: float
    eal_estimator: str
    share_of_world_output: float
    flagged: bool


class _PlausibilityReference(TypedDict):
    reference: WorldOutputRecord
    flags: PlausibilityFlagsRecord
    view: Literal["welfare"]
    interpretation: str


class AvailablePlausibility(_PlausibilityReference):
    """Each scenario's welfare EAL, and the total, as a share of world output
    (``catmodel.results.plausibility``)."""

    status: Literal["available"]
    study_year: int
    reference_year_note: str | None
    total: PlausibilityRow | Unavailable
    by_scenario: dict[str, PlausibilityRow | Unavailable]
    flagged_scenarios: list[str]


class UnavailablePlausibility(_PlausibilityReference):
    status: Literal["unavailable"]
    reason: str


class BracketHalfWidth(TypedDict):
    var: float
    tvar: NotRequired[float]


class BracketTail(TypedDict):
    var: float
    tvar: float | None
    status: Literal["available", "withheld"]
    reason: str | None
    # Absent when the states have fewer years than the tail gate's batches.
    half_width: NotRequired[BracketHalfWidth]
    minimum_batch_exceedances: NotRequired[int]


class AvailableBracketReference(TypedDict):
    status: Literal["available"]
    expected_annual_loss: float | None
    normalized_empirical_mean: float
    tail: dict[str, BracketTail]
    mean_changing: bool
    coupling: str


class UnavailableBracketReference(TypedDict):
    status: Literal["unavailable"]
    reason: str
    coupling: str


type BracketReference = AvailableBracketReference | UnavailableBracketReference


class BracketReferences(TypedDict):
    executed: BracketReference
    comonotone: BracketReference


class BracketPrecision(TypedDict):
    effective_sample_size: float
    effective_tail_observations: dict[str, float]
    interpretation: str


class AvailableDependenceBracket(TypedDict):
    """The executed and within-state comonotone scenario loss sums
    (``catmodel.results.dependence_bracket.dependence_bracket``)."""

    status: Literal["available"]
    view: str
    conditioning: str
    scenario_views: dict[str, str]
    references: BracketReferences
    eal_estimator: Literal["reference_native_mean", "reference_weighted_empirical_mean"]
    tvar_estimators: dict[str, TvarEstimator] | None
    mean_unavailable_reason: str | None
    precision: BracketPrecision
    limitations: str


type DependenceBracket = AvailableDependenceBracket | Unavailable


class BandComonotoneMethod(TypedDict):
    rank_grid_points: int
    normal_score_range: list[float]


class BandIndependentMethod(TypedDict):
    seed: int
    samples_per_epistemic_group: int
    batches: int
    tolerance: float


class BandMethod(TypedDict):
    probabilities: list[float]
    mixture: str
    comonotone: BandComonotoneMethod
    independent: BandIndependentMethod


class BandQuantiles(TypedDict):
    # By band probability.
    quantiles: dict[str, float]
    error_bounds: dict[str, float]


class IndependentBand(TypedDict):
    quantiles: dict[str, float]
    convergence: dict[str, GateStatus]
    sample_mean: float


class RankGroupBand(BandQuantiles):
    scenario_ids: list[str]
    expected_annual_loss: float


class AvailableViewBand(TypedDict):
    status: Literal["available"]
    expected_annual_loss: float
    fixed_rate_expected_annual_loss: float
    independent: IndependentBand
    comonotone: BandQuantiles
    by_rank_group: dict[str, RankGroupBand]
    # The view's own, where it counts the lab's loss (a lab-loss view or
    # gross), a state is left out of such views (POOL-EXCLUDED-SAMPLES) and
    # its band is computed at its own weights: the states it weighs, and the
    # epistemic groups with fewer of them than the others. The record's
    # fields describe every other view.
    states: NotRequired[int]
    incomplete_epistemic_groups: NotRequired[list[str]]


type ViewBand = AvailableViewBand | Unavailable


class _BandRecord(TypedDict):
    interpretation: str
    method: BandMethod
    # Epistemic groups with fewer positive-weight states than the others.
    incomplete_epistemic_groups: NotRequired[list[str]]
    incomplete_epistemic_groups_note: NotRequired[str]
    # Counts of what the band integrates, when a view can have one.
    epistemic_groups: NotRequired[int]
    states: NotRequired[int]
    rank_groups: NotRequired[int]
    ranked_laws: NotRequired[int]
    views: dict[str, ViewBand]


class AvailableParameterBand(_BandRecord):
    """The EAL parameter band of the welfare, gross and lab-loss views
    (``catmodel.results.eal_parameter_band.reported_parameter_band``)."""

    status: Literal["available"]


class UnavailableParameterBand(_BandRecord):
    status: Literal["unavailable"]
    reason: str


class WithheldParameterBand(TypedDict):
    status: Literal["unavailable"]
    reason: str
    interpretation: str


type EalParameterBand = (
    AvailableParameterBand | UnavailableParameterBand | WithheldParameterBand
)


class SensitivitySummary(TypedDict):
    """EAL effects of the executed assumption challenges, what is missing,
    and the exhibits that rest on the reference capture or the setup's
    evidence (``catmodel.results.sensitivity_results``)."""

    analysis_sha256: str
    reference_arm: str
    comparison_status: Literal["complete", "partial", "not_requested", "inapplicable"]
    # By scenario, then measure (gross, welfare or net_resource_cost).
    scenario_eal: dict[str, dict[str, ViewEal]]
    aggregate_eal: dict[str, ViewEal]
    comparison_gaps: list[ComparisonGap]
    tornado: Tornado
    respondent_sampling: RespondentSampling
    # From a public-trajectory setup's evidence.
    trajectory_quadrature: NotRequired[TrajectoryQuadrature]
    frequency_estimate: NotRequired[FrequencyEstimate | WithheldExhibit]
    # The frequency estimate's ``hierarchy`` summary, or its withheld record.
    hierarchy_pool: NotRequired[dict[str, object]]
    large_loss_anchor: NotRequired[LargeLossAnchor | WithheldExhibit]
    excluded_samples: NotRequired[ExcludedSamples]
    plausibility: AvailablePlausibility | UnavailablePlausibility
    dependence_bracket: DependenceBracket
    eal_parameter_band: EalParameterBand


# --- Capability scenarios ----------------------------------------------------


class RespondentSpread(TypedDict):
    """The respondent spread k and common level bias b the setup's count laws
    use: both one, the stated laws at face value."""

    k: float
    basis: Literal["face_value"]
    level_bias: float


class CapabilityPercentiles(TypedDict):
    assumption_id: Literal["NUM-CAPABILITY-SCENARIOS"]
    values: list[float]
    law: str


class _CurvePointGate(TypedDict):
    status: Literal["available", "withheld", "unavailable"]
    reason: str | None
    half_width: float | None
    relative_half_width: float | None
    minimum_batch_exceedances: int | None


class QuantilePoint(_CurvePointGate):
    non_exceedance_probability: float
    return_period_years: float
    annual_loss: float | None


class ExceedancePoint(_CurvePointGate):
    loss: float
    probability: float | None


class ScenarioView(TypedDict):
    view: str
    measure: str
    source_status: Literal["total", "public", "not_yet_public"]
    status: Availability
    reason: str | None
    expected_annual_loss: float | None
    sampling_standard_error: float | None
    sampling_reason: str | None
    years_per_state: int | None
    quantiles: list[QuantilePoint]
    exceedance: list[ExceedancePoint]


class NodeReference(TypedDict):
    node_id: str
    progress: float
    progress_cdf: float


class ScenarioInterpolation(TypedDict):
    lower_node: NodeReference | None
    upper_node: NodeReference | None
    weight: float | None
    exact_node: bool


class LabEci(TypedDict):
    period_start: float
    period_end: float


class CapabilityScenario(TypedDict):
    """One representative capability path's annual loss laws and EALs."""

    scenario: str
    label: str
    description: str
    status: Availability
    reason: str | None
    forecast_probability: float | None
    forecast_probability_note: str
    percentile: float | None
    progress: float | None
    progress_cdf: float | None
    eci_period_start: float | None
    eci_period_end: float | None
    eci_by_lab: dict[str, LabEci]
    arm: str | None
    interpolation: ScenarioInterpolation
    by_view: dict[str, ScenarioView]


class NodeView(TypedDict):
    measure: str
    source_status: Literal["total", "public", "not_yet_public"]
    expected_annual_loss: float | None
    status: Availability
    reason: str | None
    sampling_standard_error: float | None
    sampling_reason: str | None


class CapabilityNode(TypedDict):
    """One trajectory node's arm, weight, kept samples and EALs."""

    node_id: str
    progress: float
    progress_cdf: float
    eci_period_start: float | None
    eci_period_end: float | None
    base_weight: float | None
    arm: str
    samples: int
    by_view: dict[str, NodeView]


class ScenarioReconciliation(TypedDict):
    """Whether the current-frontier scenario reproduces the no-progress
    arm's reported EAL."""

    scenario: Literal["current_frontier"]
    arm: str
    status: Availability
    reason: str | None
    current_frontier_expected_annual_loss: float | None
    no_progress_arm_expected_annual_loss: float | None


type SafeguardLevel = Literal["weaker", "current", "stronger"]


class SafeguardGridLevel(TypedDict):
    safeguard_level: SafeguardLevel
    safeguard_arm: str
    status: Availability
    reason: str | None
    nodes: list[CapabilityNode]


class SafeguardGridCell(CapabilityScenario):
    safeguard_level: SafeguardLevel
    safeguard_arm: str
    # Whether the percentile's progress lies outside the level's nodes, and
    # the nearest node it is clamped to with the log-progress gap.
    capability_clamped: bool
    clamped_node: NodeReference | None
    clamp_log_progress_gap: float | None


class SafeguardGrid(TypedDict):
    """The capability scenarios over each safeguard level's own states."""

    status: Availability
    reason: str | None
    interpretation: str
    disclosures: list[str]
    levels: list[SafeguardGridLevel]
    cells: list[SafeguardGridCell]


class CapabilityScenarios(TypedDict):
    """Deterministic percentile-path loss scenarios reduced from retained
    trajectory-conditioned annual draws
    (``catmodel.results.capability_scenarios``)."""

    status: Availability
    reason: str | None
    interpretation: str
    disclosures: list[str]
    respondent_spread: RespondentSpread
    percentiles: CapabilityPercentiles
    levels: list[float]
    thresholds: list[float]
    scenarios: list[CapabilityScenario]
    nodes: list[CapabilityNode]
    # By view.
    reconciliation: dict[str, ScenarioReconciliation]
    # Present when the setup has safeguard-level variants.
    safeguard_grid: NotRequired[SafeguardGrid]


# --- Reports -----------------------------------------------------------------


class _ReportCore(TypedDict):
    analysis_sha256: str
    analysis_spec: AnalysisSpec
    status: Literal["complete", "partial"]
    arms: dict[str, ArmResult]
    # Each arm compared with its matched control (comparison_reference_arms),
    # or with the reference arm when it has none.
    comparisons: ComparisonTable
    comparison_reference_arms: dict[str, str]
    # Each arm whose matched control is not the reference arm, compared with
    # the reference arm; every other arm's is its entry in ``comparisons``.
    reference_comparisons: ComparisonTable
    rankings_available: bool
    rankings: dict[str, ViewRanking]
    # Each frozen requirement with its outcome and each requested view's
    # comparison status; the comparison itself is in ``comparisons``.
    comparison_requirements: list[AssessedRequirement]
    comparison_families: dict[str, ComparisonFamily]


class ReducedResults(_ReportCore):
    """What the reduction returns (``catmodel.results.reduction.reduce_captures``):
    a run or replay adds the study, the scenario coverage, the limitations and
    the exhibits to make it a ``NativeResults`` report."""

    coverage: ReducedCoverage
    diagnostics: ResultsDiagnostics


class HeadlineResults(_ReportCore):
    """The headline of a run's or replay's report, which ``results.json``
    holds: what the result describes, its identity, coverage and limitations,
    each arm's statistics, the comparisons and each scenario's contribution.
    Its layout is the one to build on, so it stays small: a report-only
    analysis is an exhibit, not a headline section."""

    schema_version: int
    study: StudyScope
    coverage: ResultCoverage
    limitations: list[ResultLimitation]
    scenario_contributions: ScenarioContributions
    # The code that produced the capture.
    producer_provenance: CapturedCodeProvenance
    # How this report was computed (worker processes); results never depend on
    # it, so it is audit context outside every identity. A run and a replay
    # record it; an export's reduction does not.
    execution_context: NotRequired[ExecutionContext]


class ResultsExhibits(TypedDict):
    """The exhibits of a run's or replay's report, which the diagnostics file
    holds: report-only analyses of the same capture beside the headline."""

    sensitivity_summary: SensitivitySummary
    workload: Workload
    small_event_aggregation: SmallEventAggregation
    capability_scenarios: CapabilityScenarios
    tail_gate: TailGateResults
    scenario_groups: ScenarioGroups


class NativeResults(HeadlineResults, ResultsExhibits):
    """A run's or replay's whole report (``catmodel.results.execution``):
    the headline, the exhibits and the view diagnostics."""

    # Each reduced arm's view diagnostics, which the diagnostics file holds
    # with the exhibits.
    diagnostics: NotRequired[ResultsDiagnostics]


# The top-level sections of a report by the file that holds them, each list
# read from its contract: results.json holds the headline sections, and the
# diagnostics file the exhibits and the ``diagnostics`` section.
HEADLINE_SECTIONS: tuple[str, ...] = tuple(
    typing.get_type_hints(HeadlineResults, include_extras=True)
)
EXHIBIT_SECTIONS: tuple[str, ...] = tuple(
    typing.get_type_hints(ResultsExhibits, include_extras=True)
)
VIEW_DIAGNOSTICS_SECTION = "diagnostics"


def reason_code(text: str) -> str:
    """A reason's code: the first ``REASON_CODE_LENGTH`` hexadecimal digits of
    the SHA-256 of its UTF-8 text, so the same reason has the same code in
    every result."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:REASON_CODE_LENGTH]


def is_reason(key: object) -> bool:
    """Whether a string value under ``key`` is a reason: ``key`` is
    ``reason`` or ends in ``_reason``."""
    return isinstance(key, str) and (key == "reason" or key.endswith("_reason"))


def holds_reasons(key: object) -> bool:
    """Whether every string in the value under ``key`` is a reason: ``key``
    ends in ``_reasons`` or contains ``_reasons_``."""
    return isinstance(key, str) and (key.endswith("_reasons") or "_reasons_" in key)


def _map_reasons(value: Any, convert: Callable[[str], str], inside: bool) -> Any:
    """``value`` with ``convert`` applied to every reason in it: a string
    under a key named ``reason`` or ending in ``_reason``, or any string inside
    the value of a key ending in ``_reasons`` or containing ``_reasons_``."""
    if isinstance(value, Mapping):
        return {
            key: convert(item)
            if isinstance(item, str) and (inside or is_reason(key))
            else _map_reasons(item, convert, inside or holds_reasons(key))
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [
            convert(item)
            if inside and isinstance(item, str)
            else _map_reasons(item, convert, inside)
            for item in value
        ]
    return value


def intern_reasons(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``payload`` with every reason replaced by its code
    (``reason_code``) and the ``reasons`` table from code to text."""
    if "reasons" in payload:
        raise ValueError("a payload to intern already has a reasons table")
    table: dict[str, str] = {}

    def convert(text: str) -> str:
        code = reason_code(text)
        if table.setdefault(code, text) != text:
            raise ValueError(f"two reasons share the code {code}")
        return code

    encoded = _map_reasons(payload, convert, False)
    encoded["reasons"] = dict(sorted(table.items()))
    return encoded


def resolve_reasons(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of an interned ``payload`` with every reason code replaced by its
    text and no ``reasons`` table.

    Refuses a table that is not a mapping of strings, an entry whose key is
    not its text's code, a code without an entry and an entry no reason uses,
    so an edited table cannot change what a reason says.
    """
    table = payload.get("reasons")
    if not isinstance(table, Mapping) or any(
        not isinstance(code, str) or not isinstance(text, str)
        for code, text in table.items()
    ):
        raise ValueError("the reasons table must map reason codes to texts")
    if any(reason_code(text) != code for code, text in table.items()):
        raise ValueError("a reasons entry's code is not the code of its text")
    used: set[str] = set()

    def convert(code: str) -> str:
        if code not in table:
            raise ValueError(f"reason code {code!r} has no entry in reasons")
        used.add(code)
        return table[code]

    decoded = _map_reasons(
        {key: value for key, value in payload.items() if key != "reasons"},
        convert,
        False,
    )
    if set(table) - used:
        raise ValueError("the reasons table has entries that no reason uses")
    return decoded
