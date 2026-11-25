"""Scientific incident specifications, independent of evidence origin."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from types import NoneType, UnionType
from typing import (
    Annotated,
    Literal,
    Self,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
)

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PlainValidator,
    WrapValidator,
    model_validator,
)

from catmodel.contracts.components import (
    AggregateLabLossLaw,
    AnnualLabLossLaw,
    ComponentAmountLaw,
    ComponentObligation,
    ComponentPayment,
    NativePaymentKernel,
    PolicyMembership,
)
from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.primitives import (
    Frozen,
    FrozenDict,
    RequiredIdentifier,
    RequiredString,
    Sha256Digest,
)
from catmodel.contracts.shared_fields import SharedFieldsSet
from catmodel.hashing import canonical_json_sha256


def _number(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("quantities must be finite JSON numbers")
    return value


type Nonnegative = Annotated[
    float, Field(ge=0, allow_inf_nan=False), BeforeValidator(_number)
]
type Positive = Annotated[
    float, Field(gt=0, allow_inf_nan=False), BeforeValidator(_number)
]
type Probability = Annotated[Nonnegative, Field(le=1)]
type Coordinate = Annotated[float, Field(allow_inf_nan=False), BeforeValidator(_number)]


class Record(SharedFieldsSet):
    """A frozen record whose every number is finite and whose every value JSON
    can encode.

    Validation checks that by encoding the record (``finite_json_record``),
    before any validator a subclass adds. A record validated from its fields
    encodes only what its own fields hold besides records: every record in
    them was validated on the way, so checked already, and a law several
    records deep is encoded once, not once for each record around a value. A
    record given as an instance is kept as it is and nothing in it is
    validated, so all of it is encoded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def digest(self) -> str:
        return canonical_json_sha256(self.model_dump(mode="json"))

    @model_validator(mode="wrap")
    @classmethod
    def finite_json_record(
        cls, value: object, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        record = handler(value)
        # Encoding refuses a number that is not finite; the check needs
        # neither the canonical key order nor a digest.
        json.dumps(
            record.model_dump(
                mode="json",
                exclude=None if record is value else _record_fields(type(record)),
            ),
            allow_nan=False,
        )
        return record


_FIELD_VALIDATORS = (AfterValidator, BeforeValidator, PlainValidator, WrapValidator)


def _only_records(annotation: object) -> bool:
    """Whether a field so annotated holds nothing but validated records: a
    record, None, or tuples of them. A validator on the field could put
    another value in their place, so such a field does not count."""
    if isinstance(annotation, TypeAliasType):
        return _only_records(annotation.__value__)
    origin = get_origin(annotation)
    if origin is Annotated:
        inner, *metadata = get_args(annotation)
        return not any(
            isinstance(row, _FIELD_VALIDATORS) for row in metadata
        ) and _only_records(inner)
    if origin in (Union, UnionType):
        return all(
            member is NoneType or _only_records(member)
            for member in get_args(annotation)
        )
    if origin is tuple:
        members = [member for member in get_args(annotation) if member is not ...]
        return bool(members) and all(_only_records(member) for member in members)
    return isinstance(annotation, type) and issubclass(annotation, Record)


_RECORD_FIELDS: dict[type[Record], set[str] | None] = {}


def _record_fields(model: type[Record]) -> set[str] | None:
    """The fields of ``model`` that hold only records, which a record
    validated from its fields need not encode again; None when it has none."""
    try:
        return _RECORD_FIELDS[model]
    except KeyError:
        names = {
            name
            for name, field in model.model_fields.items()
            if not any(isinstance(row, _FIELD_VALIDATORS) for row in field.metadata)
            and _only_records(field.annotation)
        }
        _RECORD_FIELDS[model] = names or None
        return _RECORD_FIELDS[model]


def require_unique(values, label: str) -> None:
    """Refuse repeated ``values``, naming them by ``label``."""
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _component_relations(components, obligations, payments, memberships) -> None:
    identifiers = {row.component_id for row in components}
    if any(
        row.component_id not in identifiers
        for row in (*obligations, *payments, *memberships)
    ):
        raise ValueError("component relations refer to an absent physical component")
    for relations, names in (
        (obligations, ("component_id", "entity_id")),
        (payments, ("component_id", "entity_id")),
        (memberships, ("component_id", "contract_id")),
    ):
        require_unique(
            [tuple(getattr(row, name) for name in names) for row in relations],
            "component relation keys",
        )
    obligation_fractions = {
        (row.component_id, row.entity_id): row.fraction for row in obligations
    }
    for payment in payments:
        if payment.fraction > obligation_fractions.get(
            (payment.component_id, payment.entity_id), 0
        ):
            raise ValueError("component payment exceeds its explicit legal obligation")
    for identifier in identifiers:
        if (
            math.fsum(
                row.fraction for row in payments if row.component_id == identifier
            )
            > 1 + 1e-12
        ):
            raise ValueError(
                "component payments cannot recover unique harm more than once"
            )


def _payment_kernel_relations(components, obligations, payments, kernel) -> None:
    if kernel is None:
        return
    kernel_ids = set(kernel.liability_component_ids) | set(kernel.direct_component_ids)
    if not kernel_ids <= {row.component_id for row in components}:
        raise ValueError("native payment kernel refers to an absent physical component")
    if kernel_ids & {row.component_id for row in (*obligations, *payments)}:
        raise ValueError(
            "native and authored payment mappings cannot own the same component"
        )


class Entity(Record):
    entity_id: RequiredIdentifier
    kind: Literal["model", "lab", "operator", "deployment", "population"]
    related_entity_ids: tuple[RequiredIdentifier, ...] = ()
    revision: str | None = None
    definition: RequiredString
    display_name: RequiredString
    status: Literal["actual", "hypothetical"]

    def scientific_projection(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"display_name", "status"})


class CapabilityScale(Record):
    scale_id: RequiredIdentifier
    quantity_definition: RequiredString
    unit: RequiredString
    reference_definition: RequiredString
    measurement_protocol: RequiredString


class DeploymentBinding(Record):
    deployment_id: RequiredIdentifier
    model_id: RequiredIdentifier
    use_population: RequiredString
    tools_and_permissions: RequiredString
    human_oversight: RequiredString


class Population(Record):
    scenario_id: RequiredIdentifier
    definition: RequiredString
    causal_qualification: RequiredString
    initiating_unit: RequiredString
    terminal_unit: RequiredString
    severity_unit: RequiredString
    monetary_basis: RequiredString
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    threshold: Nonnegative | None = None
    threshold_comparator: Literal["gt", "ge"] | None = None
    threshold_scope: Literal["surviving_terminal_total"] = "surviving_terminal_total"
    accounting_standing: RequiredString
    exclusions: tuple[RequiredString, ...] = ()
    deployment_ids: tuple[RequiredIdentifier, ...] = ()

    @model_validator(mode="after")
    def threshold_pair(self) -> Self:
        if (self.threshold is None) != (self.threshold_comparator is None):
            raise ValueError("threshold and comparison must be declared together")
        return self


class ReportPopulationBinding(Record):
    source_scenario_id: RequiredIdentifier
    target_scenario_id: RequiredIdentifier
    scope: Literal["same_terminal_members"] = "same_terminal_members"
    rationale: RequiredString


class Evidence(Record):
    evidence_id: RequiredIdentifier
    origin: Literal["observation", "fit", "elicitation", "stipulation"]
    original_sha256: Sha256Digest
    selected_sha256: Sha256Digest
    target: RequiredString
    conditioning: RequiredString
    units: RequiredString
    transport: RequiredString
    acquisition: Frozen[dict[str, object]] | None = None
    raw_attempt_sha256: tuple[Sha256Digest, ...] = ()
    review: RequiredString
    reliance: RequiredString
    rights: RequiredString


class PathKnot(Record):
    time: Probability
    value: Coordinate
    model_revision: RequiredString | None = None


class EntityPath(Record):
    entity_id: RequiredIdentifier
    scale_id: RequiredIdentifier
    history_status: Literal["stipulated", "captured"]
    history_rationale: RequiredString
    interpolation: Literal["linear", "step"]
    knots: tuple[PathKnot, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def complete_ordered_path(self) -> Self:
        if self.knots[0].time != 0 or self.knots[-1].time != 1:
            raise ValueError("entity paths must cover the complete study period")
        if any(a.time >= b.time for a, b in zip(self.knots, self.knots[1:])):
            raise ValueError("path knot times must be strictly increasing")
        return self


# INITIAL_ESTIMATE judgment for every capability response law, elicited or
# authored (methodology.md): a last-slope continuation extends at most one
# top-anchor spacing above the highest response coordinate and is flat beyond,
# since uncapped an exponential continuation under a heavy-tailed path has no
# finite mean. Its one-sided range is zero spacings (flat) to one.
LAST_SLOPE_CAP_SPACINGS = 1.0


def last_slope_cap(coordinates: Sequence[float]) -> float:
    """Highest coordinate at which an upper last-slope continuation still changes."""
    ordered = sorted(set(coordinates))
    if len(ordered) < 2:
        raise ValueError("last-slope continuation needs two response coordinates")
    return ordered[-1] + LAST_SLOPE_CAP_SPACINGS * (ordered[-1] - ordered[-2])


class ResponsePoint(Record):
    value: Coordinate
    multiplier: Positive


class LocalResponse(Record):
    """Piecewise response in a local coordinate.

    ``last_slope`` continues the highest segment upward to ``last_slope_cap``
    of the points and is flat beyond it and below the lowest point, exactly as
    for elicited responses.
    """

    domain: Literal["local_entity_state", "conditional_annual_kernel"] = (
        "local_entity_state"
    )
    interpolation: Literal["log", "linear"]
    extrapolation: Literal["flat", "last_slope", "unavailable"]
    points: tuple[ResponsePoint, ...] = Field(min_length=1)
    held_fixed: RequiredString

    @model_validator(mode="after")
    def ordered_points(self) -> Self:
        if self.domain == "conditional_annual_kernel" and any(
            point.multiplier != 1 for point in self.points
        ):
            raise ValueError(
                "a conditioned annual kernel cannot be relabelled as a local response curve"
            )
        if any(a.value >= b.value for a, b in zip(self.points, self.points[1:])):
            raise ValueError("response coordinates must be strictly increasing")
        return self


class UnitCountMass(Record):
    count: int = Field(ge=0, strict=True)
    probability: Probability


class ClusterCount(Record):
    """An episode's count of potential members: a finite support of counts
    and their probabilities. ``mean`` is always null."""

    family: Literal["finite"]
    support: tuple[UnitCountMass, ...] = ()
    mean: None = None

    @model_validator(mode="after")
    def identified_law(self) -> Self:
        if not self.support:
            raise ValueError("finite cluster counts require their complete support")
        require_unique([row.count for row in self.support], "cluster count support")
        if not math.isclose(
            math.fsum(row.probability for row in self.support), 1, abs_tol=1e-12
        ):
            raise ValueError("cluster count probabilities must sum to one")
        return self

    def expected_count(self) -> float:
        return math.fsum(row.count * row.probability for row in self.support)


class JointMarkOutcome(Record):
    probability: Probability
    components: tuple[ComponentAmountLaw, ...] = ()
    identified_gross_total: Nonnegative | None = None
    composition_complete: bool = True
    overlap_resolved: bool = True
    scenario_ids: tuple[RequiredIdentifier, ...] = ()
    obligations: tuple[ComponentObligation, ...] = ()
    payments: tuple[ComponentPayment, ...] = ()
    memberships: tuple[PolicyMembership, ...] = ()
    financial_mapping_complete: bool = False
    payment_kernel: NativePaymentKernel | None = None

    @model_validator(mode="after")
    def physical_component_ids(self) -> Self:
        require_unique(self.scenario_ids, "scenario report population IDs")
        _payment_kernel_relations(
            self.components, self.obligations, self.payments, self.payment_kernel
        )
        _component_relations(
            self.components, self.obligations, self.payments, self.memberships
        )
        require_unique(
            [component.component_id for component in self.components],
            "physical component IDs in an outcome",
        )
        if not self.composition_complete and self.identified_gross_total is None:
            raise ValueError(
                "incomplete composition requires an identified gross total or remains unavailable"
            )
        return self


class JointClusterOutcome(Record):
    probability: Probability
    members: tuple[JointMarkOutcome, ...]
    campaign_components: tuple[ComponentAmountLaw, ...] = ()
    campaign_obligations: tuple[ComponentObligation, ...] = ()
    campaign_payments: tuple[ComponentPayment, ...] = ()
    campaign_memberships: tuple[PolicyMembership, ...] = ()
    campaign_financial_mapping_complete: bool = False
    campaign_payment_kernel: NativePaymentKernel | None = None

    @model_validator(mode="after")
    def conditional_joint_mark(self) -> Self:
        _payment_kernel_relations(
            self.campaign_components,
            self.campaign_obligations,
            self.campaign_payments,
            self.campaign_payment_kernel,
        )
        if any(member.probability != 1 for member in self.members):
            raise ValueError(
                "members in a complete joint cluster branch have conditional probability one"
            )
        require_unique(
            [row.component_id for row in self.campaign_components],
            "joint-branch campaign component IDs",
        )
        _component_relations(
            self.campaign_components,
            self.campaign_obligations,
            self.campaign_payments,
            self.campaign_memberships,
        )
        return self


class MarkKernel(Record):
    cluster: ClusterCount | None = None
    shared_probability: Probability
    unit_probability: Probability
    campaign_components: tuple[ComponentAmountLaw, ...] = ()
    campaign_lab_losses: tuple[AggregateLabLossLaw, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    campaign_obligations: tuple[ComponentObligation, ...] = ()
    campaign_payments: tuple[ComponentPayment, ...] = ()
    campaign_memberships: tuple[PolicyMembership, ...] = ()
    campaign_financial_mapping_complete: bool = False
    campaign_payment_kernel: NativePaymentKernel | None = None
    member_outcomes: tuple[JointMarkOutcome, ...] = ()
    explicit_joint_members: tuple[JointClusterOutcome, ...] = ()
    gate_meaning: RequiredString
    selection_assumption: RequiredString
    displacement_assumption: RequiredString

    @model_validator(mode="after")
    def complete_joint_support(self) -> Self:
        require_unique(
            [law.lab_id for law in self.campaign_lab_losses], "aggregate lab targets"
        )
        _payment_kernel_relations(
            self.campaign_components,
            self.campaign_obligations,
            self.campaign_payments,
            self.campaign_payment_kernel,
        )
        _component_relations(
            self.campaign_components,
            self.campaign_obligations,
            self.campaign_payments,
            self.campaign_memberships,
        )
        if self.explicit_joint_members:
            if self.cluster is not None or self.member_outcomes:
                raise ValueError(
                    "explicit joint clusters derive their sole member count authority from each branch"
                )
            if not math.isclose(
                math.fsum(row.probability for row in self.explicit_joint_members),
                1,
                abs_tol=1e-12,
            ):
                raise ValueError("joint cluster probabilities must sum to one")
            for branch in self.explicit_joint_members:
                if set(row.component_id for row in self.campaign_components) & set(
                    row.component_id for row in branch.campaign_components
                ):
                    raise ValueError(
                        "common and joint-branch campaign components need distinct physical IDs"
                    )
        elif self.cluster is None or not self.member_outcomes:
            raise ValueError(
                "a mark kernel requires exactly one complete joint cluster or count/conditional-mark law"
            )
        elif not math.isclose(
            math.fsum(row.probability for row in self.member_outcomes), 1, abs_tol=1e-12
        ):
            raise ValueError("conditional member-mark probabilities must sum to one")
        require_unique(
            [row.component_id for row in self.campaign_components],
            "campaign component IDs",
        )
        return self

    def expected_member_count(self) -> float:
        if self.explicit_joint_members:
            return math.fsum(
                branch.probability * len(branch.members)
                for branch in self.explicit_joint_members
            )
        assert self.cluster is not None
        return self.cluster.expected_count()

    def possible_member_outcomes(self) -> tuple[JointMarkOutcome, ...]:
        if self.explicit_joint_members:
            return tuple(
                member
                for branch in self.explicit_joint_members
                if branch.probability > 0
                for member in branch.members
            )
        return tuple(
            member for member in self.member_outcomes if member.probability > 0
        )


class IncidentSegment(Record):
    start: Probability
    end: Probability
    reference_exposure_share: Probability
    initiation_probability: Probability
    kernel: MarkKernel
    source_public_status: Literal["public", "not_yet_public"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Source model's public-release status at incident initiation, held fixed throughout this segment.",
    )
    source_model_id: RequiredIdentifier | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_model_revision: RequiredString | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def interval(self) -> Self:
        if self.start >= self.end:
            raise ValueError("incident segments must have positive duration")
        source = (
            self.source_model_id,
            self.source_model_revision,
            self.source_public_status,
        )
        if any(value is not None for value in source) and not all(
            value is not None for value in source
        ):
            raise ValueError(
                "source model identity, revision and public status must be supplied together"
            )
        return self


type SpreadTailForm = Literal["pure_stretch", "calibrated_p95_peak"]
"""Tail-width rule applied after reading a count law at stretched scores.

``pure_stretch`` keeps POOL-PCHIP's local width. ``calibrated_p95_peak`` caps
the width so the mean integrand peaks at the stretched law's p95.
"""

SPREAD_TAIL_FORMS: tuple[SpreadTailForm, ...] = (
    "pure_stretch",
    "calibrated_p95_peak",
)


class ParameterRankLaw(Record):
    """A count's epistemic law, read at one parameter rank per simulated year.

    ``count`` holds the five interpolated count quantiles. A year's normal
    score ``z`` reads the law at the stretched score ``rank_spread * z``, with
    ``epistemic_tail_scale`` and the post-stretch ``tail_form``
    (``elicitation.epistemic.stretched_score_quantiles``). ``group_id`` names
    the rank within a joint state's rank streams: a state's laws sharing it
    read the same rank in a year, and states read independent ranks unless
    they declare one stream (``JointState.rank_stream_id``, ``NUM-STREAMS``).
    """

    group_id: RequiredIdentifier
    count: Quantiles
    epistemic_tail_scale: Positive = 1
    rank_spread: Positive = 1
    tail_form: SpreadTailForm


class IncidentLaw(Record):
    origin_id: RequiredIdentifier
    scenario_id: RequiredIdentifier
    entity_id: RequiredIdentifier
    deployment_id: RequiredIdentifier | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reference_annual_rate: Nonnegative = Field(
        description="The expected annual count; for a law with a parameter rank, the count law's mean over that rank."
    )
    parameter_rank: ParameterRankLaw | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="An epistemic count law read at a fresh rank each simulated year; the year's count is Poisson (or negative binomial) at the reference rate times Q(u) / E[Q(U)].",
    )
    count_family: Literal["poisson", "negative_binomial"]
    count_variance_to_mean: Positive = 1
    response: LocalResponse
    segments: tuple[IncidentSegment, ...] = Field(min_length=1)
    evidence_ids: tuple[RequiredIdentifier, ...] = ()
    input_stage: Literal["attempted_initiator", "qualified_initiator"]
    qualification_assumption: RequiredString
    conditioning_context: Frozen[dict[str, object]] = Field(default_factory=FrozenDict)
    unavailable_lab_losses: Frozen[dict[RequiredIdentifier, RequiredString]] = Field(
        default_factory=FrozenDict,
        exclude_if=lambda value: not value,
        description="Lab totals whose incident-linked cost roster is unidentified, even if the included components are zero.",
    )
    annual_lab_losses: tuple[AnnualLabLossLaw, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
        description="Lab losses drawn once per simulated year in which this origin's count is at least one, outside every episode's marks and episode cap.",
    )

    @model_validator(mode="after")
    def unique_rate_authority(self) -> Self:
        require_unique(
            [law.lab_id for law in self.annual_lab_losses], "annual lab targets"
        )
        if (
            self.annual_lab_losses
            and len({segment.source_public_status for segment in self.segments}) > 1
        ):
            raise ValueError(
                "an annual lab loss needs one source status across the law's segments"
            )
        if self.count_family == "poisson" and self.count_variance_to_mean != 1:
            raise ValueError("Poisson variance-to-mean must equal one")
        if (
            self.count_family == "negative_binomial"
            and self.count_variance_to_mean <= 1
        ):
            raise ValueError(
                "fixed-VMR negative binomial requires variance-to-mean > 1"
            )
        if any(a.end > b.start for a, b in zip(self.segments, self.segments[1:])):
            raise ValueError(
                "incident exposure segments must be ordered and nonoverlapping"
            )
        if (
            math.fsum(segment.reference_exposure_share for segment in self.segments)
            > 1 + 1e-12
        ):
            raise ValueError(
                "absolute exposure shares cannot exceed the full reference period"
            )
        if self.count_family == "negative_binomial" and any(
            segment.initiation_probability != 1 for segment in self.segments
        ):
            raise ValueError(
                "fixed-VMR negative binomial authors the final qualified annual count; Poisson-style initiation thinning is unsupported"
            )
        if self.input_stage == "qualified_initiator" and any(
            segment.initiation_probability != 1 for segment in self.segments
        ):
            raise ValueError(
                "already qualified initiators cannot repeat initiation qualification"
            )
        return self


class PopulationAbsence(Record):
    scenario_id: RequiredIdentifier
    disposition: Literal["structural_zero", "unavailable"]
    reason: RequiredString


class JointState(Record):
    state_id: RequiredIdentifier
    epistemic_group_id: RequiredIdentifier
    sampling_group_id: RequiredIdentifier | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The respondent-sampling group of a context-bound state: one trajectory node or fixed-anchor coordinate and one respondent model, whose kept samples are treated as independent draws (NUM-RESPONDENT-SAMPLING). None where states are not respondent samples.",
    )
    borrowed_leaves: tuple[RequiredIdentifier, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
        description="The leaves whose answers the state reads from a donor, a sample of another respondent model at the same node, because its own model does not answer them (POOL-UNANSWERED-SCENARIOS). The state's sampling group is then not independent of the donor's, and no respondent-sampling standard error is reported for an EAL that weights the state (NUM-RESPONDENT-SAMPLING). Empty where every answer is the respondent's own.",
    )
    lab_loss_left_out: bool = Field(
        default=False,
        exclude_if=lambda value: not value,
        description="Whether the state is left out of the views that count the lab's own loss, the lab-loss views and the gross views, because a case it reads, its own or a donor's, declares a part of the lab's own loss unavailable (POOL-EXCLUDED-SAMPLES). Such a view then gives the state weight zero and splits its sampling group's weight among the group's other states in proportion to their weights; every other view weights the state as the arm does. False where every case the state reads gives the lab's loss, and where the study withholds those views instead.",
    )
    rank_stream_id: RequiredIdentifier | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The address of the state's parameter-rank streams (NUM-STREAMS): states declaring one address read the same parameter rank for a rank group in a simulated year. None is the state's own ID, whose ranks are independent of every other state's; a variant state declares the state it is measured against.",
    )
    weight: Probability
    entity_paths: tuple[EntityPath, ...]
    laws: tuple[IncidentLaw, ...] = ()
    absent_populations: tuple[PopulationAbsence, ...] = ()
    physical_factor_state: Frozen[dict[str, Coordinate]] = Field(
        default_factory=FrozenDict
    )

    @model_validator(mode="after")
    def unique_bindings(self) -> Self:
        require_unique(
            [row.entity_id for row in self.entity_paths], "entity path bindings"
        )
        require_unique([row.origin_id for row in self.laws], "initiating origin IDs")
        require_unique(self.borrowed_leaves, "borrowed leaves")
        if self.lab_loss_left_out and self.sampling_group_id is None:
            raise ValueError(
                "a state left out of the views that count the lab's own loss names "
                "the sampling group that takes its weight in them"
            )
        require_unique(
            [row.scenario_id for row in self.absent_populations],
            "population dispositions",
        )
        if {row.scenario_id for row in self.laws} & {
            row.scenario_id for row in self.absent_populations
        }:
            raise ValueError(
                "a population cannot have both an incident law and an absence disposition"
            )
        if not {row.entity_id for row in self.laws} <= {
            row.entity_id for row in self.entity_paths
        }:
            raise ValueError("every conditional incident law needs its own entity path")
        return self


class LawOverride(Record):
    state_id: RequiredIdentifier
    law: IncidentLaw


class CountExperiment(Record):
    kind: Literal["fixed_marginal_gaussian", "shared_gamma_intensity"]
    origin_groups: Frozen[dict[RequiredIdentifier, RequiredIdentifier]]
    correlation: Probability | None = None
    coefficient_of_variation: Positive | None = None

    @model_validator(mode="after")
    def distinct_mechanisms(self) -> Self:
        if not self.origin_groups:
            raise ValueError("a count experiment must declare its origin groups")
        if self.kind == "fixed_marginal_gaussian":
            if self.correlation is None or self.coefficient_of_variation is not None:
                raise ValueError("Gaussian count coupling requires correlation only")
        elif self.coefficient_of_variation is None or self.correlation is not None:
            raise ValueError("gamma intensity requires coefficient of variation only")
        return self


class Alternative(Record):
    """One arm compared with the reference: its purpose, state weights and
    count experiment. The laws it puts in place of reference laws are held by
    the setup, by alternative (``RunSetup.overrides``), so that a setup can
    read one arm's laws at a time; the inline encoding lists them here, as
    ``overrides``."""

    alternative_id: RequiredIdentifier
    purpose: Literal["scientific_alternative", "numerical_refinement"]
    description: RequiredString
    count_experiment: CountExperiment | None = None
    state_weights: Frozen[dict[RequiredIdentifier, Nonnegative]] | None = None
    matched_control_id: RequiredIdentifier | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The arm this alternative is measured against instead of the reference; a count experiment names its own, and a count control names itself.",
    )
    selection_provenance: Frozen[dict[str, object]] = Field(default_factory=FrozenDict)

    @model_validator(mode="after")
    def coherent_terms(self) -> Self:
        if self.count_experiment is not None and self.matched_control_id is None:
            raise ValueError("a count experiment must name its matched control")
        if self.state_weights is not None and not math.isclose(
            math.fsum(self.state_weights.values()), 1, abs_tol=1e-12
        ):
            raise ValueError("alternative joint-state weights must sum to one")
        return self


class ExecutionRules(Record):
    work_limit_exceedance: Literal["abort_without_capture"] = "abort_without_capture"


SCENARIO_VIEW_MEASURES = ("gross", "welfare", "net_resource_cost")
"""The accounting measures a scenario view can name."""


def parse_scenario_view(view: str) -> tuple[str, str] | None:
    """The scenario ID and measure of a scenario view,
    ``scenario::<id>::<measure>``; ``None`` for any other view.

    A scenario view names its measure, one of ``SCENARIO_VIEW_MEASURES``.
    One that does not is refused: ``scenario::<id>`` alone is the key of a
    scenario's scope (``view_bases``, ``unavailable_views``), not a view.
    """
    if not view.startswith("scenario::"):
        return None
    identifier = view.removeprefix("scenario::")
    scenario, _, measure = identifier.rpartition("::")
    if not scenario or measure not in SCENARIO_VIEW_MEASURES:
        raise ValueError(
            f"scenario view {view!r} names no measure: a scenario view is "
            f"'scenario::<id>::<measure>' with a measure of "
            f"{list(SCENARIO_VIEW_MEASURES)}, so the scenario's gross loss is "
            f"'scenario::{identifier}::gross'"
        )
    return scenario, measure


def gross_view(view: str) -> bool:
    """Whether ``view`` is a gross view: ``gross`` or a scenario's,
    ``scenario::<id>::gross``. ``oep_gross``, a maximum over occurrences, is
    not one."""
    if view.startswith("scenario::"):
        return view.rpartition("::")[2] == "gross"
    return view == "gross"


def counts_lab_loss(view: str) -> bool:
    """Whether ``view`` counts a lab's own loss.

    The lab-loss views do: ``lab_loss::<lab>`` and a source status's,
    ``source_status::<status>::lab_loss::<lab>``. So do the gross views
    (``gross_view``), which add each node's annual lab business loss
    (``ACC-LAB-ANNUAL-BUSINESS-LOSS``). No other view reads the lab's loss:
    the welfare and net-resource views, their scenario and source-status
    views and ``oep_gross`` leave it out.

    These are the views a state can be left out of
    (``JointState.lab_loss_left_out``)."""
    if gross_view(view):
        return True
    if view.startswith("source_status::"):
        view = view.removeprefix("source_status::").partition("::")[2]
    return view.startswith("lab_loss::")


class ComparisonRequirement(Record):
    family_id: RequiredIdentifier
    arm_id: RequiredIdentifier | None = None
    scenario_ids: tuple[RequiredIdentifier, ...]
    affected_views: tuple[RequiredIdentifier, ...] | None = Field(
        default=None,
        min_length=1,
        description="Exact affected output view IDs; None includes all requested views within the scenario scope.",
    )
    status: Literal["required", "unavailable", "inapplicable"] = "required"
    reason: RequiredString | None = None
    held_fixed: tuple[RequiredString, ...]

    @model_validator(mode="after")
    def explicit_scope(self) -> Self:
        require_unique(self.scenario_ids, "comparison scenario scope")
        if self.affected_views is not None:
            require_unique(self.affected_views, "comparison view scope")
            for view in self.affected_views:
                parse_scenario_view(view)
        if not self.scenario_ids or not self.held_fixed:
            raise ValueError(
                "comparison requirements need scenario scope and held-fixed quantities"
            )
        if self.status == "required" and (
            self.arm_id is None or self.reason is not None
        ):
            raise ValueError(
                "required comparison needs an arm and no unavailability reason"
            )
        if self.status != "required" and self.reason is None:
            raise ValueError("unavailable or inapplicable comparisons need a reason")
        return self


# Import commands start each arm at DEFAULT_YEARS_PER_STATE simulated years per
# state, an INITIAL_ESTIMATE numerical judgment (NUM-BUDGETS): 64 years in each
# of the tail gate's 16 batches, few enough that a comparison arm of up to 488
# states fits its default cap. The tail gate then doubles them up to the
# reference and comparison arms' caps on simulated state-years, which are
# judgments (NUM-TAIL-GATE).
DEFAULT_YEARS_PER_STATE = 1024
DEFAULT_MAX_REFERENCE_STATE_YEARS = 2_000_000
DEFAULT_MAX_COMPARISON_STATE_YEARS = 500_000

# Work allowances per simulated year of a state (NUM-BUDGETS), each with the
# count it bounds: events drawn individually and the realized members of
# campaign episodes (those of kernels with several members or campaign terms),
# each drawn as its own mark. A state's limit on a count is its allowance times
# its simulated years, so it bounds the count's mean over those years, never
# one year's count; max_members_per_campaign bounds one episode's members. The
# defaults, which the import commands freeze and their options override, are
# INITIAL_ESTIMATE compute guards sized from the measured costs NUM-BUDGETS
# gives.
WORK_ALLOWANCES: Mapping[str, str] = {
    "max_vectorized_events_per_year": "vectorized_initiators",
    "max_campaign_members_per_year": "campaign_members",
}
DEFAULT_WORK_ALLOWANCES: Mapping[str, int] = {
    "max_vectorized_events_per_year": 50_000,
    "max_campaign_members_per_year": 50_000,
}
_INTEGER_CONTROLS = (*WORK_ALLOWANCES, "max_members_per_campaign", "event_batch_size")
NUMERICAL_CONTROLS = frozenset({*_INTEGER_CONTROLS, "small_events", "tail_gate"})
"""Every numerical control a plan may freeze, plan-wide or for an arm."""
TAIL_GATE_SETTINGS = frozenset(
    {
        "tolerance",
        "batches",
        "max_reference_state_years",
        "max_comparison_state_years",
        "views",
        "levels",
    }
)
"""Every setting of a plan's ``tail_gate`` control."""


def check_tail_gate_controls(value: object) -> None:
    """Refuse a ``tail_gate`` control that is not a record of the gate's
    settings, each of its own type and range. ``None`` is the default gate,
    and a null cap, ``views`` or ``levels`` its default; a present
    ``tolerance`` or ``batches`` needs a value, and a scenario view among
    ``views`` its measure (``parse_scenario_view``)."""
    if value is None:
        return
    if not isinstance(value, Mapping) or set(value) - TAIL_GATE_SETTINGS:
        raise ValueError(f"tail_gate controls take only {sorted(TAIL_GATE_SETTINGS)}")
    if "tolerance" in value:
        tolerance = value["tolerance"]
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, int | float)
            or not 0 < tolerance < 1
        ):
            raise ValueError("the tail gate tolerance must lie strictly in (0, 1)")
    for name, low, optional in (
        ("batches", 2, False),
        ("max_reference_state_years", 1, True),
        ("max_comparison_state_years", 1, True),
    ):
        if name not in value or (optional and value[name] is None):
            continue
        count = value[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < low:
            raise ValueError(
                f"the tail gate {name} must be an integer of at least {low}"
            )
    views = value.get("views")
    if views is not None and (
        not isinstance(views, list | tuple)
        or not views
        or any(not isinstance(view, str) or not view for view in views)
    ):
        raise ValueError("the tail gate views must be a nonempty list of view names")
    for view in views or ():
        parse_scenario_view(view)
    levels = value.get("levels")
    if levels is not None and (
        not isinstance(levels, list | tuple)
        or not levels
        or any(
            isinstance(level, bool)
            or not isinstance(level, int | float)
            or not 0 < level < 1
            for level in levels
        )
    ):
        raise ValueError(
            "the tail gate levels must be a nonempty list of probabilities in (0, 1)"
        )


def check_numerical_controls(controls: Mapping[str, object]) -> None:
    """Refuse an unsupported numerical control, a count control that is
    present and not a positive integer, or a ``tail_gate`` control of the
    wrong shape (``check_tail_gate_controls``), so a misspelt, mistyped or
    unknown control fails where the plan is built or loaded, not when it
    runs."""
    unknown = sorted(set(controls) - NUMERICAL_CONTROLS)
    if unknown:
        raise ValueError(
            f"unsupported numerical controls {unknown}; supported: "
            f"{sorted(NUMERICAL_CONTROLS)}"
        )
    for name in _INTEGER_CONTROLS:
        if name not in controls:
            continue
        value = controls[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"numerical control {name} must be a positive integer")
    check_tail_gate_controls(controls.get("tail_gate"))


class ExecutionPlan(Record):
    plan_id: RequiredIdentifier
    required_arms: tuple[RequiredIdentifier, ...]
    reference_arm: RequiredIdentifier = "reference"
    simulations: int = Field(ge=1, strict=True)
    seed: int = Field(ge=0, strict=True)
    requested_views: tuple[RequiredIdentifier, ...]
    capture_grain: Literal["annual_joint"] = "annual_joint"
    method: Literal["physical_marked_poisson"] = "physical_marked_poisson"
    staged_rules: ExecutionRules = Field(default_factory=ExecutionRules)
    numerical_controls: Frozen[dict[str, object]] = Field(default_factory=FrozenDict)
    simulations_by_arm: Frozen[
        dict[RequiredIdentifier, Annotated[int, Field(ge=1, strict=True)]]
    ] = Field(default_factory=FrozenDict)
    numerical_controls_by_arm: Frozen[dict[RequiredIdentifier, dict[str, object]]] = (
        Field(default_factory=FrozenDict)
    )
    comparison_requirements: tuple[ComparisonRequirement, ...] = ()
    requested_comparison_families: tuple[RequiredIdentifier, ...] = ()
    required_state_ids_by_arm: Frozen[
        dict[RequiredIdentifier, tuple[RequiredIdentifier, ...]]
    ] = Field(default_factory=FrozenDict)

    @model_validator(mode="after")
    def frozen_requirements(self) -> Self:
        require_unique(self.required_arms, "required comparison arms")
        require_unique(self.requested_views, "requested views")
        for view in self.requested_views:
            parse_scenario_view(view)
        require_unique(
            self.requested_comparison_families, "requested comparison families"
        )
        if self.requested_comparison_families and {
            row.family_id for row in self.comparison_requirements
        } != set(self.requested_comparison_families):
            raise ValueError(
                "frozen comparison requirements must cover exactly the requested families"
            )
        if self.reference_arm not in self.required_arms:
            raise ValueError("the reference arm must be required")
        if not self.requested_views:
            raise ValueError("a plan must request at least one observable")
        if not (
            set(self.simulations_by_arm)
            | set(self.numerical_controls_by_arm)
            | set(self.required_state_ids_by_arm)
        ) <= set(self.required_arms):
            raise ValueError("arm numerical controls must refer to required arms")
        for controls in (
            self.numerical_controls,
            *self.numerical_controls_by_arm.values(),
        ):
            check_numerical_controls(controls)
            gate = controls.get("tail_gate")
            views = gate.get("views") if isinstance(gate, Mapping) else None
            unrequested = [
                view for view in views or () if view not in self.requested_views
            ]
            if unrequested:
                raise ValueError(
                    f"tail gate views {unrequested} are not requested views of the "
                    f"plan: {list(self.requested_views)}"
                )
        for arm, state_ids in self.required_state_ids_by_arm.items():
            if not state_ids:
                raise ValueError(f"required state support for {arm!r} must be nonempty")
            require_unique(state_ids, "required arm joint states")
        if any(
            isinstance(value, bool) or value < 1
            for value in self.simulations_by_arm.values()
        ):
            raise ValueError("arm simulation counts must be positive integers")
        if any(
            row.arm_id is not None
            and (
                row.arm_id not in self.required_arms or row.arm_id == self.reference_arm
            )
            for row in self.comparison_requirements
        ):
            raise ValueError(
                "comparison requirement arms must be frozen required alternative arms"
            )
        canonical_json_sha256(self.model_dump(mode="json"))
        return self

    def for_arm(self, arm_id: str) -> ExecutionPlan:
        if arm_id not in self.required_arms:
            raise ValueError("arm execution requires a frozen required arm")
        return self.model_copy(
            update={
                "simulations": self.simulations_by_arm.get(arm_id, self.simulations),
                "numerical_controls": FrozenDict(
                    {
                        **self.numerical_controls,
                        **self.numerical_controls_by_arm.get(arm_id, {}),
                    }
                ),
            }
        )

    def required_states(
        self, arm_id: str, all_state_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        if arm_id not in self.required_arms:
            raise ValueError("state support requires a frozen required arm")
        return self.required_state_ids_by_arm.get(arm_id, all_state_ids)
