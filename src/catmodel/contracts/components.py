"""Physical loss components and optional legal and financial relations."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from catmodel.contracts.native_severity import (
    NativeSeverityLaw,
    shared_native_severity_sampler,
)
from catmodel.contracts.primitives import Frozen, RequiredIdentifier, RequiredString
from catmodel.contracts.shared_fields import SharedFieldsSet
from catmodel.economic_loss import EconomicNature
from catmodel.probability.distributions import (
    SeveritySampler,
    severity_lower_support,
    validate_monetary_amount,
    validate_scale_factor,
)
from catmodel.probability.liability import LiabilityAttenuation
from catmodel.probability.numerics import product_at_least


def _number(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("component quantities must be finite JSON numbers")
    return value


type NonnegativeAmount = Annotated[
    float, Field(ge=0, allow_inf_nan=False), BeforeValidator(_number)
]
type Fraction = Annotated[
    float, Field(ge=0, le=1, allow_inf_nan=False), BeforeValidator(_number)
]


class _Record(SharedFieldsSet):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentIdentity(_Record):
    component_id: RequiredIdentifier
    bearer_id: RequiredIdentifier | None = None
    nature: EconomicNature | None = None
    loss_type: RequiredString | None = None
    contributor_ids: tuple[RequiredIdentifier, ...] | None = None
    attribution_weights: Frozen[dict[RequiredIdentifier, Fraction]] | None = None
    lab_loss_weights: Frozen[dict[RequiredIdentifier, Fraction]] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Complete pre-recovery lab cost allocation of this unique component; null is unknown and absent keys in a supplied map mean zero.",
    )
    scenario_ids: tuple[RequiredIdentifier, ...] = ()
    report_labels: tuple[RequiredString, ...] = ()

    @model_validator(mode="after")
    def coherent_relations(self) -> Self:
        for values in (self.contributor_ids, self.scenario_ids, self.report_labels):
            if values is not None and len(values) != len(set(values)):
                raise ValueError("component relation identifiers must be unique")
        if self.attribution_weights is not None and not math.isclose(
            math.fsum(self.attribution_weights.values()), 1.0, abs_tol=1e-12
        ):
            raise ValueError("conserving attribution weights must sum to one")
        return self


class ComponentAmountLaw(ComponentIdentity):
    """One amount law, with explicit conditional rank sharing when stochastic.

    ``multiplier`` scales raw amounts before the severity law's outer cap;
    ``post_multiplier`` scales the capped amount.
    Reusing a rank group declares quantile coupling, not physical identity.
    """

    amount: NonnegativeAmount | None = None
    severity: NativeSeverityLaw | None = None
    rank_group_id: RequiredIdentifier | None = None
    rank_scope: Literal["member", "campaign"] = "member"
    multiplier: NonnegativeAmount = 1.0
    post_multiplier: NonnegativeAmount = 1.0

    @model_validator(mode="after")
    def identified_law(self) -> Self:
        if (self.amount is None) == (self.severity is None):
            raise ValueError("a component requires exactly one amount or severity law")
        if self.amount is not None:
            validate_monetary_amount("amount", self.amount)
        validate_scale_factor("multiplier", self.multiplier)
        validate_scale_factor("post_multiplier", self.post_multiplier)
        if self.severity is not None and self.rank_group_id is None:
            raise ValueError("a stochastic amount requires an explicit rank group")
        if self.amount is not None and self.rank_group_id is not None:
            raise ValueError("a deterministic amount does not use a rank group")
        if (
            self.severity is not None
            and self.multiplier > 0
            and self.post_multiplier > 0
        ):
            component_amount_lower_support(self)
        return self


class AggregateLabLossLaw(_Record):
    """Pre-recovery lab loss outside the episode's component allocations.

    It is drawn once per initiating episode and is not a physical component.
    The episode's lab view is the components' ``lab_loss_weights`` allocations
    from the same draw plus this amount, capped at ``episode_cap``; the cap
    never changes any other view. The draw is independent of physical marks,
    payment marks and member count.
    """

    lab_id: RequiredIdentifier
    amount: NonnegativeAmount | None = None
    severity: NativeSeverityLaw | None = None
    episode_cap: NonnegativeAmount | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def identified_law(self) -> Self:
        if (self.amount is None) == (self.severity is None):
            raise ValueError(
                "aggregate lab loss requires exactly one amount or severity law"
            )
        if self.amount is not None:
            validate_monetary_amount("amount", self.amount)
        if self.episode_cap is not None:
            if self.episode_cap == 0:
                raise ValueError("an episode lab-loss cap must be positive")
            validate_monetary_amount("episode_cap", self.episode_cap)
        if self.severity is not None:
            _nonnegative_lower_support(shared_native_severity_sampler(self.severity))
        return self


class AnnualLabLossLaw(_Record):
    """A lab's own loss from all of one origin's initiating episodes in a year.

    It is drawn once per simulated year in which the origin's count is at
    least one, and is zero in a year without an episode. The draw does not
    depend on how many episodes the year has, nor on any episode's marks. It
    adds to the lab's own view and to every gross view of the origin's
    scenario, never to a welfare view (``annual_views``); any cap is part of
    its own law (a severity cap, or an amount already at or below it),
    separate from the episode cap of ``AggregateLabLossLaw``.
    """

    lab_id: RequiredIdentifier
    amount: NonnegativeAmount | None = None
    severity: NativeSeverityLaw | None = None

    @model_validator(mode="after")
    def identified_law(self) -> Self:
        if (self.amount is None) == (self.severity is None):
            raise ValueError(
                "annual lab loss requires exactly one amount or severity law"
            )
        if self.amount is not None:
            validate_monetary_amount("amount", self.amount)
        if self.severity is not None:
            _nonnegative_lower_support(shared_native_severity_sampler(self.severity))
        return self


class NativeLiabilityParameters(_Record):
    mean: Fraction
    concentration: NonnegativeAmount
    recovery_rate: Fraction
    severity_slope: NonnegativeAmount
    median_severity: NonnegativeAmount | None = None
    regulatory_recovery_threshold: NonnegativeAmount
    regulatory_recovery_uplift: Fraction

    @model_validator(mode="after")
    def native_parameters(self) -> Self:
        if self.concentration <= 0 or self.regulatory_recovery_threshold <= 0:
            raise ValueError(
                "native concentration and recovery threshold must be positive"
            )
        if self.median_severity is not None and self.median_severity <= 0:
            raise ValueError("native median severity must be positive when supplied")
        if self.mean in (0.0, 1.0) or self.recovery_rate == 0:
            if self.mean > self.recovery_rate:
                raise ValueError("endpoint mean cannot exceed recovery probability")
            if self.severity_slope != 0 or self.regulatory_recovery_uplift != 0:
                raise ValueError(
                    "exact endpoint payments cannot retain severity-dependent adjustments"
                )
            return self
        LiabilityAttenuation(**self.model_dump())
        return self


class NativePaymentKernel(_Record):
    """Captured reduced-form developer payment, not separately identified legal stages."""

    basis: Literal["reduced_form_developer_share"] = "reduced_form_developer_share"
    source_severity: NativeSeverityLaw
    rank_group_id: RequiredIdentifier
    rank_scope: Literal["member", "campaign"] = "member"
    entity_id: RequiredIdentifier
    liability_component_ids: tuple[RequiredIdentifier, ...] = ()
    direct_component_ids: tuple[RequiredIdentifier, ...] = ()
    liability: NativeLiabilityParameters | None = None

    @model_validator(mode="after")
    def distinct_amounts(self) -> Self:
        identifiers = (*self.liability_component_ids, *self.direct_component_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("native payment components must be nonempty and unique")
        if native_payment_uses_severity(self):
            # Admit the full source support when payment conditioning consumes it.
            _nonnegative_lower_support(
                shared_native_severity_sampler(self.source_severity)
            )
        return self


class ComponentObligation(_Record):
    component_id: RequiredIdentifier
    entity_id: RequiredIdentifier
    fraction: Fraction
    basis: Literal["authored_obligation", "reduced_form_payment_proxy"] = (
        "authored_obligation"
    )


class ComponentPayment(_Record):
    component_id: RequiredIdentifier
    entity_id: RequiredIdentifier
    fraction: Fraction
    basis: Literal["authored_payment", "reduced_form_payment_proxy"] = (
        "authored_payment"
    )


class PolicyMembership(_Record):
    """A membership template whose occurrence scope is resolved before reduction."""

    component_id: RequiredIdentifier
    contract_id: RequiredIdentifier
    obligated_entity_id: RequiredIdentifier
    eligible: bool
    covered_fraction: Fraction = 1.0
    outside_limits: bool = False
    defense_loading: NonnegativeAmount = 0.0
    defense_outside_limits: bool = False
    occurrence_id: RequiredIdentifier | None = None
    occurrence_scope: Literal["member", "campaign", "contract_year"] = "campaign"

    @model_validator(mode="after")
    def scoped_occurrence(self) -> Self:
        if self.occurrence_id is None and self.occurrence_scope != "campaign":
            raise ValueError(
                "a non-campaign occurrence scope requires an occurrence ID"
            )
        return self


class ContractTerms(_Record):
    """Independent contract exposure; no other-insurance coordination is implied."""

    contract_id: RequiredIdentifier
    contract_year: RequiredIdentifier
    entity_id: RequiredIdentifier
    occurrence_retention: NonnegativeAmount = 0.0
    occurrence_limit: NonnegativeAmount | None = None
    annual_retention: NonnegativeAmount = 0.0
    annual_limit: NonnegativeAmount | None = None
    defense_loading: NonnegativeAmount = 0.0
    defense_outside_limits: bool = False
    outside_limits_subject_to_aggregate: bool = True


def _nonnegative_lower_support(sampler: SeveritySampler) -> tuple[float, bool]:
    lower, atom = severity_lower_support(sampler, apply_severity_cap=False)
    if lower < 0:
        raise ValueError("a physical severity source has negative reachable support")
    return lower, atom


def native_payment_uses_severity(kernel: NativePaymentKernel) -> bool:
    """Whether the kernel's liability payment is conditioned on its severity."""
    return bool(
        kernel.liability_component_ids
        and kernel.liability is not None
        and kernel.liability.mean not in (0.0, 1.0)
        and kernel.liability.recovery_rate > 0
    )


def component_amount_lower_support(law: ComponentAmountLaw) -> tuple[float, bool]:
    """The least amount a component law can take, and whether it is an atom."""
    if law.multiplier == 0 or law.post_multiplier == 0:
        return 0.0, True
    if law.amount is not None:
        return law.amount * law.multiplier * law.post_multiplier, True
    assert law.severity is not None
    sampler = shared_native_severity_sampler(law.severity)
    lower, atom = _nonnegative_lower_support(sampler)
    cap = sampler.severity_cap
    if cap is not None and product_at_least(lower, law.multiplier, cap):
        return cap * law.post_multiplier, True
    return lower * law.multiplier * law.post_multiplier, atom
