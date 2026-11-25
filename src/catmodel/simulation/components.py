"""Physical loss components: their view weights, support and defense charges."""

from __future__ import annotations

from collections.abc import Sequence

from catmodel.contracts.components import (
    AggregateLabLossLaw,
    AnnualLabLossLaw,
    ComponentAmountLaw,
    ComponentIdentity,
    ContractTerms,
    PolicyMembership,
)
from catmodel.contracts.native_severity import shared_native_severity_sampler
from catmodel.economic_loss import welfare_weight


def component_view_weight(
    component: ComponentIdentity, view: str, *, default: float = 0.0
) -> float:
    """Project an identified component onto one linear accounting view."""
    category, _, identifier = view.partition("::")
    if category == "gross":
        return 1.0
    if category == "welfare":
        if component.nature is None:
            raise ValueError("Unknown economic nature: None")
        return welfare_weight(component.nature)
    if category == "net_resource_cost":
        return float(component.nature in {"resource_cost", "injury_valuation"})
    if category == "provider_contribution":
        return float(len(component.contributor_ids or ()))
    if category == "lab_loss":
        if component.lab_loss_weights is None:
            raise ValueError("Complete lab loss allocation is unavailable")
        return component.lab_loss_weights.get(identifier, 0.0)
    if category == "bearer":
        return float(component.bearer_id == identifier)
    if category == "contributor":
        return float(identifier in (component.contributor_ids or ()))
    if category == "attribution":
        return (component.attribution_weights or {}).get(identifier, 0.0)
    if category == "scenario":
        return float(identifier in component.scenario_ids)
    return default


def component_has_positive_support(law: ComponentAmountLaw) -> bool:
    if law.multiplier == 0 or law.post_multiplier == 0:
        return False
    if law.amount is not None:
        return law.amount > 0
    assert law.severity is not None
    return shared_native_severity_sampler(law.severity).quantile(1) > 0


def aggregate_lab_loss_has_positive_support(
    law: AggregateLabLossLaw | AnnualLabLossLaw,
) -> bool:
    if law.amount is not None:
        return law.amount > 0
    assert law.severity is not None
    return shared_native_severity_sampler(law.severity).quantile(1) > 0


def episode_lab_loss_caps(laws: Sequence[AggregateLabLossLaw]) -> dict[str, float]:
    """The finite episode caps of a kernel's aggregate lab laws, by lab."""
    return {law.lab_id: law.episode_cap for law in laws if law.episode_cap is not None}


def defense_loading_state(
    loss_type: str | None,
    positive_amount: bool,
    membership: PolicyMembership,
    contract: ContractTerms,
) -> tuple[float, bool, bool, str | None]:
    """A membership's contract loading, whether a positive amount charges
    explicit defense or a loading, and why its loadings conflict, if they do.
    Admission refuses a law that can realize a conflict
    (``incident_runtime._financial_support_admission``)."""
    contract_loading = 0.0 if membership.outside_limits else contract.defense_loading
    conflict = (
        "contract and component defense loadings cannot both charge the same amount"
        if positive_amount and contract_loading > 0 and membership.defense_loading > 0
        else None
    )
    return (
        contract_loading,
        positive_amount and loss_type == "defense",
        positive_amount and (contract_loading > 0 or membership.defense_loading > 0),
        conflict,
    )


def defense_loading_scope_reason(
    explicit_defense: set, loaded_scopes: set
) -> str | None:
    return (
        "explicit defense and a defense loading cannot both charge the same physical scope"
        if explicit_defense & loaded_scopes
        else None
    )
