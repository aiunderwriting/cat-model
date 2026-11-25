"""Scenario facet schema.

Facets are reporting classifications. Hierarchy placement owns the origin and
what the AI did, and the facet projection derives ``causal_origin`` from it.
Facets are never rendered into questions; an ambiguous classification sets
``provisional``.
"""

from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator

from catmodel.scenarios.schema import StrictModel

if TYPE_CHECKING:
    from catmodel.scenarios.loader import ScenarioDefinition

PerilMechanism = Literal[
    "deception_of_humans",
    "technical_intrusion",
    "novel_vulnerability_discovery",
    "supply_chain_compromise",
    "capability_uplift",
    "erroneous_output_reliance",
    "harmful_engagement",
    "bounded_pipeline_error",
    "discriminatory_output",
    "fabricated_content",
    "autonomous_goal_pursuit",
    "cross_system_cascade",
    "shared_component_defect",
    "self_exfiltration",
    "evaluation_deception",
    "mass_manipulation",
    "enforcement_process",
    "government_designation",
    "service_availability_failure",
    "deployed_system_hijack",
    "private_civil_claim",
]

LifecycleStage = Literal[
    "training_evaluation",
    "internal_deployment",
    "external_deployment",
    "provider_enterprise_conduct",
]

# Control mode of the implicated model instance. ``any_copy`` covers every copy
# of the provider's model however obtained (served, stolen, leaked or open
# weights), which is what helping an attacker counts; ``derivative`` covers
# fine-tuned/distilled descendants; exfiltration modes distinguish human from
# model initiation, with the latter requiring developer-control-loss placement.
DeploymentControlMode = Literal[
    "any_copy",
    "api",
    "managed",
    "released_weights",
    "stolen_weights",
    "self_exfiltrated",
    "derivative",
]

HarmType = Literal[
    "economic_loss",
    "financial_theft",
    "bodily_injury",
    "property_damage",
    "discrimination_harm",
    "ip_loss",
    "regulatory_sanction",
]

# The provider appears when provider-borne components are included.
LossBearer = Literal[
    "provider",
    "deployer_or_customer",
    "third_party_organizations",
    "third_party_individuals",
    "general_public",
]

# Dominant claims basis from the insured provider's perspective; mixed bases are
# marked provisional with a YAML rationale.
PartyBasis = Literal["first_party", "third_party"]


def admissible_lifecycle_stages(
    origin_regime: str | None, ai_role: str | None
) -> frozenset[str]:
    if ai_role == "developer_control_loss":
        return frozenset({"training_evaluation", "internal_deployment"})
    if origin_regime == "provider_conduct":
        return frozenset({"provider_enterprise_conduct"})
    return frozenset({"external_deployment"})


# These mechanisms are owned by one placement coordinate; values are
# ("origin", origin_regime) or ("role", ai_role).
PLACEMENT_EXCLUSIVE_MECHANISMS: dict[str, tuple[str, str]] = {
    "self_exfiltration": ("role", "developer_control_loss"),
    "evaluation_deception": ("role", "developer_control_loss"),
    "enforcement_process": ("origin", "provider_conduct"),
    "government_designation": ("origin", "provider_conduct"),
    "service_availability_failure": ("role", "service_failure"),
    "deployed_system_hijack": ("role", "acted_for_attacker"),
    "private_civil_claim": ("origin", "provider_conduct"),
}


def _placement_coordinates(node_ids: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Return placement coordinates; missing nodes yield ``(None, None)``.

    The sentinel lets validation report coherence findings before contract
    membership checks run.
    """
    # Import lazily because hierarchy validation imports scenario loading, which
    # imports this module for facet validation.
    from catmodel.scenarios.hierarchy_contract import (
        cached_hierarchy_contract,
        scenario_placement_coordinates,
    )

    try:
        return scenario_placement_coordinates(cached_hierarchy_contract(), node_ids)[:2]
    except KeyError:
        return (None, None)


# Rollup axes and additivity: single-valued axes partition; multi-valued axes
# overlap.
FACET_ROLLUP_AXES: dict[str, Literal["partition", "multi_label_overlapping"]] = {
    "causal_origin": "partition",
    "peril_mechanism": "multi_label_overlapping",
    "lifecycle_stage": "partition",
    "deployment_control_mode": "partition",
    "harm_types": "multi_label_overlapping",
    "bearers": "multi_label_overlapping",
    "party_basis": "partition",
}


class ScenarioFacetsDef(StrictModel):
    peril_mechanism: list[PerilMechanism] = Field(min_length=1)
    # Omit when placement admits one stage; singleton stages are derived.
    lifecycle_stage: LifecycleStage | None = None
    deployment_control_mode: DeploymentControlMode
    harm_types: list[HarmType] = Field(min_length=1)
    bearers: list[LossBearer] = Field(min_length=1)
    party_basis: PartyBasis
    # An ambiguous classification is marked provisional, with a YAML rationale.
    provisional: bool = False

    @field_validator("peril_mechanism", "harm_types", "bearers")
    @classmethod
    def validate_unique_vector(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError(f"facet vector values must be unique, got {value}")
        return value


def scenario_facet_projection(scenario: "ScenarioDefinition") -> dict[str, object]:
    projection = scenario.facets.model_dump(mode="json")
    origin_regime, ai_role = _placement_coordinates(scenario.node_ids)
    if origin_regime is None:
        raise ValueError(f"cell node {scenario.node_ids!r} resolves no origin regime")
    projection["causal_origin"] = origin_regime
    allowed_stages = admissible_lifecycle_stages(origin_regime, ai_role)
    lifecycle_stage = scenario.facets.lifecycle_stage
    if lifecycle_stage is None:
        if len(allowed_stages) != 1:
            raise ValueError(
                f"lifecycle_stage is required for node {scenario.node_ids!r}"
            )
        lifecycle_stage = next(iter(allowed_stages))
    projection["lifecycle_stage"] = lifecycle_stage
    return projection


SCENARIO_GROUP_AXES: tuple[str, ...] = (
    "risk_domain",
    *(
        axis
        for axis, additivity in FACET_ROLLUP_AXES.items()
        if additivity == "partition"
    ),
)
"""The axes a result's scenarios are grouped by: the hierarchy contract's
reporting domain, then the facet axes on which every scenario has one value.
The groups of such an axis share no scenario, so their EALs add up. An axis on
which a scenario can have several values, such as its loss bearers, is not
grouped: a scenario's loss cannot be divided among its values."""


def scenario_group_values(scenario: "ScenarioDefinition") -> dict[str, str]:
    """One scenario's value on every grouping axis."""
    projection = scenario_facet_projection(scenario)
    return {
        axis: scenario.risk_domain if axis == "risk_domain" else str(projection[axis])
        for axis in SCENARIO_GROUP_AXES
    }


def scenario_facet_violations(scenario: "ScenarioDefinition") -> list[str]:
    facets = scenario.facets
    violations: list[str] = []

    origin_regime, ai_role = _placement_coordinates(scenario.node_ids)
    allowed_stages = admissible_lifecycle_stages(origin_regime, ai_role)
    if facets.lifecycle_stage is None and len(allowed_stages) > 1:
        violations.append(
            f"lifecycle_stage is required for node {scenario.node_ids!r} "
            f"(expected one of {sorted(allowed_stages)})"
        )
    elif (
        facets.lifecycle_stage is not None
        and facets.lifecycle_stage not in allowed_stages
    ):
        violations.append(
            f"lifecycle_stage {facets.lifecycle_stage!r} is incoherent with "
            f"node {scenario.node_ids!r} "
            f"(expected one of {sorted(allowed_stages)})"
        )
    coordinates = {"origin": origin_regime, "role": ai_role}
    for mechanism in facets.peril_mechanism:
        owner = PLACEMENT_EXCLUSIVE_MECHANISMS.get(mechanism)
        if owner is not None and coordinates[owner[0]] != owner[1]:
            violations.append(
                f"peril_mechanism {mechanism!r} is owned by the "
                f"{owner[1]!r} {owner[0]} and cannot be authored on a cell "
                f"placed at {scenario.node_ids!r}"
            )
    if (facets.deployment_control_mode == "any_copy") != (
        ai_role == "assisted_attacker"
    ):
        violations.append(
            "deployment_control_mode 'any_copy' is exactly the mode of cells where "
            "the AI helped the attacker, which count any copy of the model"
        )
    if (
        facets.deployment_control_mode == "self_exfiltrated"
        and ai_role != "developer_control_loss"
    ):
        violations.append(
            "deployment_control_mode 'self_exfiltrated' is only coherent on "
            "developer-control-loss cells"
        )
    if origin_regime == "provider_conduct" and facets.party_basis != "first_party":
        violations.append(
            "provider-conduct cells have the provider as respondent or "
            "defendant; party_basis must be 'first_party'"
        )
    return violations
