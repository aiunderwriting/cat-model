"""Cross-input invariants of the canonical run setup."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from catmodel.contracts.conditional import validate_conditional_contexts
from catmodel.contracts.incident import (
    IncidentLaw,
    LawOverride,
    PathKnot,
    require_unique,
)
from catmodel.contracts.law_overrides import ArmCheck
from catmodel.dateutil import add_one_year

if TYPE_CHECKING:
    from catmodel.contracts.run_setup import RunSetup


def _validate_identifiers_and_weights(setup: RunSetup) -> None:
    if setup.scope.period_end != add_one_year(setup.scope.period_start):
        raise ValueError("incident specifications require an exact one-year period")
    if not (
        setup.scope.forecast_as_of
        <= setup.scope.data_cutoff
        <= setup.scope.availability_cutoff
        <= setup.scope.information_date
        <= setup.scope.period_end
    ):
        raise ValueError(
            "require as-of <= data <= availability <= information <= period end"
        )
    for label, values in (
        ("entities", [row.entity_id for row in setup.scope.entities]),
        ("capability scales", [row.scale_id for row in setup.scope.capability_scales]),
        (
            "deployment/model bindings",
            [(row.deployment_id, row.model_id) for row in setup.scope.deployments],
        ),
        ("contracts", [row.contract_id for row in setup.contracts]),
        ("populations", [row.scenario_id for row in setup.scope.populations]),
        (
            "report population bindings",
            [
                (row.source_scenario_id, row.target_scenario_id)
                for row in setup.scope.report_population_bindings
            ],
        ),
        ("evidence", [row.evidence_id for row in setup.evidence.records]),
        ("joint states", [row.state_id for row in setup.joint_states]),
        (
            "alternatives",
            [row.alternative_id for row in setup.alternatives],
        ),
    ):
        require_unique(values, label)
    if any(row.alternative_id == "reference" for row in setup.alternatives):
        raise ValueError(
            "reference is the unmodified joint law, not an alternative override"
        )
    if not math.isclose(
        math.fsum(row.weight for row in setup.joint_states),
        1,
        abs_tol=1e-12,
    ):
        raise ValueError("joint-state weights must sum to one")


def _validate_rosters(setup: RunSetup) -> None:
    entities = {row.entity_id for row in setup.scope.entities}
    scenarios = {row.scenario_id for row in setup.scope.populations}
    if any(
        row.source_scenario_id not in scenarios
        or row.target_scenario_id not in scenarios
        for row in setup.scope.report_population_bindings
    ):
        raise ValueError(
            "report population bindings must refer to declared populations"
        )
    for entity in setup.scope.entities:
        if not set(entity.related_entity_ids) <= entities:
            raise ValueError("entity relation refers outside the roster")
    roster = {row.entity_id: row for row in setup.scope.entities}
    if any(row.entity_id not in roster for row in setup.contracts):
        raise ValueError(
            "contracts require a declared entity in the incident specification roster"
        )


def _validate_entity_paths(setup: RunSetup) -> None:
    roster = {row.entity_id: row for row in setup.scope.entities}
    scenarios = {row.scenario_id for row in setup.scope.populations}
    scales = {row.scale_id for row in setup.scope.capability_scales}
    known_histories: dict[str, tuple[PathKnot, ...]] = {}
    asof_time = (
        setup.scope.forecast_as_of - setup.scope.period_start
    ).total_seconds() / (
        setup.scope.period_end - setup.scope.period_start
    ).total_seconds()
    for binding in setup.scope.deployments:
        if (
            binding.deployment_id not in roster
            or roster[binding.deployment_id].kind != "deployment"
        ):
            raise ValueError("deployment bindings require a declared deployment entity")
        if binding.model_id not in roster or roster[binding.model_id].kind != "model":
            raise ValueError(
                "deployment bindings require an explicit model entity, not a lab frontier"
            )
    for population in setup.scope.populations:
        if not set(population.deployment_ids) <= {
            row.deployment_id for row in setup.scope.deployments
        }:
            raise ValueError(
                "population refers to an undeclared deployment/model binding"
            )
    for state in setup.joint_states:
        if {row.scenario_id for row in state.laws} | {
            row.scenario_id for row in state.absent_populations
        } != scenarios:
            raise ValueError(
                "every declared population requires an explicit incident law or absence disposition in every joint state"
            )
        for path in state.entity_paths:
            if path.entity_id not in roster or path.scale_id not in scales:
                raise ValueError(
                    "entity path refers to an undeclared entity or capability scale"
                )
            entity = roster[path.entity_id]
            if entity.kind == "model" and (
                entity.revision is None
                or any(knot.model_revision != entity.revision for knot in path.knots)
            ):
                raise ValueError(
                    "fixed-model paths must retain their declared model revision"
                )
            if entity.kind == "lab" and any(
                knot.model_revision is None for knot in path.knots
            ):
                raise ValueError("lab paths require explicit model revision lineage")
            if 0 < asof_time < 1 and asof_time not in {
                knot.time for knot in path.knots
            }:
                raise ValueError(
                    "midyear paths must contain the exact forecast as-of state"
                )
            if path.history_status == "captured":
                prefix = tuple(knot for knot in path.knots if knot.time <= asof_time)
                if (
                    path.entity_id in known_histories
                    and known_histories[path.entity_id] != prefix
                ):
                    raise ValueError(
                        "captured history must agree across future joint paths"
                    )
                known_histories[path.entity_id] = prefix


def _validate_alternatives(setup: RunSetup) -> None:
    """Check what the alternatives declare without reading their overrides:
    the arms that have overrides are declared, and weights cover the states."""
    states = {row.state_id for row in setup.joint_states}
    declared = {row.alternative_id for row in setup.alternatives}
    if undeclared := sorted(set(setup.overrides.arms()) - declared):
        raise ValueError(f"law overrides name undeclared alternatives: {undeclared}")
    for alternative in setup.alternatives:
        if (
            alternative.state_weights is not None
            and set(alternative.state_weights) != states
        ):
            raise ValueError(
                "alternative weights must cover exactly the union joint-state support"
            )


def _arm_check(setup: RunSetup, check_law: Callable[[IncidentLaw], None]) -> ArmCheck:
    """The check of one alternative's overrides: no state and origin is
    overridden twice, and each override replaces a declared law of its state,
    for the same population and with a declared entity path, and is itself a
    valid law (``check_law``). It keeps the setup's identifiers, not the setup."""
    scenarios = {
        state.state_id: {law.origin_id: law.scenario_id for law in state.laws}
        for state in setup.joint_states
    }
    paths = {
        state.state_id: {path.entity_id for path in state.entity_paths}
        for state in setup.joint_states
    }

    def check(alternative_id: str, overrides: tuple[LawOverride, ...]) -> None:
        require_unique(
            [(row.state_id, row.law.origin_id) for row in overrides],
            "state/origin overrides",
        )
        for override in overrides:
            if (
                override.state_id not in scenarios
                or override.law.origin_id not in scenarios[override.state_id]
            ):
                raise ValueError(
                    "alternatives must replace a declared state/origin law"
                )
            if (
                override.law.scenario_id
                != scenarios[override.state_id][override.law.origin_id]
            ):
                raise ValueError(
                    "an alternative cannot silently replace its origin population"
                )
            if override.law.entity_id not in paths[override.state_id]:
                raise ValueError(
                    "alternative laws require their own declared entity path"
                )
            check_law(override.law)

    return check


def _laws_by_state(
    overrides: Iterable[LawOverride],
) -> dict[str, dict[str, IncidentLaw]]:
    result: dict[str, dict[str, IncidentLaw]] = {}
    for row in overrides:
        result.setdefault(row.state_id, {})[row.law.origin_id] = row.law
    return result


def _validate_matched_controls(setup: RunSetup) -> None:
    arms = {row.alternative_id: row for row in setup.alternatives}
    controls = {
        row.alternative_id: arms.get(row.matched_control_id)
        for row in setup.alternatives
        if row.matched_control_id is not None
    }
    if None in controls.values():
        raise ValueError("a matched control must be a declared alternative")
    all_origins = {law.origin_id for state in setup.joint_states for law in state.laws}
    for alternative in setup.alternatives:
        experiment = alternative.count_experiment
        if experiment is None:
            continue
        if alternative.purpose != "scientific_alternative":
            raise ValueError("count-dependence experiments are scientific alternatives")
        if set(experiment.origin_groups) != all_origins:
            raise ValueError(
                "count experiment groups must cover exactly the initiating origins"
            )
        control_arm = controls[alternative.alternative_id]
        control = None if control_arm is None else control_arm.count_experiment
        if control is None or control.kind != experiment.kind:
            raise ValueError("count experiments require their named matched control")
        if experiment.kind == "fixed_marginal_gaussian":
            if (
                control.correlation != 0
                or control.origin_groups != experiment.origin_groups
            ):
                raise ValueError(
                    "Gaussian count stress requires the same groups at zero correlation as control"
                )
        elif (
            control.coefficient_of_variation != experiment.coefficient_of_variation
            or len(set(control.origin_groups.values())) != len(control.origin_groups)
        ):
            raise ValueError(
                "shared gamma stress requires independent origin factors at the same CV as control"
            )
    for alternative in setup.alternatives:
        experiment = alternative.count_experiment
        if experiment is None:
            continue
        control = controls[alternative.alternative_id]
        assert control is not None
        if (
            alternative.state_weights
            or {row.state_id: row.weight for row in setup.joint_states}
        ) != (
            control.state_weights
            or {row.state_id: row.weight for row in setup.joint_states}
        ):
            raise ValueError(
                "matched count controls must retain the same joint-state weights"
            )
        # Read in turn: these keep the arm's laws while its control's are read.
        stressed = _laws_by_state(setup.overrides.of(alternative.alternative_id))
        controlled = _laws_by_state(setup.overrides.of(control.alternative_id))
        for state in setup.joint_states:
            baseline = {law.origin_id: law for law in state.laws}
            stress_laws = {**baseline, **stressed.get(state.state_id, {})}
            control_laws = {**baseline, **controlled.get(state.state_id, {})}
            if stress_laws != control_laws:
                raise ValueError(
                    "matched count controls must hold every conditional scientific incident law and mark kernel fixed"
                )


def _incident_law_check(setup: RunSetup) -> Callable[[IncidentLaw], None]:
    """The check of one incident law against the setup's rosters, for every
    reference law and every override."""
    entities = {row.entity_id for row in setup.scope.entities}
    roster = {row.entity_id: row for row in setup.scope.entities}
    scenarios = {row.scenario_id for row in setup.scope.populations}
    evidence = {row.evidence_id for row in setup.evidence.records}
    contract_map = {row.contract_id: row for row in setup.contracts}
    deployment_models = {
        (binding.deployment_id, binding.model_id) for binding in setup.scope.deployments
    }
    population_deployments = {
        population.scenario_id: population.deployment_ids
        for population in setup.scope.populations
    }

    def check(law: IncidentLaw) -> None:
        if law.entity_id not in entities or law.scenario_id not in scenarios:
            raise ValueError("incident law refers outside the entity/population roster")
        if (
            law.deployment_id is not None
            and (law.deployment_id, law.entity_id) not in deployment_models
        ):
            raise ValueError(
                "incident deployment must bind its originating model entity"
            )
        if (
            law.deployment_id is not None
            and population_deployments[law.scenario_id]
            and law.deployment_id not in population_deployments[law.scenario_id]
        ):
            raise ValueError("incident deployment lies outside its population scope")
        if any(
            lab_id not in roster or roster[lab_id].kind != "lab"
            for lab_id in (
                *law.unavailable_lab_losses,
                *(annual.lab_id for annual in law.annual_lab_losses),
            )
        ):
            raise ValueError(
                "unavailable and annual lab losses must refer to declared lab entities"
            )
        for segment in law.segments:
            kernel = segment.kernel
            if any(
                aggregate.lab_id not in roster or roster[aggregate.lab_id].kind != "lab"
                for aggregate in kernel.campaign_lab_losses
            ):
                raise ValueError(
                    "aggregate lab losses must refer to declared lab entities"
                )
            tags = {
                tag
                for component in kernel.campaign_components
                for tag in component.scenario_ids
            }
            outcomes = list(kernel.member_outcomes)
            for branch in kernel.explicit_joint_members:
                tags.update(
                    tag
                    for component in branch.campaign_components
                    for tag in component.scenario_ids
                )
                outcomes.extend(branch.members)
            for outcome in outcomes:
                tags.update(outcome.scenario_ids)
                tags.update(
                    tag
                    for component in outcome.components
                    for tag in component.scenario_ids
                )
            components = [*kernel.campaign_components]
            components.extend(
                component
                for branch in kernel.explicit_joint_members
                for component in branch.campaign_components
            )
            components.extend(
                component for outcome in outcomes for component in outcome.components
            )
            for component in components:
                if any(
                    entity_id not in roster or roster[entity_id].kind != "lab"
                    for entity_id in component.lab_loss_weights or ()
                ):
                    raise ValueError(
                        "lab loss allocations must refer to declared lab entities"
                    )
            if not tags <= scenarios:
                raise ValueError(
                    "scenario report tags must refer to declared populations"
                )
            mappings = [
                (
                    kernel.campaign_obligations,
                    kernel.campaign_payments,
                    kernel.campaign_memberships,
                    kernel.campaign_payment_kernel,
                    kernel.campaign_financial_mapping_complete,
                )
            ]
            mappings.extend(
                (
                    outcome.obligations,
                    outcome.payments,
                    outcome.memberships,
                    outcome.payment_kernel,
                    outcome.financial_mapping_complete,
                )
                for outcome in kernel.member_outcomes
            )
            for branch in kernel.explicit_joint_members:
                mappings.append(
                    (
                        branch.campaign_obligations,
                        branch.campaign_payments,
                        branch.campaign_memberships,
                        branch.campaign_payment_kernel,
                        branch.campaign_financial_mapping_complete,
                    )
                )
                mappings.extend(
                    (
                        member.obligations,
                        member.payments,
                        member.memberships,
                        member.payment_kernel,
                        member.financial_mapping_complete,
                    )
                    for member in branch.members
                )
            for (
                obligations,
                payments,
                memberships,
                payment_kernel,
                complete,
            ) in mappings:
                obligation_ids = {
                    (row.component_id, row.entity_id) for row in obligations
                }
                payment_ids = {(row.component_id, row.entity_id) for row in payments}
                if payment_kernel is not None:
                    inferred = {
                        (key, payment_kernel.entity_id)
                        for key in (
                            *payment_kernel.liability_component_ids,
                            *payment_kernel.direct_component_ids,
                        )
                    }
                    obligation_ids |= inferred
                    payment_ids |= inferred
                for membership in memberships:
                    contract = contract_map.get(membership.contract_id)
                    if contract is None:
                        raise ValueError(
                            "policy membership refers to an undeclared contract"
                        )
                    if membership.obligated_entity_id != contract.entity_id:
                        raise ValueError(
                            "policy membership entity differs from the contract entity"
                        )
                    if (
                        complete
                        and membership.eligible
                        and membership.covered_fraction > 0
                        and (
                            (
                                membership.component_id,
                                membership.obligated_entity_id,
                            )
                            not in obligation_ids
                            or (
                                membership.component_id,
                                membership.obligated_entity_id,
                            )
                            not in payment_ids
                        )
                    ):
                        raise ValueError(
                            "eligible coverage requires separate explicit obligation and payment outcomes"
                        )
        if not set(law.evidence_ids) <= evidence:
            raise ValueError("incident law refers to undeclared evidence")

    return check


def validate_setup_inputs(setup: RunSetup) -> ArmCheck:
    """Check the setup's inputs against each other, its alternatives'
    overrides apart, and give the check of one alternative's overrides
    (``_arm_check``), which the setup's ``overrides`` apply."""
    _validate_identifiers_and_weights(setup)
    _validate_rosters(setup)
    _validate_entity_paths(setup)
    check_law = _incident_law_check(setup)
    for state in setup.joint_states:
        for law in state.laws:
            check_law(law)
    _validate_alternatives(setup)
    return _arm_check(setup, check_law)


def validate_setup_arms(setup: RunSetup) -> None:
    """The checks that compare an arm with its matched control or with a
    conditional context. They read the overrides of the arms they concern,
    arm by arm, and leave none held."""
    _validate_matched_controls(setup)
    validate_conditional_contexts(setup)
    setup.overrides.release()
