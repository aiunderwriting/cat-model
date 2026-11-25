"""Canonical loss taxonomy and derived placement views.

The YAML contract owns the splitting rule, its conventions, the one admission
threshold, the tree and its population predicates; scenario files author only
terminal ``node_ids`` and state the admission threshold as ``{{admission}}``.
Children, terminal status, assignments, order and domains are derived. Every
split classifies one axis and names, in its ``split_rationale``, the estimate
that differs between its children. Every leaf is either estimated by exactly
one scenario or carries an unestimated disposition, never both.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from catmodel.hashing import canonical_json_sha256
from catmodel.resources import resource_path
from catmodel.scenarios.schema import FrozenModel, Prose, Text
from catmodel.scenarios.taxonomy_registry import RiskDomainName
from catmodel.yaml_io import safe_load

DEFAULT_HIERARCHY_CONTRACT_PATH = resource_path(
    "contracts", "scenarios", "hierarchy-contract.yaml"
)
# A monetary amount in text: the admission threshold belongs to the contract,
# so no scenario text may state another amount where an incident is defined.
MONETARY_AMOUNT = re.compile(
    r"(USD|US\$|\$)\s*\d|\d[\d,.]*\s*(thousand|million|billion|trillion)?\s*"
    r"(USD|dollars)\b",
    re.IGNORECASE,
)


class Clause(FrozenModel):
    axis: str | None = None
    value: Any | None = None
    value_in: tuple[Any, ...] | None = Field(default=None, min_length=1)
    negate: bool = False
    complement_of: tuple[str, ...] | None = Field(default=None, min_length=1)
    any_of: tuple[Clause, ...] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def one_form(self) -> Clause:
        forms = (
            self.axis is not None,
            self.complement_of is not None,
            self.any_of is not None,
        )
        if sum(forms) != 1:
            raise ValueError("clause must use exactly one form")
        if self.axis is not None and (self.value is None) == (self.value_in is None):
            raise ValueError("axis clause needs exactly one of value/value_in")
        return self


class AxisDef(FrozenModel):
    values: tuple[str, ...]
    set_valued: bool = False


class AdmissionThreshold(FrozenModel):
    """The one admission threshold every leaf applies to an incident's gross loss."""

    quantity_kind: Literal["gross_loss"]
    comparator: Literal["ge"]
    threshold: float = Field(gt=0, allow_inf_nan=False)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    measurement_basis: Literal["policy_start_usd_gross"]


class LargeLossLevels(FrozenModel):
    """Two fixed gross-loss amounts every leaf's direct question counts at.

    ``large_loss`` anchors the severity tail above the respondent's p99;
    ``very_large_loss`` enters no reference law and is the second anchor of
    the two-anchor comparison. Both use the admission measure and
    comparator: gross loss before recovery, at least the amount, in
    policy-start currency.
    """

    quantity_kind: Literal["gross_loss"]
    comparator: Literal["ge"]
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    measurement_basis: Literal["policy_start_usd_gross"]
    large_loss: float = Field(gt=0, allow_inf_nan=False)
    very_large_loss: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> LargeLossLevels:
        if not self.very_large_loss > self.large_loss:
            raise ValueError(
                "the very-large-loss level must exceed the large-loss level"
            )
        return self


# A terminal without a scenario owner is either a named population that no
# scenario estimates (``unestimated``) or a history with no named qualifying
# origin (``unsupported``). Neither has a numerical estimate; neither is zero.
UnownedTerminalDisposition = Literal["unestimated", "unsupported"]


class TerminalDisposition(FrozenModel):
    disposition: UnownedTerminalDisposition


class ClassificationQuestion(FrozenModel):
    """One existing authored fact classified by the immediate children.

    A split classifies a declared axis. Loss size is never a split: admission is
    the contract's one threshold, and how far an incident spreads is severity.
    """

    text: Text
    dimension: str

    @field_validator("dimension")
    @classmethod
    def axis_dimension(cls, value: str) -> str:
        if value.startswith("threshold:"):
            raise ValueError(
                f"split dimension {value!r} is a loss threshold; a split never "
                "classifies loss size, which the admission threshold and severity own"
            )
        if re.fullmatch(r"axis:[a-z][a-z0-9_]*", value) is None:
            raise ValueError(
                f"split dimension {value!r} must name a declared axis as axis:<name>"
            )
        return value


class HierarchyNode(FrozenModel):
    node_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    parent_id: str | None = None
    path_label: str
    display_label: str = Field(min_length=1)
    display_summary: str = Field(min_length=1)
    display_prose: str
    predicate: tuple[Clause, ...] = ()
    classification_question: ClassificationQuestion | None = None
    # Required on every split: the estimate (exposure unit, severity evidence or
    # capability response) that differs between its children.
    split_rationale: Prose | None = None
    reporting_domain: RiskDomainName | None = None
    residual: bool = False
    unestimated_disposition: TerminalDisposition | None = None


class HierarchyContract(FrozenModel):
    contract_id: str
    hierarchy_version: str
    # The splitting rule, quoted verbatim from methodology.md#loss-taxonomy.
    rule: str = Field(min_length=1)
    conventions: str = Field(min_length=1)
    admission: AdmissionThreshold
    large_loss_levels: LargeLossLevels
    axes: dict[str, AxisDef]
    nodes: tuple[HierarchyNode, ...]

    @model_validator(mode="after")
    def large_loss_levels_above_admission(self) -> HierarchyContract:
        levels, admission = self.large_loss_levels, self.admission
        if (
            levels.currency != admission.currency
            or levels.measurement_basis != admission.measurement_basis
            or not levels.large_loss > admission.threshold
        ):
            raise ValueError(
                "large-loss levels must exceed the admission threshold on its basis"
            )
        return self


class ScenarioPlacement(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def node_ids(self) -> tuple[str, ...]: ...


class HierarchyAuthority(FrozenModel):
    hierarchy_version: str
    hierarchy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HierarchyValidationResult(FrozenModel):
    errors: tuple[str, ...]


def load_hierarchy_contract(
    path: Path | str = DEFAULT_HIERARCHY_CONTRACT_PATH,
) -> HierarchyContract:
    resolved = Path(path)
    try:
        payload = safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("contract must be a YAML mapping")
        contract = HierarchyContract.model_validate(payload)
        errors = validate_hierarchy_contract(contract=contract).errors
        if errors:
            raise ValueError("invalid hierarchy: " + "; ".join(errors))
        return contract
    except Exception as exc:
        raise ValueError(f"Error loading hierarchy contract {resolved}: {exc}") from exc


@lru_cache(maxsize=1)
def cached_hierarchy_contract() -> HierarchyContract:
    return load_hierarchy_contract()


def nodes_by_id(contract: HierarchyContract) -> dict[str, HierarchyNode]:
    return {node.node_id: node for node in contract.nodes}


def children_by_id(contract: HierarchyContract) -> dict[str, tuple[str, ...]]:
    children: dict[str, list[str]] = {node.node_id: [] for node in contract.nodes}
    for node in contract.nodes:
        if node.parent_id in children:
            children[node.parent_id].append(node.node_id)
    return {node_id: tuple(values) for node_id, values in children.items()}


def terminal_node_ids(contract: HierarchyContract) -> frozenset[str]:
    return frozenset(
        node_id
        for node_id, children in children_by_id(contract).items()
        if not children
    )


def node_path(contract: HierarchyContract, node_id: str) -> str:
    nodes = nodes_by_id(contract)
    current = nodes[node_id]
    labels: list[str] = []
    seen: set[str] = set()
    while current.parent_id is not None:
        if current.node_id in seen:
            raise ValueError(f"cycle contains {current.node_id}")
        seen.add(current.node_id)
        labels.append(current.path_label)
        current = nodes[current.parent_id]
    return "/".join(reversed(labels))


def reporting_domain_of(contract: HierarchyContract, node_id: str) -> str:
    nodes = nodes_by_id(contract)
    current: HierarchyNode | None = nodes[node_id]
    while current is not None:
        if current.reporting_domain is not None:
            return current.reporting_domain
        current = nodes.get(current.parent_id) if current.parent_id else None
    raise ValueError(f"node {node_id!r} has no reporting-domain ancestor")


def origin_role_of(
    contract: HierarchyContract, node_id: str
) -> tuple[str | None, str | None]:
    """The origin and, below misuse or AI failure, what the AI did."""
    nodes = nodes_by_id(contract)
    origin: str | None = None
    role: str | None = None
    current: HierarchyNode | None = nodes[node_id]
    while current is not None:
        for clause in current.predicate:
            if clause.axis == "origin_regime" and not clause.negate:
                origin = str(clause.value)
            elif clause.axis in ("misuse_role", "failure_role") and not clause.negate:
                role = str(clause.value)
        current = nodes.get(current.parent_id) if current.parent_id else None
    return origin, role


def admission_clause(contract: HierarchyContract | None = None) -> str:
    """The admission threshold as the relative clause that ends a scenario's
    terminal-unit predicate, where the scenario writes ``{{admission}}``."""
    admission = (contract or cached_hierarchy_contract()).admission
    amount = f"{admission.currency} {admission.threshold:,.0f}"
    return (
        "whose gross loss, summed over all affected parties and loss kinds before "
        f"any recovery and valued in policy-start {admission.currency}, is at least "
        f"{amount} (exactly {amount} is admitted)"
    )


def admission_threshold(contract: HierarchyContract | None = None) -> float:
    """The one gross-loss amount at which every leaf admits an incident."""
    return (contract or cached_hierarchy_contract()).admission.threshold


def admission_statement(contract: HierarchyContract | None = None) -> str:
    """The admission threshold as a sentence, the same for every leaf."""
    admission = (contract or cached_hierarchy_contract()).admission
    return (
        "An incident is admitted when its gross loss, summed over all affected "
        "parties and loss kinds before any recovery and valued in policy-start "
        f"{admission.currency}, is at least {admission.currency} "
        f"{admission.threshold:,.0f}; exactly {admission.currency} "
        f"{admission.threshold:,.0f} is admitted."
    )


def large_loss_levels(contract: HierarchyContract | None = None) -> LargeLossLevels:
    """The two fixed amounts every leaf's large-loss counts use."""
    return (contract or cached_hierarchy_contract()).large_loss_levels


def scenario_placement_coordinates(
    contract: HierarchyContract, node_ids: Sequence[str]
) -> tuple[str | None, str | None, str]:
    """Resolve one forecast's common origin, AI role and reporting domain."""
    if not node_ids or len(node_ids) != len(set(node_ids)):
        raise ValueError("scenario placement needs nonempty unique terminal node_ids")
    coordinates = {
        (*origin_role_of(contract, node_id), reporting_domain_of(contract, node_id))
        for node_id in node_ids
    }
    if len(coordinates) != 1:
        raise ValueError(
            "scenario terminal union must share origin, AI role and reporting "
            f"domain; node_ids={tuple(node_ids)}"
        )
    return next(iter(coordinates))


def hierarchy_authority(
    contract: HierarchyContract | None = None,
) -> HierarchyAuthority:
    loaded = contract or cached_hierarchy_contract()
    semantic = loaded.model_dump(mode="json", exclude_none=True)
    return HierarchyAuthority(
        hierarchy_version=loaded.hierarchy_version,
        hierarchy_sha256=canonical_json_sha256(semantic),
    )


def _clause_errors(
    clause: Clause, contract: HierarchyContract, node_ids: frozenset[str]
) -> list[str]:
    if clause.any_of is not None:
        return [
            error
            for inner in clause.any_of
            for error in _clause_errors(inner, contract, node_ids)
        ]
    if clause.complement_of is not None:
        return [
            f"complement references unknown node {node_id}"
            for node_id in clause.complement_of
            if node_id not in node_ids
        ]
    if clause.axis is not None:
        definition = contract.axes.get(clause.axis)
        if definition is None:
            return [f"unknown axis {clause.axis}"]
        values = clause.value_in or (clause.value,)
        return [
            f"axis {clause.axis} has unknown value {value}"
            for value in values
            if value not in definition.values
        ]
    return []


def _classification_clause_errors(
    clause: Clause, dimension: str, siblings: frozenset[str]
) -> list[str]:
    """Check that a child clause uses the parent's dimension and complements only siblings."""
    if clause.any_of is not None:
        return [
            error
            for inner in clause.any_of
            for error in _classification_clause_errors(inner, dimension, siblings)
        ]
    if clause.complement_of is not None:
        invalid = sorted(set(clause.complement_of) - siblings)
        return [f"complement references non-siblings {invalid}"] if invalid else []
    actual = f"axis:{clause.axis}"
    return (
        []
        if actual == dimension
        else [f"child predicate uses {actual}, but parent classifies {dimension}"]
    )


def _ai_rd_sibling_severity_scope_errors(
    contract: HierarchyContract,
    scenarios: Sequence[ScenarioPlacement],
) -> list[str]:
    siblings: list[ScenarioPlacement] = []
    for scenario in scenarios:
        try:
            is_ai_rd = (
                scenario_placement_coordinates(contract, scenario.node_ids)[2]
                == "ai_rd"
            )
        except (KeyError, ValueError):
            # Placement validation reports the malformed assignment separately.
            continue
        if is_ai_rd:
            siblings.append(scenario)
    siblings.sort(key=lambda scenario: scenario.id)
    errors: list[str] = []
    exclusion_sets: dict[str, tuple[str, ...]] = {}
    for scenario in siblings:
        governance = getattr(scenario, "severity_governance", None)
        if governance is None:
            exclusion_sets[scenario.id] = ()
            continue

        exclusions = tuple(governance.severity_scope_exclusions)
        exclusion_sets[scenario.id] = exclusions
        if exclusions and governance.cap_posture == "uncapped":
            errors.append(
                f"{scenario.id}: AI R&D severity scope combines typed "
                "out-of-estimand exclusions with cap_posture='uncapped'; the "
                "shared ontology requires a finite base posture"
            )

    if len(set(exclusion_sets.values())) > 1:
        errors.append(
            "AI R&D siblings have divergent severity_scope_exclusions: "
            f"{dict(sorted(exclusion_sets.items()))}"
        )
    return errors


def validate_hierarchy_contract(
    *,
    contract: HierarchyContract | None = None,
    scenarios: Sequence[ScenarioPlacement] | None = None,
) -> HierarchyValidationResult:
    loaded = contract or cached_hierarchy_contract()
    errors: list[str] = []
    ids = [node.node_id for node in loaded.nodes]
    duplicates = sorted(node_id for node_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate node ids: {duplicates}")
    nodes = nodes_by_id(loaded)
    children = children_by_id(loaded)
    roots = [node.node_id for node in loaded.nodes if node.parent_id is None]
    if len(roots) != 1:
        errors.append(f"expected one root; got {roots}")
    for node in loaded.nodes:
        if node.parent_id is not None and node.parent_id not in nodes:
            errors.append(f"{node.node_id} has unknown parent {node.parent_id}")
        if node.parent_id is not None and not node.predicate:
            errors.append(f"{node.node_id}: non-root nodes need a local predicate")
        child_ids = children.get(node.node_id, ())
        question = node.classification_question
        if child_ids:
            if len(child_ids) < 2:
                errors.append(
                    f"{node.node_id}: classification requires at least two children"
                )
            if question is None:
                errors.append(
                    f"{node.node_id}: internal node needs a classification question"
                )
            else:
                if question.dimension.removeprefix("axis:") not in loaded.axes:
                    errors.append(
                        f"{node.node_id}: unknown classification dimension {question.dimension}"
                    )
                for child_id in child_ids:
                    for clause in nodes[child_id].predicate:
                        errors.extend(
                            f"{child_id}: {error}"
                            for error in _classification_clause_errors(
                                clause,
                                question.dimension,
                                frozenset(child_ids) - {child_id},
                            )
                        )
            if node.split_rationale is None:
                errors.append(
                    f"{node.node_id}: a split needs a split_rationale naming the "
                    "estimate that differs between its children"
                )
        elif question is not None or node.split_rationale is not None:
            errors.append(
                f"{node.node_id}: a terminal node cannot carry a classification "
                "question or split_rationale"
            )
        for clause in node.predicate:
            errors.extend(
                f"{node.node_id}: {error}"
                for error in _clause_errors(clause, loaded, frozenset(ids))
            )
        try:
            node_path(loaded, node.node_id)
        except (KeyError, ValueError) as exc:
            errors.append(f"{node.node_id}: {exc}")
        if children.get(node.node_id) and node.unestimated_disposition is not None:
            errors.append(
                f"{node.node_id}: only terminal nodes may carry an unestimated disposition"
            )
        sibling_ids = set(children.get(node.parent_id or "", ()))
        for clause in node.predicate:
            if clause.complement_of is None:
                continue
            invalid_complements = sorted(set(clause.complement_of) - sibling_ids)
            if invalid_complements:
                errors.append(
                    f"{node.node_id}: complement references non-siblings "
                    f"{invalid_complements}"
                )

    sibling_labels: dict[tuple[str | None, str], list[str]] = {}
    residuals_by_parent: dict[str | None, list[str]] = {}
    for node in loaded.nodes:
        sibling_labels.setdefault((node.parent_id, node.path_label), []).append(
            node.node_id
        )
        if node.residual:
            residuals_by_parent.setdefault(node.parent_id, []).append(node.node_id)
            complement_clauses = [
                clause for clause in node.predicate if clause.complement_of is not None
            ]
            expected_named_siblings = set(children.get(node.parent_id or "", ())) - {
                node.node_id
            }
            if (
                len(complement_clauses) != 1
                or set(complement_clauses[0].complement_of or ())
                != expected_named_siblings
            ):
                errors.append(
                    f"{node.node_id}: residual complement must name every other sibling"
                )
    for (parent_id, path_label), node_ids in sibling_labels.items():
        if len(node_ids) > 1:
            errors.append(
                f"siblings under {parent_id!r} reuse path label {path_label!r}: "
                f"{sorted(node_ids)}"
            )
    for parent_id, node_ids in residuals_by_parent.items():
        if len(node_ids) > 1:
            errors.append(
                f"parent {parent_id!r} has multiple residual children: "
                f"{sorted(node_ids)}"
            )

    if scenarios is not None:
        terminals = terminal_node_ids(loaded)
        node_ids = [node_id for scenario in scenarios for node_id in scenario.node_ids]
        if len(node_ids) != len(set(node_ids)):
            errors.append("scenario node assignments must be unique")
        for scenario in scenarios:
            invalid = [
                node_id for node_id in scenario.node_ids if node_id not in terminals
            ]
            if invalid:
                errors.append(
                    f"{scenario.id} must assign terminal nodes; got {invalid}"
                )
            try:
                scenario_placement_coordinates(loaded, scenario.node_ids)
            except (KeyError, ValueError) as exc:
                errors.append(f"{scenario.id}: {exc}")
        owners = {
            node_id: scenario.id
            for scenario in scenarios
            for node_id in scenario.node_ids
        }
        for terminal_id in sorted(terminals):
            disposition = nodes[terminal_id].unestimated_disposition
            if terminal_id not in owners and disposition is None:
                errors.append(
                    f"unassigned terminal {terminal_id} needs an explicit disposition"
                )
            elif terminal_id in owners and disposition is not None:
                errors.append(
                    f"terminal {terminal_id} is estimated by {owners[terminal_id]} but "
                    f"is marked {disposition.disposition}; remove its "
                    "unestimated_disposition"
                )
        errors.extend(_ai_rd_sibling_severity_scope_errors(loaded, scenarios))
    return HierarchyValidationResult(errors=tuple(errors))
