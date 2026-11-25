"""Whole-category EAL questions for the taxonomy parents an instrument declares.

Each parent is asked, in its own call that never sees its children's answers,
for the expected annual societal harm of its whole category and, when it has
modeled children, of the part they miss (the remainder). That answer and the
sum of its children's mean EALs plus the remainder are two views of one
quantity, compared as epistemic means, integrated over each stated uncertainty
law: means add whatever the dependence between the parts, and medians do not.
A fixed rule pools the two (``pool_parent``, ``HIER-CATEGORY-POOL``): the
pooled category value is their weighted geometric mean, with the category
answer's weight set by the parent's height (``category_pool_weight``). No
question asks the respondent which view to revise. Parents are pooled after
every parent below them, and leaves carry their pooled direct and chain counts
after the direct EAL cross-check. A pool's factor is a frequency factor on the
expected count of every leaf beneath the parent and scales every remainder at
or beneath it; remainders are reported, never simulated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import ClassVar, NamedTuple

from catmodel.contracts.contextual_risk import (
    ContextualParentNode,
    ContextualRiskInstrument,
    ContextualRiskNode,
    ParentAnswer,
    ParentPoolStatus,
)
from catmodel.contracts.elicitation import Reasoning, Respondent
from catmodel.elicitation.contextual_frequency import (
    CHAIN_SYSTEM_PROMPT,
)
from catmodel.elicitation.contextual_risk import (
    RenderedOnce,
)
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.elicitation.question_text import (
    SOCIETAL_HARM_BASIS,
    amounts_text,
    answer_schema_text,
    category_evidence_text,
    factor_shared,
    question_preamble,
    roles_text,
    status_rule,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import strict_json_loads
from catmodel.scenarios.base_rates import DatedEvidenceText
from catmodel.scenarios.hierarchy_contract import (
    admission_statement,
    cached_hierarchy_contract,
    children_by_id,
    nodes_by_id,
)
from catmodel.scenarios.loader import ScenarioDefinition
from catmodel.scenarios.threat_models import ThreatModel
from catmodel.validation_errors import EvidenceValidationError

ROOT_ID = "omega"


HIERARCHY_EAL_TOLERANCE = 0.01
"""Relative tolerance between a parent case's stored child EALs and pool and
their replay.

Stored values are rounded to four significant figures; a larger gap means
they do not come from these answers and this code.
"""


def _text(value: str) -> str:
    return " ".join(value.split())


def hierarchy_parents(
    nodes: Sequence[ContextualRiskNode],
    scenarios: Mapping[str, ScenarioDefinition],
) -> tuple[ContextualParentNode, ...]:
    """Every split whose subtree holds an estimated scenario, over an
    instrument's nodes, the root first and each parent before its descendants.

    A parent's children are its immediate children in the tree: each child
    split that is itself a parent, and each leaf whose scenario's node is
    collectible. Every other child, labelled in ``unmodeled_children``, falls
    into its remainder: an unestimated or unsupported leaf, a child split with
    no estimated scenario, and a leaf whose node cannot be collected. No split
    is skipped. A split's
    question would only repeat a child's if that child were its whole
    category, and every split has at least two children, so a split whose
    only estimated child is one leaf also has an unestimated child whose loss
    its remainder estimates. Every parent states the one admission threshold
    that every leaf shares (``HIER-THRESHOLD``). A scenario forecasting several
    leaves is refused: where it sits among the parents is decided when one is
    authored.
    """
    contract = cached_hierarchy_contract()
    tree = children_by_id(contract)
    labels = nodes_by_id(contract)
    threshold = admission_statement(contract)
    owners: dict[str, str] = {}
    for scenario in scenarios.values():
        if len(scenario.node_ids) != 1:
            raise ValueError(
                f"{scenario.id}: the taxonomy parents place only scenarios that "
                "forecast one leaf"
            )
        owners[scenario.node_ids[0]] = scenario.id
    collectible = {
        node.population.scenario_id: node.node_id
        for node in nodes
        if node.unavailable_reason is None and node.population.scenario_id in scenarios
    }

    @cache
    def estimated(node_id: str) -> bool:
        return node_id in owners or any(estimated(child) for child in tree[node_id])

    def modeled(child: str) -> str | None:
        if tree[child]:
            return child if estimated(child) else None
        return collectible.get(owners.get(child, ""))

    def roster(node_id: str) -> list[str]:
        if not (tree[node_id] and estimated(node_id)):
            return []
        return [node_id, *(found for child in tree[node_id] for found in roster(child))]

    return tuple(
        ContextualParentNode(
            parent_id=split,
            level="root" if split == ROOT_ID else "branch",
            label=labels[split].display_label,
            definition=_text(labels[split].display_prose),
            threshold=threshold,
            children=tuple(
                child
                for child in (modeled(child) for child in tree[split])
                if child is not None
            ),
            unmodeled_children=tuple(
                labels[child].display_label
                for child in tree[split]
                if modeled(child) is None
            ),
        )
        for split in roster(ROOT_ID)
    )


def category_text(
    instrument: ContextualRiskInstrument, parent: ContextualParentNode
) -> str:
    """A taxonomy parent's category, its modeled children, each defined once,
    and the categories directly beneath it that no modeled child covers."""
    nodes = {node.node_id: node for node in instrument.nodes}
    parents = {row.parent_id: row for row in instrument.parents}
    scenarios = [child for child in parent.children if child in nodes]
    categories = [child for child in parent.children if child not in nodes]
    own, shared = factor_shared(
        {
            child: nodes[child].population.definition
            if child in nodes
            else parents[child].definition
            for child in parent.children
        }
    )
    lines = [
        f"## Category {parent.parent_id}: {parent.label}",
        parent.definition,
        f"Admission: {parent.threshold}",
    ]
    if scenarios:
        lines.append("Modeled scenarios in it, each defined by its qualified unit:")
        lines.extend(f"- {child}: {own[child]}" for child in scenarios)
    if categories:
        lines.append("Modeled sub-categories in it:")
        lines.extend(
            f"- {child} ({parents[child].label}): {own[child]}" for child in categories
        )
    lines.extend(f"Also for {roles_text(keys)}: {text}" for keys, text in shared)
    if parent.unmodeled_children:
        lines.append(
            "Categories directly in it that no modeled scenario or sub-category "
            "covers: " + "; ".join(parent.unmodeled_children) + "."
        )
    currency = nodes[next(iter(nodes))].population.currency
    lines.append(amounts_text(currency))
    return "\n".join(lines)


def subtree_leaves(
    instrument: ContextualRiskInstrument, child_id: str
) -> tuple[str, ...]:
    """Every modeled node at or beneath a child: itself for a node, and each
    node in a parent's subtree for a parent."""
    parents = {row.parent_id: row for row in instrument.parents}
    if child_id not in parents:
        return (child_id,)
    return tuple(
        leaf
        for child in parents[child_id].children
        for leaf in subtree_leaves(instrument, child)
    )


class CategoryEvidence(NamedTuple):
    """The evidence a parent's question quotes, which is all its answer may
    cite."""

    cards: tuple[DatedEvidenceText, ...]
    """The source cards of the modeled scenarios beneath the parent, each
    once, in order of first appearance."""
    threat_models: tuple[tuple[str, ThreatModel], ...]
    """The node ID and the threat model of each of those scenarios that has
    one. The question quotes its summary only."""


def category_evidence(
    instrument: ContextualRiskInstrument, parent: ContextualParentNode
) -> CategoryEvidence:
    """The evidence of every modeled scenario beneath a parent, read from the
    instrument's nodes: the scenarios in the order ``subtree_leaves`` gives
    them, each one's source cards in its own order, and a card that several
    of them show once.

    An answer cites evidence by ID, so beneath one parent an ID names one
    thing. Two different cards with one ID, two threat models with one ID and
    an ID shared by a card and a threat model are refused.
    """
    nodes = {node.node_id: node for node in instrument.nodes}
    cards: dict[str, DatedEvidenceText] = {}
    threat_models: dict[str, tuple[str, ThreatModel]] = {}
    for leaf in subtree_leaves(instrument, parent.parent_id):
        node = nodes[leaf]
        for card in node.source_cards:
            if cards.setdefault(card.card_id, card) != card:
                raise ValueError(
                    f"category {parent.parent_id}: two different source cards "
                    f"beneath it share the ID {card.card_id!r}"
                )
        threat_model = node.threat_model
        if threat_model is not None:
            if threat_model.threat_model_id in threat_models:
                raise ValueError(
                    f"category {parent.parent_id}: two threat models beneath it "
                    f"share the ID {threat_model.threat_model_id!r}"
                )
            threat_models[threat_model.threat_model_id] = (leaf, threat_model)
    if shared := sorted(cards.keys() & threat_models.keys()):
        raise ValueError(
            f"category {parent.parent_id}: a source card and a threat model "
            f"beneath it share the IDs {shared}"
        )
    return CategoryEvidence(tuple(cards.values()), tuple(threat_models.values()))


def check_category_citations(reasoning: Reasoning, evidence: CategoryEvidence) -> None:
    """``evidence_used`` cites only what a parent's question quotes: its
    source cards and the threat models whose summaries it shows. The error
    locates the first other entry, so a retry says which one to change."""
    supplied = {card.card_id for card in evidence.cards} | {
        threat_model.threat_model_id for _, threat_model in evidence.threat_models
    }
    for index, identity in enumerate(reasoning.evidence_used):
        if identity not in supplied:
            raise EvidenceValidationError(
                "evidence_used_must_cite_supplied_cards",
                f"evidence_used cites {identity!r}, which this question does not supply",
                location=("reasoning", "evidence_used", index),
            )


def _remainder_task(parent: ContextualParentNode) -> str:
    if not parent.children:
        return (
            "remainder_eal: no part of this category is modeled, so the whole "
            "category is unmodeled: omit remainder_eal or set it to null."
        )
    outside = (
        "the categories listed above that no modeled scenario or sub-category covers"
        if parent.unmodeled_children
        else "only the losses that the modeled parts' definitions miss, which "
        "should be small"
    )
    return (
        "remainder_eal: uncertainty quantiles of the part of that expected harm "
        f"outside the modeled scenarios and sub-categories: {outside}. The "
        "modeled parts and the remainder together make up the whole category, "
        "each loss once. No estimate for any part is shown or asked here: this "
        "is a separate, top-down estimate, so do not build it by summing guesses "
        "for the parts."
    )


@dataclass(frozen=True)
class ContextualParentPrompt(RenderedOnce):
    """One parent's whole-category question at one coordinate.

    It quotes the evidence of the modeled scenarios beneath the parent
    (``category_evidence``): the source cards their own questions show, and
    the summary of each one's threat model.
    """

    system_prompt: ClassVar[str] = CHAIN_SYSTEM_PROMPT

    instrument: ContextualRiskInstrument
    parent: ContextualParentNode
    respondent: Respondent
    anchor_score: float

    def __post_init__(self) -> None:
        if (
            self.parent not in self.instrument.parents
            or self.respondent not in self.instrument.respondents
            or self.anchor_score not in self.instrument.anchor_scores
        ):
            raise ValueError("parent question is outside the frozen instrument")

    @property
    def call_id(self) -> str:
        return canonical_json_sha256(
            [
                self.instrument.context_id,
                self.parent.parent_id,
                self.anchor_score,
                "taxonomy_parent",
            ]
        )

    def _render(self) -> str:
        evidence = category_evidence(self.instrument, self.parent)
        return "\n\n".join(
            (
                question_preamble(self.instrument, self.anchor_score),
                category_text(self.instrument, self.parent),
                category_evidence_text(evidence.cards, evidence.threat_models),
                "## Questions\n"
                + "\n\n".join(
                    (
                        "category_eal: uncertainty quantiles of the expected annual "
                        "societal harm from the whole category above, for this lab "
                        "and deployment: the expected total over every in-scope "
                        "incident in the category that meets the admission "
                        f"threshold, each loss counted once. {SOCIETAL_HARM_BASIS}",
                        _remainder_task(self.parent),
                        status_rule(
                            whole="If the category cannot be estimated, return "
                            "status unavailable with only unavailable_reason and "
                            "reasoning."
                        ),
                    )
                ),
                answer_schema_text(ParentAnswer.json_schema_for(self.parent)),
            )
        )


def parent_prompts(
    instrument: ContextualRiskInstrument,
) -> tuple[ContextualParentPrompt, ...]:
    return tuple(
        ContextualParentPrompt(instrument, parent, respondent, score)
        for parent in instrument.parents
        for respondent in instrument.respondents
        for score in instrument.anchor_scores
    )


def parse_parent_answer(text: str, prompt: ContextualParentPrompt) -> ParentAnswer:
    answer = ParentAnswer.model_validate(strict_json_loads(text))
    answer.validate_against(prompt.parent)
    check_category_citations(
        answer.reasoning, category_evidence(prompt.instrument, prompt.parent)
    )
    return answer


def law_mean(law) -> float | None:
    """Epistemic mean of five stated quantiles, or None without a finite one."""
    if law is None:
        return None
    try:
        return epistemic_mean(law)
    except ValueError:
        return None


def finite_sum(values: Iterable[float]) -> float | None:
    """The sum of finite values, or None where it leaves the float range."""
    try:
        total = math.fsum(values)
    except OverflowError:
        return None
    return total if math.isfinite(total) else None


EXHAUSTED_PARENT_REASON = (
    "The parent question exhausted its validation attempts, so the parent has no "
    "answer to pool."
)


def category_pool_weight(height: int) -> float:
    """INITIAL_ESTIMATE weight of a parent's own category answer in its pool
    with what is beneath it (``HIER-CATEGORY-POOL``): one over the parent's
    height plus one.

    A parent whose modeled children are all leaves has height one and weight
    one half, and each level above takes a smaller share, so that along the
    tallest branch beneath a parent the leaves' sum and every category answer
    above it weigh equally. A judgment, not calibrated.
    """
    if height < 1:
        raise ValueError("a taxonomy parent's height is at least one")
    return 1 / (height + 1)


@dataclass(frozen=True)
class CategoryPool:
    """One parent's category answer pooled with what is beneath it.

    ``direct`` is the mean of the parent's category law and ``remainder`` of
    its remainder law, and ``children`` maps each child to its mean EAL going
    in: a leaf's own, and a parent child's pooled value. ``factor`` is
    ``(direct / modeled) ** weight``, so ``pooled`` is the weighted geometric
    mean of ``modeled`` (weight ``1 - weight``) and ``direct`` (``weight``).
    The factor is one unless ``status`` is ``pooled``, and ``reason`` then
    says why the parent is not pooled.
    """

    parent_id: str
    status: ParentPoolStatus
    reason: str | None
    height: int
    weight: float
    direct: float | None
    remainder: float | None
    children: Mapping[str, float | None]
    factor: float = 1.0

    @property
    def children_sum(self) -> float | None:
        """The sum of the children's mean EALs; None when one is missing or
        the sum is not finite."""
        values = list(self.children.values())
        if any(value is None for value in values):
            return None
        return finite_sum(value for value in values if value is not None)

    @property
    def modeled(self) -> float | None:
        """The children's mean EALs plus the remainder mean; None when any of
        them is missing or the sum is not finite."""
        children = self.children_sum
        if children is None or self.remainder is None:
            return None
        return finite_sum((children, self.remainder))

    @property
    def pooled(self) -> float | None:
        """The pooled category value: ``modeled`` times the factor."""
        modeled = self.modeled
        return None if modeled is None else modeled * self.factor


def pool_parent(
    parent: ContextualParentNode,
    answer: ParentAnswer | None,
    children: Mapping[str, float | None],
    height: int,
    *,
    weight: float | None = None,
) -> CategoryPool:
    """A parent's category answer pooled with its children plus its remainder
    (``HIER-CATEGORY-POOL``).

    ``children`` holds each child's mean EAL going in. The parent's direct
    and remainder means integrate its stated laws over their epistemic ranks
    (``epistemic_mean``): means add whatever the dependence between the
    parts, and a sum of medians is not the median of a sum. With ``D`` the
    direct mean and ``M`` the children's means plus the remainder mean, the
    factor is ``(D / M) ** weight`` whatever the size of the gap, so the
    pooled value is continuous in it. ``weight`` is the height's
    (``category_pool_weight``) unless given: zero leaves what is beneath the
    parent as it is, and one makes it sum to the category answer.

    A parent is not pooled, and its factor is one, when ``answer`` is None
    (its question exhausted its validation attempts), when ``D`` or ``M`` is
    not identified, or when either is zero: no factor scales zero harm, and a
    wholly zero answer is not pooled (as in ``chain_law.pooled_count``).
    """
    if weight is None:
        weight = category_pool_weight(height)
    if not 0 <= weight <= 1:
        raise ValueError("the category pool weight lies in [0, 1]")
    category = None if answer is None else answer.category_eal
    stated_remainder = None if answer is None else answer.remainder_eal
    direct, remainder = law_mean(category), law_mean(stated_remainder)
    missing = sorted(child for child in parent.children if children[child] is None)

    def pool(status: ParentPoolStatus, reason: str | None, factor: float = 1.0):
        return CategoryPool(
            parent.parent_id,
            status,
            reason,
            height,
            weight,
            direct,
            remainder,
            dict(children),
            factor,
        )

    if answer is None:
        return pool("parent_exhausted", EXHAUSTED_PARENT_REASON)
    reason = (
        "The category has no modeled child to pool with."
        if not parent.children
        else "The parent answer is unavailable."
        if category is None
        else "The parent answer has no remainder estimate."
        if stated_remainder is None
        else "The parent's category or remainder law has no finite mean."
        if direct is None or remainder is None
        else "No finite mean EAL is identified for children "
        + ", ".join(missing)
        + ": an answer or law is unavailable, or the severity law has no finite mean."
        if missing
        else None
    )
    if reason is not None:
        return pool("comparison_unavailable", reason)
    modeled = pool("pooled", None).modeled
    assert direct is not None
    if modeled is not None and (direct == 0 or modeled == 0):
        return pool(
            "not_pooled",
            "The category answer and everything beneath it have zero expected "
            "harm, so there is nothing to pool."
            if direct == modeled
            else "The category answer's mean is zero, and a wholly zero answer is "
            "not pooled: what is beneath it stands."
            if direct == 0
            else "Everything beneath the category has zero expected harm, which no "
            "factor can scale, so it stands.",
        )
    ratio = math.nan if modeled is None else direct / modeled
    if not 0 < ratio < math.inf:
        return pool(
            "comparison_unavailable",
            "The category answer and what is beneath it are further apart than "
            "the numerical range, so no factor is identified.",
        )
    return pool("pooled", None, ratio**weight)
