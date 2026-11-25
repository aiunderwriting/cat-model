"""Whole-category EAL questions for every estimated split of the taxonomy, and
the fixed pool of each category answer with what is beneath it."""

import json
import math
import re
from collections import Counter
from dataclasses import replace
from functools import cache
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.collection.contextual_collection import run_contextual_collection
from catmodel.config import ModelProfile
from catmodel.contracts.components import ComponentAmountLaw
from catmodel.contracts.contextual_risk import (
    ContextualRiskArtifact,
    ContextualRiskInstrument,
    ParentAnswer,
    ParentPool,
)
from catmodel.contracts.elicitation import QUANTILES, Quantiles
from catmodel.contracts.incident import IncidentLaw
from catmodel.elicitation.contextual_hierarchy import (
    ContextualParentPrompt,
    category_pool_weight,
    hierarchy_parents,
    pool_parent,
)
from catmodel.elicitation.contextual_risk import contextual_node_from_scenario
from catmodel.elicitation.eal_crosscheck import pooled_answer
from catmodel.elicitation.epistemic import epistemic_mean, epistemic_quantiles
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import ProviderResponse
from catmodel.probability.distributions import Distribution
from catmodel.resources import scenarios_path
from catmodel.results.results_summary import results_summary
from catmodel.scenarios.hierarchy_contract import (
    admission_statement,
    cached_hierarchy_contract,
    children_by_id,
)
from catmodel.scenarios.loader import load_scenarios
from catmodel.setup.contextual_hierarchy import (
    hierarchy_factor_limitation,
    hierarchy_outcomes,
    hierarchy_summary,
    parent_heights,
    parent_order,
    validate_hierarchy_cases,
)
from catmodel.setup.contextual_inputs import load_contextual_collection
from catmodel.setup.contextual_node import (
    CompileOptions,
    LeafFactors,
    _severity_scaled,
    admitted_contextual_node,
    compile_contextual_node,
    leaf_eal,
    leaf_eals,
)
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.excluded_samples import pooled_weights
from catmodel.setup.rank_spread import REFERENCE_SPREAD_TAIL_FORM
from test_support.capability_sources import write_source_bundle
from test_support.contextual_risk import (
    CHAIN_MARKER,
    contextual_answer,
    contextual_instrument,
    fixture_artifact,
    fixture_response,
    governed_contextual_answer,
    hierarchy_artifact,
    hierarchy_instrument,
    parent_answer,
    pool_artifact,
    pool_instrument,
)
from test_support.count_law_oracle import CountLawOracle
from tests.test_contextual_study import _capture

# Two collected leaves. Every split that holds an estimated scenario is still a
# parent, so most parents here have no modeled child.
LEAF = "compromise_organizations"
DESIGNATION = "government_designation"
SCENARIOS = (LEAF, DESIGNATION)
COMPROMISE = "n_compromise"  # LEAF's own parent, three splits below the root
ASSISTED = "n_misuse_assisted"
MISUSE = "n_misuse"
GOVERNANCE = "n_conduct"  # DESIGNATION's own parent, a top-level branch
CLAIMS = "n_conduct_private_claims"  # a parent child of GOVERNANCE
NO_CHILD = "n_fraud"  # a parent whose leaves are not collected here
REMAINDER = 1e9
CAPPED = "compromise_critical_infrastructure"  # a node whose lab cap binds
FLOORED = "enforcement_matter"  # a node with an inclusive admission floor


@cache
def _scenarios():
    return {row.id: row for row in load_scenarios(scenarios_path())}


def _splits() -> set[str]:
    """Every split of the current tree; each holds an estimated scenario."""
    return {
        node
        for node, children in children_by_id(cached_hierarchy_contract()).items()
        if children
    }


@cache
def _instrument():
    return hierarchy_instrument(SCENARIOS)


def _leaf(prompt):
    return governed_contextual_answer(prompt.node)


def _pooled_means(instrument) -> dict[str, float]:
    """Every node's mean EAL from its fixture case's pooled answer, each
    finite: what the node brings to its parent's pool."""
    means = {}
    for case in fixture_artifact(instrument, _leaf).cases:
        if case.anchor_score == 0.5:
            _, mean = leaf_eal(instrument, case.node_id, pooled_answer(case), 0.5)
            assert mean is not None
            means[case.node_id] = mean
    assert means.keys() == {node.node_id for node in instrument.nodes}
    return means


@cache
def _means():
    return _pooled_means(_instrument())


class _Answers:
    """Synthetic parent answers by an independent recursion over the tree.

    Each category answer is what lies beneath its parent going in, its
    children's values plus ``REMAINDER``, times the parent's entry in
    ``scales``. A leaf child's value is its mean EAL and a parent child's its
    pooled value. A parent whose answer is ``scale`` times what is beneath it
    has the factor ``scale ** (1 / (height + 1))``, which ``factor`` states
    from the scale and the height alone.
    """

    def __init__(self, instrument, means, *, scales=None):
        self.parents = {row.parent_id: row for row in instrument.parents}
        self.heights = parent_heights(instrument)
        self.means = means
        self.scales = scales or {}

    def modeled(self, parent_id):
        row = self.parents[parent_id]
        return math.fsum(self.value(child) for child in row.children) + REMAINDER

    def direct(self, parent_id):
        scale = self.scales.get(parent_id, 1.0)
        if not self.parents[parent_id].children:
            # No modeled child: the whole category is unmodeled.
            return scale * REMAINDER
        return scale * self.modeled(parent_id)

    def factor(self, parent_id):
        if not self.parents[parent_id].children:
            return 1.0
        return self.scales.get(parent_id, 1.0) ** (1 / (self.heights[parent_id] + 1))

    def value(self, child):
        if child not in self.parents:
            return self.means[child]
        if not self.parents[child].children:
            return self.direct(child)
        return self.modeled(child) * self.factor(child)

    def __call__(self, parent_id, _score):
        row = self.parents[parent_id]
        return parent_answer(
            self.direct(parent_id), REMAINDER if row.children else None
        )


def _outcome(instrument, artifact):
    return hierarchy_outcomes(instrument, artifact)[
        0.5, instrument.respondents[0].key()
    ]


def _law(outcome, parent_id):
    law = outcome.remainder_laws[parent_id]
    assert law is not None
    return law


def _frequency_record(instrument, artifact, **kwargs):
    setup = compile_contextual_setup(
        instrument,
        artifact,
        setup_id="hierarchy",
        years_per_state=1,
        **kwargs,
    )
    return next(
        cast(dict, row.acquisition["frequency_estimate"])
        for row in setup.evidence.records
        if row.acquisition and "frequency_estimate" in row.acquisition
    )


def _render(instrument, parent):
    return ContextualParentPrompt(
        instrument, parent, instrument.respondents[0], 0.5
    ).render()


UNMODELED = (
    "Categories directly in it that no modeled scenario or sub-category covers: "
)


def _unmodeled(text):
    """The unmodeled categories a parent question lists."""
    lines = [line for line in text.splitlines() if line.startswith(UNMODELED)]
    if not lines:
        return []
    (line,) = lines
    return line.removeprefix(UNMODELED).removesuffix(".").split("; ")


def _owning_parent(instrument, child):
    return next(row for row in instrument.parents if child in row.children)


def _tree_child(child):
    """A parent child's place in the tree: itself for a split, and the leaf
    its scenario forecasts for a node."""
    scenario = _scenarios().get(child)
    return child if scenario is None else scenario.node_ids[0]


def _represented(row):
    return {_tree_child(child) for child in row.children}


def _full_nodes():
    return [
        contextual_node_from_scenario(scenario, "deployment", "public")
        for scenario in _scenarios().values()
    ]


# The parents an instrument declares, and their question


def test_every_estimated_split_is_a_parent_over_its_immediate_children():
    nodes = _full_nodes()
    parents = {row.parent_id: row for row in hierarchy_parents(nodes, _scenarios())}
    tree = children_by_id(cached_hierarchy_contract())
    # Every split of the current tree holds an estimated scenario.
    assert set(parents) == _splits()
    # The root first, and each parent before the parents beneath it.
    order = [row.parent_id for row in hierarchy_parents(nodes, _scenarios())]
    assert order[0] == "omega"
    assert all(
        order.index(row.parent_id) < order.index(child)
        for row in parents.values()
        for child in row.children
        if child in parents
    )
    root = parents["omega"]
    assert root.level == "root"
    assert {row.level for row in parents.values() if row is not root} == {"branch"}
    assert root.children == (MISUSE, "n_failure", GOVERNANCE)
    assert parents[ASSISTED].children == (
        COMPROMISE,
        NO_CHILD,
        "influence_manipulation",
        "n_cbrn",
    )
    assert parents[GOVERNANCE].children == (
        DESIGNATION,
        "enforcement_matter",
        CLAIMS,
    )
    for row in parents.values():
        # Children are the parent's own children in the tree, in tree order: a
        # child split, or the node of the scenario that owns a leaf.
        assert [_tree_child(child) for child in row.children] == [
            child for child in tree[row.parent_id] if child in _represented(row)
        ]
    owned = [child for row in parents.values() for child in row.children]
    assert sorted(owned) == sorted(
        [*(node.node_id for node in nodes), *(set(parents) - {"omega"})]
    )
    threshold = admission_statement()
    assert {row.threshold for row in parents.values()} == {threshold}
    assert "at least USD 100,000" in threshold
    # A node that cannot be collected is no modeled child: its losses fall
    # into its parent's remainder.
    blocked = [
        node.model_copy(update={"unavailable_reason": "Synthetic gap"})
        if node.node_id == "input_hijack"
        else node
        for node in nodes
    ]
    takeover = next(
        row
        for row in hierarchy_parents(blocked, _scenarios())
        if row.parent_id == "n_misuse_takeover"
    )
    assert takeover.children == ("component_tampering",)
    payload = _instrument().model_dump(mode="json")
    owner = next(row for row in payload["parents"] if LEAF in row["children"])
    owner["children"] = []
    with pytest.raises(
        ValueError, match="every collectible node the child of exactly one parent"
    ):
        ContextualRiskInstrument.model_validate(payload)


def test_unestimated_and_unsupported_leaves_fall_into_their_parents_remainders():
    nodes = _full_nodes()
    parents = hierarchy_parents(nodes, _scenarios())
    tree = children_by_id(cached_hierarchy_contract())
    estimated = {leaf for row in _scenarios().values() for leaf in row.node_ids}
    omitted = {
        child
        for row in parents
        for child in tree[row.parent_id]
        if child not in _represented(row)
    }
    # With every scenario collected, the only children a parent omits are the
    # leaves no scenario estimates: the "other" leaves under open-ended
    # questions and the unsupported other-origin leaf.
    assert not omitted & estimated
    assert omitted == {
        node.node_id
        for node in cached_hierarchy_contract().nodes
        if node.unestimated_disposition is not None
    }
    assert "n_other_origin" in omitted
    # Each parent labels the categories directly beneath it that no modeled
    # child covers; nine of the fourteen have none.
    instrument = hierarchy_instrument(tuple(_scenarios()))
    unmodeled = {row.parent_id: row.unmodeled_children for row in instrument.parents}
    assert {parent: labels for parent, labels in unmodeled.items() if labels} == {
        "omega": ("Other origin",),
        ASSISTED: ("Other process",),
        "n_failure": ("Other AI failure",),
        "n_failure_service": ("Other service failure",),
        GOVERNANCE: ("Other process",),
    }
    assert sum(not labels for labels in unmodeled.values()) == 9
    service = next(
        row for row in instrument.parents if row.parent_id == "n_failure_service"
    )
    text = _render(instrument, service)
    assert "remainder_eal: uncertainty quantiles" in text
    assert "the categories listed above that no modeled scenario" in text
    assert _unmodeled(text) == ["Other service failure"]
    assert "should be small" not in text
    # A parent whose every tree child is modeled asks for a small remainder,
    # never for a category one of its children already covers.
    compromise = next(row for row in instrument.parents if row.parent_id == COMPROMISE)
    text = _render(instrument, compromise)
    assert _unmodeled(text) == []
    assert (
        "only the losses that the modeled parts' definitions miss, which should "
        "be small" in text
    )
    misuse = next(row for row in instrument.parents if row.parent_id == MISUSE)
    # The other process beneath helped the attacker belongs to that child's
    # remainder, not to the misuse branch's.
    assert misuse.unmodeled_children == ()
    root = next(row for row in instrument.parents if row.level == "root")
    assert _unmodeled(_render(instrument, root)) == ["Other origin"]


def test_a_split_with_no_estimated_scenario_is_neither_a_parent_nor_a_child():
    """Without the CBRN scenarios, the CBRN split and its biological split hold
    no estimated scenario: neither is a parent, and the split above labels
    the CBRN category among those its remainder estimates."""
    cbrn = {
        scenario_id
        for scenario_id, row in _scenarios().items()
        if row.node_ids[0].startswith("n_cbrn")
    }
    assert len(cbrn) == 4
    scenarios = {key: row for key, row in _scenarios().items() if key not in cbrn}
    nodes = [node for node in _full_nodes() if node.node_id not in cbrn]
    parents = {row.parent_id: row for row in hierarchy_parents(nodes, scenarios)}
    assert "n_cbrn" not in parents and "n_cbrn_biological" not in parents
    assert len(parents) == 12
    assisted = parents[ASSISTED]
    assert assisted.children == (COMPROMISE, NO_CHILD, "influence_manipulation")
    assert assisted.unmodeled_children == ("CBRN weapon", "Other process")


def test_no_split_is_skipped_and_a_multi_leaf_scenario_is_refused():
    """A split whose only estimated child is one leaf still has an
    unestimated child, since every split has at least two, so it stays a
    parent and its remainder estimates that child's loss."""
    nodes = _full_nodes()
    service = next(
        row
        for row in hierarchy_parents(nodes, _scenarios())
        if row.parent_id == "n_failure_service"
    )
    tree = children_by_id(cached_hierarchy_contract())
    assert service.children == ("service_outage_degradation",)
    assert len(tree["n_failure_service"]) == 2
    # Where a scenario forecasting several leaves would sit among the parents
    # is undecided, so the roster refuses one.
    scenarios = dict(_scenarios())
    scenarios["fraud_individuals"] = scenarios["fraud_individuals"].model_copy(
        update={"node_ids": ("n_fraud_individuals", "n_fraud_organizations")}
    )
    del scenarios["fraud_organizations"]
    with pytest.raises(ValueError, match="forecast one leaf"):
        hierarchy_parents(nodes, scenarios)


def test_parent_question_shows_definitions_not_answers():
    instrument = _instrument()
    parent = next(row for row in instrument.parents if row.parent_id == COMPROMISE)
    text = ContextualParentPrompt(
        instrument, parent, instrument.respondents[0], 0.5
    ).render()
    assert "category_eal" in text and "remainder_eal" in text
    assert "No estimate for any part is shown or asked here" in text
    assert "With status available, set unavailable_reason to null or omit it" in text
    assert (
        "No source excerpts are supplied for this question, so evidence_used is "
        "empty." in text
    )
    assert f"- {LEAF}: " in text
    assert "rate:" not in text and "total_loss" not in text
    # The one admission threshold is stated once.
    assert text.count(admission_statement()) == 1
    assisted = next(row for row in instrument.parents if row.parent_id == ASSISTED)
    text = ContextualParentPrompt(
        instrument, assisted, instrument.respondents[0], 0.5
    ).render()
    # A parent child shows its label and definition; its threshold is the
    # category's own, stated once.
    parents = {row.parent_id: row for row in instrument.parents}
    for child in (COMPROMISE, NO_CHILD, "n_cbrn"):
        assert f"- {child} ({parents[child].label}): " in text
    assert text.count(admission_statement()) == 1
    root = next(row for row in instrument.parents if row.level == "root")
    root_text = _render(instrument, root)
    assert "the categories listed above that no modeled scenario" in root_text
    assert _unmodeled(root_text) == ["Other origin"]


def test_parent_levels_must_match_the_tree():
    payload = _instrument().model_dump(mode="json")
    payload["parents"][1]["level"] = "root"
    with pytest.raises(ValueError, match="the only parent at level root"):
        ContextualRiskInstrument.model_validate(payload)


# The pool of one parent


def test_the_category_answer_weighs_one_over_its_height_plus_one():
    assert [category_pool_weight(height) for height in (1, 2, 3, 4, 5)] == [
        1 / 2,
        1 / 3,
        1 / 4,
        1 / 5,
        1 / 6,
    ]
    with pytest.raises(ValueError, match="at least one"):
        category_pool_weight(0)
    # A parent over leaves alone, or with no modeled child, has height one;
    # each parent above is one taller than its tallest parent child.
    heights = parent_heights(_instrument())
    assert {
        parent: heights[parent]
        for parent in (COMPROMISE, NO_CHILD, "n_cbrn", ASSISTED, MISUSE, "omega")
    } == {COMPROMISE: 1, NO_CHILD: 1, "n_cbrn": 2, ASSISTED: 3, MISUSE: 4, "omega": 5}
    assert heights[CLAIMS] == 1 and heights[GOVERNANCE] == 2
    # The parents are pooled lowest first.
    assert [
        {heights[row.parent_id] for row in level}
        for level in parent_order(_instrument())
    ] == [{1}, {2}, {3}, {4}, {5}]
    # Along the tallest branch the leaves' sum keeps the product of what each
    # pool leaves it, which is the weight of every category answer above it:
    # the root's own, and each lower answer's times what the pools above keep.
    branch = [heights[parent] for parent in ("n_cbrn_biological", "n_cbrn")]
    branch += [heights[parent] for parent in (ASSISTED, MISUSE, "omega")]
    assert branch == [1, 2, 3, 4, 5]
    kept = [1 - category_pool_weight(height) for height in branch]
    assert math.prod(kept) == pytest.approx(1 / 6)
    for index, height in enumerate(branch):
        assert category_pool_weight(height) * math.prod(
            kept[index + 1 :]
        ) == pytest.approx(1 / 6)


def test_a_parent_pools_to_the_weighted_geometric_mean_of_its_two_views():
    parent = next(row for row in _instrument().parents if row.parent_id == COMPROMISE)
    # Category answer 4e9; the leaf's 0.75e9 plus a remainder of 0.25e9.
    pool = pool_parent(parent, parent_answer(4e9, 0.25e9), {LEAF: 0.75e9}, 1)
    assert (pool.status, pool.reason) == ("pooled", None)
    assert pool.direct == pytest.approx(4e9)
    assert pool.modeled == pytest.approx(1e9)
    assert (pool.height, pool.weight) == (1, 0.5)
    # (4e9 / 1e9) ** (1 / 2) = 2, so the pooled value is sqrt(4e9 * 1e9).
    assert pool.factor == pytest.approx(2.0, rel=1e-12)
    assert pool.pooled == pytest.approx(2e9, rel=1e-12)
    # One level up the category answer weighs one third: 8 ** (1 / 3) = 2.
    taller = pool_parent(parent, parent_answer(8e9, 0.25e9), {LEAF: 0.75e9}, 2)
    assert taller.weight == 1 / 3
    assert taller.factor == pytest.approx(2.0, rel=1e-12)
    # The two ends of the weight: what is beneath stands, or sums to the
    # category answer.
    answer = parent_answer(4e9, 0.25e9)
    alone = pool_parent(parent, answer, {LEAF: 0.75e9}, 1, weight=0)
    assert (alone.status, alone.factor, alone.pooled) == ("pooled", 1.0, pool.modeled)
    only = pool_parent(parent, answer, {LEAF: 0.75e9}, 1, weight=1)
    assert only.factor == pytest.approx(4.0, rel=1e-12)
    assert only.pooled == pytest.approx(pool.direct, rel=1e-12)
    with pytest.raises(ValueError, match=r"lies in \[0, 1\]"):
        pool_parent(parent, answer, {LEAF: 0.75e9}, 1, weight=1.5)


@pytest.mark.parametrize("ratio", (1.0, 1.02, 2.9, 3.0, 3.1, 1e-3))
def test_every_comparable_parent_is_pooled_whatever_its_gap(ratio):
    """No threshold decides which parents are pooled, so the factor is
    continuous in the gap."""
    parent = next(row for row in _instrument().parents if row.parent_id == COMPROMISE)
    pool = pool_parent(parent, parent_answer(ratio * 1e9, 0.5e9), {LEAF: 0.5e9}, 1)
    assert pool.status == "pooled"
    assert pool.factor == pytest.approx(math.sqrt(ratio), rel=1e-12)


def test_a_zero_or_unidentified_side_is_not_pooled_and_says_why():
    parent = next(row for row in _instrument().parents if row.parent_id == COMPROMISE)

    def pooled(answer, mean):
        return pool_parent(parent, answer, {LEAF: mean}, 1)

    # A wholly zero category answer leaves what is beneath it standing.
    zero_answer = pooled(parent_answer(0, 0), 5.0)
    assert (zero_answer.status, zero_answer.factor) == ("not_pooled", 1.0)
    assert zero_answer.reason is not None
    assert "category answer's mean is zero" in zero_answer.reason
    # No factor scales zero harm up to a positive category answer.
    zero_beneath = pooled(parent_answer(5.0, 0), 0.0)
    assert (zero_beneath.status, zero_beneath.factor) == ("not_pooled", 1.0)
    assert zero_beneath.reason is not None
    assert "zero expected harm, which no factor can scale" in zero_beneath.reason
    both = pooled(parent_answer(0, 0), 0.0)
    assert (both.status, both.factor) == ("not_pooled", 1.0)
    assert both.reason is not None and "nothing to pool" in both.reason
    # An exhausted parent question has no answer to pool.
    exhausted = pooled(None, 5.0)
    assert (exhausted.status, exhausted.factor) == ("parent_exhausted", 1.0)
    assert (exhausted.direct, exhausted.modeled, exhausted.pooled) == (None, None, None)
    # Each way a mean is not identified is unavailable, with its reason.
    unavailable = ParentAnswer.model_validate(
        {
            "status": "unavailable",
            "unavailable_reason": "Synthetic gap.",
            "reasoning": parent_answer(1, 1).reasoning.model_dump(),
        }
    )
    reasons = {
        "The parent answer is unavailable.": pooled(unavailable, 5.0),
        "The parent answer has no remainder estimate.": pooled(
            parent_answer(3e9, None), 5.0
        ),
        "No finite mean EAL is identified for children "
        f"{LEAF}: an answer or law is unavailable, or the severity law has no "
        "finite mean.": pooled(parent_answer(3e9, 1e9), None),
    }
    for reason, pool in reasons.items():
        assert (pool.status, pool.factor) == ("comparison_unavailable", 1.0)
        assert pool.reason == reason
        assert pool.pooled is None
    # Two views further apart than a float ratio holds have no finite factor.
    apart = pooled(parent_answer(1e308, 0), 1e-10)
    assert (apart.status, apart.factor) == ("comparison_unavailable", 1.0)
    assert apart.reason is not None and "numerical range" in apart.reason


def test_the_stored_pool_record_states_its_status_coherently():
    pooled = {
        "status": "pooled",
        "height": 1,
        "weight": 0.5,
        "direct_eal_mean": 4e9,
        "modeled_eal_mean": 1e9,
        "factor": 2.0,
    }
    assert ParentPool.model_validate(pooled).reason is None
    unpooled = {
        **pooled,
        "status": "not_pooled",
        "reason": "Synthetic zero side.",
        "direct_eal_mean": 0.0,
        "factor": 1.0,
    }
    assert ParentPool.model_validate(unpooled).factor == 1
    for invalid, message in (
        ({**pooled, "reason": "Synthetic."}, "states its reason"),
        ({**unpooled, "reason": None}, "states its reason"),
        ({**unpooled, "factor": 2.0}, "has factor one"),
        ({**pooled, "modeled_eal_mean": None}, "positive direct and modeled"),
        ({**pooled, "direct_eal_mean": 0.0}, "positive direct and modeled"),
        ({**pooled, "height": 0}, "greater than or equal to 1"),
        ({**pooled, "weight": 1.5}, "less than or equal to 1"),
        ({**pooled, "factor": 0.0}, "greater than 0"),
        ({**pooled, "status": "reconciled"}, "Input should be"),
    ):
        with pytest.raises(ValidationError, match=message):
            ParentPool.model_validate(invalid)


def test_a_parent_with_no_modeled_child_is_asked_only_for_its_category():
    instrument = _instrument()
    other = next(row for row in instrument.parents if row.parent_id == NO_CHILD)
    assert other.children == ()
    text = ContextualParentPrompt(
        instrument, other, instrument.respondents[0], 0.5
    ).render()
    assert "omit remainder_eal" in text
    assert "remainder_eal: uncertainty quantiles" not in text
    with pytest.raises(ValueError, match="supplies none"):
        parent_answer(1e9, 1e8).validate_against(other)
    pool = pool_parent(other, parent_answer(1e9, None), {}, 1)
    assert (pool.status, pool.factor) == ("comparison_unavailable", 1.0)
    assert pool.reason == "The category has no modeled child to pool with."
    artifact = hierarchy_artifact(instrument, _leaf, _Answers(instrument, _means()))
    outcome = _outcome(instrument, artifact)
    # Its parent pools with the whole category as the child's value.
    assert outcome.pools[ASSISTED].children[NO_CHILD] == pytest.approx(REMAINDER)
    assert _law(outcome, NO_CHILD).p50 == pytest.approx(REMAINDER)
    assert outcome.pools[NO_CHILD].status == "comparison_unavailable"
    assert (
        outcome.pools[NO_CHILD].reason
        == "The category has no modeled child to pool with."
    )
    # Every parent with a modeled child is pooled; these answers agree with
    # what is beneath them, so no pool moves anything.
    pooled = [
        outcome.pools[row.parent_id] for row in instrument.parents if row.children
    ]
    assert {pool.status for pool in pooled} == {"pooled"}
    for pool in pooled:
        assert pool.factor == pytest.approx(1, rel=1e-12)
    assert all(
        factors.frequency == pytest.approx(1, rel=1e-12)
        for factors in outcome.factors.values()
    )


def test_the_pool_adds_means_not_medians():
    """Two heavy-tailed children with median EAL 1 each.

    Their independent sum has a median near 15, not 2, and a mean equal to
    the sum of their means. A parent who states that sum's law agrees with
    its children as means, so its pool moves them little; pooling it with a
    sum of medians would scale them up several times.
    """
    parent = next(row for row in _instrument().parents if row.parent_id == COMPROMISE)
    child = Quantiles(p5=0.001, p25=0.05, p50=1, p75=20, p95=1000)
    rng = np.random.default_rng(0)
    ranks = rng.random((2, 400_000)).clip(1e-12, 1 - 1e-12)
    total = epistemic_quantiles(child, ranks[0]) + epistemic_quantiles(child, ranks[1])
    quantiles = np.quantile(total, [0.05, 0.25, 0.5, 0.75, 0.95])
    category = Quantiles(**dict(zip(QUANTILES, quantiles, strict=True)))
    answer = ParentAnswer.model_validate(
        {
            "status": "available",
            "category_eal": category.model_dump(),
            "remainder_eal": Quantiles(p5=0, p25=0, p50=0, p75=0, p95=0).model_dump(),
            "reasoning": parent_answer(1, 1).reasoning.model_dump(),
        }
    )
    mean = epistemic_mean(child)
    pool = pool_parent(parent, answer, {LEAF: 2 * mean}, 1)
    assert pool.children_sum == 2 * mean
    assert pool.direct == pytest.approx(epistemic_mean(category))
    # The two views of the mean are within three times of each other, so the
    # factor is within the square root of three of one.
    assert abs(math.log(pool.factor)) < math.log(3) / 2
    # The sum of the children's medians, 2, lies more than three times below
    # the parent's median: a pool of medians would not agree.
    assert category.p50 > 3 * 2
    medians = pool_parent(parent, answer, {LEAF: 2.0}, 1)
    assert medians.factor > 2 * pool.factor


# Leaf EALs


def test_leaf_eals_are_count_median_and_count_mean_times_the_unit_count_mean():
    instrument = _instrument()
    answers = {
        node.node_id: governed_contextual_answer(node) for node in instrument.nodes
    }
    medians, means = leaf_eals(instrument, answers, 0.5)
    rate = answers[LEAF].rate
    assert rate is not None
    median, mean = medians[LEAF], means[LEAF]
    assert median is not None and mean is not None
    per_count = median / rate.p50
    assert mean == pytest.approx(epistemic_mean(rate) * per_count, rel=1e-12)
    assert mean > median


def test_leaf_eal_reads_the_count_at_the_options_spread_and_level_bias():
    instrument = _instrument()
    node = next(row for row in instrument.nodes if row.node_id == LEAF)
    answer = governed_contextual_answer(node)
    assert answer.rate is not None
    median, mean = leaf_eal(instrument, LEAF, answer, 0.5)
    biased_median, biased_mean = leaf_eal(
        instrument,
        LEAF,
        answer,
        0.5,
        options=CompileOptions(
            rank_spread=1.5,
            spread_tail_form=REFERENCE_SPREAD_TAIL_FORM,
            count_factor=0.8,
        ),
    )
    assert median is not None and mean is not None
    # The level bias scales every count quantile, so the median and the mean;
    # the spread keeps the median and moves the mean, here by the ratio of the
    # count law's means that CountLawOracle integrates.
    assert biased_median == pytest.approx(0.8 * median, rel=1e-15)
    stretched = CountLawOracle(
        answer.rate, spread=1.5, tail_form=REFERENCE_SPREAD_TAIL_FORM
    ).mean()
    stated = CountLawOracle(answer.rate, tail_form=REFERENCE_SPREAD_TAIL_FORM).mean()
    assert biased_mean == pytest.approx(0.8 * mean * stretched / stated, rel=1e-12)


def test_a_leaf_whose_law_the_run_would_refuse_has_no_stored_eal():
    """The stored parent values compile through the run's admission, so a law
    the run refuses gives no leaf EAL rather than a value the run never uses."""
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"][0]["population"].update(threshold=200, threshold_comparator="gt")
    refused = ContextualRiskInstrument.model_validate(payload)
    node_id = refused.nodes[0].node_id
    # The fixture's USD 120 per event never exceeds the USD 200 screen.
    assert leaf_eal(refused, node_id, contextual_answer(), 0.5) == (None, None)
    assert leaf_eal(contextual_instrument(), node_id, contextual_answer(), 0.5) == (
        100.0,
        100.0,
    )


# The pools of a tree, on the fixture whose leaves have exact EALs: leaf_a 100
# and leaf_b 200 under mid, with leaf_c 100 beside mid under the root, top.

EXACT_MID = (1600.0, 100.0)  # four times the 300 of its leaves plus 100
EXACT_TOP = (8000.0, 100.0)  # eight times mid's pooled 800 plus 100 plus 100


def _exact(mid=EXACT_MID, top=EXACT_TOP):
    """The exact fixture with each parent's (category, remainder) answer;
    None for an exhausted question."""
    instrument = pool_instrument()
    answers = [None if row is None else parent_answer(*row) for row in (mid, top)]
    return instrument, pool_artifact(instrument, *answers)


def test_two_levels_compound_with_weights_one_half_then_one_third():
    """mid, over leaves alone, pools 1600 with 400 at weight one half:
    factor 2, pooled 800. top, one level up, then pools 8000 with mid's 800
    plus leaf_c's 100 plus its remainder 100 at weight one third: factor 2,
    pooled 2000. The factors multiply down the tree."""
    instrument, artifact = _exact()
    validate_hierarchy_cases(artifact, instrument)
    assert parent_heights(instrument) == {"top": 2, "mid": 1}
    outcome = _outcome(instrument, artifact)
    mid, top = outcome.pools["mid"], outcome.pools["top"]
    assert (mid.status, mid.height, mid.weight) == ("pooled", 1, 1 / 2)
    assert mid.children == {"leaf_a": 100.0, "leaf_b": 200.0}
    assert (mid.direct, mid.modeled) == (1600.0, 400.0)
    assert mid.factor == pytest.approx(2.0, rel=1e-12)
    assert mid.pooled == pytest.approx(800.0, rel=1e-12)
    # top has a parent child and a leaf child: it is one taller than mid, and
    # mid goes in at its pooled value, the leaf at its own.
    assert (top.status, top.height, top.weight) == ("pooled", 2, 1 / 3)
    assert top.children == {"mid": pytest.approx(800.0, rel=1e-12), "leaf_c": 100.0}
    assert top.direct == 8000.0
    assert top.modeled == pytest.approx(1000.0, rel=1e-12)
    assert top.factor == pytest.approx(2.0, rel=1e-12)
    assert top.pooled == pytest.approx(2000.0, rel=1e-12)
    # A leaf's factor is the product over the parents above it; only the
    # count is scaled.
    assert {leaf: factors.severity for leaf, factors in outcome.factors.items()} == {
        "leaf_a": 1.0,
        "leaf_b": 1.0,
        "leaf_c": 1.0,
    }
    assert {
        leaf: factors.frequency for leaf, factors in outcome.factors.items()
    } == pytest.approx({"leaf_a": 4.0, "leaf_b": 4.0, "leaf_c": 2.0}, rel=1e-12)
    assert outcome.clamped == {}
    # Each parent case stores the child EALs going in and its pool.
    cases = {
        case.parent_id: case
        for case in artifact.parent_cases
        if case.anchor_score == 0.5
    }
    assert cases["mid"].children_median_eal == {"leaf_a": 100.0, "leaf_b": 200.0}
    assert cases["top"].children_median_eal == {"mid": None, "leaf_c": 100.0}
    assert cases["top"].children_mean_eal == {"mid": 800.0, "leaf_c": 100.0}
    assert cases["mid"].pool == ParentPool(
        status="pooled",
        height=1,
        weight=1 / 2,
        direct_eal_mean=1600.0,
        modeled_eal_mean=400.0,
        factor=2.0,
    )
    assert cases["top"].pool == ParentPool(
        status="pooled",
        height=2,
        weight=1 / 3,
        direct_eal_mean=8000.0,
        modeled_eal_mean=1000.0,
        factor=2.0,
    )


def test_the_exhibit_arms_are_the_pool_and_the_two_ends_of_its_weight():
    """Leaves alone (weight zero) total 100 + 200 + 100 = 400. The pool gives
    4 x 100 + 4 x 200 + 2 x 100 = 1400. With weight one mid's leaves sum to
    its 1600 less its remainder's share, a factor of 4, and top then scales
    1600 + 100 + 100 = 1800 to its 8000, a factor of 40 / 9."""
    instrument, artifact = _exact()
    outcome = _outcome(instrument, artifact)
    assert outcome.parents_only == pytest.approx(
        {"leaf_a": 160 / 9, "leaf_b": 160 / 9, "leaf_c": 40 / 9}, rel=1e-12
    )
    record = _frequency_record(instrument, artifact)
    welfare = record["views"]["welfare"]
    by_scenario = welfare["by_scenario"]
    arms = ("before_hierarchy", "final", "hierarchy_parents_only")
    assert {leaf: [by_scenario[leaf][arm] for arm in arms] for leaf in by_scenario} == {
        "leaf_a": pytest.approx([100.0, 400.0, 16000 / 9], rel=1e-12),
        "leaf_b": pytest.approx([200.0, 800.0, 32000 / 9], rel=1e-12),
        "leaf_c": pytest.approx([100.0, 200.0, 4000 / 9], rel=1e-12),
    }
    before, final, only = (
        welfare["by_arm"][arm]["expected_annual_loss"] for arm in arms
    )
    assert before == pytest.approx(400.0, rel=1e-12)
    assert final == pytest.approx(1400.0, rel=1e-12)
    # 8000 less the two remainders' shares: 8000 x (300 x 4 + 100) / 1800.
    assert only == pytest.approx(52000 / 9, rel=1e-12)
    assert before < final < only
    assert all(welfare["by_arm"][arm]["status"] == "available" for arm in arms)
    # The gross and lab views add each leaf's annual business loss, which
    # does not scale with its count, so they move less than in proportion.
    for view in ("gross", f"lab_loss::{instrument.lab.entity_id}"):
        rows = record["views"][view]["by_arm"]
        low, pooled, high = (rows[arm]["expected_annual_loss"] for arm in arms)
        assert low < pooled < high
        assert pooled < 3.5 * low
    # No follow-up call belongs to the taxonomy parents.
    assert set(record["follow_ups"]) == set(welfare["follow_up_effects"])
    assert "hierarchy" not in record["follow_ups"]
    summary = record["hierarchy"]
    assert {
        parent: (row["height"], row["weight"], row["cases_pooled"])
        for parent, row in summary.items()
    } == {"top": (2, 1 / 3, 3), "mid": (1, 1 / 2, 3)}


def test_across_levels_the_pool_can_lie_outside_the_two_ends_of_its_weight():
    """A category between the root and the leaves pulls the pool towards its
    own answer, which neither end of the weight keeps. mid answers 3 for
    leaves of 300, a factor of one tenth, and top answers the leaves' own
    400. Both ends of the weight then total 400, and the pool totals
    130 x (400 / 130) ** (1 / 3), about 189."""
    instrument, artifact = _exact(mid=(3.0, 0.0), top=(400.0, 0.0))
    validate_hierarchy_cases(artifact, instrument)
    outcome = _outcome(instrument, artifact)
    assert outcome.pools["mid"].factor == pytest.approx(0.1, rel=1e-12)
    assert outcome.pools["top"].modeled == pytest.approx(130.0, rel=1e-12)
    eals = {"leaf_a": 100.0, "leaf_b": 200.0, "leaf_c": 100.0}
    before = sum(eals.values())
    pooled = sum(outcome.factors[leaf].frequency * eal for leaf, eal in eals.items())
    only = sum(outcome.parents_only[leaf] * eal for leaf, eal in eals.items())
    assert before == 400.0
    assert only == pytest.approx(400.0, rel=1e-12)
    assert pooled == pytest.approx(130 * (400 / 130) ** (1 / 3), rel=1e-12)
    assert pooled < min(before, only)


def test_remainders_are_scaled_by_the_pools_and_never_simulated(tmp_path):
    """Each remainder is scaled by its own parent's factor and by every
    factor above it: mid's 100 by 2 x 2 and top's 100 by 2. They are reported
    lines; the run simulates the three leaves alone, 1400."""
    instrument, artifact = _exact()
    outcome = _outcome(instrument, artifact)
    assert _law(outcome, "mid") == Quantiles(**dict.fromkeys(QUANTILES, 400.0))
    assert _law(outcome, "top") == Quantiles(**dict.fromkeys(QUANTILES, 200.0))
    weights = pooled_weights(instrument.respondents, {})
    summary = hierarchy_summary(((artifact, instrument, weights),))
    lines = {parent: row["remainder_line"] for parent, row in summary.items()}
    assert {parent: line["status"] for parent, line in lines.items()} == {
        "mid": "estimated_unsimulated",
        "top": "estimated_unsimulated",
    }
    assert lines["mid"]["expected_annual_loss"] == pytest.approx(400.0)
    assert lines["top"]["expected_annual_loss"] == pytest.approx(200.0)
    # The leaves and the remainders beneath top make up its pooled 2000.
    leaves = 4 * 100 + 4 * 200 + 2 * 100
    assert leaves + 400 + 200 == 2000
    assert outcome.pools["top"].pooled == pytest.approx(2000, rel=1e-12)
    _, _, report, _ = _capture(tmp_path, instrument, artifact)
    metrics = report["arms"]["reference"]["metrics"]
    assert metrics["welfare"]["expected_annual_loss"] == pytest.approx(leaves)
    assert {
        leaf: metrics[f"scenario::{leaf}::welfare"]["expected_annual_loss"]
        for leaf in ("leaf_a", "leaf_b", "leaf_c")
    } == pytest.approx({"leaf_a": 400.0, "leaf_b": 800.0, "leaf_c": 200.0})


POOL_HEADING = "## Effect of the taxonomy parents' pools"


def _pool_section(markdown: str) -> list[str]:
    """The lines of the summary's section on the taxonomy parents' pools."""
    lines = markdown.split("\n")
    start = lines.index(POOL_HEADING)
    end = next(
        index
        for index in range(start + 1, len(lines))
        if lines[index].startswith("## ")
    )
    return lines[start:end]


def test_the_summary_reports_the_pools_effect_and_its_parent_cases(tmp_path):
    """The exact fixture's leaves total 400 before the pools, 1400 after them
    and 52000 / 9 with every category answer at weight one; both parents are
    pooled at each of its three coordinates."""
    instrument, artifact = _exact()
    (tmp_path / "pooled").mkdir()
    _, _, report, _ = _capture(tmp_path / "pooled", instrument, artifact)
    markdown, rows = results_summary(report)
    section = _pool_section(markdown)
    assert [line for line in section if line.startswith("| ")] == [
        "| Parent cases pooled | Parent cases not pooled | EAL before the pools | EAL after the pools | Change |",
        "| --- | --- | --- | --- | --- |",
        "| 6 | 0 | 400 | 1.4K | 250.0% |",
    ]
    text = "\n".join(section)
    assert "Of 6 parent cases, 6 were pooled and 0 were not" in text
    assert "weight one at every parent, the EAL is 5.78K." in text
    assert "with no further call" in text
    # The pool is no follow-up call: its effect has its own section, after
    # the follow-up calls' table, which has no row for it.
    lines = markdown.split("\n")
    follow_ups = lines.index("## Effect of follow-up calls")
    assert follow_ups < lines.index(POOL_HEADING)
    assert [
        line.split(" | ")[0]
        for line in lines[follow_ups : lines.index(POOL_HEADING)]
        if line.startswith("| ")
    ] == [
        "| Follow-up call",
        "| ---",
        "| Large-loss coherence retry",
        "| Chain guardrail retry",
        "| Leaf EAL cross-check",
    ]
    selected = {
        row["statistic"]: row for row in rows if row["section"] == "hierarchy_pool"
    }
    assert {name: row["value"] for name, row in selected.items()} == pytest.approx(
        {
            "cases_pooled": 6,
            "cases_not_pooled": 0,
            "expected_annual_loss_before": 400.0,
            "expected_annual_loss_after": 1400.0,
            "expected_annual_loss_parents_only": 52000 / 9,
            "relative_change": 2.5,
        }
    )
    assert {name: row["unit"] for name, row in selected.items()} == {
        "cases_pooled": "count",
        "cases_not_pooled": "count",
        "expected_annual_loss_before": "USD",
        "expected_annual_loss_after": "USD",
        "expected_annual_loss_parents_only": "USD",
        "relative_change": "ratio",
    }
    assert {row["status"] for row in selected.values()} == {"available"}
    assert {row["reason"] for row in selected.values()} == {None}
    assert selected["cases_pooled"]["view"] is None
    assert selected["relative_change"]["view"] == "welfare"
    assert not [row for row in rows if row["subject"] == "hierarchy"]
    # A parent that is not pooled is counted as such. mid's wholly zero
    # answer leaves its 300 standing, and top pools 4000 with 500 at factor
    # two, so the leaves' 400 double; with weight one top's factor is eight.
    instrument, artifact = _exact(mid=(0.0, 0.0), top=(4000.0, 100.0))
    (tmp_path / "zero").mkdir()
    _, _, report, _ = _capture(tmp_path / "zero", instrument, artifact)
    markdown, _ = results_summary(report)
    section = _pool_section(markdown)
    assert "| 3 | 3 | 400 | 800 | 100.0% |" in section
    text = "\n".join(section)
    assert "Of 6 parent cases, 3 were pooled and 3 were not" in text
    assert "weight one at every parent, the EAL is 3.2K." in text


def test_parents_that_are_not_pooled_leave_what_is_beneath_them_standing():
    weights = pooled_weights(pool_instrument().respondents, {})
    # mid's question exhausted its attempts: it has no answer to pool, and
    # top cannot add up a child without a value, so no leaf moves.
    instrument, artifact = _exact(mid=None)
    validate_hierarchy_cases(artifact, instrument)
    assert {row.kind for row in artifact.exhausted_calls()} == {"parent"}
    outcome = _outcome(instrument, artifact)
    assert outcome.pools["mid"].status == "parent_exhausted"
    top = outcome.pools["top"]
    assert top.status == "comparison_unavailable"
    assert top.reason is not None and "children mid" in top.reason
    assert set(outcome.factors.values()) == {LeafFactors()}
    assert outcome.remainder_laws["mid"] is None
    assert _law(outcome, "top").p50 == 100.0
    stored = {
        case.parent_id: case.pool
        for case in artifact.parent_cases
        if case.anchor_score == 0.5
    }
    assert stored["mid"] is not None and stored["top"] is not None
    assert stored["mid"].status == "parent_exhausted"
    assert (stored["mid"].direct_eal_mean, stored["mid"].factor) == (None, 1.0)
    assert stored["top"].modeled_eal_mean is None
    summary = hierarchy_summary(((artifact, instrument, weights),))
    assert summary["mid"]["cases_by_status"]["parent_exhausted"] == 3
    assert summary["top"]["cases_by_status"]["comparison_unavailable"] == 3
    assert (summary["mid"]["cases_pooled"], summary["mid"]["cases_not_pooled"]) == (
        0,
        3,
    )
    assert summary["mid"]["remainder_line"]["status"] == "unavailable"
    # Nor does the weight-one arm move a leaf.
    assert set(outcome.parents_only.values()) == {1.0}
    # A wholly zero category answer is not pooled either: the 300 beneath mid
    # stands, and top pools 4000 with 300 + 100 + 100 at factor two.
    instrument, artifact = _exact(mid=(0.0, 0.0), top=(4000.0, 100.0))
    validate_hierarchy_cases(artifact, instrument)
    outcome = _outcome(instrument, artifact)
    mid = outcome.pools["mid"]
    assert (mid.status, mid.factor) == ("not_pooled", 1.0)
    assert mid.reason is not None and "mean is zero" in mid.reason
    assert outcome.pools["top"].children["mid"] == 300.0
    assert outcome.pools["top"].factor == pytest.approx(2.0, rel=1e-12)
    assert {
        leaf: factors.frequency for leaf, factors in outcome.factors.items()
    } == pytest.approx({"leaf_a": 2.0, "leaf_b": 2.0, "leaf_c": 2.0}, rel=1e-12)
    summary = hierarchy_summary(((artifact, instrument, weights),))
    assert summary["mid"]["cases_by_status"]["not_pooled"] == 3
    assert summary["mid"]["absolute_log_ratio_before"] == {
        "median": None,
        "infinite": 3,
        "unavailable": 0,
    }
    # A results file holds no infinite number: each such case states None.
    assert {
        row["absolute_log_ratio_before"] for row in summary["mid"]["case_results"]
    } == {None}
    json.dumps(summary, allow_nan=False)
    assert summary["top"]["cases_pooled"] == 3


def test_a_parent_with_leaf_and_parent_children_of_different_heights():
    """The helped-attacker split has a leaf child, two parent children of
    height one and one of height two, so its height is three and its answer
    weighs one quarter. Each pool's factor is two: 4 ** (1/2) below, 8 **
    (1/3) at the CBRN split and 16 ** (1/4) here."""
    influence, chemical = "influence_manipulation", "cbrn_chemical"
    instrument = hierarchy_instrument((LEAF, influence, chemical, DESIGNATION))
    parents = {row.parent_id: row for row in instrument.parents}
    assert parents[ASSISTED].children == (COMPROMISE, NO_CHILD, influence, "n_cbrn")
    assert parents["n_cbrn"].children == ("n_cbrn_biological", chemical)
    heights = parent_heights(instrument)
    assert [heights[child] for child in (COMPROMISE, NO_CHILD, "n_cbrn")] == [1, 1, 2]
    assert heights[ASSISTED] == 3
    means = _pooled_means(instrument)
    answers = _Answers(
        instrument, means, scales={COMPROMISE: 4, "n_cbrn": 8, ASSISTED: 16}
    )
    artifact = hierarchy_artifact(instrument, _leaf, answers)
    validate_hierarchy_cases(artifact, instrument)
    outcome = _outcome(instrument, artifact)
    assert {
        parent: outcome.pools[parent].weight
        for parent in (COMPROMISE, "n_cbrn", ASSISTED)
    } == {COMPROMISE: 1 / 2, "n_cbrn": 1 / 3, ASSISTED: 1 / 4}
    for parent in (COMPROMISE, "n_cbrn", ASSISTED):
        assert outcome.pools[parent].factor == pytest.approx(2.0, rel=1e-9)
    # The parent children go in at their pooled values and the leaf at its
    # own mean EAL.
    going_in = outcome.pools[ASSISTED].children
    assert going_in[COMPROMISE] == pytest.approx(
        2 * (means[LEAF] + REMAINDER), rel=1e-9
    )
    assert going_in["n_cbrn"] == pytest.approx(
        2 * (means[chemical] + 2 * REMAINDER), rel=1e-9
    )
    assert going_in[NO_CHILD] == pytest.approx(REMAINDER)
    assert going_in[influence] == pytest.approx(means[influence], rel=1e-12)
    # Every other parent agrees with what is beneath it.
    for parent, pool in outcome.pools.items():
        if parent not in (COMPROMISE, "n_cbrn", ASSISTED) and parents[parent].children:
            assert pool.factor == pytest.approx(1.0, rel=1e-9), parent
    assert {
        leaf: factors.frequency for leaf, factors in outcome.factors.items()
    } == pytest.approx(
        {LEAF: 4.0, influence: 2.0, chemical: 4.0, DESIGNATION: 1.0}, rel=1e-9
    )
    # A pool also scales its own remainder and every remainder beneath it,
    # the whole category of a parent with no modeled child included.
    assert {
        parent: _law(outcome, parent).p50 / REMAINDER
        for parent in (
            COMPROMISE,
            "n_cbrn",
            "n_cbrn_biological",
            NO_CHILD,
            ASSISTED,
            MISUSE,
            GOVERNANCE,
        )
    } == pytest.approx(
        {
            COMPROMISE: 4.0,
            "n_cbrn": 4.0,
            "n_cbrn_biological": 4.0,
            NO_CHILD: 2.0,
            ASSISTED: 2.0,
            MISUSE: 1.0,
            GOVERNANCE: 1.0,
        },
        rel=1e-9,
    )


def test_the_root_s_answer_moves_every_leaf_of_the_full_roster():
    """The repository roster with every scenario collected. Each parent's
    answer weighs one over its height plus one, and a root answer 64 times
    what is beneath it, at weight one sixth, doubles every leaf and every
    remainder."""
    instrument = hierarchy_instrument(tuple(_scenarios()))
    heights = parent_heights(instrument)
    assert heights["omega"] == 5 and len(heights) == len(_splits())
    answers = _Answers(instrument, _pooled_means(instrument), scales={"omega": 64})
    artifact = hierarchy_artifact(instrument, _leaf, answers)
    validate_hierarchy_cases(artifact, instrument)
    outcome = _outcome(instrument, artifact)
    assert {pool.status for pool in outcome.pools.values()} == {"pooled"}
    assert {parent: pool.weight for parent, pool in outcome.pools.items()} == {
        parent: 1 / (height + 1) for parent, height in heights.items()
    }
    assert outcome.pools["omega"].factor == pytest.approx(2.0, rel=1e-9)
    assert set(outcome.factors) == {node.node_id for node in instrument.nodes}
    for leaf, factors in outcome.factors.items():
        assert factors.frequency == pytest.approx(2.0, rel=1e-9), leaf
    for parent in outcome.pools:
        assert _law(outcome, parent).p50 == pytest.approx(2 * REMAINDER, rel=1e-9)


# The clamp on a leaf's cumulative factor


def test_cumulative_factors_are_clamped_to_the_range_of_one_factor():
    assert LeafFactors(2.0, 0.5).bounded() == LeafFactors(2.0, 0.5)
    assert LeafFactors(1e6, 1e-6).bounded() == LeafFactors(1e4, 1e-4)
    # A computed product within the tie tolerance of a bound stands.
    tie = LeafFactors(1e4 * (1 + 1e-12), 1e-4 * (1 - 1e-12))
    assert tie.bounded() == tie


def test_a_binding_clamp_is_applied_in_the_run_and_disclosed():
    """mid's answer is 1e12 times the 400 beneath it, so its pool's factor is
    1e6, beyond the range of a leaf's factor: its leaves are scaled by 1e4 and
    its remainder by the whole 1e6. top's answer agrees with what is then
    beneath it. The exhibit lists the clamp and the result discloses it."""
    clamped_mid = 1e4 * (100 + 200) + 1e6 * 100
    instrument, artifact = _exact(
        mid=(4e14, 100.0), top=(clamped_mid + 100 + 100, 100.0)
    )
    validate_hierarchy_cases(artifact, instrument)
    outcome = _outcome(instrument, artifact)
    assert outcome.pools["mid"].factor == pytest.approx(1e6, rel=1e-12)
    assert outcome.pools["top"].children["mid"] == pytest.approx(clamped_mid)
    assert outcome.pools["top"].factor == pytest.approx(1.0, rel=1e-12)
    assert {
        leaf: factors.frequency for leaf, factors in outcome.factors.items()
    } == pytest.approx({"leaf_a": 1e4, "leaf_b": 1e4, "leaf_c": 1.0}, rel=1e-12)
    assert set(outcome.clamped) == {"leaf_a", "leaf_b"}
    for product in outcome.clamped.values():
        assert product.frequency == pytest.approx(1e6, rel=1e-12)
        assert product.severity == 1.0
    assert _law(outcome, "mid").p50 == pytest.approx(1e8, rel=1e-12)
    setup = compile_contextual_setup(
        instrument, artifact, setup_id="clamped", years_per_state=1
    )
    record = next(
        cast(dict, row.acquisition["frequency_estimate"])
        for row in setup.evidence.records
        if row.acquisition and "frequency_estimate" in row.acquisition
    )
    view = record["views"]["welfare"]["by_scenario"]
    assert view["leaf_a"]["final"] == pytest.approx(
        1e4 * view["leaf_a"]["before_hierarchy"], rel=1e-9
    )
    assert view["leaf_c"]["final"] == pytest.approx(view["leaf_c"]["before_hierarchy"])
    summary = record["hierarchy"]
    anchors = len(instrument.anchor_scores)
    # The clamp binds beneath mid after its pool, and still beneath top.
    assert summary["mid"]["cases_clamped"] == anchors
    assert summary["top"]["cases_clamped"] == anchors
    clamp = summary["mid"]["case_results"][0]["clamped_leaves"]
    assert set(clamp) == {"leaf_a", "leaf_b"}
    assert clamp["leaf_a"] == {
        "cumulative_factor": pytest.approx(1e6, rel=1e-12),
        "applied_factor": 1e4,
    }
    weights = pooled_weights(instrument.respondents, {})
    (disclosure,) = hierarchy_factor_limitation(((artifact, instrument, weights),))
    assert "clamped to 0.0001 to 10,000" in disclosure
    assert f"binds in {2 * anchors} leaf cases" in disclosure
    assert "leaves: leaf_a, leaf_b." in disclosure
    assert disclosure in setup.limitations
    # Without a binding clamp nothing is disclosed.
    instrument, unclamped = _exact()
    assert hierarchy_factor_limitation(((unclamped, instrument, weights),)) == ()


# Stored values, replay and collection


def test_validation_holds_stored_child_eals_and_pools_to_their_replay():
    instrument, artifact = _exact()
    validate_hierarchy_cases(artifact, instrument)
    payload = artifact.model_dump(mode="json")

    def changed(edit) -> ContextualRiskArtifact:
        tampered = json.loads(json.dumps(payload))
        for case in tampered["parent_cases"]:
            if case["parent_id"] == "mid":
                edit(case)
        return ContextualRiskArtifact.model_validate(tampered)

    # Stored child EALs within tolerance stand; others do not come from these
    # answers.
    for factor, accepted in ((1.005, True), (1.05, False)):
        for field in ("children_median_eal", "children_mean_eal"):

            def scale(case, field=field, factor=factor):
                case[field] = {
                    key: None if value is None else value * factor
                    for key, value in case[field].items()
                }

            if accepted:
                validate_hierarchy_cases(changed(scale), instrument)
            else:
                with pytest.raises(ValueError, match="stored child EALs of mid"):
                    validate_hierarchy_cases(changed(scale), instrument)
    # Nor does a pool with another factor, mean, height, weight or status.
    for update in (
        {"factor": 2.1},
        {"direct_eal_mean": 1700.0},
        {"modeled_eal_mean": 430.0},
        {"height": 2},
        {"weight": 1 / 3},
        {"status": "not_pooled", "reason": "Synthetic zero side.", "factor": 1.0},
    ):
        with pytest.raises(ValueError, match="stored pool of mid differs"):
            validate_hierarchy_cases(
                changed(lambda case, update=update: case["pool"].update(update)),
                instrument,
            )
    # A factor and means within tolerance of the replay stand.
    validate_hierarchy_cases(
        changed(
            lambda case: case["pool"].update(
                factor=2.01, direct_eal_mean=1601.0, modeled_eal_mean=400.4
            )
        ),
        instrument,
    )
    # A complete group's parent case stores its pool.
    with pytest.raises(ValidationError, match="child means and pool together"):
        changed(lambda case: case.update(pool=None))
    # An exhausted parent, and only one, is recorded as parent_exhausted.
    with pytest.raises(ValidationError, match="parent_exhausted exactly when"):
        changed(lambda case: case.update(answer=None))


def _pooled_artifact(instrument):
    """A capture whose leaf's own parent and its top-level branch each double
    what is beneath them: 4 ** (1/2) and, at height four, 32 ** (1/5)."""
    answers = _Answers(instrument, _means(), scales={COMPROMISE: 4, MISUSE: 32})
    return hierarchy_artifact(instrument, _leaf, answers)


def test_selections_from_a_checked_collection_s_outcomes_are_the_same(monkeypatch):
    """A caller that checked a collection where it read it carries the
    outcomes to where the selections are made, across processes: they are the
    same selections, and nothing is checked or compiled for them again."""
    import pickle

    import catmodel.worker_pool  # noqa: F401  (read-only mappings pickle)
    from catmodel.setup import contextual_setup
    from catmodel.setup.contextual_setup import (
        checked_hierarchy_outcomes,
        contextual_respondent_selections,
    )

    instrument = _instrument()
    artifact = _pooled_artifact(instrument)
    weights = pooled_weights(instrument.respondents, {})
    checked = contextual_respondent_selections(instrument, artifact, weights)
    outcomes = pickle.loads(
        pickle.dumps(checked_hierarchy_outcomes(instrument, artifact))
    )
    # The outcomes carry the factor the selections apply.
    for row in outcomes.values():
        assert row.factors[LEAF].frequency == pytest.approx(4.0, rel=1e-9)
        assert row.factors[DESIGNATION].frequency == pytest.approx(1.0, rel=1e-9)
    case = next(
        row for row in artifact.cases if row.node_id == LEAF and row.anchor_score == 0.5
    )
    before, selected = pooled_answer(case).rate, checked[0].answers_by_node[LEAF][0.5]
    assert before is not None and selected.rate is not None
    assert selected.rate.p50 == pytest.approx(4 * before.p50, rel=1e-9)
    assert checked[0].severity_factors_by_node[LEAF][0.5] == 1.0

    def refuse(*_args, **_kwargs):
        raise AssertionError("a checked collection is not checked again")

    for name in (
        "validate_contextual_artifact",
        "validate_eal_crosschecks",
        "validate_hierarchy_cases",
        "hierarchy_outcomes",
    ):
        monkeypatch.setattr(contextual_setup, name, refuse)
    assert (
        contextual_respondent_selections(
            instrument, artifact, weights, outcomes=outcomes
        )
        == checked
    )
    # Without them, making the selections checks the collection.
    with pytest.raises(AssertionError, match="not checked again"):
        contextual_respondent_selections(instrument, artifact, weights)


def test_unit_laws_worked_out_in_one_process_are_kept_in_another(monkeypatch):
    """Checking a collection works out every case's unit law. They go back to
    the importing process with the collection, which then compiles none of
    them again for the checks, outcomes and summaries it runs."""
    import pickle

    from catmodel.setup import contextual_node
    from catmodel.setup.contextual_node import keep_unit_laws, unit_laws_of

    instrument = _instrument()
    artifact = _pooled_artifact(instrument)
    monkeypatch.setattr(contextual_node, "_UNIT_LAWS", {})
    validate_hierarchy_cases(artifact, instrument)
    outcomes = hierarchy_outcomes(instrument, artifact)
    laws = unit_laws_of(instrument)
    assert laws
    assert {query.digest for query in laws} == {instrument.digest()}
    assert unit_laws_of(contextual_instrument()) == {}
    # As the importing process receives them, with the instrument they name.
    received_instrument, received = pickle.loads(pickle.dumps((instrument, laws)))
    assert received == laws

    def refuse(_query):
        raise AssertionError("a kept unit law is not compiled again")

    monkeypatch.setattr(contextual_node, "_UNIT_LAWS", {})
    monkeypatch.setattr(contextual_node, "_unit_law", refuse)
    with pytest.raises(AssertionError, match="not compiled again"):
        validate_hierarchy_cases(artifact, received_instrument)
    keep_unit_laws(received)
    validate_hierarchy_cases(artifact, received_instrument)
    assert hierarchy_outcomes(received_instrument, artifact) == outcomes


class _Provider:
    """A fake transport over one instrument: governed node answers and parent
    answers from ``answers``. Every question is one of the three kinds a
    collection opens with."""

    model_id = "fixture-model"
    provider_name = "openai"
    reasoning_effort = "low"

    def __init__(self, instrument, answers):
        self.nodes = {row.node_id: row for row in instrument.nodes}
        self.answers = answers
        self.calls = []
        self.parents = Counter()

    async def complete(self, *, user_prompt, **_kwargs):
        if "societal harm from the whole category above" in user_prompt:
            found = re.search(r"^## Category ([a-z_]+):", user_prompt, re.MULTILINE)
            assert found is not None
            kind = "parent"
            self.parents[found.group(1)] += 1
            text = self.answers(found.group(1), 0).model_dump_json()
        else:
            node = re.search(r"^## Scenario ([a-z_]+)$", user_prompt, re.MULTILINE)
            assert node is not None
            answer = governed_contextual_answer(self.nodes[node.group(1)])
            kind = "chain" if CHAIN_MARKER in user_prompt else "direct"
            text = fixture_response(user_prompt, answer)
        self.calls.append(kind)
        return ProviderResponse(
            text=text,
            model_id=self.model_id,
            provider_name=self.provider_name,
            input_tokens=5,
            output_tokens=10,
        )


def _collect(tmp_path, monkeypatch, base, answers):
    bundle = write_source_bundle(tmp_path / "source")
    payload = base.model_dump(mode="json")
    payload["source_bundle_manifest_sha256"] = capture_source_bundle_snapshot(
        bundle
    ).manifest_sha256
    for node in payload["nodes"]:
        node["source_cards"] = []
    instrument = ContextualRiskInstrument.model_validate(payload)
    provider = _Provider(instrument, answers)
    profile = ModelProfile(
        models=(
            ModelSpec(
                provider="openai",
                model_id="fixture-model",
                reasoning_effort="low",
                max_output_tokens=100_000,
            ),
        )
    )
    monkeypatch.setattr("catmodel.config.load_model_profile", lambda **_kwargs: profile)
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers", lambda *_args, **_kwargs: [provider]
    )
    run = tmp_path / "capture"
    artifact = run_contextual_collection(
        instrument=instrument,
        source_bundle=bundle,
        run_directory=run,
        profile="test",
    )
    assert artifact is not None
    _, replayed, _ = load_contextual_collection(run, bundle)
    assert replayed == artifact
    return instrument, artifact, provider, run


def test_collection_asks_each_parent_one_question_and_pools_by_rule(
    tmp_path, monkeypatch
):
    """A parent whose answer is four times what is beneath it, a gap that
    needs no further call: every parent is asked once per coordinate, and the
    capture stores the pools the rule gives."""
    base = _instrument()
    answers = _Answers(base, _means(), scales={COMPROMISE: 4})
    instrument, artifact, provider, run = _collect(tmp_path, monkeypatch, base, answers)
    anchors = len(instrument.anchor_scores)
    assert Counter(provider.calls) == {
        "direct": len(SCENARIOS) * anchors,
        "chain": len(SCENARIOS) * anchors,
        "parent": len(instrument.parents) * anchors,
    }
    assert provider.parents == {row.parent_id: anchors for row in instrument.parents}
    assert {row.parent_id for row in instrument.parents} == _splits()
    # The stored calls are the questions asked: none follows a parent's.
    assert len(list((run / "calls").iterdir())) == len(provider.calls)
    assert artifact.exhausted_calls() == ()
    pools = {
        case.parent_id: case.pool
        for case in artifact.parent_cases
        if case.anchor_score == 0.5
    }
    assert all(pool is not None for pool in pools.values())
    stored = pools[COMPROMISE]
    assert stored is not None
    assert (stored.status, stored.height, stored.weight) == ("pooled", 1, 0.5)
    assert stored.factor == pytest.approx(2.0, rel=1e-3)
    assert stored.direct_eal_mean == pytest.approx(
        4 * (_means()[LEAF] + REMAINDER), rel=1e-3
    )
    assert stored.modeled_eal_mean == pytest.approx(
        _means()[LEAF] + REMAINDER, rel=1e-3
    )
    outcome = _outcome(instrument, artifact)
    assert outcome.factors[LEAF].frequency == pytest.approx(2.0, rel=1e-9)
    assert outcome.factors[DESIGNATION].frequency == pytest.approx(1.0, rel=1e-9)


def test_complete_catalog_collects_every_parent_and_runs_with_every_view(
    tmp_path, monkeypatch
):
    """Every scenario and every parent through fake collection, one deep
    parent pooled away from its leaves, then a run whose every reference view
    is available."""
    base = hierarchy_instrument(tuple(_scenarios()))
    answers = _Answers(base, _pooled_means(base), scales={NO_CHILD: 16})
    instrument, artifact, provider, _ = _collect(
        tmp_path / "collect", monkeypatch, base, answers
    )
    anchors = len(instrument.anchor_scores)
    scenarios, parents = len(_scenarios()), len(_splits())
    assert (len(instrument.nodes), len(instrument.parents)) == (scenarios, parents)
    assert Counter(provider.calls) == {
        "direct": scenarios * anchors,
        "chain": scenarios * anchors,
        "parent": parents * anchors,
    }
    outcome = _outcome(instrument, artifact)
    assert {pool.status for pool in outcome.pools.values()} == {"pooled"}
    assert outcome.pools[NO_CHILD].factor == pytest.approx(4.0, rel=1e-9)
    for parent, pool in outcome.pools.items():
        if parent != NO_CHILD:
            assert pool.factor == pytest.approx(1.0, rel=1e-9), parent
    fraud = set(
        next(row for row in instrument.parents if row.parent_id == NO_CHILD).children
    )
    assert {
        leaf for leaf, factors in outcome.factors.items() if factors.frequency > 2
    } == fraud
    _, _, report, _ = _capture(
        tmp_path / "run",
        instrument,
        artifact,
    )
    metrics = report["arms"]["reference"]["metrics"]
    for name in ("gross", "welfare", f"lab_loss::{instrument.lab.entity_id}"):
        assert metrics[name]["status"] == "available", name
    summary = report["sensitivity_summary"]["hierarchy_pool"]
    assert set(summary) == {row.parent_id for row in instrument.parents}
    assert summary[NO_CHILD]["cases_pooled"] == anchors


# The summary at the import's spread and level bias


def test_hierarchy_summary_reads_child_means_at_reference_spread():
    instrument = _instrument()
    artifact = hierarchy_artifact(
        instrument, _leaf, _Answers(instrument, _means(), scales={COMPROMISE: 4})
    )
    weights = pooled_weights(instrument.respondents, {})
    captures = ((artifact, instrument, weights),)
    uncalibrated = hierarchy_summary(captures)
    face_value = hierarchy_summary(
        captures,
        options=CompileOptions(
            rank_spread=1.0, spread_tail_form=REFERENCE_SPREAD_TAIL_FORM
        ),
    )
    calibrated = hierarchy_summary(
        captures,
        options=CompileOptions(
            rank_spread=1.5, spread_tail_form=REFERENCE_SPREAD_TAIL_FORM
        ),
    )

    assert face_value == uncalibrated
    stated, stretched = uncalibrated[COMPROMISE], calibrated[COMPROMISE]
    # At the stated count laws the pooled value is the geometric mean of the
    # category answer, 4 M, and the M beneath it: 2 M, so the gap from the
    # category answer is -2 M after the pool and -3 M before it.
    modeled = _means()[LEAF] + REMAINDER
    assert stated["pooled_eal"]["median"] == pytest.approx(2 * modeled, rel=1e-9)
    assert stated["coherence_gap_before"]["median"] == pytest.approx(
        -3 * modeled, rel=1e-9
    )
    assert stated["coherence_gap_after"]["median"] == pytest.approx(
        -2 * modeled, rel=1e-9
    )
    assert stated["absolute_log_ratio_before"]["median"] == pytest.approx(math.log(4))
    assert stated["absolute_log_ratio_after"]["median"] == pytest.approx(math.log(2))
    # A wider spread raises the leaf's count mean; the factor, derived at the
    # stated laws, and the parent's answer stay, so the gap narrows.
    assert stretched["children_sum"]["median"] > stated["children_sum"]["median"]
    assert stretched["factor"] == stated["factor"]
    assert stretched["direct_eal"] == stated["direct_eal"]
    assert (
        stretched["coherence_gap_after"]["median"]
        > stated["coherence_gap_after"]["median"]
    )
    for before, after in zip(
        stated["case_results"], stretched["case_results"], strict=True
    ):
        assert (after["status"], after["factor"]) == (
            before["status"],
            before["factor"],
        )
        assert after["pooled_eal"] > before["pooled_eal"]


def test_hierarchy_summary_scales_child_means_by_the_reference_level_bias():
    instrument = _instrument()
    artifact = hierarchy_artifact(
        instrument, _leaf, _Answers(instrument, _means(), scales={COMPROMISE: 4})
    )
    captures = ((artifact, instrument, pooled_weights(instrument.respondents, {})),)
    at_spread = CompileOptions(
        rank_spread=1.5, spread_tail_form=REFERENCE_SPREAD_TAIL_FORM
    )
    calibrated = hierarchy_summary(captures, options=at_spread)
    biased = hierarchy_summary(captures, options=replace(at_spread, count_factor=0.8))
    # COMPROMISE's only modeled child is a leaf, whose count mean the bias
    # scales before and after its pool; the parent's direct and remainder
    # answers are monetary and stay as stated, and so does its factor.
    for before, after in zip(
        calibrated[COMPROMISE]["case_results"],
        biased[COMPROMISE]["case_results"],
        strict=True,
    ):
        assert after["direct_eal"] == before["direct_eal"]
        assert after["status"] == before["status"]
        assert after["factor"] == before["factor"]
    assert biased[COMPROMISE]["children_sum"]["median"] == pytest.approx(
        0.8 * calibrated[COMPROMISE]["children_sum"]["median"], rel=1e-12
    )
    for field in ("direct_eal", "remainder"):
        assert biased[COMPROMISE][field] == calibrated[COMPROMISE][field]
    assert (
        biased[COMPROMISE]["coherence_gap_after"]["median"]
        < calibrated[COMPROMISE]["coherence_gap_after"]["median"]
    )


def test_the_pool_factor_scales_the_rank_integrated_count_in_every_view():
    """A node with an uncertain count whose lab cap binds, under a parent
    whose pool doubles it. The final arm is the scaled count law's mean over
    its rank times the unchanged law's mean per count, so welfare doubles
    exactly; the lab view, with its annual business loss, less than doubles."""
    instrument = hierarchy_instrument((CAPPED,))
    node = instrument.nodes[0]
    branch = _owning_parent(instrument, node.node_id)
    assert parent_heights(instrument)[branch.parent_id] == 1
    answers = _Answers(
        instrument, _pooled_means(instrument), scales={branch.parent_id: 4}
    )
    artifact = hierarchy_artifact(instrument, _leaf, answers)
    outcome = _outcome(instrument, artifact)
    assert outcome.factors[node.node_id].frequency == pytest.approx(2.0, rel=1e-9)
    case = next(
        row
        for row in artifact.cases
        if row.node_id == node.node_id and row.anchor_score == 0.5
    )
    _, doubled = leaf_eal(
        instrument, node.node_id, pooled_answer(case), 0.5, LeafFactors(2.0)
    )
    assert doubled is not None
    lab = f"lab_loss::{instrument.lab.entity_id}"
    for seed in (1, 2):
        record = _frequency_record(instrument, artifact, seed=seed)
        rows = {
            view: record["views"][view]["by_scenario"][node.population.scenario_id]
            for view in ("welfare", "gross", lab)
        }
        assert rows["welfare"]["final"] == pytest.approx(doubled, rel=1e-9)
        assert rows["welfare"]["final"] == pytest.approx(
            2 * rows["welfare"]["before_hierarchy"], rel=1e-9
        )
        for view in ("gross", lab):
            assert (
                rows[view]["before_hierarchy"]
                < rows[view]["final"]
                < 2 * rows[view]["before_hierarchy"]
            )


# The severity factor a compiled law can carry; no pool sets one.


def test_a_severity_factor_moves_a_capped_leaf_less_than_proportionally():
    """A severity factor scales each event's loss before caps; on a node whose
    cap binds, its effect lies strictly between one and the factor."""
    instrument = hierarchy_instrument((CAPPED,))
    node = instrument.nodes[0]
    answer = governed_contextual_answer(node)
    plain = compile_contextual_node(
        instrument,
        node,
        {0.5: answer},
        0.5,
    )
    scaled = compile_contextual_node(
        instrument, node, {0.5: answer}, 0.5, severity_factors={0.5: 3.0}
    )
    assert isinstance(plain, IncidentLaw) and isinstance(scaled, IncidentLaw)
    (before,), (after,) = plain.segments, scaled.segments
    # The separately elicited business loss does not move.
    assert before.kernel.campaign_lab_losses == after.kernel.campaign_lab_losses
    _, unscaled = leaf_eal(instrument, node.node_id, answer, 0.5)
    assert unscaled is not None and unscaled > 0
    effects = {}
    for factor in (3.0, 0.3):
        _, revised = leaf_eal(
            instrument, node.node_id, answer, 0.5, LeafFactors(severity=factor)
        )
        assert revised is not None
        effects[factor] = revised / unscaled
    # The cap binds: strictly less than proportional in both directions.
    assert 1 < effects[3.0] < 3 * (1 - 1e-3)
    assert 0.3 * (1 + 1e-3) < effects[0.3] < 1


def test_a_severity_factor_below_one_keeps_a_floored_leaf_admitted():
    """A node whose total must be at least the USD 100,000 admission
    threshold: the factor scales only the loss above the floor, so the node's
    admission holds."""
    instrument = hierarchy_instrument(("compromise_organizations", FLOORED))
    node = next(row for row in instrument.nodes if row.node_id == FLOORED)
    assert node.population.threshold == 1e5
    answer = governed_contextual_answer(node)
    law = admitted_contextual_node(
        instrument, node, {0.5: answer}, 0.5, severity_factors={0.5: 0.5}
    )
    assert isinstance(law, IncidentLaw)
    floors = {
        component.severity.components[0].severity_minimum
        for outcome in law.segments[0].kernel.member_outcomes
        for component in outcome.components
        if component.severity is not None
        and component.severity.components[0].severity_minimum
    }
    assert floors == {1e5}


def test_a_severity_factor_scales_a_fixed_amount_proportionally():
    fine = ComponentAmountLaw(component_id="fine", nature="transfer", amount=20.0)
    (scaled,) = _severity_scaled((fine,), 3.0)
    assert scaled.amount == 60.0
    assert scaled.model_dump(exclude={"amount"}) == fine.model_dump(exclude={"amount"})


def test_excess_scaling_preserves_the_qualifying_floor_and_the_cap():
    floored = Distribution(
        2e6, 5e6, 1e7, 5e7, 2e8, severity_minimum=1e6, severity_cap=1e9
    )
    scaled = floored.with_excess_scale(0.5)
    assert scaled.severity_minimum == 1e6 and scaled.severity_cap == 1e9
    for before, after in zip(
        (floored.p5, floored.p25, floored.p50, floored.p75, floored.p95),
        (scaled.p5, scaled.p25, scaled.p50, scaled.p75, scaled.p95),
        strict=True,
    ):
        assert after - 1e6 == pytest.approx(0.5 * (before - 1e6))
    uniforms = np.linspace(0.001, 0.999, 41)
    assert np.all(scaled.sample_from_uniforms(uniforms) >= 1e6)
    # Without a floor the scaling is proportional.
    plain = Distribution(2e6, 5e6, 1e7, 5e7, 2e8, gpd_shape=0.5, gpd_tail_ratio=4)
    halved = plain.with_excess_scale(0.5).sample_from_uniforms_uncapped(uniforms)
    assert halved == pytest.approx(0.5 * plain.sample_from_uniforms_uncapped(uniforms))
