"""Children's mean EALs and the pool of each taxonomy parent with what is
beneath it.

methodology.md and ``HIER-CATEGORY-POOL`` define the pool. A leaf's EAL at a
coordinate is its expected count times the analytic welfare mean per unit
count of its compiled law (``contextual_node.leaf_eal``); a parent child's is
its pooled value, everything beneath it after its own pool. The pools follow
whichever parents the instrument declares, each after every parent beneath
it, and are always replayed from the answers (``hierarchy_outcome``). Each
parent case stores every child's mean EAL, each leaf child's median EAL and
its pool at four significant figures, and replay holds those stored values to
the answers.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple, get_args

from catmodel.contracts.contextual_risk import (
    ContextualParentCase,
    ContextualParentNode,
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskCase,
    ContextualRiskInstrument,
    GroupKey,
    ParentAnswer,
    ParentPool,
    ParentPoolStatus,
)
from catmodel.contracts.elicitation import QUANTILES, Quantiles
from catmodel.elicitation.contextual_hierarchy import (
    HIERARCHY_EAL_TOLERANCE,
    CategoryPool,
    category_evidence,
    check_category_citations,
    finite_sum,
    law_mean,
    parent_prompts,
    pool_parent,
    subtree_leaves,
)
from catmodel.elicitation.eal_crosscheck import pooled_answer
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.probability.numerics import scaled_product, significant
from catmodel.setup.contextual_node import (
    LEAF_FACTOR_RANGE,
    CompileOptions,
    ContextualLawUnavailable,
    LeafFactors,
    bracket_anchors,
    interpolate,
    leaf_eals,
)


def _significant(value: float | None) -> float | None:
    return None if value is None else significant(value)


def parent_heights(instrument: ContextualRiskInstrument) -> dict[str, int]:
    """Each declared parent's height: one where no child is itself a parent,
    else one more than the tallest parent among its children."""
    parents = {row.parent_id: row for row in instrument.parents}

    @cache
    def height(parent_id: str) -> int:
        return 1 + max(
            (
                height(child)
                for child in parents[parent_id].children
                if child in parents
            ),
            default=0,
        )

    return {parent_id: height(parent_id) for parent_id in parents}


def parent_order(
    instrument: ContextualRiskInstrument,
) -> tuple[tuple[ContextualParentNode, ...], ...]:
    """The declared parents by height, lowest first: each parent is pooled
    after every parent beneath it."""
    heights = parent_heights(instrument)
    return tuple(
        tuple(row for row in instrument.parents if heights[row.parent_id] == level)
        for level in sorted(set(heights.values()))
    )


@dataclass(frozen=True)
class _Tree:
    """An instrument's declared parents, as the pools traverse them."""

    parents: dict[str, ContextualParentNode]
    heights: dict[str, int]
    order: tuple[ContextualParentNode, ...]
    """The parents by height, lowest first (``parent_order``)."""
    leaves: dict[str, tuple[str, ...]]
    """Every modeled node beneath each parent."""
    chain: dict[str, tuple[str, ...]]
    """For each parent, itself and then every parent above it up to the root;
    for each modeled node, every parent above it, its own first."""
    root: str


def _tree(instrument: ContextualRiskInstrument) -> _Tree:
    parents = {row.parent_id: row for row in instrument.parents}
    owner = {
        child: row.parent_id for row in instrument.parents for child in row.children
    }
    (root,) = (parent_id for parent_id in parents if parent_id not in owner)

    def chain(node: str) -> tuple[str, ...]:
        above = [node] if node in parents else []
        while node in owner:
            node = owner[node]
            above.append(node)
        return tuple(above)

    return _Tree(
        parents,
        parent_heights(instrument),
        tuple(row for level in parent_order(instrument) for row in level),
        {parent_id: subtree_leaves(instrument, parent_id) for parent_id in parents},
        {node: chain(node) for node in (*parents, *owner)},
        root,
    )


type ParentAnswers = Mapping[str, ParentAnswer | None]
"""Each parent's own answer; None where its question exhausted its
validation attempts."""


def _remainder_law(
    parent: ContextualParentNode, answer: ParentAnswer | None
) -> Quantiles | None:
    """A parent's unmodeled part: its remainder law, or its whole category
    when no child is modeled; None without an answer."""
    if answer is None:
        return None
    return answer.remainder_eal if parent.children else answer.category_eal


def _scale(tree: _Tree, node: str, top: str, factors: Mapping[str, float]) -> float:
    """The product of the pool factors that reach ``node`` within ``top``'s
    subtree: those of the parents above a modeled node, or of a parent and
    the parents above it, up to and including ``top``. A parent ``factors``
    omits has factor one."""
    chain = tree.chain[node]
    reached = chain[: chain.index(top) + 1]
    return scaled_product(*(factors.get(parent_id, 1.0) for parent_id in reached))


def _leaf_products(
    tree: _Tree, top: str, factors: Mapping[str, float]
) -> dict[str, LeafFactors]:
    """Each leaf beneath ``top`` at its cumulative factor within ``top``'s
    subtree, before the clamp."""
    return {
        leaf: LeafFactors(_scale(tree, leaf, top, factors)) for leaf in tree.leaves[top]
    }


def _binding(products: Mapping[str, LeafFactors]) -> dict[str, LeafFactors]:
    """The cumulative factors the clamp binds (``LeafFactors.bounded``)."""
    return {
        leaf: product
        for leaf, product in products.items()
        if product.bounded() != product
    }


def _mean_beneath(
    tree: _Tree,
    top: str,
    leaf_means: Mapping[str, float | None],
    remainder_means: Mapping[str, float | None],
    factors: Mapping[str, float],
) -> tuple[float | None, float | None]:
    """A parent's (total, own remainder) mean EAL with its own pool and the
    pools within it: every leaf beneath it at its clamped cumulative factor,
    and its own and every remainder beneath it at the unclamped product of
    the factors that reach it. A value is None when a mean it adds is missing
    or it is not finite."""
    leaves = [
        None
        if (mean := leaf_means[leaf]) is None
        else product.bounded().frequency * mean
        for leaf, product in _leaf_products(tree, top, factors).items()
    ]
    remainders = {
        parent_id: None
        if (mean := remainder_means[parent_id]) is None
        else _scale(tree, parent_id, top, factors) * mean
        for parent_id in tree.parents
        if top in tree.chain[parent_id]
    }
    own = remainders[top]
    if own is not None and not math.isfinite(own):
        own = None
    values = (*leaves, *remainders.values())
    if any(value is None for value in values):
        return None, own
    return finite_sum(value for value in values if value is not None), own


def _pools(
    tree: _Tree,
    parent_answers: ParentAnswers,
    leaf_means: Mapping[str, float | None],
    remainder_means: Mapping[str, float | None],
    weight: float | None,
) -> dict[str, CategoryPool]:
    """Each parent's pool, each after the parents beneath it. A leaf child
    goes in at its own mean EAL and a parent child at its pooled value."""
    pools: dict[str, CategoryPool] = {}
    factors: dict[str, float] = {}
    for parent in tree.order:
        children = {
            child: _mean_beneath(tree, child, leaf_means, remainder_means, factors)[0]
            if child in tree.parents
            else leaf_means[child]
            for child in parent.children
        }
        pool = pool_parent(
            parent,
            parent_answers[parent.parent_id],
            children,
            tree.heights[parent.parent_id],
            weight=weight,
        )
        pools[parent.parent_id] = pool
        factors[parent.parent_id] = pool.factor
    return pools


def _scaled(law: Quantiles | None, factor: float) -> Quantiles | None:
    """A law with every quantile scaled; None without a law or where a scaled
    quantile is not finite."""
    if law is None:
        return None
    values = {key: getattr(law, key) * factor for key in QUANTILES}
    if not all(math.isfinite(value) for value in values.values()):
        return None
    return Quantiles(**values)


@dataclass(frozen=True)
class HierarchyOutcome:
    """One coordinate and respondent's pools and their effects.

    ``pools`` hold each parent's pool with the child means going in, and
    ``children_medians`` each parent's leaf children's median EALs going in
    (None for a parent child, whose EAL is a sum of means). ``factors`` are
    every leaf's cumulative factor, the product of the pool factors of the
    parents above it, clamped (``LeafFactors.bounded``); ``clamped`` holds the
    unclamped factors of each leaf whose clamp binds, and ``clamped_within``
    those of the leaves beneath each parent whose clamp binds right after its
    pool. ``remainder_laws`` are each parent's remainder law (its whole
    category when no child is modeled) scaled by its own pool's factor and
    those of the parents above it. ``parents_only`` is every leaf's clamped
    cumulative factor with weight one at every parent, where each category's
    own answer stands for its category and is allocated to what is beneath it
    in proportion: the ``hierarchy_parents_only`` arm of the
    ``frequency_estimate`` exhibit.
    """

    pools: Mapping[str, CategoryPool]
    children_medians: Mapping[str, Mapping[str, float | None]]
    factors: Mapping[str, LeafFactors]
    remainder_laws: Mapping[str, Quantiles | None]
    parents_only: Mapping[str, float]
    clamped: Mapping[str, LeafFactors]
    clamped_within: Mapping[str, Mapping[str, LeafFactors]]


def _factors(pools: Mapping[str, CategoryPool]) -> dict[str, float]:
    return {parent_id: pool.factor for parent_id, pool in pools.items()}


def _remainder_means(
    tree: _Tree, parent_answers: ParentAnswers
) -> dict[str, float | None]:
    """The mean of each parent's remainder law (``_remainder_law``)."""
    return {
        parent_id: law_mean(_remainder_law(row, parent_answers[parent_id]))
        for parent_id, row in tree.parents.items()
    }


def _outcome(
    tree: _Tree,
    instrument: ContextualRiskInstrument,
    leaf_answers: Mapping[str, ContextualRiskAnswer],
    parent_answers: ParentAnswers,
    anchor_score: float,
) -> HierarchyOutcome:
    medians, means = leaf_eals(instrument, leaf_answers, anchor_score)
    remainder_means = _remainder_means(tree, parent_answers)
    pools = _pools(tree, parent_answers, means, remainder_means, None)
    factors = _factors(pools)
    products = _leaf_products(tree, tree.root, factors)
    only = _factors(_pools(tree, parent_answers, means, remainder_means, 1.0))
    return HierarchyOutcome(
        pools=pools,
        children_medians={
            parent_id: {
                child: None if child in tree.parents else medians[child]
                for child in row.children
            }
            for parent_id, row in tree.parents.items()
        },
        factors={leaf: product.bounded() for leaf, product in products.items()},
        remainder_laws={
            parent_id: _scaled(
                _remainder_law(row, parent_answers[parent_id]),
                _scale(tree, parent_id, tree.root, factors),
            )
            for parent_id, row in tree.parents.items()
        },
        parents_only={
            leaf: product.bounded().frequency
            for leaf, product in _leaf_products(tree, tree.root, only).items()
        },
        clamped=_binding(products),
        clamped_within={
            parent_id: _binding(_leaf_products(tree, parent_id, factors))
            for parent_id in tree.parents
        },
    )


def hierarchy_outcome(
    instrument: ContextualRiskInstrument,
    leaf_answers: Mapping[str, ContextualRiskAnswer],
    parent_answers: ParentAnswers,
    anchor_score: float,
) -> HierarchyOutcome:
    """Every parent pooled with what is beneath it, each after the parents
    beneath it, replayed from the answers.

    ``leaf_answers`` holds each leaf's pooled answer after its EAL
    cross-check (``pooled_answer``) and ``parent_answers`` each parent's own
    answer. Leaf EALs are read at the stated count laws, k = 1 and b = 1.
    Collection and every reader of a collection compute the outcome here.
    """
    return _outcome(
        _tree(instrument), instrument, leaf_answers, parent_answers, anchor_score
    )


class RecordedParent(NamedTuple):
    """What a parent case stores of its group's outcome, at four significant
    figures."""

    children_median_eal: dict[str, float | None]
    children_mean_eal: dict[str, float | None]
    pool: ParentPool


def recorded_parents(outcome: HierarchyOutcome) -> dict[str, RecordedParent]:
    """What each parent case stores of an outcome (``RecordedParent``)."""
    return {
        parent_id: RecordedParent(
            {
                child: _significant(value)
                for child, value in outcome.children_medians[parent_id].items()
            },
            {child: _significant(value) for child, value in pool.children.items()},
            ParentPool(
                status=pool.status,
                reason=pool.reason,
                height=pool.height,
                weight=pool.weight,
                direct_eal_mean=_significant(pool.direct),
                modeled_eal_mean=_significant(pool.modeled),
                factor=significant(pool.factor),
            ),
        )
        for parent_id, pool in outcome.pools.items()
    }


def _close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=HIERARCHY_EAL_TOLERANCE, abs_tol=1e-9)


def _pool_matches(stored: ParentPool | None, replayed: ParentPool) -> bool:
    """A stored pool is the replayed one, its means and its factor within
    ``HIERARCHY_EAL_TOLERANCE``."""
    return (
        stored is not None
        and (stored.status, stored.reason, stored.height, stored.weight)
        == (replayed.status, replayed.reason, replayed.height, replayed.weight)
        and _close(stored.direct_eal_mean, replayed.direct_eal_mean)
        and _close(stored.modeled_eal_mean, replayed.modeled_eal_mean)
        and math.isclose(
            stored.factor, replayed.factor, rel_tol=HIERARCHY_EAL_TOLERANCE
        )
    )


@dataclass(frozen=True)
class _Group:
    """One complete group's cases: its leaves and its parents by ID."""

    leaves: list[ContextualRiskCase]
    parents: dict[str, ContextualParentCase]

    def answers(
        self,
    ) -> tuple[dict[str, ContextualRiskAnswer], dict[str, ParentAnswer | None]]:
        """Each leaf's pooled answer and each parent's own answer."""
        return (
            {case.node_id: pooled_answer(case) for case in self.leaves},
            {parent_id: case.answer for parent_id, case in self.parents.items()},
        )


def _groups(artifact: ContextualRiskArtifact) -> dict[GroupKey, _Group]:
    """The complete groups' leaf and parent cases. An incomplete group, one
    with an unanswered leaf, is never pooled."""
    incomplete = artifact.incomplete_groups()
    groups: dict[GroupKey, _Group] = {}
    for case in artifact.cases:
        key = case.anchor_score, case.respondent.key()
        if key not in incomplete:
            groups.setdefault(key, _Group([], {})).leaves.append(case)
    for case in artifact.parent_cases:
        key = case.anchor_score, case.respondent.key()
        if key not in incomplete:
            groups[key].parents[case.parent_id] = case
    return groups


def hierarchy_outcomes(
    instrument: ContextualRiskInstrument,
    artifact: ContextualRiskArtifact,
    respondents: Collection[tuple[str, str, int]] | None = None,
) -> dict[GroupKey, HierarchyOutcome]:
    """Every complete group's outcome, replayed from its answers; empty
    without parents. With ``respondents``, only the groups of those
    respondent keys."""
    if not instrument.parents:
        return {}
    tree = _tree(instrument)
    return {
        key: _outcome(tree, instrument, *group.answers(), key[0])
        for key, group in _groups(artifact).items()
        if respondents is None or key[1] in respondents
    }


def validate_hierarchy_cases(
    artifact: ContextualRiskArtifact,
    instrument: ContextualRiskInstrument,
    outcomes: Mapping[GroupKey, HierarchyOutcome] | None = None,
) -> None:
    """Each parent answer matches its frozen question and cites only evidence
    that question quotes, and each complete group's stored child EALs and
    pools are those a replay from its answers gives, within
    ``HIERARCHY_EAL_TOLERANCE``.

    ``outcomes`` are the collection's own (``hierarchy_outcomes``) where the
    caller has replayed them; without them they are replayed here.
    """
    if not instrument.parents:
        return
    prompts = {
        (row.parent.parent_id, row.anchor_score, row.respondent.key()): row
        for row in parent_prompts(instrument)
    }
    quoted = {
        row.parent_id: category_evidence(instrument, row) for row in instrument.parents
    }
    for case in artifact.parent_cases:
        if case.prompt_sha256 != prompts[case.key()].prompt_sha256():
            raise ValueError("parent answer differs from its frozen prompt")
        if case.answer is not None:
            check_category_citations(case.answer.reasoning, quoted[case.parent_id])
    if outcomes is None:
        outcomes = hierarchy_outcomes(instrument, artifact)
    for key, group in _groups(artifact).items():
        replayed = recorded_parents(outcomes[key])
        for parent_id, case in group.parents.items():
            expected = replayed[parent_id]
            for stored, values in (
                (case.children_median_eal, expected.children_median_eal),
                (case.children_mean_eal, expected.children_mean_eal),
            ):
                if stored is None or any(
                    not _close(stored[child], value) for child, value in values.items()
                ):
                    raise ValueError(
                        f"stored child EALs of {parent_id} differ from those its answers imply"
                    )
            if not _pool_matches(case.pool, expected.pool):
                raise ValueError(
                    f"stored pool of {parent_id} differs from the one its answers imply"
                )


def _median(values: list[float | None]) -> dict[str, Any]:
    """Median of the finite values, with how many were infinite or missing."""
    present = [value for value in values if value is not None]
    finite = [value for value in present if math.isfinite(value)]
    return {
        "median": statistics.median(finite) if finite else None,
        "infinite": len(present) - len(finite),
        "unavailable": len(values) - len(present),
    }


def _reference_remainder(
    instrument: ContextualRiskInstrument,
    laws: Mapping[float, Quantiles | None],
) -> float | None:
    """Mean of a parent's pooled remainder law at the reference coordinate.

    The law is interpolated between the bracketing anchors like a count's
    knots, then integrated over its epistemic rank.
    """
    if instrument.public_trajectory is not None:
        (law,) = laws.values()
        return None if law is None else epistemic_mean(law)
    try:
        lower, upper, fraction = bracket_anchors(laws, instrument.reference_eci)
    except ContextualLawUnavailable:
        return None
    left, right = laws[lower], laws[upper]
    if left is None or right is None:
        return None
    return epistemic_mean(
        Quantiles(
            **{
                key: interpolate(getattr(left, key), getattr(right, key), fraction)
                for key in QUANTILES
            }
        )
    )


def _finite(value: float | None) -> float | None:
    """A value a results file can hold: None where it is missing or not
    finite."""
    return value if value is not None and math.isfinite(value) else None


def _absolute_log_ratio(left: float, right: float) -> float:
    """Absolute log ratio of two nonnegative values; infinite when exactly
    one is zero."""
    if left == right == 0:
        return 0.0
    if left == 0 or right == 0:
        return math.inf
    return abs(math.log(left / right))


def _parts(
    values: Mapping[str, float | None] | None,
) -> tuple[float, float, float] | None:
    """(direct, children, remainder) when all three are present."""
    if values is None:
        return None
    direct, children, remainder = (
        values["direct"],
        values["children"],
        values["remainder"],
    )
    if direct is None or children is None or remainder is None:
        return None
    return direct, children, remainder


def _plus(values: Mapping[str, float | None] | None) -> float | None:
    if values is None or values["children"] is None or values["remainder"] is None:
        return None
    return values["children"] + values["remainder"]


def _ratio(values: Mapping[str, float | None] | None) -> float | None:
    """Absolute log ratio of a parent's direct mean EAL to its children plus
    remainder; None when any part is missing."""
    parts = _parts(values)
    return None if parts is None else _absolute_log_ratio(parts[0], parts[1] + parts[2])


def _signed_gap(values: Mapping[str, float | None] | None) -> float | None:
    """Children plus remainder minus the parent's direct mean EAL."""
    parts = _parts(values)
    return None if parts is None else math.fsum((parts[1], parts[2], -parts[0]))


_STATUSES: tuple[ParentPoolStatus, ...] = get_args(ParentPoolStatus)
_COMPARED: tuple[ParentPoolStatus, ...] = ("pooled", "not_pooled")
"""The statuses of a parent whose direct, children and remainder means are
all identified."""


def hierarchy_summary(
    captures: Iterable[
        tuple[
            ContextualRiskArtifact,
            ContextualRiskInstrument,
            Mapping[tuple[str, str, int], float],
        ]
    ],
    *,
    options: CompileOptions = CompileOptions(),
) -> dict[str, Any]:
    """Per parent: its height and weight, every case's pool, counts by
    status, the medians across cases of the pooled means, factors and log
    ratios, the leaves whose cumulative factors were clamped right after a
    pool and the remainder line.

    Each capture is a collection or, for a setup, the cases its states read
    from one (``SampleExclusions.run_view``), in which a leaf a respondent
    borrows is pooled as its own. It carries the reference weight of each
    respondent the run keeps, by respondent key; the groups of every other
    respondent are left out, as are incomplete groups, which are never
    pooled. A case's values before its pool are its category answer and its
    children plus remainder going in; its values after are the same answer
    and what is beneath it with the pool's factor applied. A case value that
    is not finite, as the log ratio where one side is zero, is None, and the
    medians count such values as infinite. The remainder line is the weighted
    mean of each kept respondent's pooled remainder law at the reference
    coordinate, never simulated or added to totals; each parent's remainder
    covers only what lies directly beneath it outside its modeled children,
    so the lines of every parent add without overlap.

    Leaf count means are read at ``options``'s spread and scaled by its
    count factor (``leaf_eal``), while each parent's category and remainder
    answers stay at their stated monetary values and every factor is the one
    derived from the stated count laws, at k = 1 and b = 1. The coherence
    gaps, children plus remainder less the category answer, therefore show
    what the collection-time pools leave between the parents' answers and
    the leaves as ``options`` reads them.
    """
    rows: dict[str, dict[str, Any]] = {}
    for artifact, instrument, weights in captures:
        if not instrument.parents:
            continue
        tree = _tree(instrument)
        laws: dict[tuple[str, tuple[str, str, int]], dict[float, Quantiles | None]] = {}
        for (anchor, respondent), group in _groups(artifact).items():
            if respondent not in weights:
                continue
            leaf_answers, parent_answers = group.answers()
            outcome = _outcome(tree, instrument, leaf_answers, parent_answers, anchor)
            factors = _factors(outcome.pools)
            _, means = leaf_eals(instrument, leaf_answers, anchor, options=options)
            remainder_means = _remainder_means(tree, parent_answers)
            for parent in instrument.parents:
                parent_id = parent.parent_id
                pool = outcome.pools[parent_id]
                children = [
                    _mean_beneath(tree, child, means, remainder_means, factors)[0]
                    if child in tree.parents
                    else means[child]
                    for child in parent.children
                ]
                before = {
                    "direct": pool.direct,
                    "children": None
                    if any(value is None for value in children)
                    else math.fsum(value for value in children if value is not None),
                    "remainder": pool.remainder,
                }
                after = None
                if pool.status == "pooled":
                    total, remainder = _mean_beneath(
                        tree, parent_id, means, remainder_means, factors
                    )
                    after = {
                        "direct": pool.direct,
                        "children": None
                        if total is None or remainder is None
                        else total - remainder,
                        "remainder": remainder,
                    }
                row = rows.setdefault(
                    parent_id,
                    {
                        "label": parent.label,
                        "level": parent.level,
                        "threshold": parent.threshold,
                        "height": pool.height,
                        "weight": pool.weight,
                        "statuses": Counter(dict.fromkeys(_STATUSES, 0)),
                        "direct": [],
                        "children_sum": [],
                        "remainder": [],
                        "pooled": [],
                        "factor": [],
                        "before": [],
                        "after": [],
                        "coherence_gap_before": [],
                        "coherence_gap_after": [],
                        "cases": [],
                        "remainder_line": 0.0,
                        "remainder_line_available": True,
                    },
                )
                laws.setdefault((parent_id, respondent), {})[anchor] = (
                    outcome.remainder_laws[parent_id]
                )
                row["statuses"][pool.status] += 1
                before_ratio, after_ratio = _ratio(before), _ratio(after)
                row["cases"].append(
                    {
                        "anchor_score": anchor,
                        "respondent": list(respondent),
                        "status": pool.status,
                        "reason": pool.reason,
                        "factor": pool.factor,
                        "direct_eal": before["direct"],
                        "children_plus_remainder_eal": _finite(_plus(before)),
                        "absolute_log_ratio_before": _finite(before_ratio),
                        "coherence_gap_before": _finite(_signed_gap(before)),
                        "pooled_eal": _finite(_plus(after)),
                        "absolute_log_ratio_after": _finite(after_ratio),
                        "coherence_gap_after": _finite(_signed_gap(after)),
                        "clamped_leaves": {
                            leaf: {
                                "cumulative_factor": _finite(product.frequency),
                                "applied_factor": product.bounded().frequency,
                            }
                            for leaf, product in sorted(
                                outcome.clamped_within[parent_id].items()
                            )
                        },
                    }
                )
                if pool.status not in _COMPARED:
                    continue
                row["direct"].append(before["direct"])
                row["children_sum"].append(before["children"])
                row["remainder"].append(before["remainder"])
                row["before"].append(before_ratio)
                row["coherence_gap_before"].append(_signed_gap(before))
                if after is None:
                    continue
                row["factor"].append(pool.factor)
                row["pooled"].append(_plus(after))
                row["after"].append(after_ratio)
                row["coherence_gap_after"].append(_signed_gap(after))
        for (parent_id, respondent), by_anchor in laws.items():
            value = _reference_remainder(instrument, by_anchor)
            if value is None:
                rows[parent_id]["remainder_line_available"] = False
            else:
                rows[parent_id]["remainder_line"] += weights[respondent] * value
    summary = {}
    for parent_id, row in rows.items():
        statuses = row["statuses"]
        summary[parent_id] = {
            "label": row["label"],
            "level": row["level"],
            "threshold": row["threshold"],
            "height": row["height"],
            "weight": row["weight"],
            "cases": len(row["cases"]),
            "cases_by_status": dict(statuses),
            "cases_pooled": statuses["pooled"],
            "cases_not_pooled": len(row["cases"]) - statuses["pooled"],
            "cases_clamped": sum(bool(case["clamped_leaves"]) for case in row["cases"]),
            "direct_eal": _median(row["direct"]),
            "children_sum": _median(row["children_sum"]),
            "remainder": _median(row["remainder"]),
            "pooled_eal": _median(row["pooled"]),
            "factor": _median(row["factor"]),
            "absolute_log_ratio_before": _median(row["before"]),
            "absolute_log_ratio_after": _median(row["after"]),
            "coherence_gap_before": _median(row["coherence_gap_before"]),
            "coherence_gap_after": _median(row["coherence_gap_after"]),
            "case_results": row["cases"],
            "remainder_line": {
                "status": "estimated_unsimulated",
                "expected_annual_loss": row["remainder_line"],
                "note": "Mean of each respondent's remainder law, scaled by the pools at and above its parent, at the reference coordinate, weighted like the reference arm; never simulated and not in any total.",
            }
            if row["remainder_line_available"]
            else {
                "status": "unavailable",
                "reason": "A respondent's remainder law is unavailable at an anchor bracketing the reference coordinate.",
            },
        }
    return summary


def hierarchy_factor_limitation(
    captures: Iterable[
        tuple[
            ContextualRiskArtifact,
            ContextualRiskInstrument,
            Mapping[tuple[str, str, int], float],
        ]
    ],
) -> tuple[str, ...]:
    """Disclose the leaf cases whose cumulative pool factor the run clamps
    (``HIER-CATEGORY-POOL``); nothing where the clamp never binds.

    Captures are as ``hierarchy_summary`` takes them, and only the groups of
    the respondents each keeps count. A leaf case is one leaf at one
    coordinate for one respondent.
    """
    cases = 0
    leaves: set[str] = set()
    for artifact, instrument, weights in captures:
        for (_, respondent), outcome in hierarchy_outcomes(
            instrument, artifact
        ).items():
            if respondent not in weights:
                continue
            cases += len(outcome.clamped)
            leaves.update(outcome.clamped)
    if not leaves:
        return ()
    lower, upper = LEAF_FACTOR_RANGE
    return (
        "Taxonomy category pool (HIER-CATEGORY-POOL): each leaf's cumulative "
        "frequency factor, the product of the pool factors of every taxonomy "
        f"parent above it, is clamped to {lower:g} to {upper:,g}. The clamp "
        f"binds in {cases} leaf cases (one leaf at one coordinate for one "
        "respondent), which move less than the pools above them compound to; "
        "leaves: "
        + ", ".join(sorted(leaves))
        + ". The hierarchy exhibit's case results list, for each parent, the "
        "leaves beneath it whose clamp binds after its pool, with their "
        "unclamped factors.",
    )
