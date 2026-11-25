"""Samples and scenario answers a study leaves out for an unanswered leaf.

An exhausted direct question (``contracts.contextual_risk.ExhaustedCall``)
leaves its leaf unanswered for its sample, one respondent at one node. A
public-study node holds both source statuses' contexts, so a sample lacks a
leaf that is unanswered in either; a fixed-anchor setup is one node, whose
sample is the respondent's states at every coordinate. Import decides over the
whole study what an unanswered leaf leaves out (``sample_exclusions``):

- A model whose samples lacking a leaf exceed the excluded-sample control
  does not answer that leaf (``POOL-UNANSWERED-SCENARIOS``). Its answers to
  the leaf are set aside at every node, and each of its kept samples takes
  the whole case of a kept sample of a model that answers it
  (``leaf_donors``).
- Any other unanswered leaf excludes its sample with every state it holds
  (``POOL-EXCLUDED-SAMPLES``).

Other exhausted calls leave nothing out.

A direct answer may also declare the lab's own loss unavailable, its
compensation share or its annual business loss
(``lab_loss_unavailable_parts``). Its leaf then has every law but the lab's,
and no annual business-loss law. Such an answer excludes no sample and changes
no other decision of import: the kept samples, their weights, the unanswered
pairs and the donors are decided from the exhausted direct questions alone.

A view that counts the lab's own loss is computed on the samples that give
it (``contracts.incident.counts_lab_loss``). Those views are the lab-loss
views and the gross views: gross adds each node's annual lab business loss
(``ACC-LAB-ANNUAL-BUSINESS-LOSS``), which a state without that law would
omit. A kept sample is left out of them when a case its states read declares
a part of the lab's loss unavailable: its own case of a leaf, or the donor's
case of a leaf its model does not answer
(``SampleExclusions.lab_loss_left_out``, ``JointState.lab_loss_left_out``).
Such a view gives a left-out state weight zero and splits its sampling
group's weight in an arm among the group's other states in proportion to
their weights there (``lab_loss_weights``). Every other view, welfare and net
resource cost with their scenario and source-status views, keeps every
sample at its weight, so it is the same whether or not the lab's loss was
given.

The guards that bound the excluded samples bound the excluded and left-out
samples together. Where they would breach one, no sample is left out: the
states that lack the lab's loss keep their weight, and the run withholds
both families: the lab-loss views, which those states do not hold, and the
gross views, which would omit their business loss (``without_lab_loss``).
``SampleExclusions.lab_loss_withheld`` names the guard.

A context variant study also takes
its headline study's unanswered leaves and donors (``Headline``), so that it
and its matched control read a leaf from the same answers. The stored captures
stay as collected: ``SampleExclusions.kept_view`` and
``SampleExclusions.run_view`` derive from them the cases a run keeps and the
cases its states read, and ``kept_cases`` the kept cases from the exhibit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

from catmodel.contracts.contextual_risk import (
    EXHAUSTED_CALL_KINDS,
    ContextualRiskArtifact,
    ContextualRiskCase,
    ContextualRiskInstrument,
    ExhaustedCall,
)
from catmodel.contracts.elicitation import Respondent
from catmodel.contracts.incident import Evidence, JointState
from catmodel.contracts.run_setup import EvidenceSet
from catmodel.frozen_calls import EXHAUSTION_CAUSES, PARSE_ATTEMPTS, exhaustion_causes

type RespondentKey = tuple[str, str, int]
type Capture = tuple[ContextualRiskArtifact, Mapping[str, bytes]]
"""An artifact and its capture's files, which hold each call's attempts."""
type Donors = Mapping[tuple[str, RespondentKey], RespondentKey]
"""One node's donors: by leaf and borrowing respondent, the respondent whose
case of the leaf the borrower's states read (``leaf_donors``)."""
type StudyNodes = Mapping[
    str, Sequence[tuple[ContextualRiskInstrument, ContextualRiskArtifact]]
]
"""Each node of a study with its captures, which share one respondent roster
and ask the same leaves."""

MAX_EXCLUDED_SAMPLE_SHARE = 0.10
"""The largest share of a study's samples, and of each model's, that import
may exclude, and the share of a model's samples lacking one leaf above which
the model does not answer that leaf: a judgment, with a sensitivity
range of 0.05 to 0.20 (the import functions' ``max_excluded_sample_share``).
``POOL-EXCLUDED-SAMPLES`` gives its calibration and the selection it guards
against, and ``POOL-UNANSWERED-SCENARIOS`` what a model's unanswered leaf
uses instead."""


LAB_LOSS_PARTS = ("lab_compensation", "lab_annual_business_loss")
"""The two answers a direct question gives about the lab's own loss: its
compensation share and its annual business loss. Each may be declared
unavailable with a reason (``ContextualRiskAnswer``)."""

type Declared = frozenset[tuple[str, RespondentKey, str, str]]
"""The leaves whose direct answer declares the lab's loss unavailable, each as
(node, respondent key, context ID, node ID)."""
type LeftOut = Mapping[
    str, Mapping[RespondentKey, tuple[tuple[str, str, RespondentKey], ...]]
]
"""Each node's samples left out of the views that count the lab's loss: by
respondent, the cases its states read whose direct answer declares the lab's
loss unavailable, each as (context ID, node ID, answering respondent), the
respondent itself or the donor whose case of the leaf it borrows."""


def lab_loss_unavailable_parts(case: ContextualRiskCase) -> tuple[str, ...]:
    """The parts of the lab's loss that ``case``'s direct answer declares
    unavailable, in ``LAB_LOSS_PARTS`` order: none where it gives both, and
    none for a direct answer that is itself unavailable and so holds neither.

    A resolving coherence retry and an EAL cross-check replace the count and
    the loss quantiles only, so the stored direct answer's parts are the ones
    every law of the case reads."""
    return tuple(
        name
        for name in LAB_LOSS_PARTS
        if (part := getattr(case.answer, name)) is not None
        and part.status == "unavailable"
    )


type PairBasis = Literal["own", "own_and_headline", "headline"]
"""What makes a leaf one that a model does not answer in a study: ``own``, the
study's captures, in which the share of the model's samples lacking the leaf
is above the excluded-sample control; for a variant study, ``own_and_headline``
where its headline study has the pair too, and ``headline`` where only the
headline has it and the variant follows."""


@dataclass(frozen=True)
class UnansweredPair:
    """A leaf that a model does not answer (``POOL-UNANSWERED-SCENARIOS``):
    ``lacking`` of the model's ``samples`` over every node of the study lack
    the leaf, a share above the excluded-sample control unless the pair
    follows the headline alone (``PairBasis``)."""

    leaf: str
    model: str
    lacking: int
    samples: int
    basis: PairBasis = "own"

    @property
    def share(self) -> float:
        return self.lacking / self.samples


@dataclass(frozen=True)
class Headline:
    """What a variant study follows of its headline study, whose captures its
    matched control reweights (``POOL-UNANSWERED-SCENARIOS``): the headline's
    unanswered pairs, as (model, leaf), and its donors at each of the
    variant's nodes (``SampleExclusions.followed_at``)."""

    pairs: frozenset[tuple[str, str]]
    donors: Mapping[str, Donors]


@dataclass(frozen=True)
class SampleExclusions:
    """What a study leaves out for its unanswered leaves: each node's kept
    respondents and pooled weights, and each leaf a model does not answer with
    its donors.

    ``weights`` maps each node, a public study's trajectory node or a
    fixed-anchor context, to each kept respondent's weight there;
    ``unanswered`` maps each node to each excluded respondent's unanswered
    leaves, those of its model's unanswered pairs left out, as (context ID,
    node ID) pairs; ``node_weights`` holds each node's reference weight.
    ``leaves`` maps each leaf the study asks to its scenario. ``pairs`` are
    the unanswered pairs, by leaf and then by model in roster order, and
    ``donors`` maps each node to its donors (``Donors``). ``headline`` is
    what a variant study followed of its headline study, and None for any
    other study.

    ``lab_loss_unavailable`` holds every leaf whose direct answer declares the
    lab's loss unavailable (``Declared``); none of them excludes a sample or
    enters any field above. ``lab_loss_left_out`` holds each node's kept
    samples that are left out of the views that count the lab's loss
    (``LeftOut``). ``lab_loss_withheld`` is None where the samples that read
    such a leaf are left out, and otherwise the guard that leaving them out
    would breach: none is then left out, their states hold no lab-loss law
    and the run withholds the views that count the lab's loss.
    """

    roster: tuple[Respondent, ...]
    weights: Mapping[str, Mapping[RespondentKey, float]]
    unanswered: Mapping[str, Mapping[RespondentKey, tuple[tuple[str, str], ...]]]
    node_weights: Mapping[str, float]
    leaves: Mapping[str, str]
    pairs: tuple[UnansweredPair, ...]
    donors: Mapping[str, Donors]
    headline: Headline | None = None
    lab_loss_unavailable: Declared = frozenset()
    lab_loss_left_out: LeftOut = MappingProxyType({})
    lab_loss_withheld: str | None = None

    @property
    def samples(self) -> int:
        return len(self.weights) * len(self.roster)

    @property
    def excluded(self) -> int:
        return sum(len(rows) for rows in self.unanswered.values())

    @property
    def weighted_share(self) -> float:
        """The excluded share with each node at its reference weight."""
        total = math.fsum(self.node_weights.values())
        if not total:
            return 0.0
        return (
            math.fsum(
                self.node_weights[node] * len(rows) / len(self.roster)
                for node, rows in self.unanswered.items()
            )
            / total
        )

    def left_out(self, node: str) -> frozenset[RespondentKey]:
        """The kept respondents of ``node`` that are left out of the views
        that count the lab's loss."""
        return frozenset(self.lab_loss_left_out.get(node, ()))

    def by_model(self) -> dict[str, dict[str, float]]:
        """Each model's samples over every node, those excluded and their share."""
        counts = Counter(row.model_id for row in self.roster)
        models = {row.key(): row.model_id for row in self.roster}
        excluded = Counter(
            models[key] for rows in self.unanswered.values() for key in rows
        )
        return {
            model: {
                "samples": count * len(self.weights),
                "excluded": excluded[model],
                "excluded_share": excluded[model] / (count * len(self.weights)),
            }
            for model, count in sorted(counts.items())
        }

    def _set_aside(self) -> frozenset[tuple[str, str]]:
        """The unanswered pairs, as (model, leaf)."""
        return frozenset((pair.model, pair.leaf) for pair in self.pairs)

    def answering(self, leaf: str) -> tuple[str, ...]:
        """The models that answer ``leaf``, in roster order."""
        set_aside = self._set_aside()
        return tuple(
            model
            for model in roster_models(self.roster)
            if (model, leaf) not in set_aside
        )

    def borrowers(self, node: str) -> frozenset[RespondentKey]:
        """The kept respondents of ``node`` whose states read a donor's case
        of some leaf."""
        return frozenset(borrower for _, borrower in self.donors[node])

    def kept_view(
        self, node: str, artifact: ContextualRiskArtifact
    ) -> ContextualRiskArtifact:
        """The cases a run keeps from one capture of ``node``: those of its
        kept respondents, without a model's answers to a leaf it does not
        answer. Each is a stored case and appears once, however many states
        read it, so the exhibits that report what was asked and answered count
        these. Not a capture: it is never validated or stored."""
        return _kept(artifact, self.unanswered[node], self._set_aside())

    def followed_at(self, nodes: Mapping[str, str]) -> Headline:
        """What a variant study follows of this one, its headline: ``nodes``
        maps each of the variant's nodes to this study's node on the same
        trajectory."""
        return Headline(
            self._set_aside(), {node: self.donors[own] for node, own in nodes.items()}
        )

    def matched_control(self) -> dict[str, object] | None:
        """How a variant study's unanswered pairs and donors compare with
        its headline's, and so with its matched control's; None for a study
        that follows no headline.

        ``pairs_not_in_headline`` counts the variant's pairs the headline
        answers. ``donors`` counts, over the pairs of both studies, each
        borrowing sample at each node, and ``donors_not_headline`` those
        whose donor is not the one the sample has in the headline at that
        node. ``differs`` says whether either count is above zero.
        """
        if self.headline is None:
            return None
        same = [
            donor == self.headline.donors[node].get((leaf, borrower))
            for node, donors in self.donors.items()
            for (leaf, borrower), donor in donors.items()
            if (borrower[1], leaf) in self.headline.pairs
        ]
        alone = sum(pair.basis == "own" for pair in self.pairs)
        return {
            "pairs_not_in_headline": alone,
            "donors": len(same),
            "donors_not_headline": same.count(False),
            "differs": bool(alone) or not all(same),
        }

    def run_view(
        self, node: str, artifact: ContextualRiskArtifact
    ) -> ContextualRiskArtifact:
        """The cases the states of ``node`` read from one of its captures:
        the kept cases (``kept_view``) and, for each leaf a kept respondent's
        model does not answer, its donor's case under the respondent's own
        identity. Whatever compiles or pools a respondent's cases therefore
        reads a borrowed case as that respondent's own.

        A borrowing respondent's parent cases keep their answers and drop
        their stored child EALs and pools, which its group never had or drew
        from the answers set aside; every reader replays a pool from the
        answers. Not a capture: it is never validated or stored."""
        kept = self.kept_view(node, artifact)
        donors = self.donors[node]
        if not donors:
            return kept
        respondents = {row.key(): row for row in self.roster}
        own: dict[tuple[str, RespondentKey], list[ContextualRiskCase]] = {}
        for case in kept.cases:
            own.setdefault((case.node_id, case.respondent.key()), []).append(case)
        borrowers = self.borrowers(node)
        return kept.model_copy(
            update={
                "cases": (
                    *kept.cases,
                    *(
                        case.model_copy(update={"respondent": respondents[borrower]})
                        for (leaf, borrower), donor in sorted(donors.items())
                        for case in own[leaf, donor]
                    ),
                ),
                "parent_cases": tuple(
                    case.model_copy(
                        update={
                            "children_median_eal": None,
                            "children_mean_eal": None,
                            "pool": None,
                        }
                    )
                    if case.respondent.key() in borrowers
                    else case
                    for case in kept.parent_cases
                ),
            }
        )

    def case_weights(self, node: str) -> dict[tuple[str, RespondentKey], float]:
        """The weight at ``node`` of the states that read each kept case, by
        leaf and respondent: the respondent's own weight and that of every
        respondent that borrows the case. A model's cases of a leaf it does
        not answer are not kept and have no entry."""
        weights = self.weights[node]
        set_aside = self._set_aside()
        result = {
            (leaf, key): weight
            for leaf in self.leaves
            for key, weight in weights.items()
            if (key[1], leaf) not in set_aside
        }
        for (leaf, borrower), donor in self.donors[node].items():
            result[leaf, donor] += weights[borrower]
        return result

    def pool_shares(self, leaf: str) -> dict[str, float]:
        """Each answering model's share of ``leaf``'s pooled weight over the
        study, each node at its reference weight: the weight of its own kept
        samples and of the samples that borrow from them."""
        shares = dict.fromkeys(self.answering(leaf), 0.0)
        total = math.fsum(self.node_weights.values())
        for node, weights in self.weights.items():
            donors = self.donors[node]
            for key, weight in weights.items():
                source = donors.get((leaf, key), key)
                shares[source[1]] += self.node_weights[node] * weight
        return {
            model: share / total if total else 0.0 for model, share in shares.items()
        }


def roster_models(roster: Sequence[Respondent]) -> tuple[str, ...]:
    """The roster's models in roster order, the order in which the sorted
    respondent keys first name them."""
    ordered = sorted(roster, key=lambda row: row.key())
    return tuple(dict.fromkeys(row.model_id for row in ordered))


def pooled_weights(
    roster: Sequence[Respondent], excluded: Mapping[RespondentKey, object]
) -> dict[RespondentKey, float]:
    """Each kept respondent's weight: an equal share for each model of the
    roster (``POOL-WEIGHTS``), split equally among the model's kept samples."""
    models = {row.model_id for row in roster}
    kept = Counter(row.model_id for row in roster if row.key() not in excluded)
    return {
        row.key(): 1 / len(models) / kept[row.model_id]
        for row in roster
        if row.key() not in excluded
    }


def lab_loss_weights(
    weights: Mapping[str, float],
    groups: Mapping[str, str | None],
    left_out: Collection[str],
) -> tuple[Mapping[str, float] | None, str | None]:
    """An arm's weights in the views that count the lab's own loss, or why
    it has none.

    ``weights`` are the arm's state weights, ``groups`` each state's sampling
    group and ``left_out`` the states left out of those views, all under one
    key. A left-out state weighs zero. Its sampling group keeps its weight in
    the arm, which the group's other states share in proportion to their
    weights there:

    ```text
    w_lab(s) = w(s) * W_g / (W_g - L_g)
    ```

    for a state ``s`` of group ``g`` that is not left out, with ``W_g`` the
    arm's weight on the group and ``L_g`` its weight on the group's left-out
    states. A group without a left-out state at positive weight keeps the
    arm's weights, and an arm without one returns ``weights`` itself. Every
    lab-loss view and every gross view of an arm uses these weights
    (``contracts.incident.counts_lab_loss``), so the source statuses'
    lab-loss EALs add up to the lab's and the scenarios' gross EALs to gross.

    Those views are unavailable in an arm where a group with weight has no
    weight on a state that is not left out, or a left-out state with weight
    names no group: None, with the reason.
    """
    leaving = {state for state in left_out if weights.get(state, 0.0) > 0}
    if not leaving:
        return weights, None
    members: dict[str | None, list[str]] = {}
    for state, weight in weights.items():
        if weight > 0:
            members.setdefault(groups.get(state), []).append(state)
    result = dict(weights)
    for group in sorted({groups.get(state) for state in leaving}, key=str):
        states = members[group]
        remaining = [state for state in states if state not in leaving]
        if group is None:
            return None, (
                "A state with weight is left out of the views that count the "
                "lab's own loss (POOL-EXCLUDED-SAMPLES) and names no sampling "
                "group, so no state takes its weight in them."
            )
        if not remaining:
            return None, (
                f"Every state with weight in sampling group {group!r} is left out "
                "of the views that count the lab's own loss, because a case it "
                "reads declares that loss unavailable (POOL-EXCLUDED-SAMPLES), so "
                "no state carries the group's weight in them."
            )
        scale = math.fsum(weights[state] for state in states) / math.fsum(
            weights[state] for state in remaining
        )
        for state in states:
            result[state] = 0.0 if state in leaving else weights[state] * scale
    return result, None


def lab_loss_state_weights(
    states: Iterable[JointState],
) -> tuple[Mapping[str, float] | None, str | None]:
    """How a setup's own weights serve the views that count the lab's loss.

    First, each of its ``states``' weight in those views, by state ID
    (``lab_loss_weights``); None where no state with weight is left out of
    them, so the states' own weights serve every view. Import leaves a sample
    out only where its model keeps another in those views at the node
    (``_lab_loss_breach``), so a setup's own weights always have them.
    Second, why the gross views are unavailable at those weights
    (``gross_withheld_reason``), or None."""
    states = tuple(states)
    weights = {state.state_id: state.weight for state in states}
    found, reason = lab_loss_weights(
        weights,
        {state.state_id: state.sampling_group_id for state in states},
        {state.state_id for state in states if state.lab_loss_left_out},
    )
    if found is None:
        raise ValueError(reason)
    withheld = gross_withheld_reason(
        found,
        {state.state_id: state.state_id for state in states if lacks_lab_loss(state)},
    )
    return (None if found is weights else found), withheld


def lacks_lab_loss(state: JointState) -> bool:
    """Whether a respondent sample's state holds no law of the lab's loss
    without being left out of the views that count it.

    A law with incidents then records the lab's loss unavailable
    (``IncidentLaw.unavailable_lab_losses``), so the state has no annual
    business-loss law for that leaf and its gross loss would omit the term.
    Import leaves such a state unmarked where leaving its sample out would
    breach a sample guard (``SampleExclusions.lab_loss_withheld``)."""
    return (
        state.sampling_group_id is not None
        and not state.lab_loss_left_out
        and any(
            law.unavailable_lab_losses and law.reference_annual_rate > 0
            for law in state.laws
        )
    )


def gross_withheld_reason(
    weights: Mapping[str, float], lacking: Mapping[str, str]
) -> str | None:
    """Why an arm reports no gross view, or None where it can.

    ``weights`` are the arm's weights in the views that count the lab's loss
    (``lab_loss_weights``) and ``lacking`` the states that hold no law of
    that loss without being left out (``lacks_lab_loss``), by the weights'
    key with their IDs. One of them with weight withholds the gross views:
    its gross loss would omit the lab's annual business loss. The arm's
    lab-loss views are unavailable in the same arm because the state does
    not hold them."""
    found = sorted(
        state_id for key, state_id in lacking.items() if weights.get(key, 0.0) > 0
    )
    if not found:
        return None
    return (
        f"State {found[0]!r} has weight and holds no law of the lab's own loss, "
        "so its gross loss would omit the lab's annual business loss, and it is "
        "not left out of the views that count that loss (POOL-EXCLUDED-SAMPLES)."
    )


def _refused(text: str) -> ValueError:
    return ValueError(
        text + " because a leaf's direct question exhausted its validation "
        "attempts (POOL-EXCLUDED-SAMPLES); recollect the study"
    )


def _missing(
    nodes: StudyNodes,
) -> tuple[
    tuple[Respondent, ...],
    dict[str, str],
    dict[str, dict[RespondentKey, tuple[tuple[str, str], ...]]],
    dict[str, dict[RespondentKey, tuple[tuple[str, str], ...]]],
]:
    """A study's roster in key order, its leaves with their scenarios, each
    node's unanswered leaves by respondent, as (context ID, node ID) pairs,
    and, in the same form, each node's leaves whose direct answer declares
    the lab's loss unavailable."""
    roster: tuple[Respondent, ...] | None = None
    leaves: dict[str, str] | None = None
    missing: dict[str, dict[RespondentKey, tuple[tuple[str, str], ...]]] = {}
    declared: dict[str, dict[RespondentKey, tuple[tuple[str, str], ...]]] = {}
    for node, captures in sorted(nodes.items()):
        rows: dict[RespondentKey, list[tuple[str, str]]] = {}
        gaps: dict[RespondentKey, list[tuple[str, str]]] = {}
        for instrument, artifact in captures:
            ordered = tuple(sorted(instrument.respondents, key=lambda row: row.key()))
            asked = {
                row.node_id: row.population.scenario_id
                for row in instrument.collectible_nodes
            }
            if roster is None or leaves is None:
                roster, leaves = ordered, asked
            if ordered != roster:
                raise ValueError(
                    "every capture of a study shares one respondent roster"
                )
            if asked != leaves:
                raise ValueError("every capture of a study asks the same leaves")
            for case in artifact.unanswered_cases:
                rows.setdefault(case.respondent.key(), []).append(
                    (instrument.context_id, case.node_id)
                )
            for answered in artifact.cases:
                if lab_loss_unavailable_parts(answered):
                    gaps.setdefault(answered.respondent.key(), []).append(
                        (instrument.context_id, answered.node_id)
                    )
        missing[node] = {key: tuple(sorted(set(value))) for key, value in rows.items()}
        declared[node] = {key: tuple(sorted(set(value))) for key, value in gaps.items()}
    if roster is None or leaves is None:
        raise ValueError("a study needs a capture")
    return roster, leaves, missing, declared


def _kept(
    artifact: ContextualRiskArtifact,
    excluded: Collection[RespondentKey],
    set_aside: Collection[tuple[str, str]],
) -> ContextualRiskArtifact:
    """The cases a run keeps from one capture: those of the respondents not
    ``excluded`` at its node, without the answers ``set_aside``, each a
    (model, leaf)."""
    return artifact.model_copy(
        update={
            "cases": tuple(
                case
                for case in artifact.cases
                if case.respondent.key() not in excluded
                and (case.respondent.model_id, case.node_id) not in set_aside
            ),
            "unanswered_cases": (),
            "parent_cases": tuple(
                case
                for case in artifact.parent_cases
                if case.respondent.key() not in excluded
            ),
        }
    )


def kept_cases(
    record: Mapping[str, Any], node: str, artifact: ContextualRiskArtifact
) -> ContextualRiskArtifact:
    """The cases a run keeps from one capture of ``node``, read from the
    study's ``excluded_samples`` record: ``SampleExclusions.kept_view`` for
    whoever holds the exhibit and not the decision."""
    return _kept(
        artifact,
        {tuple(row["respondent"]) for row in record["excluded"] if row["node"] == node},
        {
            (pair["model_id"], pair["node_id"])
            for pair in record.get("unanswered_scenarios", {}).get("pairs", ())
        },
    )


def _pairs(
    roster: Sequence[Respondent],
    missing: Mapping[str, Mapping[RespondentKey, Sequence[tuple[str, str]]]],
    max_share: float,
    followed: Collection[tuple[str, str]] = (),
) -> tuple[UnansweredPair, ...]:
    """The unanswered pairs of a study, by leaf and then by model in roster
    order: those its own captures make (``unanswered_pairs``) and
    ``followed``, its headline's, each a (model, leaf)."""
    samples = {
        model: count * len(missing)
        for model, count in Counter(row.model_id for row in roster).items()
    }
    lacking: Counter[tuple[str, str]] = Counter()
    for rows in missing.values():
        for key, cases in rows.items():
            lacking.update((key[1], leaf) for leaf in {leaf for _, leaf in cases})
    own = {
        pair for pair, count in lacking.items() if count / samples[pair[0]] > max_share
    }
    basis: dict[tuple[bool, bool], PairBasis] = {
        (True, False): "own",
        (True, True): "own_and_headline",
        (False, True): "headline",
    }
    models = roster_models(roster)
    return tuple(
        UnansweredPair(
            leaf,
            model,
            lacking[model, leaf],
            samples[model],
            basis[(model, leaf) in own, (model, leaf) in followed],
        )
        for leaf in sorted({leaf for _, leaf in (*own, *followed)})
        for model in models
        if (model, leaf) in own or (model, leaf) in followed
    )


def unanswered_pairs(
    nodes: StudyNodes, max_share: float = MAX_EXCLUDED_SAMPLE_SHARE
) -> tuple[UnansweredPair, ...]:
    """Each leaf that a model does not answer over ``nodes``
    (``POOL-UNANSWERED-SCENARIOS``).

    A sample, one respondent at one node, lacks a leaf whose direct question
    is exhausted in any of the node's captures. A model does not answer a
    leaf when the share of its samples, over every node, that lack the leaf
    exceeds ``max_share``.
    """
    roster, _, missing, _ = _missing(nodes)
    return _pairs(roster, missing, max_share)


def _by_model(
    respondents: Sequence[Respondent], models: Sequence[str]
) -> list[list[Respondent]]:
    """Each of ``models``' samples among ``respondents``, by sample index,
    in the order the models are given."""
    return [
        sorted(
            (row for row in respondents if row.model_id == model),
            key=lambda row: row.sample_index,
        )
        for model in models
    ]


def _in_rounds(samples: Sequence[Sequence[Respondent]]) -> list[Respondent]:
    """Several models' samples (``_by_model``) listed in rounds: the first
    sample of each model, then the second of each, and so on, passing over a
    model whose samples are used up."""
    return [
        rows[turn]
        for turn in range(max(map(len, samples), default=0))
        for rows in samples
        if turn < len(rows)
    ]


def leaf_donors(
    leaf: str,
    kept: Mapping[str, Sequence[Respondent]],
    unanswering: Sequence[str],
    answering: Sequence[str],
) -> dict[str, dict[RespondentKey, RespondentKey]]:
    """One leaf's donors by node: for each kept sample of a model that does
    not answer the leaf, the kept sample whose case of it the sample takes.

    ``kept`` maps each node to its kept respondents, and ``unanswering`` and
    ``answering`` hold the models that do not and that do answer the leaf,
    each in roster order. The assignment is a fixed rotation with no
    randomness. At a node the donors, the kept samples of the answering
    models, are listed in rounds (``_in_rounds``), and the borrowers, the
    kept samples of the models that do not answer, model by model
    (``_by_model``). With the nodes in ID order, at the node at position n
    the borrower at position b of its list takes the donor at position
    (b + n) mod L of the L listed donors.

    Consecutive borrowers therefore take consecutive donors. At a node every
    donor lends the same number of times or one more, and a borrowing
    model's samples take donors that alternate between the answering models,
    while each has a sample left in the round, and that repeat no sample
    unless the model keeps more samples than there are donors. The list is
    entered one donor further on at each node, so over any L consecutive
    nodes that keep the same samples every answering sample lends equally
    often. With one answering model and equal sample counts each of its
    samples lends equally often at every node, so the leaf's pool there is
    that model's samples in equal shares.

    Raises ``ValueError`` where no answering model keeps a sample at a node.
    """
    donors: dict[str, dict[RespondentKey, RespondentKey]] = {}
    for position, (node, respondents) in enumerate(sorted(kept.items())):
        lenders = _in_rounds(_by_model(respondents, answering))
        if not lenders:
            raise ValueError(
                f"no model that answers leaf {leaf!r} keeps a sample at node "
                f"{node!r} (POOL-UNANSWERED-SCENARIOS); recollect the study"
            )
        borrowers = [
            row for rows in _by_model(respondents, unanswering) for row in rows
        ]
        donors[node] = {
            borrower.key(): lenders[(index + position) % len(lenders)].key()
            for index, borrower in enumerate(borrowers)
        }
    return donors


def _unanswering(pairs: Sequence[UnansweredPair], max_share: float) -> str:
    """Why each of one leaf's pairs is unanswered, for the refusal of a leaf
    that no model answers."""
    parts = []
    if own := [pair for pair in pairs if pair.basis != "headline"]:
        parts.append(
            "; ".join(
                f"model {pair.model} lacks it in {pair.lacking} of its "
                f"{pair.samples} samples ({pair.share:.1%})"
                for pair in own
            )
            + f", above the frozen control max_excluded_sample_share {max_share:g}"
        )
    parts += [
        f"model {pair.model} does not answer it in the headline study, which a "
        "variant study follows"
        for pair in pairs
        if pair.basis == "headline"
    ]
    return "; ".join(parts)


def sample_exclusions(
    nodes: StudyNodes,
    node_weights: Mapping[str, float],
    *,
    max_share: float = MAX_EXCLUDED_SAMPLE_SHARE,
    headline: Headline | None = None,
) -> SampleExclusions:
    """Decide what a study's unanswered leaves leave out, pool what is kept
    and refuse what the guards forbid.

    ``nodes`` maps each node to its captures, which share one respondent
    roster and ask the same leaves, and ``node_weights`` each node to its
    reference weight. The unanswered pairs are found first
    (``unanswered_pairs``): a leaf of one excludes no sample, and each kept
    sample of its model takes a donor's case of it (``leaf_donors``). Any
    other unanswered leaf, in any of a node's captures, excludes its
    respondent at that node.

    A variant study passes what it follows of its ``headline``, so that it
    and its matched control, the headline captures reweighted, read a leaf
    from the same answers. Its pairs are its own and every pair of the
    headline: for one that its own captures do not make, the model's answers
    to the leaf are set aside all the same, and its samples lacking the leaf
    are not excluded for it. For a pair of both studies a borrowing sample
    takes the donor it has in the headline at the same node, where the
    variant keeps that donor at the node and its model answers the leaf in
    the variant; every other borrowing sample takes the donor of the
    variant's own rotation.

    Refused are a leaf that no model answers; a model that does not answer
    more than half of the leaves, which should not be in the roster; a model
    keeping fewer than half its samples, rounded up, at some node, so a
    pooled model mass never rests on a minority of its samples; a model whose
    excluded share over every node exceeds ``max_share``; and a study whose
    excluded share, counted over every node, does. The first two count a
    variant's pairs with those it follows.

    A leaf whose direct answer declares the lab's loss unavailable
    (``lab_loss_unavailable_parts``) is no part of any of that: it makes no
    pair, no exclusion and no refusal, and changes no weight or donor. Once
    the study has passed on its exhausted direct questions alone, each kept
    sample whose states read such a leaf's case, its own or a donor's
    (``_left_out``), is left out of the views that count the lab's loss,
    provided the excluded and left-out samples together still pass the last
    three guards (``_lab_loss_breach``). If they would not, none is left out:
    the states keep every other law, the run withholds the views that count
    the lab's loss, and ``lab_loss_withheld`` names the guard.
    """
    if not 0 <= max_share <= 1 or not math.isfinite(max_share):
        raise ValueError("the excluded-sample share control must lie in [0, 1]")
    roster, leaves, missing, declared = _missing(nodes)
    models = roster_models(roster)
    if headline is not None and (
        set(headline.donors) != set(nodes)
        or any(
            model not in models or leaf not in leaves for model, leaf in headline.pairs
        )
    ):
        raise ValueError(
            "a variant study's headline asks the variant's leaves of the variant's "
            "models and gives its donors at each of the variant's nodes"
        )
    decided = _decide(roster, leaves, missing, node_weights, max_share, headline)
    gaps: Declared = frozenset(
        (node, key, context, leaf)
        for node, rows in declared.items()
        for key, cases in rows.items()
        for context, leaf in cases
    )
    if not gaps:
        return decided
    left_out = _left_out(decided, declared)
    breach = _lab_loss_breach(decided, left_out, max_share)
    if breach is not None:
        return replace(decided, lab_loss_unavailable=gaps, lab_loss_withheld=breach)
    return replace(decided, lab_loss_unavailable=gaps, lab_loss_left_out=left_out)


def _left_out(
    exclusions: SampleExclusions,
    declared: Mapping[str, Mapping[RespondentKey, Sequence[tuple[str, str]]]],
) -> dict[str, dict[RespondentKey, tuple[tuple[str, str, RespondentKey], ...]]]:
    """Each node's kept samples whose states read a case that declares the
    lab's loss unavailable (``LeftOut``): the sample's own case of a leaf its
    model answers, or its donor's case of a leaf it borrows. ``declared``
    gives each node's declaring leaves by respondent, as (context ID, node
    ID) pairs. A case that no kept state reads, an excluded sample's or a
    model's answer to a leaf it does not answer, leaves no sample out."""
    set_aside = exclusions._set_aside()
    result: dict[
        str, dict[RespondentKey, tuple[tuple[str, str, RespondentKey], ...]]
    ] = {}
    for node, weights in exclusions.weights.items():
        gaps = declared.get(node, {})
        read: dict[RespondentKey, set[tuple[str, str, RespondentKey]]] = {}
        for key in weights:
            for context, leaf in gaps.get(key, ()):
                if (key[1], leaf) not in set_aside:
                    read.setdefault(key, set()).add((context, leaf, key))
        for (leaf, borrower), donor in exclusions.donors[node].items():
            for context, declaring in gaps.get(donor, ()):
                if declaring == leaf:
                    read.setdefault(borrower, set()).add((context, leaf, donor))
        result[node] = {key: tuple(sorted(read[key])) for key in sorted(read)}
    return result


def _lab_loss_breach(
    exclusions: SampleExclusions,
    left_out: Mapping[str, Collection[RespondentKey]],
    max_share: float,
) -> str | None:
    """The sample guard that leaving ``left_out`` out of the views that count
    the lab's loss would breach, or None: the last three guards of
    ``sample_exclusions``, applied to the excluded and left-out samples
    together. A model keeps at least half its samples, rounded up, in those
    views at every node; its share of excluded or left-out samples over every
    node is at most ``max_share``; and so is the study's."""
    counts = Counter(row.model_id for row in exclusions.roster)
    for node, kept in sorted(exclusions.weights.items()):
        leaving = left_out.get(node, ())
        for model, count in sorted(counts.items()):
            keeps = sum(key[1] == model and key not in leaving for key in kept)
            if keeps < math.ceil(count / 2):
                return (
                    f"model {model} would keep {keeps} of its {count} samples in "
                    f"the views that count the lab's own loss at node {node!r}, "
                    "fewer than half"
                )
    removed = Counter(
        key[1]
        for rows in (exclusions.unanswered, left_out)
        for keys in rows.values()
        for key in keys
    )
    nodes = len(exclusions.weights)
    for model, count in sorted(counts.items()):
        share = removed[model] / (count * nodes)
        if share > max_share:
            return (
                f"model {model} would have {removed[model]} of its "
                f"{count * nodes} samples ({share:.1%}) excluded or left out of "
                "the views that count the lab's own loss, above the frozen "
                f"control max_excluded_sample_share {max_share:g}"
            )
    total = sum(removed.values())
    share = total / exclusions.samples
    if share > max_share:
        return (
            f"{total} of {exclusions.samples} samples ({share:.1%}) would be "
            "excluded or left out of the views that count the lab's own loss, "
            f"above the frozen control max_excluded_sample_share {max_share:g}"
        )
    return None


def _decide(
    roster: tuple[Respondent, ...],
    leaves: Mapping[str, str],
    missing: Mapping[str, Mapping[RespondentKey, Sequence[tuple[str, str]]]],
    node_weights: Mapping[str, float],
    max_share: float,
    headline: Headline | None,
) -> SampleExclusions:
    """``sample_exclusions`` from the study's exhausted direct questions
    alone: ``missing`` gives each node's unanswered leaves by respondent."""
    models = roster_models(roster)
    pairs = _pairs(
        roster, missing, max_share, () if headline is None else headline.pairs
    )
    pairs_by_leaf: dict[str, list[UnansweredPair]] = {}
    for pair in pairs:
        pairs_by_leaf.setdefault(pair.leaf, []).append(pair)
    for leaf, rows in pairs_by_leaf.items():
        if len(rows) == len(models):
            raise ValueError(
                f"no model answers leaf {leaf!r}: "
                + _unanswering(rows, max_share)
                + " (POOL-UNANSWERED-SCENARIOS); recollect the study with a model "
                "that answers it"
            )
    for model in models:
        own = sorted(pair.leaf for pair in pairs if pair.model == model)
        if 2 * len(own) > len(leaves):
            raise ValueError(
                f"model {model} does not answer {len(own)} of the study's "
                f"{len(leaves)} leaves, more than half ({', '.join(own)}), so it "
                "should not be in the roster (POOL-UNANSWERED-SCENARIOS); "
                "recollect the study without it"
            )
    set_aside = {(pair.model, pair.leaf) for pair in pairs}
    weights: dict[str, dict[RespondentKey, float]] = {}
    unanswered: dict[str, dict[RespondentKey, tuple[tuple[str, str], ...]]] = {}
    for node, rows in missing.items():
        unanswered[node] = {
            key: excluding
            for key in sorted(rows)
            if (
                excluding := tuple(
                    case for case in rows[key] if (key[1], case[1]) not in set_aside
                )
            )
        }
        kept = pooled_weights(roster, unanswered[node])
        for model, count in sorted(Counter(row.model_id for row in roster).items()):
            keeps = sum(row.model_id == model and row.key() in kept for row in roster)
            if keeps < math.ceil(count / 2):
                excluding_leaves = sorted(
                    {
                        leaf
                        for key, cases in unanswered[node].items()
                        if key[1] == model
                        for _, leaf in cases
                    }
                )
                raise _refused(
                    f"model {model} keeps {keeps} of its {count} samples at node "
                    f"{node!r}, fewer than half (unanswered leaves: "
                    f"{', '.join(excluding_leaves)}),"
                )
        weights[node] = kept
    kept_respondents = {
        node: [row for row in roster if row.key() in kept]
        for node, kept in weights.items()
    }
    donors: dict[str, dict[tuple[str, RespondentKey], RespondentKey]] = {
        node: {} for node in weights
    }
    for leaf, rows in pairs_by_leaf.items():
        unanswering = [pair.model for pair in rows]
        answering = [model for model in models if model not in unanswering]
        for node, by_borrower in leaf_donors(
            leaf, kept_respondents, unanswering, answering
        ).items():
            followed = {} if headline is None else headline.donors[node]
            for borrower, donor in by_borrower.items():
                theirs = followed.get((leaf, borrower))
                donors[node][leaf, borrower] = (
                    theirs
                    if theirs is not None
                    and theirs in weights[node]
                    and theirs[1] in answering
                    else donor
                )
    exclusions = SampleExclusions(
        roster, weights, unanswered, dict(node_weights), leaves, pairs, donors, headline
    )
    for model, row in exclusions.by_model().items():
        if row["excluded_share"] > max_share:
            raise _refused(
                f"model {model} has {row['excluded']:g} of its {row['samples']:g} "
                f"samples ({row['excluded_share']:.1%}) excluded, above the frozen "
                f"control max_excluded_sample_share {max_share:g},"
            )
    share = exclusions.excluded / exclusions.samples
    if share > max_share:
        raise _refused(
            f"{exclusions.excluded} of {exclusions.samples} samples ({share:.1%}) "
            "are excluded, above the frozen control max_excluded_sample_share "
            f"{max_share:g},"
        )
    return exclusions


def _exhausted(captures: Sequence[Capture]) -> list[tuple[ExhaustedCall, str]]:
    """Every exhausted call of the captures with the cause of its last
    attempt (``exhaustion_causes``)."""
    rows: list[tuple[ExhaustedCall, str]] = []
    for artifact, files in captures:
        calls = artifact.exhausted_calls()
        causes = exhaustion_causes(
            files, ((row.prompt_sha256, row.respondent) for row in calls)
        )
        rows.extend((row, causes[row.prompt_sha256, row.respondent]) for row in calls)
    return rows


def _tally(rows: Sequence[tuple[ExhaustedCall, str]]) -> dict[str, Any]:
    """Exhausted calls counted by kind, by cause and by leaf or parent."""
    by_subject: dict[str, dict[str, Counter[str]]] = {}
    by_kind: Counter[str] = Counter(dict.fromkeys(EXHAUSTED_CALL_KINDS, 0))
    by_cause: Counter[str] = Counter(dict.fromkeys(EXHAUSTION_CAUSES, 0))
    for row, cause in rows:
        by_subject.setdefault(row.subject_id, {}).setdefault(row.kind, Counter())[
            cause
        ] += 1
        by_kind[row.kind] += 1
        by_cause[cause] += 1
    return {
        "parse_attempts": PARSE_ATTEMPTS,
        "calls": sum(by_kind.values()),
        "by_kind": dict(by_kind),
        "by_cause": dict(by_cause),
        "by_subject": {
            subject: {
                kind: {
                    "calls": sum(causes.values()),
                    "attempts": sum(causes.values()) * PARSE_ATTEMPTS,
                    "by_cause": dict(sorted(causes.items())),
                }
                for kind, causes in sorted(kinds.items())
            }
            for subject, kinds in sorted(by_subject.items())
        },
    }


def _calls(captures: Sequence[Capture]) -> dict[str, Any]:
    """Every exhausted call, counted by kind, by cause and by leaf or parent."""
    return _tally(_exhausted(captures))


def _lab_loss_unavailable(
    exclusions: SampleExclusions, captures: Sequence[Capture]
) -> dict[str, Any]:
    """The ``lab_loss_unavailable`` record: the cases whose direct answer
    declares a part of the lab's loss unavailable, counted in all, by part
    and by leaf, and what follows for the views that count the lab's loss:
    the samples left out of them, with the cases each reads and their number
    by model, or the guard that withholds the views. A case that declares
    both parts counts once as a case and once under each part."""
    by_part: Counter[str] = Counter(dict.fromkeys(LAB_LOSS_PARTS, 0))
    by_subject: dict[str, list[tuple[str, ...]]] = {}
    for artifact, _ in captures:
        for case in artifact.cases:
            if parts := lab_loss_unavailable_parts(case):
                by_part.update(parts)
                by_subject.setdefault(case.node_id, []).append(parts)
    cases = sum(map(len, by_subject.values()))
    if not cases:
        disposition = "none"
    elif exclusions.lab_loss_withheld is None:
        disposition = "samples_left_out"
    else:
        disposition = "views_withheld"
    left_out = [
        (node, respondent, read)
        for node, rows in sorted(exclusions.lab_loss_left_out.items())
        for respondent, read in sorted(rows.items())
    ]
    by_model = Counter(respondent[1] for _, respondent, _ in left_out)
    return {
        "cases": cases,
        "by_part": dict(by_part),
        "by_subject": {
            leaf: {
                "cases": len(rows),
                "by_part": {
                    part: sum(part in parts for parts in rows)
                    for part in LAB_LOSS_PARTS
                },
            }
            for leaf, rows in sorted(by_subject.items())
        },
        "disposition": disposition,
        **(
            {
                "left_out_samples": len(left_out),
                "left_out_by_model": {
                    model: by_model[model] for model in roster_models(exclusions.roster)
                },
                "left_out": [
                    {
                        "node": node,
                        "respondent": list(respondent),
                        "reads": [
                            {
                                "context_id": context,
                                "node_id": leaf,
                                "respondent": list(answering),
                            }
                            for context, leaf, answering in read
                        ],
                    }
                    for node, respondent, read in left_out
                ],
            }
            if disposition == "samples_left_out"
            else {"withheld_because": exclusions.lab_loss_withheld}
            if disposition == "views_withheld"
            else {}
        ),
    }


def excluded_samples_limitation(
    exclusions: SampleExclusions, captures: Sequence[Capture], max_share: float
) -> str:
    """Disclose the excluded samples, the exhausted calls, the lab losses
    declared unavailable and the selection they may carry
    (``POOL-EXCLUDED-SAMPLES``)."""
    calls = _calls(captures)
    declared = _lab_loss_unavailable(exclusions, captures)
    kinds = ", ".join(
        f"{count} {kind}" for kind, count in calls["by_kind"].items() if count
    )
    causes = ", ".join(
        f"{count} {cause}" for cause, count in calls["by_cause"].items() if count
    )
    text = (
        f"Excluded samples (POOL-EXCLUDED-SAMPLES): {exclusions.excluded} of "
        f"{exclusions.samples} samples (each a respondent at one trajectory node, "
        "or at every coordinate of a fixed-anchor setup) are excluded because a "
        "leaf's direct question exhausted its validation attempts, "
        f"{exclusions.weighted_share:.1%} of the reference weight; "
        f"{calls['calls']} calls exhausted theirs in all"
        + (f" ({kinds}; last attempt by cause: {causes})" if kinds else "")
        + ". "
    )
    if declared["cases"]:
        parts = ", ".join(
            f"{count} {part}" for part, count in declared["by_part"].items() if count
        )
        text += (
            f"{declared['cases']} direct "
            f"{'answer' if declared['cases'] == 1 else 'answers'} declared the "
            f"lab's own loss unavailable ({parts}). Such an answer excludes no "
            "sample and changes no weight, so the welfare views, which count no "
            "part of the lab's loss, are as if it had been given. The views that "
            "count the lab's own loss are computed on the samples that give it: "
            "the lab-loss views and the gross views, since gross adds the lab's "
            "annual business loss, which a state that reads such an answer does "
            "not hold. "
            + (
                f"{declared['left_out_samples']} "
                + (
                    "sample whose states read such an answer, its own or that "
                    "of a sample whose answers it uses, is"
                    if declared["left_out_samples"] == 1
                    else "samples whose states read such an answer, their own "
                    "or that of a sample whose answers they use, are"
                )
                + " left out of those views ("
                + ", ".join(
                    f"{model} {count}"
                    for model, count in declared["left_out_by_model"].items()
                )
                + "): there each weighs zero and its model's other samples at "
                "the node share its weight. A respondent that declares the lab's "
                "loss unavailable may be one that would have judged it larger or "
                "less knowable. "
                if exclusions.lab_loss_withheld is None
                else "No sample is left out of those views, because leaving out "
                "the samples whose states read such an answer would breach an "
                f"import guard ({exclusions.lab_loss_withheld}); their states "
                "hold every other law, and the lab-loss views and the gross views "
                "are unavailable. "
            )
        )
    if not exclusions.excluded and not calls["calls"]:
        return text + "Every model keeps every sample at every node."
    return text + (
        "At each node a model keeps its equal share, split equally among its "
        "remaining samples; other exhausted calls exclude nothing and leave the "
        "answers they would have revised standing. "
        + (
            "An exhausted direct question of a scenario its model does not answer "
            "excludes nothing either: the model's samples use other models' "
            "answers to it (POOL-UNANSWERED-SCENARIOS). "
            if exclusions.pairs
            else ""
        )
        + "Excluded samples may differ "
        "systematically from the ones kept: refusals are content-linked and may "
        "concentrate on high-severity leaves, output-limit stops on the longest "
        "reasoning, and other validation failures on harder cases, so the pooled "
        "answers can be biased in either direction. No result with them exists, "
        "because their answers do not; their other answers remain in the capture "
        "for inspection. Import refuses a study in which a model keeps fewer than "
        "half its samples at a node, or which excludes more than "
        f"{max_share:g} of a model's samples or of all its samples."
    )


def _listed(names: Sequence[str]) -> str:
    """Names in a sentence: ``a``, ``a and b`` or ``a, b and c``."""
    if len(names) < 3:
        return " and ".join(names)
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _lacking(pair: UnansweredPair) -> str:
    """A pair's samples lacking the leaf, with what makes it a pair where
    that is not the study's captures alone."""
    counts = (
        f"unanswered in {pair.lacking} of its {pair.samples} samples, {pair.share:.1%}"
    )
    return {
        "own": counts,
        "own_and_headline": f"{counts}, and unanswered in the headline study",
        "headline": f"unanswered in the headline study, which the variant "
        f"follows; here {counts}",
    }[pair.basis]


def _against_control(exclusions: SampleExclusions) -> str:
    """For a variant study, how its unanswered scenarios compare with its
    matched control's: a caveat exactly where they differ."""
    matched = exclusions.matched_control()
    if matched is None:
        return ""
    rule = (
        " A variant study takes every such pair of its headline study too, "
        "whatever its own samples lack, and for a pair of both studies a "
        "borrowing sample takes the donor it has in the headline at the same "
        "node, where the variant keeps that donor and its model answers the "
        "scenario in the variant; otherwise the variant's own rotation gives "
        "the donor."
    )
    if not matched["differs"]:
        return rule + (
            " Every pair here is the headline's and every donor the headline's "
            "at its node, so the variant and its matched control, the headline "
            "captures reweighted to the embedded rule, take each scenario from "
            "the same models and each borrowed case from the same sample."
        )
    differences = []
    if matched["pairs_not_in_headline"]:
        differences.append(
            "pairs that the headline study answers: "
            f"{matched['pairs_not_in_headline']} of {len(exclusions.pairs)}"
        )
    if matched["donors_not_headline"]:
        differences.append(
            "donors other than the headline's, for the pairs of both studies: "
            f"{matched['donors_not_headline']} of {matched['donors']}"
        )
    return rule + (
        " The variant and its matched control, the headline captures reweighted "
        "to the embedded rule, differ in which models answer a scenario or in "
        f"whose answers a model's samples borrow ({'; '.join(differences)}), so "
        "the variant's effect also carries that difference."
    )


def unanswered_scenarios_limitation(
    exclusions: SampleExclusions, max_share: float
) -> tuple[str, ...]:
    """Disclose each scenario a model does not answer and whose answers its
    samples use, the scenarios that rest on one model, what results do not
    report because borrowed answers are not independent and, for a variant
    study, whether it differs from its matched control in them
    (``POOL-UNANSWERED-SCENARIOS``); nothing for a study without an
    unanswered pair."""
    if not exclusions.pairs:
        return ()
    pairs = "; ".join(
        f"model {pair.model} does not answer scenario "
        f"{exclusions.leaves[pair.leaf]} ({_lacking(pair)}): its samples use the "
        f"answers of {_listed(exclusions.answering(pair.leaf))}"
        for pair in exclusions.pairs
    )
    single = [
        f"{exclusions.leaves[leaf]} ({exclusions.answering(leaf)[0]})"
        for leaf in dict.fromkeys(pair.leaf for pair in exclusions.pairs)
        if len(exclusions.answering(leaf)) == 1
    ]
    return (
        f"Unanswered scenarios (POOL-UNANSWERED-SCENARIOS): {pairs}. A model does "
        "not answer a scenario when more than "
        f"{max_share:g} of its samples, over every node, lack the scenario's "
        "direct answer. Its answers to that scenario are then set aside at every "
        "node, those of the samples that answered included, and each of its kept "
        "samples takes, for that scenario alone, the whole case of one kept "
        "sample of an answering model at the same node, by a fixed rotation in "
        "which the answering models' samples lend in turn. The "
        "borrowing sample keeps its own answers to every other scenario and to "
        "every taxonomy-parent question, so the borrowed answers enter its "
        "category pools, where its own category answers still scale them. "
        "Scenarios that rest on one model's answers: "
        + (", ".join(single) if single else "none")
        + ". The models that answer a scenario may differ systematically from "
        "one that does not, in an unknown direction, and not answering may go "
        "with how severe a model would have judged the scenario. A borrowed case "
        "repeats a donor's answers, so the samples are not independent. Results "
        "therefore report no respondent-sampling standard error and no EAL "
        "parameter band for an EAL or a change that weights a borrowing sample, "
        "and no standard deviation of the samples' EALs at a trajectory node "
        "that holds one. A model-pooling arm that leaves out an answering model still "
        "carries that model's answers wherever another model's samples borrowed "
        "them. The rotation balances the answering samples, not the answering "
        "models: an answering model that keeps more samples lends more often, so "
        "with several answering models their shares of a scenario's pool are "
        "equal only where they keep equally many samples, and can differ from "
        "node to node; the excluded_samples exhibit reports the realized shares."
        + _against_control(exclusions),
    )


def _unanswered_scenarios(
    exclusions: SampleExclusions, rows: Sequence[tuple[ExhaustedCall, str]]
) -> dict[str, object]:
    """The ``unanswered_scenarios`` record of the ``excluded_samples``
    exhibit: each unanswered pair, each such leaf's answering models with
    their shares of its pool, the leaves one model answers and, for a variant
    study, how it compares with its matched control."""
    causes: Counter[tuple[str, str, str]] = Counter(
        (call.subject_id, call.respondent[1], cause)
        for call, cause in rows
        if call.kind == "direct"
    )

    def direct_calls(pair: UnansweredPair) -> dict[str, object]:
        by_cause = {
            cause: causes[pair.leaf, pair.model, cause] for cause in EXHAUSTION_CAUSES
        }
        calls = sum(by_cause.values())
        return {
            "calls": calls,
            "attempts": calls * PARSE_ATTEMPTS,
            "by_cause": by_cause,
        }

    by_leaf = {}
    for leaf in sorted({pair.leaf for pair in exclusions.pairs}):
        shares = exclusions.pool_shares(leaf)
        by_leaf[leaf] = {
            "scenario_id": exclusions.leaves[leaf],
            "answering_models": {
                model: {"realized_share": share, "equal_share": 1 / len(shares)}
                for model, share in shares.items()
            },
        }
    matched = exclusions.matched_control()
    return {
        "register_id": "POOL-UNANSWERED-SCENARIOS",
        "pairs": [
            {
                "node_id": pair.leaf,
                "scenario_id": exclusions.leaves[pair.leaf],
                "model_id": pair.model,
                "basis": pair.basis,
                "samples": pair.samples,
                "lacking_samples": pair.lacking,
                "lacking_share": pair.share,
                "exhausted_direct_calls": direct_calls(pair),
            }
            for pair in exclusions.pairs
        ],
        "by_leaf": by_leaf,
        "leaves_on_one_model": sum(
            len(row["answering_models"]) == 1 for row in by_leaf.values()
        ),
        **({} if matched is None else {"matched_control": matched}),
        "interpretation": "A model does not answer a leaf when the share of its "
        "samples, over every node, that lack the leaf (its direct question "
        "exhausted in a context of the sample's node) exceeds "
        "max_excluded_sample_share. Its answers to that leaf are then set aside "
        "at every node, those of the samples that answered included, because "
        "answers that pass a filter removing more than that share are a selected "
        "subset; such a leaf excludes no sample. At each node each kept sample of "
        "the model takes, for that leaf alone, the whole case of one kept sample "
        "of an answering model in each context of the node: the direct and chain "
        "answers, the EAL cross-check and every follow-up. The donor follows a "
        "fixed rotation with no randomness: at a node the kept samples of the "
        "answering models are listed in rounds, each model's first sample by "
        "sample index, then each model's second, and so on, and the borrowing "
        "samples, listed model by model and by sample index within a model, "
        "take them in turn, starting one donor further on at each node in "
        "node-ID order. At a node every answering "
        "sample therefore lends the same number of times or one more, over any "
        "run of as many nodes as there are donors equally often, and with one "
        "answering model and equal sample counts equally often at every node. "
        "The borrowing sample keeps its own answers to every other leaf "
        "and to every taxonomy-parent question, whose pools read the borrowed "
        "case as its own, so its own category answers still scale the borrowed "
        "counts. pairs lists each unanswered pair with the model's "
        "samples, those lacking the leaf and the exhausted direct calls behind "
        "them by the cause of their last attempt"
        + (
            "; its basis is own, the study's captures making the pair by the "
            "rule above."
            if matched is None
            else ". A variant study takes every pair of its headline study too, "
            "so that it and its matched control, the headline captures "
            "reweighted, leave the same models out of a leaf. basis says what "
            "makes a pair: own, the variant's captures by the rule above; "
            "own_and_headline, those and the headline study; headline, the "
            "headline study alone, in which case the model's answers to the "
            "leaf are set aside all the same and its samples lacking the leaf "
            "are not excluded for it. For a pair of both "
            "studies a borrowing sample takes the donor it has in the headline "
            "at the same node, where the variant keeps that donor at the node "
            "and its model answers the leaf in the variant, and otherwise the "
            "donor of the variant's own rotation. matched_control reports what "
            "still differs: pairs_not_in_headline counts the pairs of basis own; "
            "donors counts, over the pairs of both studies, each borrowing "
            "sample at each node, and donors_not_headline those whose donor is "
            "not the headline's; differs says whether either count is above "
            "zero, and the variant's effect then also carries a difference in "
            "which models answer a leaf or in whose answers a sample borrows."
        )
        + " by_leaf gives each such leaf's "
        "answering models with realized_share, the model's share of the leaf's "
        "pooled weight over the study with each node at its reference weight "
        "(its own samples and those that borrow from them), beside equal_share, "
        "one over the number of answering models: the share each would hold if "
        "the answering models kept equally many samples at every node and "
        "every answering sample lent equally often. The rotation balances the "
        "answering samples, not the models, so an answering model that keeps "
        "more samples than another, by the roster or because a sample of the "
        "other is excluded, lends more often and holds a larger share, and the "
        "shares can differ from node to node. An arm that weighs the nodes "
        "differently from the reference, such as a forecast alternative or a "
        "variant's matched control, can pool the leaf in other shares. "
        "leaves_on_one_model counts the "
        "listed leaves that one model answers. The models that answer a leaf "
        "may differ systematically from one that does not, in an unknown "
        "direction. A borrowed case repeats a donor's answers, so the "
        "samples are not independent. Results therefore report no "
        "respondent-sampling standard error and no EAL parameter band for an "
        "EAL or a change that weights a borrowing sample, and no standard "
        "deviation of the samples' EALs at a trajectory node that holds one. "
        "A model-pooling arm "
        "that leaves out an answering model still carries that model's answers "
        "wherever another model's samples borrowed them. The "
        "exhibits that count cases and follow-up calls count "
        "each stored case once, whichever states read it; the model's own "
        "answers to the leaf, set aside, are in none of them and remain in the "
        "capture.",
    }


def excluded_samples_record(
    exclusions: SampleExclusions, captures: Sequence[Capture], max_share: float
) -> dict[str, object]:
    """The ``excluded_samples`` exhibit: every exhausted call by leaf or parent,
    kind and cause, every lab loss declared unavailable with the samples left
    out of the views that count the lab's loss for it, the excluded samples
    with the leaves each lacks, their shares and the weights they
    renormalized, and, for a study with an unanswered pair, its
    ``unanswered_scenarios`` record."""
    roster = exclusions.roster
    samples = Counter(row.model_id for row in roster)
    renormalized = {}
    for node, kept in sorted(exclusions.weights.items()):
        models = {}
        for model, count in sorted(samples.items()):
            remaining = [
                weight
                for row in roster
                if row.model_id == model and (weight := kept.get(row.key()))
            ]
            if len(remaining) < count:
                models[model] = {
                    "samples": count,
                    "kept": len(remaining),
                    "model_weight": 1 / len(samples),
                    "weight_per_sample": 1 / len(samples) / count,
                    "weight_per_kept_sample": remaining[0],
                }
        if models:
            renormalized[node] = models
    exhausted = _exhausted(captures)
    record: dict[str, object] = {
        "register_id": "POOL-EXCLUDED-SAMPLES",
        "max_excluded_sample_share": max_share,
        "samples": exclusions.samples,
        "excluded_samples": exclusions.excluded,
        "excluded_share": exclusions.excluded / exclusions.samples,
        "excluded_share_weighted": exclusions.weighted_share,
        "by_model": exclusions.by_model(),
        "excluded": [
            {
                "node": node,
                "respondent": list(respondent),
                "unanswered": [
                    {"context_id": context, "node_id": leaf} for context, leaf in leaves
                ],
            }
            for node, rows in sorted(exclusions.unanswered.items())
            for respondent, leaves in sorted(rows.items())
        ],
        "renormalized_weights": renormalized,
        "exhausted_calls": _tally(exhausted),
        "lab_loss_unavailable": _lab_loss_unavailable(exclusions, captures),
        "interpretation": "A call is exhausted when every validation attempt "
        "returned a response that failed validation, a provider's refusal or "
        "output-limit stop included. A provider failure, an error rather than a "
        "response, is never exhausted: a transient one fails only its call's "
        "coordinate and respondent group, any other stops the study, and a rerun "
        "sends the call again with the same prompt, using no validation attempt "
        "(NUM-PROVIDER-FAILURES). Each exhausted call's "
        "cause is its last attempt's: refusal or output_limit where the attempt "
        "records that stop, otherwise validation. A sample is one respondent at "
        "one trajectory node (in a fixed-anchor setup, the respondent's states at "
        "every coordinate). A sample with a leaf whose direct question is exhausted"
        + (
            ", other than a leaf its model does not answer (unanswered_scenarios),"
            if exclusions.pairs
            else ""
        )
        + " has no law for that leaf and is excluded with every state it holds; at "
        "each node a model keeps its equal share, split equally among its "
        "remaining samples. excluded_share counts samples over every node; "
        "excluded_share_weighted weights each node by its reference weight. Any "
        "other exhausted call leaves the answers it would have revised standing: a "
        "chain question leaves no chain count, so the direct count stands; a "
        "guardrail retry, EAL cross-check, coherence retry or parent question "
        "leaves the answers before it. Calls are counted over every capture, those of "
        "excluded samples included, whose other answers remain in the capture for "
        "inspection; each exhausted call used every one of its validation attempts, "
        "and a subject's attempts count those, calls times parse_attempts, not the "
        "provider failures a call stored before a rerun re-sent it. "
        "A direct answer gives the lab's own loss in two parts, its compensation "
        "share (lab_compensation) and its annual business loss "
        "(lab_annual_business_loss), and may declare either unavailable; "
        "lab_loss_unavailable counts the cases that do, in all, by part and by "
        "leaf, a case that declares both once as a case and once under each part. "
        "Such a leaf has every law but the lab's, and no annual business-loss "
        "law. It excludes no sample and changes no weight, unanswered pair or "
        "donor, so a view that counts no part of the lab's loss (welfare, net "
        "resource cost and their scenario and source-status views) keeps every "
        "sample and is the same whether or not the lab's loss was given. A "
        "view that counts the lab's own loss is computed on the samples that "
        "give it: the lab-loss views and the gross views (gross and each "
        "scenario's), since gross adds the lab's annual business loss, which a "
        "state reading such a case would omit. "
        "Under disposition samples_left_out, each kept sample whose "
        "states read such a case, its own or the donor's case of a leaf it "
        "borrows, is left out of those views: there its states weigh zero, "
        "and each one's sampling group, its model at its node, keeps its "
        "weight, split among the group's other states in proportion to their "
        "weights. left_out lists those samples, each with the cases it reads "
        "and the respondent that answered them, left_out_samples counts them "
        "and left_out_by_model counts them by model. Under "
        "views_withheld, leaving them out would have breached the "
        "guard that withheld_because names, the sample guards counting the "
        "excluded and the left-out samples together, so none is left out: "
        "their states hold every other law, and the lab-loss views and the "
        "gross views are unavailable. "
        "A lab loss declared unavailable never makes a model one that does not "
        "answer a leaf. "
        "A result that excludes each excluded respondent at every node, a "
        "reweighting of the same states, is computable but not generated.",
    }
    if exclusions.pairs:
        record["unanswered_scenarios"] = _unanswered_scenarios(exclusions, exhausted)
    return record


EXCLUDED_SAMPLES_PATH = "derived/excluded-samples.json"
_NO_VARIANTS: Mapping[str, Mapping[str, Mapping[str, object]]] = MappingProxyType({})


def excluded_samples_evidence(
    record: Mapping[str, object],
    variants: Mapping[str, Mapping[str, Mapping[str, object]]] = _NO_VARIANTS,
) -> EvidenceSet:
    """Carry the ``excluded_samples`` exhibit to results with the setup's
    other evidence: the study's ``record`` and, under each key of
    ``variants``, a family's context variant studies' records by variant ID."""
    borrowed = any(
        "unanswered_scenarios" in row
        for row in (
            record,
            *(row for by_variant in variants.values() for row in by_variant.values()),
        )
    )
    disclosure = {**record, **{key: dict(rows) for key, rows in variants.items()}}
    data = json.dumps(disclosure, sort_keys=True, allow_nan=False).encode()
    digest = hashlib.sha256(data).hexdigest()
    return EvidenceSet(
        records=(
            Evidence(
                evidence_id="excluded_samples",
                origin="elicitation",
                original_sha256=digest,
                selected_sha256=digest,
                target="Every call whose validation attempts were exhausted, by leaf or parent, kind and cause, every direct answer that declares the lab's loss unavailable and the samples left out of the views that count the lab's own loss for it, the samples excluded for an unanswered leaf, their shares and the pooled weights they renormalized."
                + (
                    " Each leaf a model does not answer, with the samples lacking it and each answering model's share of the leaf's pool."
                    if borrowed
                    else ""
                ),
                conditioning="Every replayed capture of the setup.",
                units="Counts of calls, attempts and samples; pooled weights.",
                transport="Excluded samples are left out of the run; at each node a model keeps its equal share, split equally among its remaining samples. A sample left out of the views that count the lab's own loss, the lab-loss views and the gross views, keeps its states and its weight in every other view."
                + (
                    " For a leaf its model does not answer, each kept sample reads the whole case of a kept sample of an answering model at the same node."
                    if borrowed
                    else ""
                ),
                acquisition={"excluded_samples": disclosure},
                review="Derived from replayed captures; exhaustion is a validation outcome, not a respondent judgment, and a lab loss declared unavailable is the respondent's own statement.",
                reliance="A disclosure of missing answers and the reweighting they caused; no result with the excluded samples exists."
                + (
                    " Nor does one with a model's own answers to a leaf it does not answer."
                    if borrowed
                    else ""
                ),
                rights="Derived from the captured answers; their rights are retained.",
            ),
        ),
        files={EXCLUDED_SAMPLES_PATH: data},
    )
