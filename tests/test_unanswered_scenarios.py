"""A model that does not answer a scenario is left out of that scenario only.

A (model, leaf) pair is unanswered when more than the excluded-sample control's
share of the model's samples lack the leaf. Import then sets the model's
answers to the leaf aside at every node and gives each of its kept samples
the whole case of a kept sample of an answering model, by a fixed rotation
(``POOL-UNANSWERED-SCENARIOS``); any other unanswered leaf excludes its
sample (``POOL-EXCLUDED-SAMPLES``).
"""

import asyncio
import csv
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import cast

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.collection.contextual_collection import collect_study
from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskInstrument,
    UnansweredCase,
)
from catmodel.contracts.elicitation import Respondent
from catmodel.contracts.public_frontier import PublicFrontierQuadrature
from catmodel.contracts.results import UnavailableParameterBand
from catmodel.contracts.run_setup import EvidenceSet
from catmodel.elicitation.contextual_frequency import frequency_collection_summary
from catmodel.elicitation.contextual_risk import ContextualPrompt
from catmodel.elicitation.eal_crosscheck import (
    eal_crosscheck_summary,
    follow_up_counts,
    pooled_answer,
)
from catmodel.llm.transport import LLMProvider
from catmodel.results.eal_parameter_band import NO_BAND
from catmodel.results.execution import CAPTURE_MANIFEST, replay_capture, run_setup
from catmodel.results.respondent_sampling import borrowed_reason
from catmodel.results.results_export import (
    _chain_summary,
    export_results,
    load_results_export,
)
from catmodel.results.results_file import load_results
from catmodel.results.results_summary import _Summary, write_results_summary
from catmodel.results.sensitivity_results import NO_NODE_SD
from catmodel.setup.contextual_hierarchy import (
    hierarchy_outcome,
    hierarchy_outcomes,
    hierarchy_summary,
)
from catmodel.setup.contextual_inputs import read_contextual_capture_files
from catmodel.setup.contextual_setup import (
    checked_hierarchy_outcomes,
    compile_contextual_setup,
    contextual_respondent_selections,
    selection_cases,
)
from catmodel.setup.excluded_samples import (
    MAX_EXCLUDED_SAMPLE_SHARE,
    Headline,
    UnansweredPair,
    leaf_donors,
    roster_models,
    sample_exclusions,
    unanswered_pairs,
)
from catmodel.setup.public_frontier import public_state_node
from catmodel.setup.public_frontier_comparisons import (
    CONTEXT_VARIANT_FAMILIES,
    EXPOSURE_CONTROL,
)
from catmodel.stored_files import StoredFiles
from test_support.collection_trace import request_kind
from test_support.contextual_risk import (
    POOL_FIXTURE_COUNTS,
    contextual_answer,
    counted_answer,
    fixture_artifact,
    fixture_response,
    hierarchy_artifact,
    leaves_instrument,
    parent_answer,
    path_linked_payload,
    pool_instrument,
)
from test_support.results import available_metric, present, reduced_arm
from tests.test_excluded_samples import (
    CONFIG,
    _anchor,
    _context,
    _direct,
    _exhibit,
    _Immediate,
    _import,
    _public_study,
    _respondent,
    _sending,
    _subject,
)
from tests.test_exposure_scale_variants import _importer

LEAVES = ("leaf_a", "leaf_b", "leaf_c")
STATUSES = ("public", "private")


def _key(model: str, index: int) -> tuple[str, str, int]:
    return f"{model}-{index}", model, index


def _instrument(
    context_id: str, models: Mapping[str, int], leaves: Sequence[str] = LEAVES
) -> ContextualRiskInstrument:
    """``leaves`` without taxonomy parents, asked of ``models``' samples in
    one context."""
    return leaves_instrument(leaves, models).model_copy(
        update={"context_id": context_id}
    )


def _capture(
    instrument: ContextualRiskInstrument,
    lacking: Iterable[tuple[str, str, int]],
    declaring: Collection[tuple[str, str, int]] = (),
) -> tuple[ContextualRiskInstrument, ContextualRiskArtifact]:
    """One context's capture as far as import's decision reads it: its
    unanswered cases, each a (leaf, model, sample index), and the cases
    whose direct answer declares the lab's annual business loss unavailable,
    each the same."""
    respondents = {row.key(): row for row in instrument.respondents}
    declared = ContextualRiskAnswer.model_validate(
        contextual_answer().model_dump(mode="json")
        | {
            "lab_annual_business_loss": {
                "status": "unavailable",
                "unavailable_reason": "No basis for it",
            }
        }
    )
    cases = (
        tuple(
            case
            for case in fixture_artifact(instrument, lambda _prompt: declared).cases
            if (case.node_id, *case.respondent.key()[1:]) in declaring
        )
        if declaring
        else ()
    )
    return instrument, ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=cases,
        unanswered_cases=tuple(
            UnansweredCase(
                node_id=leaf,
                anchor_score=instrument.anchor_scores[0],
                respondent=respondents[_key(model, index)],
                prompt_sha256="a" * 64,
                chain_answer=None,
                chain_prompt_sha256="b" * 64,
            )
            for leaf, model, index in lacking
        ),
    )


def _study(
    models: Mapping[str, int],
    lacking: Collection[tuple[str, str, str, str, int]] = (),
    *,
    nodes: Sequence[str] = ("n0", "n1"),
    leaves: Sequence[str] = LEAVES,
    declaring: Collection[tuple[str, str, str, str, int]] = (),
):
    """A study of ``nodes``, each with a capture for either source status,
    in which each of ``lacking``, a (node, status, leaf, model, sample
    index), is an unanswered case, and the direct answer of each of
    ``declaring``, the same, declares the lab's annual business loss
    unavailable."""
    assert {(node, status) for node, status, *_ in (*lacking, *declaring)} <= {
        (node, status) for node in nodes for status in STATUSES
    }
    return {
        node: [
            _capture(
                _instrument(f"{node}-{status}", models, leaves),
                [row[2:] for row in lacking if row[:2] == (node, status)],
                [row[2:] for row in declaring if row[:2] == (node, status)],
            )
            for status in STATUSES
        ]
        for node in nodes
    }


def _equal(nodes: Iterable[str]) -> dict[str, float]:
    nodes = list(nodes)
    return dict.fromkeys(nodes, 1 / len(nodes))


def test_a_model_does_not_answer_a_leaf_its_samples_lack_beyond_the_control():
    models = {"a": 5, "b": 5}
    # Two nodes of five samples: ten samples a model. One of them lacking the
    # leaf is the control's share, 10%, and not above it, so the model
    # answers the leaf and the sample is excluded.
    at_control = _study(models, [("n0", "public", "leaf_a", "a", 1)])
    assert unanswered_pairs(at_control) == ()
    exclusions = sample_exclusions(at_control, _equal(at_control))
    assert exclusions.pairs == ()
    assert exclusions.unanswered == {
        "n0": {_key("a", 1): (("n0-public", "leaf_a"),)},
        "n1": {},
    }
    assert all(not donors for donors in exclusions.donors.values())
    assert exclusions.weights["n0"] == pytest.approx(
        {
            **{_key("a", index): 1 / 8 for index in (0, 2, 3, 4)},
            **{_key("b", index): 1 / 10 for index in range(5)},
        }
    )
    # A second sample lacking it, at the other node, makes two of ten: the
    # model does not answer the leaf, and the leaf excludes no sample.
    above = _study(
        models,
        [("n0", "public", "leaf_a", "a", 1), ("n1", "private", "leaf_a", "a", 4)],
    )
    pair = UnansweredPair(leaf="leaf_a", model="a", lacking=2, samples=10)
    assert pair.share == 0.2
    assert unanswered_pairs(above) == (pair,)
    exclusions = sample_exclusions(above, _equal(above))
    assert exclusions.pairs == (pair,)
    assert exclusions.unanswered == {"n0": {}, "n1": {}}
    assert exclusions.excluded == 0
    for node in above:
        assert exclusions.weights[node] == pytest.approx(
            {_key(model, index): 1 / 10 for model in models for index in range(5)}
        )
        # Every sample of the model takes a donor, those that answered too.
        assert exclusions.borrowers(node) == {_key("a", index) for index in range(5)}
    assert exclusions.answering("leaf_a") == ("b",)
    assert exclusions.answering("leaf_b") == ("a", "b")
    # The control is the excluded-sample control: at 0.2 the same captures
    # exclude both samples instead, each within the model's share.
    assert unanswered_pairs(above, 0.2) == ()
    relaxed = sample_exclusions(above, _equal(above), max_share=0.2)
    assert relaxed.pairs == () and relaxed.excluded == 2


def test_a_sample_lacks_a_leaf_once_whichever_of_its_contexts_lack_it():
    models = {"a": 5, "b": 5}
    study = _study(
        models,
        [
            ("n0", "public", "leaf_a", "a", 1),
            ("n0", "private", "leaf_a", "a", 1),
        ],
    )
    # One sample, though two of its contexts lack the leaf: 10%, no pair.
    assert unanswered_pairs(study) == ()
    exclusions = sample_exclusions(study, _equal(study))
    assert exclusions.unanswered["n0"] == {
        _key("a", 1): (("n0-private", "leaf_a"), ("n0-public", "leaf_a"))
    }


def test_only_a_leaf_outside_its_models_unanswered_pairs_excludes_a_sample():
    models = {"a": 5, "b": 5}
    study = _study(
        models,
        [
            *(("n0", "public", "leaf_a", "a", index) for index in (0, 1, 2)),
            # One of those samples also lacks a leaf the model answers.
            ("n0", "private", "leaf_b", "a", 2),
        ],
    )
    exclusions = sample_exclusions(study, _equal(study))
    assert exclusions.pairs == (UnansweredPair("leaf_a", "a", 3, 10),)
    # The sample is excluded for the other leaf alone, and the guards count
    # what remains: one of the model's ten samples, at the control.
    assert exclusions.unanswered == {
        "n0": {_key("a", 2): (("n0-private", "leaf_b"),)},
        "n1": {},
    }
    assert exclusions.by_model()["a"] == {
        "samples": 10,
        "excluded": 1,
        "excluded_share": 0.1,
    }
    # An excluded sample has no state, so it borrows nothing.
    assert exclusions.borrowers("n0") == {_key("a", index) for index in (0, 1, 3, 4)}
    assert exclusions.borrowers("n1") == {_key("a", index) for index in range(5)}


def _declines(models: Collection[str], nodes: Sequence[str], samples: int):
    """Every sample of ``models`` lacking leaf_a at every one of ``nodes``."""
    return [
        (node, "public", "leaf_a", model, index)
        for node in nodes
        for model in models
        for index in range(samples)
    ]


def _lends(exclusions, node: str) -> Counter:
    """How many borrowers take each donor at ``node``."""
    return Counter(exclusions.donors[node].values())


def test_with_one_answering_model_each_of_its_samples_lends_equally_at_every_node():
    models = {"a": 3, "b": 3}
    nodes = ("n0", "n1", "n2")
    study = _study(models, _declines({"a"}, nodes, 3), nodes=nodes)
    exclusions = sample_exclusions(study, {"n0": 0.5, "n1": 0.3, "n2": 0.2})
    # The borrowers take the answering model's samples in turn, starting one
    # sample further on at each node.
    assert exclusions.donors == {
        "n0": {
            ("leaf_a", _key("a", 0)): _key("b", 0),
            ("leaf_a", _key("a", 1)): _key("b", 1),
            ("leaf_a", _key("a", 2)): _key("b", 2),
        },
        "n1": {
            ("leaf_a", _key("a", 0)): _key("b", 1),
            ("leaf_a", _key("a", 1)): _key("b", 2),
            ("leaf_a", _key("a", 2)): _key("b", 0),
        },
        "n2": {
            ("leaf_a", _key("a", 0)): _key("b", 2),
            ("leaf_a", _key("a", 1)): _key("b", 0),
            ("leaf_a", _key("a", 2)): _key("b", 1),
        },
    }
    for node in nodes:
        # So the leaf's pool at every node is the answering model's samples
        # in equal shares: each one's own sixth and its borrower's sixth.
        weights = exclusions.case_weights(node)
        assert {key: weights["leaf_a", key] for _, key in weights if key[1] == "b"} == (
            pytest.approx({_key("b", index): 1 / 3 for index in range(3)})
        )
        assert not any(key[1] == "a" for leaf, key in weights if leaf == "leaf_a")
        # Every other leaf keeps each respondent's own weight.
        assert {key: weights["leaf_b", key] for _, key in weights} == pytest.approx(
            exclusions.weights[node]
        )
    assert exclusions.pool_shares("leaf_a") == pytest.approx({"b": 1.0})


def test_two_models_that_do_not_answer_each_read_both_samples_of_the_one_that_does():
    models = {"a": 2, "b": 2, "c": 2}
    study = _study(models, _declines({"a", "c"}, ("n0",), 2))
    exclusions = sample_exclusions(study, _equal(study))
    assert [pair.model for pair in exclusions.pairs] == ["a", "c"]
    # The borrowers are listed model by model, a-0, a-1, c-0, c-1: four of
    # them for two donors, so each donor lends twice at either node, once to
    # either model.
    assert exclusions.donors == {
        "n0": {
            ("leaf_a", _key("a", 0)): _key("b", 0),
            ("leaf_a", _key("a", 1)): _key("b", 1),
            ("leaf_a", _key("c", 0)): _key("b", 0),
            ("leaf_a", _key("c", 1)): _key("b", 1),
        },
        "n1": {
            ("leaf_a", _key("a", 0)): _key("b", 1),
            ("leaf_a", _key("a", 1)): _key("b", 0),
            ("leaf_a", _key("c", 0)): _key("b", 1),
            ("leaf_a", _key("c", 1)): _key("b", 0),
        },
    }
    for node in study:
        weights = exclusions.case_weights(node)
        assert [weights["leaf_a", _key("b", index)] for index in range(2)] == (
            pytest.approx([1 / 2, 1 / 2])
        )
    assert exclusions.pool_shares("leaf_a") == pytest.approx({"b": 1.0})


def test_two_answering_models_of_two_samples_lend_all_four_over_the_nodes():
    models = {"a": 2, "b": 2, "c": 2}
    nodes = ("n0", "n1", "n2", "n3")
    study = _study(models, _declines({"a"}, nodes, 2), nodes=nodes)
    exclusions = sample_exclusions(study, {"n0": 0.4, "n1": 0.3, "n2": 0.2, "n3": 0.1})
    # The donors in rounds are b-0, c-0, b-1, c-1, and the two borrowers
    # take two in a row, one of either model, from one further on at each
    # node.
    assert exclusions.donors == {
        "n0": {
            ("leaf_a", _key("a", 0)): _key("b", 0),
            ("leaf_a", _key("a", 1)): _key("c", 0),
        },
        "n1": {
            ("leaf_a", _key("a", 0)): _key("c", 0),
            ("leaf_a", _key("a", 1)): _key("b", 1),
        },
        "n2": {
            ("leaf_a", _key("a", 0)): _key("b", 1),
            ("leaf_a", _key("a", 1)): _key("c", 1),
        },
        "n3": {
            ("leaf_a", _key("a", 0)): _key("c", 1),
            ("leaf_a", _key("a", 1)): _key("b", 0),
        },
    }
    # Over the four nodes every one of the four answering samples lends, and
    # equally often.
    assert sum((_lends(exclusions, node) for node in nodes), Counter()) == {
        _key(model, index): 2 for model in ("b", "c") for index in range(2)
    }
    # One sample of either answering model lends at every node, so their
    # shares of the leaf are equal whatever the nodes weigh.
    for node in nodes:
        weights = exclusions.case_weights(node)
        assert sorted(
            weights["leaf_a", _key(model, index)]
            for model in ("b", "c")
            for index in range(2)
        ) == pytest.approx([1 / 6, 1 / 6, 1 / 3, 1 / 3])
    assert exclusions.pool_shares("leaf_a") == pytest.approx({"b": 0.5, "c": 0.5})


def test_an_odd_number_of_borrowers_leaves_two_answering_models_unequal_at_a_node():
    models = {"a": 3, "b": 3, "c": 3}
    nodes = ("n0", "n1", "n2")
    study = _study(models, _declines({"a"}, nodes, 3), nodes=nodes)
    exclusions = sample_exclusions(study, {"n0": 0.5, "n1": 0.25, "n2": 0.25})
    # Six donors in rounds, b-0, c-0, b-1, c-1, b-2, c-2, and three borrowers.
    assert exclusions.donors == {
        "n0": {
            ("leaf_a", _key("a", 0)): _key("b", 0),
            ("leaf_a", _key("a", 1)): _key("c", 0),
            ("leaf_a", _key("a", 2)): _key("b", 1),
        },
        "n1": {
            ("leaf_a", _key("a", 0)): _key("c", 0),
            ("leaf_a", _key("a", 1)): _key("b", 1),
            ("leaf_a", _key("a", 2)): _key("c", 1),
        },
        "n2": {
            ("leaf_a", _key("a", 0)): _key("b", 1),
            ("leaf_a", _key("a", 1)): _key("c", 1),
            ("leaf_a", _key("a", 2)): _key("b", 2),
        },
    }
    # A node gives one answering model two of the three borrowers: its share
    # of the leaf there is 1/3 + 2/9 against 1/3 + 1/9. The two models keep
    # equally many samples, so their shares even out over the nodes, not at
    # each, and with unequal node weights not exactly.
    assert exclusions.pool_shares("leaf_a") == pytest.approx(
        {
            "b": 0.5 * 5 / 9 + 0.25 * 4 / 9 + 0.25 * 5 / 9,
            "c": 0.5 * 4 / 9 + 0.25 * 5 / 9 + 0.25 * 4 / 9,
        }
    )
    equal = sample_exclusions(
        {node: study[node] for node in ("n0", "n1")}, _equal(("n0", "n1"))
    )
    assert equal.pool_shares("leaf_a") == pytest.approx({"b": 0.5, "c": 0.5})
    # Over three nodes the six donors do not lend equally often, and one of
    # them not at all; over six, as many nodes as donors, each lends three
    # times.
    assert sum((_lends(exclusions, node) for node in nodes), Counter()) == {
        _key("b", 0): 1,
        _key("c", 0): 2,
        _key("b", 1): 3,
        _key("c", 1): 2,
        _key("b", 2): 1,
    }
    six = tuple(f"n{index}" for index in range(6))
    whole = sample_exclusions(
        _study(models, _declines({"a"}, six, 3), nodes=six), _equal(six)
    )
    assert sum((_lends(whole, node) for node in six), Counter()) == {
        _key(model, index): 3 for model in ("b", "c") for index in range(3)
    }


def test_two_models_that_do_not_answer_share_no_donor_while_one_is_unused():
    models = {"a": 2, "b": 2, "c": 2, "d": 2}
    study = _study(models, _declines({"a", "d"}, ("n0", "n1"), 2))
    exclusions = sample_exclusions(study, _equal(study))
    assert roster_models(exclusions.roster) == ("a", "b", "c", "d")
    # Four borrowers model by model, a-0, a-1, d-0, d-1, for the four donors
    # in rounds, b-0, c-0, b-1, c-1: each takes its own.
    assert exclusions.donors == {
        "n0": {
            ("leaf_a", _key("a", 0)): _key("b", 0),
            ("leaf_a", _key("a", 1)): _key("c", 0),
            ("leaf_a", _key("d", 0)): _key("b", 1),
            ("leaf_a", _key("d", 1)): _key("c", 1),
        },
        "n1": {
            ("leaf_a", _key("a", 0)): _key("c", 0),
            ("leaf_a", _key("a", 1)): _key("b", 1),
            ("leaf_a", _key("d", 0)): _key("c", 1),
            ("leaf_a", _key("d", 1)): _key("b", 0),
        },
    }
    for node in study:
        assert set(_lends(exclusions, node).values()) == {1}
        # Either model's two samples read a sample of either answering model.
        for model in ("a", "d"):
            assert {
                donor[1]
                for (_, borrower), donor in exclusions.donors[node].items()
                if borrower[1] == model
            } == {"b", "c"}


def test_the_rotation_balances_the_answering_samples_and_not_their_models():
    # The answering models keep four samples and two: six donors in rounds,
    # b-0, c-0, b-1, c-1, b-2, b-3, and two borrowers.
    models = {"a": 2, "b": 4, "c": 2}
    nodes = tuple(f"n{index}" for index in range(6))
    study = _study(models, _declines({"a"}, nodes, 2), nodes=nodes)
    exclusions = sample_exclusions(study, _equal(nodes))
    # Over six nodes, as many as there are donors, each donor lends twice.
    assert sum((_lends(exclusions, node) for node in nodes), Counter()) == {
        **{_key("b", index): 2 for index in range(4)},
        **{_key("c", index): 2 for index in range(2)},
    }
    # So the borrowing model's third of the pool goes to the answering
    # models four to two, beside their own thirds: not a half each.
    assert exclusions.pool_shares("leaf_a") == pytest.approx(
        {"b": 1 / 3 + 1 / 3 * 4 / 6, "c": 1 / 3 + 1 / 3 * 2 / 6}
    )
    assert exclusions.pool_shares("leaf_a") == pytest.approx({"b": 5 / 9, "c": 4 / 9})
    # An excluded sample has the same effect. Two answering models of two
    # samples each lend one sample each at every one of five nodes, a half
    # of the pool either.
    even = {"a": 2, "b": 2, "c": 2}
    five = tuple(f"n{index}" for index in range(5))
    lacking = _declines({"a"}, five, 2)
    whole = sample_exclusions(_study(even, lacking, nodes=five), _equal(five))
    assert all(set(_lends(whole, node).values()) == {1} for node in five)
    assert whole.pool_shares("leaf_a") == pytest.approx({"b": 0.5, "c": 0.5})
    # Where c loses a sample for another leaf, the donors are b-0, c-0, b-1,
    # and at that node, the third, the borrowers take the two of b: two
    # thirds of the pool there against c's own third.
    excluded = sample_exclusions(
        _study(even, [*lacking, ("n2", "public", "leaf_b", "c", 1)], nodes=five),
        _equal(five),
    )
    assert excluded.unanswered["n2"] == {_key("c", 1): (("n2-public", "leaf_b"),)}
    assert _lends(excluded, "n2") == {_key("b", 0): 1, _key("b", 1): 1}
    assert excluded.pool_shares("leaf_a") == pytest.approx(
        {"b": 0.8 * 1 / 2 + 0.2 * 2 / 3, "c": 0.8 * 1 / 2 + 0.2 * 1 / 3}
    )


def _samples(model: str, *indexes: int) -> list[Respondent]:
    return [
        Respondent(respondent_id=f"{model}-{index}", model_id=model, sample_index=index)
        for index in indexes
    ]


def test_models_that_keep_unequal_samples_are_listed_until_each_runs_out():
    # One answering model keeps one sample and the other three: the donors
    # in rounds are b-0, c-0, c-1, c-2. The models that do not answer keep
    # two and three: the borrowers model by model are a-0, a-1, d-0, d-1,
    # d-2, one more than the donors.
    kept = [
        *_samples("a", 0, 1),
        *_samples("b", 0),
        *_samples("c", 0, 1, 2),
        *_samples("d", 0, 1, 2),
    ]
    assert leaf_donors("leaf_a", {"n0": kept, "n1": kept}, ["a", "d"], ["b", "c"]) == {
        "n0": {
            _key("a", 0): _key("b", 0),
            _key("a", 1): _key("c", 0),
            _key("d", 0): _key("c", 1),
            _key("d", 1): _key("c", 2),
            _key("d", 2): _key("b", 0),
        },
        "n1": {
            _key("a", 0): _key("c", 0),
            _key("a", 1): _key("c", 1),
            _key("d", 0): _key("c", 2),
            _key("d", 1): _key("b", 0),
            _key("d", 2): _key("c", 0),
        },
    }
    # A model that keeps more samples than there are donors takes one of
    # them twice: each donor lends once or twice.
    crowded = [*_samples("a", 0, 1, 2), *_samples("b", 0), *_samples("c", 0)]
    assert leaf_donors("leaf_a", {"n0": crowded}, ["a"], ["b", "c"]) == {
        "n0": {
            _key("a", 0): _key("b", 0),
            _key("a", 1): _key("c", 0),
            _key("a", 2): _key("b", 0),
        }
    }


def test_donors_follow_sample_indexes_and_skip_excluded_samples():
    respondents = {
        model: [
            Respondent(
                respondent_id=f"{model}:{index}", model_id=model, sample_index=index
            )
            for index in indexes
        ]
        for model, indexes in {"a": (0, 2, 10), "b": (1, 2, 11)}.items()
    }
    # Respondent IDs sort "b:11" before "b:2"; the rotation reads sample
    # indexes. At the second node the answering model keeps two samples.
    kept = {
        "n0": [*respondents["a"], *respondents["b"]],
        "n1": [*respondents["a"], respondents["b"][2], respondents["b"][0]],
    }
    donors = leaf_donors("leaf_a", kept, ["a"], ["b"])
    assert donors == {
        "n0": {
            ("a:0", "a", 0): ("b:1", "b", 1),
            ("a:2", "a", 2): ("b:2", "b", 2),
            ("a:10", "a", 10): ("b:11", "b", 11),
        },
        # Position one: the borrowers start at the second of the samples
        # kept there, 1 and 11.
        "n1": {
            ("a:0", "a", 0): ("b:11", "b", 11),
            ("a:2", "a", 2): ("b:1", "b", 1),
            ("a:10", "a", 10): ("b:11", "b", 11),
        },
    }


def test_import_refuses_a_leaf_that_no_model_answers():
    models = {"a": 5, "b": 5}
    study = _study(
        models,
        [
            ("n0", "public", "leaf_a", "a", 0),
            ("n1", "public", "leaf_a", "a", 0),
            *(("n1", "private", "leaf_a", "b", index) for index in range(3)),
        ],
    )
    with pytest.raises(
        ValueError,
        match=r"no model answers leaf 'leaf_a': model a lacks it in 2 of its 10 "
        r"samples \(20\.0%\); model b lacks it in 3 of its 10 samples \(30\.0%\), "
        r"above the frozen control max_excluded_sample_share 0\.1 "
        r"\(POOL-UNANSWERED-SCENARIOS\); recollect the study with a model that "
        r"answers it",
    ):
        sample_exclusions(study, _equal(study))


def test_import_refuses_a_model_that_does_not_answer_most_leaves():
    models = {"a": 2, "b": 2}
    lacking = [
        ("n0", "public", leaf, "a", index)
        for leaf in ("leaf_a", "leaf_b")
        for index in range(2)
    ]
    study = _study(models, lacking)
    with pytest.raises(
        ValueError,
        match=r"model a does not answer 2 of the study's 3 leaves, more than half "
        r"\(leaf_a, leaf_b\), so it should not be in the roster "
        r"\(POOL-UNANSWERED-SCENARIOS\); recollect the study without it",
    ):
        sample_exclusions(study, _equal(study))
    # Half of the leaves is not more than half.
    four = (*LEAVES, "leaf_d")
    half = sample_exclusions(_study(models, lacking, leaves=four), _equal(study))
    assert [(pair.leaf, pair.model) for pair in half.pairs] == [
        ("leaf_a", "a"),
        ("leaf_b", "a"),
    ]


def test_the_rotation_refuses_a_node_where_no_answering_model_keeps_a_sample():
    borrower = _samples("a", 0)
    with pytest.raises(
        ValueError,
        match=r"no model that answers leaf 'leaf_a' keeps a sample at node 'n0' "
        r"\(POOL-UNANSWERED-SCENARIOS\); recollect the study",
    ):
        leaf_donors("leaf_a", {"n0": borrower}, ["a"], ["b", "c"])
    # An answering model without a kept sample there is passed over while
    # another keeps one.
    assert leaf_donors(
        "leaf_a", {"n0": [*borrower, *_samples("c", 1)]}, ["a"], ["b", "c"]
    ) == {"n0": {_key("a", 0): _key("c", 1)}}


def test_the_sample_guards_count_what_the_unanswered_pairs_leave():
    # Ten nodes of two samples: twenty samples a model.
    nodes = tuple(f"n{index}" for index in range(10))
    models = {"a": 2, "b": 2}
    declined = [
        (node, "public", "leaf_c", "a", index) for node in nodes for index in range(2)
    ]
    # Both samples at one node lack a leaf the model answers elsewhere: two
    # of twenty, no pair, and no sample left to carry the model there.
    floor = _study(
        models,
        [*declined, *(("n3", "private", "leaf_a", "a", index) for index in range(2))],
        nodes=nodes,
    )
    with pytest.raises(
        ValueError,
        match=r"model a keeps 0 of its 2 samples at node 'n3', fewer than half "
        r"\(unanswered leaves: leaf_a\), because a leaf's direct question exhausted "
        r"its validation attempts \(POOL-EXCLUDED-SAMPLES\); recollect the study",
    ):
        sample_exclusions(floor, _equal(nodes))
    # Three samples excluded for three leaves, none of them the model's
    # unanswered pair: 15% of its samples.
    share = _study(
        {"a": 2, "b": 2},
        [
            *declined,
            ("n1", "public", "leaf_a", "a", 0),
            ("n2", "public", "leaf_b", "a", 0),
            ("n3", "private", "leaf_a", "a", 1),
        ],
        nodes=nodes,
    )
    with pytest.raises(
        ValueError,
        match=r"model a has 3 of its 20 samples \(15\.0%\) excluded, above the "
        r"frozen control max_excluded_sample_share 0\.1, because a leaf's direct "
        r"question exhausted its validation attempts \(POOL-EXCLUDED-SAMPLES\)",
    ):
        sample_exclusions(share, _equal(nodes))
    # Without the third, two of twenty are within it, beside the pair.
    kept = sample_exclusions(
        _study(
            models,
            [
                *declined,
                ("n1", "public", "leaf_a", "a", 0),
                ("n2", "public", "leaf_b", "a", 0),
            ],
            nodes=nodes,
        ),
        _equal(nodes),
    )
    assert kept.pairs == (UnansweredPair("leaf_c", "a", 20, 20),)
    assert kept.excluded == 2


def _left_out(*rows: tuple[str, str, str, int, str, int]):
    """The samples of one node left out of the views that count the lab's
    loss, the lab-loss views and the gross views, from rows of
    (context ID, leaf, model, sample index, answering model, its sample
    index): each sample with the declaring cases its states read."""
    found: dict = {}
    for context, leaf, model, index, answering, answering_index in rows:
        found.setdefault(_key(model, index), []).append(
            (context, leaf, _key(answering, answering_index))
        )
    return {key: tuple(sorted(cases)) for key, cases in found.items()}


def test_a_lab_loss_declared_unavailable_changes_no_other_decision_of_import():
    """A direct answer that declares the lab's loss unavailable leaves its
    sample out of the views that count the lab's loss and changes nothing
    else: the kept
    samples, their weights, the unanswered pairs and the donors are those of
    the study without it."""
    models = {"a": 5, "b": 5}
    lacking = [
        # One sample excluded for an exhausted direct question, within the
        # control, and model b not answering leaf_c at all.
        ("n1", "private", "leaf_b", "b", 4),
        *(("n0", "public", "leaf_c", "b", index) for index in range(5)),
        *(("n1", "public", "leaf_c", "b", index) for index in range(5)),
    ]
    given = sample_exclusions(_study(models, lacking), _equal(("n0", "n1")))
    study = _study(models, lacking, declaring=[("n0", "public", "leaf_a", "a", 2)])
    exclusions = sample_exclusions(study, _equal(study))
    for field in ("weights", "unanswered", "pairs", "donors", "node_weights"):
        assert getattr(exclusions, field) == getattr(given, field), field
    assert exclusions.excluded == 1 and exclusions.by_model() == given.by_model()
    assert exclusions.pairs == (UnansweredPair("leaf_c", "b", 10, 10),)
    assert given.lab_loss_unavailable == frozenset()
    assert given.lab_loss_left_out == {} and given.lab_loss_withheld is None
    assert exclusions.lab_loss_unavailable == {
        ("n0", _key("a", 2), "n0-public", "leaf_a")
    }
    assert exclusions.lab_loss_left_out == {
        "n0": _left_out(("n0-public", "leaf_a", "a", 2, "a", 2)),
        "n1": {},
    }
    assert exclusions.left_out("n0") == {_key("a", 2)}
    assert exclusions.left_out("n1") == frozenset()
    assert exclusions.lab_loss_withheld is None
    # The sample keeps its weight, a tenth, like its model's other samples.
    assert exclusions.weights["n0"][_key("a", 2)] == pytest.approx(1 / 10)


def test_a_sample_that_borrows_a_case_declaring_the_lab_loss_unavailable_is_left_out():
    """Model a does not answer leaf_a, so each of its samples reads a sample
    of b's case of it. Where that donor's answer declares the lab's loss
    unavailable, the donor and the sample that borrows from it at that node
    are both left out of the views that count the lab's loss; the samples
    that borrow the
    donor's case at another node, or another donor's, are not."""
    models = {"a": 5, "b": 5}
    nodes = ("n0", "n1")
    lacking = _declines({"a"}, nodes, 5)
    study = _study(
        models, lacking, nodes=nodes, declaring=[("n0", "private", "leaf_a", "b", 3)]
    )
    exclusions = sample_exclusions(study, _equal(nodes))
    given = sample_exclusions(_study(models, lacking, nodes=nodes), _equal(nodes))
    for field in ("weights", "unanswered", "pairs", "donors"):
        assert getattr(exclusions, field) == getattr(given, field), field
    # At n0 the borrowers take b's samples in order, so a-3 reads b-3; at n1
    # the rotation starts one donor further on, and a-2 reads b-3 there.
    assert exclusions.donors["n0"]["leaf_a", _key("a", 3)] == _key("b", 3)
    assert exclusions.donors["n1"]["leaf_a", _key("a", 2)] == _key("b", 3)
    assert exclusions.lab_loss_left_out == {
        "n0": _left_out(
            ("n0-private", "leaf_a", "a", 3, "b", 3),
            ("n0-private", "leaf_a", "b", 3, "b", 3),
        ),
        "n1": {},
    }
    assert exclusions.lab_loss_withheld is None
    # A borrower's own answer to another leaf leaves it out by itself.
    own = sample_exclusions(
        _study(
            models,
            lacking,
            nodes=nodes,
            declaring=[("n1", "public", "leaf_b", "a", 0)],
        ),
        _equal(nodes),
    )
    assert own.lab_loss_left_out == {
        "n0": {},
        "n1": _left_out(("n1-public", "leaf_b", "a", 0, "a", 0)),
    }


def test_a_case_no_kept_state_reads_leaves_no_sample_out():
    """A declaring answer leaves a sample out only through a state that reads
    it. An excluded sample's answers are read by none, and so are a model's
    answers to a leaf it does not answer, which are set aside."""
    models = {"a": 5, "b": 5}
    lacking = [
        ("n0", "public", "leaf_b", "a", 1),
        # Two of a's ten samples lack leaf_a: the model does not answer it,
        # and its eight answers to it are set aside.
        ("n0", "public", "leaf_a", "a", 0),
        ("n1", "public", "leaf_a", "a", 4),
    ]
    study = _study(
        models,
        lacking,
        declaring=[
            # The excluded sample's answer to another leaf.
            ("n0", "private", "leaf_c", "a", 1),
            # An answer of the model's to the leaf it does not answer.
            ("n1", "public", "leaf_a", "a", 2),
        ],
    )
    exclusions = sample_exclusions(study, _equal(study))
    assert exclusions.unanswered["n0"] == {_key("a", 1): (("n0-public", "leaf_b"),)}
    assert exclusions.pairs == (UnansweredPair("leaf_a", "a", 2, 10),)
    assert len(exclusions.lab_loss_unavailable) == 2
    assert exclusions.lab_loss_left_out == {"n0": {}, "n1": {}}
    assert exclusions.lab_loss_withheld is None


def test_the_sample_guards_count_the_excluded_and_the_left_out_samples_together():
    """Where leaving samples out of the views that count the lab's loss
    would breach a sample
    guard, none is left out and the guard is named; the import of everything
    else is the same, and nothing is refused."""
    models = {"a": 5, "b": 5}
    excluded = [("n1", "private", "leaf_b", "a", 4)]
    given = sample_exclusions(_study(models, excluded), _equal(("n0", "n1")))
    # One sample of a excluded and one left out: two of its ten.
    study = _study(models, excluded, declaring=[("n0", "public", "leaf_a", "a", 2)])
    exclusions = sample_exclusions(study, _equal(study))
    assert exclusions.lab_loss_left_out == {}
    assert exclusions.lab_loss_withheld == (
        "model a would have 2 of its 10 samples (20.0%) excluded or left out of "
        "the views that count the lab's own loss, above the frozen control "
        "max_excluded_sample_share 0.1"
    )
    assert len(exclusions.lab_loss_unavailable) == 1
    for field in ("weights", "unanswered", "pairs", "donors"):
        assert getattr(exclusions, field) == getattr(given, field), field
    # At their share the two fit, and the one that declared is left out.
    fitted = sample_exclusions(study, _equal(study), max_share=0.2)
    assert fitted.lab_loss_withheld is None
    assert fitted.lab_loss_left_out == {
        "n0": _left_out(("n0-public", "leaf_a", "a", 2, "a", 2)),
        "n1": {},
    }
    assert fitted.excluded == 1
    # A model keeps at least half its samples, rounded up, in the lab-loss
    # views at every node: three of five here, whatever the shares.
    three = [("n0", "public", "leaf_a", "b", index) for index in (0, 1, 2)]
    floor = sample_exclusions(
        _study(models, declaring=three), _equal(("n0", "n1")), max_share=1.0
    )
    assert floor.lab_loss_left_out == {} and floor.excluded == 0
    assert floor.lab_loss_withheld == (
        "model b would keep 2 of its 5 samples in the views that count the lab's "
        "own loss at node 'n0', fewer than half"
    )
    two = sample_exclusions(
        _study(models, declaring=three[:2]), _equal(("n0", "n1")), max_share=1.0
    )
    assert two.lab_loss_withheld is None and len(two.left_out("n0")) == 2
    # The study's share is counted over both models.
    assert sample_exclusions(
        _study(models, declaring=three[:2]), _equal(("n0", "n1"))
    ).lab_loss_withheld == (
        "model b would have 2 of its 10 samples (20.0%) excluded or left out of "
        "the views that count the lab's own loss, above the frozen control "
        "max_excluded_sample_share 0.1"
    )


def test_a_study_shares_one_roster_and_one_set_of_leaves():
    models = {"a": 2, "b": 2}
    study = _study(models)
    other_roster = {
        **study,
        "n1": [_capture(_instrument("n1-public", {"a": 2, "c": 2}), ())],
    }
    with pytest.raises(ValueError, match="shares one respondent roster"):
        sample_exclusions(other_roster, _equal(study))
    other_leaves = {
        **study,
        "n1": [_capture(_instrument("n1-public", models, LEAVES[:2]), ())],
    }
    with pytest.raises(ValueError, match="asks the same leaves"):
        sample_exclusions(other_leaves, _equal(study))
    with pytest.raises(ValueError, match="a study needs a capture"):
        sample_exclusions({}, {})
    for control in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=r"control must lie in \[0, 1\]"):
            sample_exclusions(study, _equal(study), max_share=control)
    assert MAX_EXCLUDED_SAMPLE_SHARE == 0.10


# A variant study and its headline


def _shared(nodes: Sequence[str]) -> dict[str, str]:
    """Each of a variant's nodes mapped to the headline node of its ID."""
    return {node: node for node in nodes}


def _donors(leaf: str, table: Mapping[tuple[str, int], tuple[str, int]]):
    """One node's donors for ``leaf``: each borrower with its donor, either
    a (model, sample index)."""
    return {(leaf, _key(*borrower)): _key(*donor) for borrower, donor in table.items()}


def test_a_variant_takes_its_headlines_pairs_and_donors():
    models = {"a": 2, "b": 2}
    nodes = ("n0", "n1", "n2", "n3")
    headline = sample_exclusions(
        _study(models, _declines({"a"}, nodes, 2), nodes=nodes), _equal(nodes)
    )
    assert headline.pairs == (UnansweredPair("leaf_a", "a", 8, 8, "own"),)
    assert headline.headline is None and headline.matched_control() is None
    # A variant on two of the headline's nodes, in which every sample
    # answers: decided on its own captures it has no pair.
    shared = ("n1", "n3")
    variant = _study(models, nodes=shared)
    assert sample_exclusions(variant, _equal(shared)).pairs == ()
    followed = sample_exclusions(
        variant, _equal(shared), headline=headline.followed_at(_shared(shared))
    )
    # It leaves the model out of the leaf all the same, and excludes nothing.
    assert followed.pairs == (UnansweredPair("leaf_a", "a", 0, 4, "headline"),)
    assert followed.answering("leaf_a") == ("b",)
    assert followed.excluded == 0
    assert followed.borrowers("n1") == {_key("a", 0), _key("a", 1)}
    # Each borrower takes the donor it has in the headline at the node: the
    # headline's second and fourth nodes both start at its second donor.
    crossed = _donors("leaf_a", {("a", 0): ("b", 1), ("a", 1): ("b", 0)})
    assert followed.donors == {"n1": crossed, "n3": crossed}
    assert followed.donors == {node: headline.donors[node] for node in shared}
    assert followed.matched_control() == {
        "pairs_not_in_headline": 0,
        "donors": 4,
        "donors_not_headline": 0,
        "differs": False,
    }
    # The variant's own rotation would start at its first node with the
    # first donor, as it does where its captures alone make the pair.
    declined = _study(models, _declines({"a"}, shared, 2), nodes=shared)
    alone = sample_exclusions(declined, _equal(shared))
    assert alone.donors == {
        "n1": _donors("leaf_a", {("a", 0): ("b", 0), ("a", 1): ("b", 1)}),
        "n3": crossed,
    }
    # Where its captures make the pair too, it still takes the headline's
    # donors.
    both = sample_exclusions(
        declined, _equal(shared), headline=headline.followed_at(_shared(shared))
    )
    assert both.pairs == (UnansweredPair("leaf_a", "a", 4, 4, "own_and_headline"),)
    assert both.donors == followed.donors
    assert both.matched_control() == followed.matched_control()


def test_a_variants_sample_lacking_a_leaf_it_follows_is_not_excluded():
    models = {"a": 5, "b": 5}
    nodes = ("n0", "n1")
    headline = sample_exclusions(
        _study(models, _declines({"a"}, nodes, 5)), _equal(nodes)
    )
    # One of the model's ten samples lacks the leaf in the variant, the
    # control's share and not above it: on its own captures the variant
    # excludes the sample.
    variant = _study(models, [("n0", "private", "leaf_a", "a", 3)])
    alone = sample_exclusions(variant, _equal(nodes))
    assert alone.pairs == () and alone.excluded == 1
    # Following the headline it sets the model's answers to the leaf aside,
    # the nine given included, and keeps the sample.
    followed = sample_exclusions(
        variant, _equal(nodes), headline=headline.followed_at(_shared(nodes))
    )
    assert followed.pairs == (UnansweredPair("leaf_a", "a", 1, 10, "headline"),)
    assert followed.excluded == 0 and followed.unanswered == {"n0": {}, "n1": {}}
    assert followed.borrowers("n0") == {_key("a", index) for index in range(5)}
    assert ("leaf_a", _key("a", 0)) not in followed.case_weights("n0")
    assert followed.donors == headline.donors
    # A leaf outside the pairs still excludes its sample.
    other = sample_exclusions(
        _study(models, [("n0", "private", "leaf_b", "a", 3)]),
        _equal(nodes),
        headline=headline.followed_at(_shared(nodes)),
    )
    assert other.unanswered["n0"] == {_key("a", 3): (("n0-private", "leaf_b"),)}
    assert other.borrowers("n0") == {_key("a", index) for index in (0, 1, 2, 4)}


def test_a_pair_of_the_variant_alone_takes_the_variants_own_rotation():
    models = {"a": 2, "b": 2}
    nodes = ("n0", "n1", "n2", "n3")
    headline = sample_exclusions(_study(models, nodes=nodes), _equal(nodes))
    assert headline.pairs == ()
    shared = ("n1", "n3")
    variant = _study(models, _declines({"a"}, shared, 2), nodes=shared)
    followed = sample_exclusions(
        variant, _equal(shared), headline=headline.followed_at(_shared(shared))
    )
    assert followed.pairs == (UnansweredPair("leaf_a", "a", 4, 4, "own"),)
    assert followed.donors == sample_exclusions(variant, _equal(shared)).donors
    # The headline answers the leaf with both models, so the variant and its
    # matched control differ.
    assert followed.matched_control() == {
        "pairs_not_in_headline": 1,
        "donors": 0,
        "donors_not_headline": 0,
        "differs": True,
    }


def test_a_headline_donor_the_variant_cannot_use_gives_way_to_its_rotation():
    # A donor the variant excludes at the node.
    models = {"a": 5, "b": 5}
    nodes = ("n0", "n1")
    headline = sample_exclusions(
        _study(models, _declines({"a"}, nodes, 5)), _equal(nodes)
    )
    straight = {("a", index): ("b", index) for index in range(5)}
    assert headline.donors["n0"] == _donors("leaf_a", straight)
    variant = _study(models, [("n0", "public", "leaf_b", "b", 2)])
    followed = sample_exclusions(
        variant, _equal(nodes), headline=headline.followed_at(_shared(nodes))
    )
    assert followed.unanswered["n0"] == {_key("b", 2): (("n0-public", "leaf_b"),)}
    # Its borrower takes the donor of the variant's rotation over the four
    # samples kept there, b-0, b-1, b-3, b-4: the third of them for the third
    # borrower. Every other borrower keeps its headline donor.
    assert followed.donors == {
        "n0": _donors("leaf_a", {**straight, ("a", 2): ("b", 3)}),
        "n1": headline.donors["n1"],
    }
    assert followed.matched_control() == {
        "pairs_not_in_headline": 0,
        "donors": 10,
        "donors_not_headline": 1,
        "differs": True,
    }
    # A donor whose model does not answer the leaf in the variant.
    models = {"a": 2, "b": 2, "c": 2}
    headline = sample_exclusions(
        _study(models, _declines({"a"}, nodes, 2)), _equal(nodes)
    )
    assert headline.donors == {
        "n0": _donors("leaf_a", {("a", 0): ("b", 0), ("a", 1): ("c", 0)}),
        "n1": _donors("leaf_a", {("a", 0): ("c", 0), ("a", 1): ("b", 1)}),
    }
    followed = sample_exclusions(
        _study(models, _declines({"c"}, nodes, 2)),
        _equal(nodes),
        headline=headline.followed_at(_shared(nodes)),
    )
    assert followed.pairs == (
        UnansweredPair("leaf_a", "a", 0, 4, "headline"),
        UnansweredPair("leaf_a", "c", 4, 4, "own"),
    )
    # The variant's rotation gives its four borrowers, a-0, a-1, c-0, c-1,
    # the two samples of b in turn. The borrowers of the headline's pair keep
    # a headline donor of b and take the rotation's in place of one of c.
    assert followed.donors == {
        "n0": _donors(
            "leaf_a",
            {
                ("a", 0): ("b", 0),
                ("a", 1): ("b", 1),
                ("c", 0): ("b", 0),
                ("c", 1): ("b", 1),
            },
        ),
        "n1": _donors(
            "leaf_a",
            {
                ("a", 0): ("b", 1),
                ("a", 1): ("b", 1),
                ("c", 0): ("b", 1),
                ("c", 1): ("b", 0),
            },
        ),
    }
    assert followed.matched_control() == {
        "pairs_not_in_headline": 1,
        "donors": 4,
        "donors_not_headline": 2,
        "differs": True,
    }


def test_the_guards_count_a_variants_pairs_with_those_it_follows():
    models = {"a": 2, "b": 2}
    nodes = ("n0", "n1")
    headline = sample_exclusions(
        _study(models, _declines({"a"}, nodes, 2)), _equal(nodes)
    ).followed_at(_shared(nodes))
    # The model that answers the leaf in the headline does not in the
    # variant, so no model answers it there.
    with pytest.raises(
        ValueError,
        match=r"no model answers leaf 'leaf_a': model b lacks it in 4 of its 4 "
        r"samples \(100\.0%\), above the frozen control max_excluded_sample_share "
        r"0\.1; model a does not answer it in the headline study, which a variant "
        r"study follows \(POOL-UNANSWERED-SCENARIOS\); recollect the study with a "
        r"model that answers it",
    ):
        sample_exclusions(
            _study(models, _declines({"b"}, nodes, 2)), _equal(nodes), headline=headline
        )
    # A second leaf the model does not answer, in the variant alone, makes
    # two of the three leaves.
    second = [
        (node, "public", "leaf_b", "a", index) for node in nodes for index in (0, 1)
    ]
    with pytest.raises(
        ValueError,
        match=r"model a does not answer 2 of the study's 3 leaves, more than half "
        r"\(leaf_a, leaf_b\)",
    ):
        sample_exclusions(_study(models, second), _equal(nodes), headline=headline)
    assert [
        (pair.leaf, pair.basis)
        for pair in sample_exclusions(
            _study(models, second, leaves=(*LEAVES, "leaf_d")),
            _equal(nodes),
            headline=headline,
        ).pairs
    ] == [("leaf_a", "headline"), ("leaf_b", "own")]
    # A headline of other models, leaves or nodes is not the variant's.
    study = _study(models)
    for other in (
        Headline(frozenset({("c", "leaf_a")}), headline.donors),
        Headline(frozenset({("a", "leaf_z")}), headline.donors),
        Headline(headline.pairs, {"n0": headline.donors["n0"]}),
    ):
        with pytest.raises(ValueError, match="a variant study's headline asks"):
            sample_exclusions(study, _equal(nodes), headline=other)


# The views of a capture, and a borrower's taxonomy pools

POOL_PANEL = {"fixture-a": 2, "fixture-b": 2}
SCALES = {
    _key("fixture-a", 0): 1,
    _key("fixture-a", 1): 2,
    _key("fixture-b", 0): 3,
    _key("fixture-b", 1): 5,
}
"""Each respondent's multiple of the pool fixture's counts, so that every
respondent's answers to a leaf differ."""
A_0, A_1, B_0, B_1 = SCALES
POOL_PARENTS = {"mid": parent_answer(1600, 100), "top": parent_answer(8000, 100)}
"""Each parent's answer, the same from every respondent at every anchor."""


def _scaled_leaf(prompt: ContextualPrompt) -> ContextualRiskAnswer:
    return counted_answer(
        POOL_FIXTURE_COUNTS[prompt.node.node_id] * SCALES[prompt.respondent.key()]
    )


def _declined(
    artifact: ContextualRiskArtifact, lacking: Collection[tuple[str, float, tuple]]
) -> ContextualRiskArtifact:
    """``artifact`` as collection records it where the direct question of
    each case key in ``lacking`` exhausted its attempts: the case unanswered,
    with its chain answer, and its group's parents answered but not pooled."""
    incomplete = {(anchor, respondent) for _, anchor, respondent in lacking}
    return artifact.model_copy(
        update={
            "cases": tuple(
                case for case in artifact.cases if case.key() not in lacking
            ),
            "unanswered_cases": tuple(
                UnansweredCase(
                    node_id=case.node_id,
                    anchor_score=case.anchor_score,
                    respondent=case.respondent,
                    prompt_sha256=case.prompt_sha256,
                    chain_answer=case.chain_answer,
                    chain_prompt_sha256=case.chain_prompt_sha256,
                )
                for case in artifact.cases
                if case.key() in lacking
            ),
            "parent_cases": tuple(
                case.model_copy(
                    update={
                        "children_median_eal": None,
                        "children_mean_eal": None,
                        "pool": None,
                    }
                )
                if (case.anchor_score, case.respondent.key()) in incomplete
                else case
                for case in artifact.parent_cases
            ),
        }
    )


@pytest.fixture(scope="module")
def pooled():
    """The pool fixture asked of two models' two samples, in which one sample
    of model a lacks leaf_b at one anchor: the instrument, the capture and
    import's decision, in which model a does not answer leaf_b."""
    instrument = pool_instrument(POOL_PANEL)
    complete = hierarchy_artifact(
        instrument, _scaled_leaf, lambda parent_id, _score: POOL_PARENTS[parent_id]
    )
    artifact = _declined(complete, {("leaf_b", 0.5, A_0)})
    # The capture is one collection could have stored.
    checked_hierarchy_outcomes(instrument, artifact)
    node = instrument.context_id
    exclusions = sample_exclusions({node: ((instrument, artifact),)}, {node: 1.0})
    assert exclusions.pairs == (UnansweredPair("leaf_b", "fixture-a", 1, 2),)
    assert exclusions.donors[node] == {("leaf_b", A_0): B_0, ("leaf_b", A_1): B_1}
    return instrument, artifact, exclusions


def test_the_kept_view_holds_each_stored_case_once_without_the_answers_set_aside(
    pooled,
):
    instrument, artifact, exclusions = pooled
    kept = exclusions.kept_view(instrument.context_id, artifact)
    # Four respondents, three leaves and three anchors, less the model's six
    # cases of the leaf: the unanswered one and the five its samples answered.
    assert len(artifact.cases) == 35 and len(kept.cases) == 30
    assert not kept.unanswered_cases
    stored = {case.key(): case for case in artifact.cases}
    assert all(case is stored[case.key()] for case in kept.cases)
    assert not any(
        case.node_id == "leaf_b" and case.respondent.model_id == "fixture-a"
        for case in kept.cases
    )
    assert kept.parent_cases == artifact.parent_cases
    assert kept.instrument_sha256 == artifact.instrument_sha256


def test_the_run_view_gives_each_borrower_its_donors_whole_case_at_every_anchor(
    pooled,
):
    instrument, artifact, exclusions = pooled
    node = instrument.context_id
    kept = exclusions.kept_view(node, artifact)
    read = exclusions.run_view(node, artifact)
    assert read.cases[:30] == kept.cases
    borrowed = read.cases[30:]
    stored = {case.key(): case for case in artifact.cases}
    assert {case.key() for case in borrowed} == {
        ("leaf_b", anchor, borrower)
        for anchor in instrument.anchor_scores
        for borrower in (A_0, A_1)
    }
    respondents = {row.key(): row for row in instrument.respondents}
    for case in borrowed:
        donor = {A_0: B_0, A_1: B_1}[case.respondent.key()]
        # Everything the donor's case holds, under the borrower's identity.
        assert case == stored["leaf_b", case.anchor_score, donor].model_copy(
            update={"respondent": respondents[case.respondent.key()]}
        )
        assert case.chain_answer is not None and case.eal_crosscheck is not None
    # The view has every leaf of every kept respondent once, so no group of
    # it is incomplete, and a borrower's parent cases keep their answers and
    # not the pools stored for other leaf answers.
    assert len({case.key() for case in read.cases}) == len(read.cases) == 36
    assert not read.incomplete_groups()
    parents = {case.key(): case for case in artifact.parent_cases}
    for case in read.parent_cases:
        own = parents[case.key()]
        assert case.answer == own.answer and case.answer is not None
        if case.respondent.model_id == "fixture-a":
            assert case.pool is None and case.children_mean_eal is None
        else:
            assert case == own and case.pool is not None
    # A study without an unanswered pair reads its kept cases.
    complete = sample_exclusions(
        {node: ((instrument, artifact),)}, {node: 1.0}, max_share=0.5
    )
    assert complete.pairs == ()
    assert complete.run_view(node, artifact) == complete.kept_view(node, artifact)
    assert selection_cases(instrument, artifact, complete, node) == (artifact, None)


def test_a_borrowers_pools_read_the_borrowed_leaf_beside_its_own_answers(pooled):
    instrument, artifact, exclusions = pooled
    node = instrument.context_id
    cases, outcomes = selection_cases(instrument, artifact, exclusions, node)
    assert cases == exclusions.run_view(node, artifact) and outcomes is not None
    selections = {
        row.respondent.key(): row
        for row in contextual_respondent_selections(
            instrument,
            cases,
            exclusions.weights[node],
            outcomes=outcomes,
            donors=exclusions.donors[node],
        )
    }
    assert list(selections) == [A_0, A_1, B_0, B_1]
    stored = {case.key(): case for case in artifact.cases}
    parents = {case.key(): case.answer for case in artifact.parent_cases}
    collected = hierarchy_outcomes(instrument, artifact)
    for respondent, selection in selections.items():
        donor = {A_0: B_0, A_1: B_1}.get(respondent)
        assert selection.answered_by("leaf_b") == (donor or respondent)
        assert selection.answered_by("leaf_a") == respondent
        for anchor in instrument.anchor_scores:
            # The pools of the respondent's own parent answers with the
            # leaf answers its states read, replayed from those answers.
            leaves = {
                leaf: pooled_answer(
                    stored[
                        leaf,
                        anchor,
                        donor if donor and leaf == "leaf_b" else respondent,
                    ]
                )
                for leaf in POOL_FIXTURE_COUNTS
            }
            expected = hierarchy_outcome(
                instrument,
                leaves,
                {
                    parent.parent_id: parents[parent.parent_id, anchor, respondent]
                    for parent in instrument.parents
                },
                anchor,
            )
            assert outcomes[anchor, respondent] == expected
            if donor is None:
                assert expected == collected[anchor, respondent]
            for leaf, answer in leaves.items():
                assert answer.rate is not None
                final = selection.answers_by_node[leaf][anchor].rate
                assert final is not None
                assert final.p50 == pytest.approx(
                    answer.rate.p50 * expected.factors[leaf].frequency
                )
    # By hand for the first borrower, whose own counts are the fixture's
    # (EALs of 100 for leaf_a and leaf_c) and whose donor's leaf_b count is
    # six: mid's category answer 1,600 against 100 + 600 + its remainder 100
    # at weight one half, then top's 8,000 against mid's pooled value, leaf_c
    # and its remainder at weight one third.
    mid = math.sqrt(1600 / 800)
    top = (8000 / (mid * 800 + 200)) ** (1 / 3)
    outcome = outcomes[0.5, A_0]
    assert outcome.pools["mid"].factor == pytest.approx(mid)
    assert outcome.pools["top"].factor == pytest.approx(top)
    rate = selections[A_0].answers_by_node["leaf_b"][0.5].rate
    assert rate is not None and rate.p50 == pytest.approx(6 * mid * top)
    # The donor's own states read the same case at another count, pooled
    # with the donor's leaves, three and three: a borrowed leaf's law is the
    # donor's answers scaled by the borrower's own category pools.
    mid = math.sqrt(1600 / 1000)
    top = (8000 / (mid * 1000 + 400)) ** (1 / 3)
    lent = selections[B_0].answers_by_node["leaf_b"][0.5].rate
    assert lent is not None and lent.p50 == pytest.approx(6 * mid * top)
    assert rate.p50 != pytest.approx(lent.p50)
    # With its own answer, two, the collection pooled the same sample's
    # group at the two other anchors to a factor of two at each parent.
    assert collected[0.0, A_0].pools["mid"].factor == pytest.approx(2)


def test_the_exhibit_inputs_count_stored_cases_once_and_weigh_them_as_the_run_does(
    pooled,
):
    instrument, artifact, exclusions = pooled
    node = instrument.context_id
    kept = exclusions.kept_view(node, artifact)
    read = exclusions.run_view(node, artifact)
    leaves = {row.node_id: row.population.scenario_id for row in instrument.nodes}
    weights = exclusions.weights[node]
    assert weights == pytest.approx(dict.fromkeys(SCALES, 1 / 4))
    summary = eal_crosscheck_summary(
        (
            (
                kept,
                leaves,
                {
                    (leaf, instrument.reference_eci, key): weight
                    for (leaf, key), weight in exclusions.case_weights(node).items()
                },
            ),
        )
    )
    # Each stored case once: the leaf two models answer has six cases, and
    # the borrowed leaf the answering model's six.
    assert summary["cases"] == 30
    assert {
        leaf: sum(row["cases_by_status"].values())
        for leaf, row in summary["by_leaf"].items()
    } == {"leaf_a": 12, "leaf_b": 6, "leaf_c": 12}
    # The borrowed leaf's weighted EAL is the answering model's samples at
    # the weight of the states that read them, a half each: counts of two
    # times three and times five at USD 100 an incident.
    assert summary["by_leaf"]["leaf_b"]["weighted_implied_eal"] == pytest.approx(
        0.5 * 600 + 0.5 * 1000
    )
    assert summary["by_leaf"]["leaf_a"]["weighted_implied_eal"] == pytest.approx(
        0.25 * 100 * (1 + 2 + 3 + 5)
    )
    assert frequency_collection_summary(((kept, set(leaves)),))["cases"] == 30
    assert follow_up_counts(((kept, set(leaves)),)).keys() == {
        "severity_retry",
        "chain_guardrail",
        "eal_crosscheck",
    }
    # The taxonomy parents' pools are the run's: one group for each kept
    # respondent and anchor, the borrowers' included.
    hierarchy = hierarchy_summary(((read, instrument, weights),))
    assert {parent: row["cases"] for parent, row in hierarchy.items()} == {
        "mid": 12,
        "top": 12,
    }
    assert {
        tuple(case["respondent"])
        for case in hierarchy["mid"]["case_results"]
        if case["factor"] == pytest.approx(math.sqrt(2))
    } == {A_0}


# A fixed-anchor capture, collected from fake providers


def _pool_answers(request):
    """A responder that gives ``request``'s question the pool fixture's
    answer: a leaf at its respondent's count (``SCALES``), a parent at
    ``POOL_PARENTS``."""
    kind, _, subject, _ = request["question"]
    if kind == "parent":
        text = POOL_PARENTS[subject].model_dump_json()
        return lambda _system_prompt, _user_prompt: text
    answer = counted_answer(POOL_FIXTURE_COUNTS[subject] * SCALES[_respondent(request)])
    return lambda _system_prompt, user_prompt: fixture_response(user_prompt, answer)


def test_a_fixed_anchor_import_reads_the_answering_models_cases_for_the_leaf(
    tmp_path, monkeypatch, pooled
):
    instrument, expected, _ = pooled
    _sending(
        monkeypatch,
        when=lambda request: (
            _respondent(request) == A_0
            and request_kind(request["user_prompt"]) == "direct"
            and _subject(request) == "leaf_b"
            and _anchor(request) == 0.5
        ),
        respond=_pool_answers,
    )
    (artifact,) = asyncio.run(
        collect_study(
            [(instrument, tmp_path)],
            providers=cast(
                list[LLMProvider], [_Immediate(model, None) for model in POOL_PANEL]
            ),
            config=CONFIG,
        )
    )
    assert isinstance(artifact, ContextualRiskArtifact)
    # Collection stored the capture the hand-built one stands for: the leaf
    # unanswered for one sample at one anchor, its group's parents unpooled.
    assert artifact.model_copy(update={"evidence_status": "fixture"}) == expected
    files = read_contextual_capture_files(tmp_path)
    setup = compile_contextual_setup(
        instrument, artifact, setup_id="borrowed", files=files
    )
    # Every respondent keeps its states: a quarter each at the reference score.
    rates = {
        state.epistemic_group_id: {
            law.scenario_id: law.reference_annual_rate for law in state.laws
        }
        for state in setup.joint_states
        if state.weight
    }
    assert [state.weight for state in setup.joint_states if state.weight] == [0.25] * 4
    # The states of fixture-a's two samples, at every coordinate, record the
    # leaf they read from a donor; the answering model's record none.
    assert {
        (state.epistemic_group_id, state.borrowed_leaves)
        for state in setup.joint_states
    } == {
        ("respondent-0", ("leaf_b",)),
        ("respondent-1", ("leaf_b",)),
        ("respondent-2", ()),
        ("respondent-3", ()),
    }
    # The first sample of fixture-a reads fixture-b's first sample for
    # leaf_b, a count of six, pooled with its own leaves and parent answers
    # (the factors are worked by hand in the test of the pools above).
    mid = math.sqrt(1600 / 800)
    top = (8000 / (mid * 800 + 200)) ** (1 / 3)
    assert rates["respondent-0"] == pytest.approx(
        {"leaf_a": mid * top, "leaf_b": 6 * mid * top, "leaf_c": top}
    )
    # The donor's own states: counts three, six and three.
    mid = math.sqrt(1600 / 1000)
    top = (8000 / (mid * 1000 + 400)) ** (1 / 3)
    assert rates["respondent-2"] == pytest.approx(
        {"leaf_a": 3 * mid * top, "leaf_b": 6 * mid * top, "leaf_c": 3 * top}
    )
    record = _exhibit(setup, "excluded_samples")
    assert (record["excluded_samples"], record["samples"]) == (0, 4)
    assert record["exhausted_calls"]["by_subject"] == {
        "leaf_b": {"direct": {"calls": 1, "attempts": 3, "by_cause": {"validation": 1}}}
    }
    unanswered = record["unanswered_scenarios"]
    assert unanswered["pairs"] == [
        {
            "node_id": "leaf_b",
            "scenario_id": "leaf_b",
            "model_id": "fixture-a",
            "basis": "own",
            "samples": 2,
            "lacking_samples": 1,
            "lacking_share": 0.5,
            "exhausted_direct_calls": {
                "calls": 1,
                "attempts": 3,
                "by_cause": {"validation": 1, "refusal": 0, "output_limit": 0},
            },
        }
    ]
    assert unanswered["by_leaf"]["leaf_b"]["answering_models"] == {
        "fixture-b": {"realized_share": 1.0, "equal_share": 1.0}
    }
    assert unanswered["leaves_on_one_model"] == 1
    assert any(
        text.startswith(
            "Unanswered scenarios (POOL-UNANSWERED-SCENARIOS): model fixture-a does "
            "not answer scenario leaf_b (unanswered in 1 of its 2 samples, 50.0%): "
            "its samples use the answers of fixture-b."
        )
        for text in setup.limitations
    )
    # The exhibits count each kept stored case once and weigh a borrowed
    # case as the states that read it: thirty cases, the leaf at the
    # answering model's two reference cases, a half each.
    frequency = _exhibit(setup, "frequency_estimate")
    assert frequency["frequency_collection"]["cases"] == 30
    assert frequency["eal_crosscheck"]["cases"] == 30
    leaf = frequency["eal_crosscheck"]["by_leaf"]["leaf_b"]
    assert len(leaf["cases"]) == 6
    assert leaf["weighted_implied_eal"] == pytest.approx(0.5 * 600 + 0.5 * 1000)
    assert all(row["cases"] == 12 for row in frequency["hierarchy"].values())

    def chain_panel(setup):
        """An export's chain panel of ``setup`` packaged with its capture's
        instrument and artifact, where setup import puts them, by scenario."""
        captured = {
            "captured/contextual-instrument.json": instrument.model_dump_json(),
            "captured/contextual-risk.json": artifact.model_dump_json(),
        }
        packaged = replace(
            setup,
            evidence=EvidenceSet(
                records=setup.evidence.records,
                files=StoredFiles.merged(
                    {name: data.encode() for name, data in captured.items()},
                    setup.evidence.files,
                ),
            ),
        )
        return {
            row["scenario_id"]: (
                row["anchor_score"],
                row["case_count"],
                row["direct_count_p50"],
            )
            for row in _chain_summary(packaged)
        }

    # An export's chain panel reads the kept cases at the centre anchor,
    # each once: every respondent's of the other leaves, counts of one, two,
    # three and five, and the answering model's two of this leaf, six and
    # ten. The model's own answers to it are not there: its second sample's
    # four, and its first sample's unanswered case.
    assert chain_panel(setup) == {
        "leaf_a": (0.5, 4, 2.5),
        "leaf_b": (0.5, 2, 8.0),
        "leaf_c": (0.5, 4, 2.5),
    }
    # With the control at a half the same capture has no pair and excludes
    # the sample that lacks the leaf: the panel holds the three others' cases.
    excluding = compile_contextual_setup(
        instrument,
        artifact,
        setup_id="excluded",
        files=files,
        max_excluded_sample_share=0.5,
    )
    assert _exhibit(excluding, "excluded_samples")["excluded_samples"] == 1
    assert chain_panel(excluding) == {
        "leaf_a": (0.5, 3, 3.0),
        "leaf_b": (0.5, 3, 6.0),
        "leaf_c": (0.5, 3, 3.0),
    }
    # Validating the collection alone gives advance notice of the pair.
    monkeypatch.setattr(
        "catmodel.setup.contextual_inputs.load_contextual_collection",
        lambda directory: (instrument, artifact, files),
    )
    result = CliRunner().invoke(
        cli, ["capability-risk", "validate", "--run-directory", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["unanswered_cases"] == [["leaf_b", 0.5, list(A_0)]]
    assert report["unanswered_scenarios"] == [
        {
            "node_id": "leaf_b",
            "model_id": "fixture-a",
            "samples": 2,
            "lacking_samples": 1,
            "lacking_share": 0.5,
        }
    ]


# Whole public studies, collected from fake providers

PANEL = {"fixture-a": 2, "fixture-b": 2, "fixture-c": 2}
PANEL_ROSTER = [
    {"respondent_id": f"{model}-{index}", "model_id": model, "sample_index": index}
    for model, count in PANEL.items()
    for index in range(count)
]
PANEL_NODES = ("node-00", "node-1-2", "node-1-4", "node-3-4")
"""The trajectory nodes of the three-node fixture study, in ID order; the
first is the no-progress node, which the base forecast weights at zero."""
SOURCE_STATUSES = ("public", "not_yet_public")
LEAF_COUNTS = {"leaf_a": 1.0, "leaf_b": 3.0}
MODEL_FACTORS = {"fixture-a": 1.0, "fixture-b": 1.25, "fixture-c": 1.6}


def _stated_count(leaf: str, model: str, index: int, status: str) -> float:
    """The annual count a panel respondent states for a leaf in one source
    status's context. No two respondents state the same count for a leaf, so
    a law's count tells whose answer it holds."""
    return (
        LEAF_COUNTS[leaf]
        * MODEL_FACTORS[model]
        * (1 + index / 2)
        * (2 if status == "not_yet_public" else 1)
    )


def _stated(request):
    """A responder that answers ``request``'s question at its respondent's
    count for the leaf and context."""
    _, model, index = _respondent(request)
    status = next(
        status
        for status in reversed(SOURCE_STATUSES)
        if _context(request).endswith(status)
    )
    answer = counted_answer(_stated_count(_subject(request), model, index, status))
    return lambda _system_prompt, user_prompt: fixture_response(user_prompt, answer)


def _panel_study(tmp_path, monkeypatch, unanswering: Collection[str]):
    """The panel's public study of two leaves, in which no direct question
    about leaf_b gets a valid answer from the models ``unanswering``."""
    return _public_study(
        tmp_path,
        monkeypatch,
        lambda request: (
            _respondent(request)[1] in unanswering
            and request_kind(request["user_prompt"]) == "direct"
            and _subject(request) == "leaf_b"
        ),
        respond=_stated,
        instrument=leaves_instrument(tuple(LEAF_COUNTS)),
        roster=PANEL_ROSTER,
    )


def _panel_laws(setup):
    """Each state's weight and its law of each leaf and source status, by
    trajectory node and respondent, a (model, sample index)."""
    respondents = sorted(
        (row["respondent_id"], row["model_id"], row["sample_index"])
        for row in PANEL_ROSTER
    )
    weights, laws = {}, {}
    for state in setup.joint_states:
        node = public_state_node(state.state_id)
        _, model, index = respondents[int(state.epistemic_group_id.rsplit("-", 1)[1])]
        weights[node, (model, index)] = state.weight
        for law in state.laws:
            (status,) = {segment.source_public_status for segment in law.segments}
            laws[node, (model, index), law.scenario_id, status] = law
    assert len(laws) == len(weights) * len(LEAF_COUNTS) * len(SOURCE_STATUSES)
    return weights, laws


def _pooled_eal(sources: Mapping[tuple[str, int], tuple[str, int]], leaf: str) -> float:
    """A leaf's welfare EAL at one node where each respondent of ``sources``
    carries the answers of the respondent it maps to: USD 100 an incident,
    both source statuses, every model at an equal share."""
    models = {model for model, _ in sources}
    return math.fsum(
        100
        * _stated_count(leaf, *source, status)
        / len(models)
        / sum(model == respondent[0] for model, _ in sources)
        for respondent, source in sources.items()
        for status in SOURCE_STATUSES
    )


def test_a_public_study_leaves_a_model_out_of_the_scenario_it_never_answers(
    tmp_path, monkeypatch
):
    folder = _panel_study(tmp_path, monkeypatch, {"fixture-a"})
    package = _import(folder, tmp_path / "package")
    # No sample is excluded: four nodes of six respondents, each at its
    # model's equal share of the node.
    weights, laws = _panel_laws(package)
    assert len(package.joint_states) == len(weights) == 24
    assert sorted({node for node, _ in weights}) == list(PANEL_NODES)
    for node in PANEL_NODES:
        shares = {weights[node, (model, 0)] for model in PANEL}
        shares |= {weights[node, (model, 1)] for model in PANEL}
        assert len(shares) == 1
    # The donors in rounds are the first samples of fixture-b and fixture-c
    # and then their second samples. The two samples of fixture-a take two
    # in a row, one of either model, from one further on at each node, so
    # over the four nodes each of the four answering samples lends twice.
    donors = {
        "node-00": {0: ("fixture-b", 0), 1: ("fixture-c", 0)},
        "node-1-2": {0: ("fixture-c", 0), 1: ("fixture-b", 1)},
        "node-1-4": {0: ("fixture-b", 1), 1: ("fixture-c", 1)},
        "node-3-4": {0: ("fixture-c", 1), 1: ("fixture-b", 0)},
    }
    sources = {}
    for node in PANEL_NODES:
        for model, samples in PANEL.items():
            for index in range(samples):
                own = model, index
                source = donors[node][index] if model == "fixture-a" else own
                sources[node, own] = source
                for status in SOURCE_STATUSES:
                    # Every other leaf is the respondent's own.
                    assert laws[node, own, "leaf_a", status].reference_annual_rate == (
                        pytest.approx(_stated_count("leaf_a", *own, status))
                    )
                    # The leaf's law is the donor's in each context, whole.
                    law = laws[node, own, "leaf_b", status]
                    assert law.reference_annual_rate == pytest.approx(
                        _stated_count("leaf_b", *source, status)
                    )
                    assert law == laws[node, source, "leaf_b", status]
    record = _exhibit(package, "excluded_samples")
    assert record["excluded_samples"] == 0 and record["excluded"] == []
    assert record["exhausted_calls"]["by_kind"]["direct"] == 16
    unanswered = record["unanswered_scenarios"]
    assert unanswered["register_id"] == "POOL-UNANSWERED-SCENARIOS"
    # Eight samples of the model, each lacking the leaf in both its contexts.
    assert unanswered["pairs"] == [
        {
            "node_id": "leaf_b",
            "scenario_id": "leaf_b",
            "model_id": "fixture-a",
            "basis": "own",
            "samples": 8,
            "lacking_samples": 8,
            "lacking_share": 1.0,
            "exhausted_direct_calls": {
                "calls": 16,
                "attempts": 48,
                "by_cause": {"validation": 16, "refusal": 0, "output_limit": 0},
            },
        }
    ]
    assert unanswered["leaves_on_one_model"] == 0
    assert unanswered["by_leaf"] == {
        "leaf_b": {
            "scenario_id": "leaf_b",
            "answering_models": {
                model: {
                    "realized_share": pytest.approx(0.5),
                    "equal_share": 0.5,
                }
                for model in ("fixture-b", "fixture-c")
            },
        }
    }
    # The shares are at the base forecast's node weights, and the equal
    # share is what equal kept counts and exact balance would give.
    for sentence in (
        "An arm that weighs the nodes differently from the reference, such as a "
        "forecast alternative or a variant's matched control, can pool the leaf in "
        "other shares.",
        "beside equal_share, one over the number of answering models: the share "
        "each would hold if the answering models kept equally many samples at "
        "every node and every answering sample lent equally often. The rotation "
        "balances the answering samples, not the models,",
    ):
        assert sentence in unanswered["interpretation"]
    (limitation,) = (
        text for text in package.limitations if text.startswith("Unanswered scenarios")
    )
    assert limitation.startswith(
        "Unanswered scenarios (POOL-UNANSWERED-SCENARIOS): model fixture-a does not "
        "answer scenario leaf_b (unanswered in 8 of its 8 samples, 100.0%): its "
        "samples use the answers of fixture-b and fixture-c."
    )
    assert "Scenarios that rest on one model's answers: none." in limitation
    assert (
        "A borrowed case repeats a donor's answers, so the samples are not "
        "independent. Results therefore report no respondent-sampling standard "
        "error and no EAL parameter band for an EAL or a change that weights a "
        "borrowing sample, and no standard deviation of the samples' EALs at a "
        "trajectory node that holds one."
    ) in limitation
    assert "understated" not in limitation + unanswered["interpretation"]
    assert any(
        text.startswith("Excluded samples (POOL-EXCLUDED-SAMPLES): 0 of 24 samples")
        for text in package.limitations
    )
    # The exhibits count each stored case once: leaf_b has the cases of two
    # models, eight samples in two contexts each, and its EAL by either
    # route weighs them as the states that read them.
    frequency = _exhibit(package, "frequency_estimate")
    checks = frequency["eal_crosscheck"]["by_leaf"]
    assert {leaf: len(row["cases"]) for leaf, row in checks.items()} == {
        "leaf_a": 48,
        "leaf_b": 32,
    }
    assert frequency["frequency_collection"]["cases"] == 80
    # A node's pool of leaf_b depends on the donors fixture-a's samples read
    # there, so the scenario's EAL is each node's pool at the node's
    # forecast weight; the no-progress node has none.
    plan = json.loads((folder / "public-frontier.json").read_text())
    path_weights = dict(
        zip(
            (row["draw_id"] for row in plan["nodes"]),
            plan["base"]["weights"],
            strict=True,
        )
    )
    assert path_weights[PANEL_NODES[0]] == 0

    def pooled(leaf: str, models: Collection[str] = tuple(PANEL)) -> float:
        """The leaf's EAL with only ``models``' respondents, each reading
        its donor's answers to leaf_b and its own to any other leaf."""
        return math.fsum(
            path_weights[node]
            * _pooled_eal(
                {
                    own: source if leaf == "leaf_b" else own
                    for (at, own), source in sources.items()
                    if at == node and own[0] in models
                },
                leaf,
            )
            for node in PANEL_NODES
        )

    expected = {leaf: pooled(leaf) for leaf in LEAF_COUNTS}
    assert expected["leaf_a"] == pytest.approx(481.25)
    # By hand: the answering models state 3.75, 5.625, 4.8 and 7.2 incidents
    # in the public context and twice that in the other, at USD 100 each and
    # a sixth of a node a respondent; fixture-a's samples add the counts of
    # the two donors they read at the node.
    by_node = {
        "node-1-2": 50 * (21.375 + 4.8 + 5.625),
        "node-1-4": 50 * (21.375 + 5.625 + 7.2),
        "node-3-4": 50 * (21.375 + 7.2 + 3.75),
    }
    assert expected["leaf_b"] == pytest.approx(
        math.fsum(path_weights[node] * value for node, value in by_node.items())
    )
    for leaf, value in expected.items():
        assert checks[leaf]["weighted_implied_eal"] == pytest.approx(value)
        assert checks[leaf]["weighted_direct_eal"] == pytest.approx(value)
    output = tmp_path / "run"
    report = run_setup(setup_path=package.root, output=output, code_commit="fixture")
    results = load_results(output / "results.json", diagnostics=True)
    assert results["sensitivity_summary"]["excluded_samples"] == record
    assert limitation in {row["text"] for row in results["limitations"]}
    # The run's own EAL of each scenario is the pool with the borrowed
    # answers; with the model's own it would be another figure.
    by_scenario = results["scenario_contributions"]
    assert by_scenario["status"] == "available"
    contributions = {
        row["scenario_id"]: row["expected_annual_loss"]
        for row in by_scenario["scenarios"]
    }
    assert contributions == pytest.approx(expected)
    # The band mixes the respondents as separate sources of the count laws,
    # and fixture-a's repeat their donors' laws of leaf_b: no view in which a
    # state of fixture-a has weight reports a band, and each says why. The
    # capture holds the band of every view as before, so it replays below.
    borrowing = sorted(
        state.state_id
        for state in package.joint_states
        if state.borrowed_leaves and state.weight > 0
    )
    no_band = borrowed_reason(borrowing[0], ("leaf_b",), NO_BAND)
    assert no_band.endswith(
        "(POOL-UNANSWERED-SCENARIOS), so the respondents the band mixes are not "
        "independent and no band is reported."
    )
    band = cast(
        UnavailableParameterBand, results["sensitivity_summary"]["eal_parameter_band"]
    )
    assert (band["status"], band["reason"]) == ("unavailable", no_band)
    assert {"welfare", "gross"} <= set(band["views"])
    assert all(
        row == {"status": "unavailable", "reason": no_band}
        for row in band["views"].values()
    )
    captured = json.loads((output / "capture" / CAPTURE_MANIFEST).read_text())
    assert captured["eal_parameter_band"]["views"].keys() == band["views"].keys()
    assert all(
        row["status"] == "available"
        for row in captured["eal_parameter_band"]["views"].values()
    )
    # No respondent lacks a node, and the borrowing model keeps its states
    # in every model-pooling arm: with fixture-b left out, leaf_b still
    # holds the fixture-b sample that fixture-a's states read.
    assert "incomplete_epistemic_groups" not in band
    without_b = available_metric(
        reduced_arm(results["arms"]["model_pooling_leave_one_out::fixture-b"])[
            "metrics"
        ]["scenario::leaf_b::welfare"]
    )
    assert without_b["expected_annual_loss"] == pytest.approx(
        pooled("leaf_b", ("fixture-a", "fixture-c"))
    )
    # A donor's answers are in its own sampling group and its borrower's, so
    # the groups are not independent: no EAL or change that weights a state of
    # fixture-a has a respondent-sampling standard error, and each says why.
    withheld = borrowed_reason(borrowing[0], ("leaf_b",))
    summary = results["sensitivity_summary"]
    sampling = summary["respondent_sampling"]
    assert sampling["status"] == "available"
    assert sampling["views"]["welfare"] == {"status": "unavailable", "reason": withheld}
    changes = [
        row for rows in sampling["comparisons"].values() for row in rows.values()
    ]
    assert changes and all(row["status"] == "unavailable" for row in changes)
    assert summary["tornado"]["rows"]
    for row in summary["tornado"]["rows"]:
        assert (row["sampling_standard_error"], row["sampling_z"]) == (None, None)
        assert "(POOL-UNANSWERED-SCENARIOS)" in present(row["sampling_reason"])
    indicator = summary["trajectory_quadrature"]
    assert indicator["status"] == "available"
    quadrature = indicator["by_arm"]["reference"]
    assert (quadrature["sampling_standard_error"], quadrature["sampling_reason"]) == (
        None,
        withheld,
    )
    # Every node holds the two samples of fixture-a, so none reports the
    # standard deviation of its state EALs, and each names its own first
    # borrowing state: the no-progress node too, whose states the
    # no-progress arm weights. The node's mean and contributions are
    # reported.
    assert set(quadrature["by_node"]) == set(PANEL_NODES)
    for node, row in quadrature["by_node"].items():
        first = min(
            state.state_id
            for state in package.joint_states
            if state.borrowed_leaves and public_state_node(state.state_id) == node
        )
        assert (row["samples"], row["state_eal_sd"]) == (6, None)
        assert row["state_eal_sd_reason"] == borrowed_reason(
            first, ("leaf_b",), NO_NODE_SD
        )
        assert row["state_eal_mean"] > 0
    assert math.fsum(
        row["difference_contribution"] for row in quadrature["by_node"].values()
    ) == pytest.approx(quadrature["absolute_difference"], rel=1e-9, abs=1e-12)
    node_views = [
        view
        for node in results["capability_scenarios"]["nodes"]
        for view in node["by_view"].values()
    ]
    assert node_views
    for view in node_views:
        assert view["sampling_standard_error"] is None
        assert "(POOL-UNANSWERED-SCENARIOS)" in present(view["sampling_reason"])
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=output / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert replay == report
    # An export's results, the disclosure among them, reproduce from its
    # capture when it is loaded.
    index = export_results(
        setup_path=package.root,
        capture_directory=output / "capture",
        results_path=output / "results.json",
        output=tmp_path / "export",
        code_commit="fixture",
    )
    assert load_results_export(tmp_path / "export") == index
    # The scenario table states why a row has no standard error, as the node
    # table does: each path's reason is the one its results give.
    with (tmp_path / "export" / "capability_scenarios.csv").open() as handle:
        scenario_rows = list(csv.DictReader(handle))
    stated = {
        (row["scenario"], view): data["sampling_reason"]
        for row in results["capability_scenarios"]["scenarios"]
        for view, data in row["by_view"].items()
    }
    assert scenario_rows
    for row in scenario_rows:
        assert row["sampling_standard_error"] == ""
        assert row["sampling_reason"] == stated[row["scenario"], row["view"]]
        assert "(POOL-UNANSWERED-SCENARIOS)" in row["sampling_reason"]
    # The export leaves out every context's calls and parsed answers and
    # keeps the study's plan and each context's source bundle. Its chain
    # panel is the one the omitted files give, read from the setup.
    copied = {
        path.relative_to(tmp_path / "export/native/setup").as_posix()
        for path in (tmp_path / "export/native/setup").rglob("*")
        if path.is_file()
    }
    left_out = set(package.files) - copied
    assert left_out and all(
        "/calls/" in name or name.endswith("/contextual-risk.json") for name in left_out
    )
    assert not any("/calls/" in name or "contextual-risk" in name for name in copied)
    assert {
        "evidence/captured/public-study.json",
        "evidence/captured/public-frontier.json",
        "evidence/captured/contexts/0/source-bundle/manifest.yaml",
    } <= copied
    assert load_results_export(tmp_path / "export", answers=package.root) == index
    # The export's chain panel, at the centre node in either source status,
    # holds the kept cases: six respondents' of leaf_a and the four
    # answering respondents' of leaf_b, without the unanswered cases of
    # fixture-a. Each count is the median of the counts stated there.
    panel = {
        (row["scenario_id"], row["source_status"]): row for row in index.chain_summary
    }
    assert {row["trajectory_draw_id"] for row in panel.values()} == {"node-1-2"}
    for status in SOURCE_STATUSES:
        for leaf, models in (
            ("leaf_a", tuple(PANEL)),
            ("leaf_b", ("fixture-b", "fixture-c")),
        ):
            row = panel[leaf, status]
            assert row["case_count"] == 2 * len(models)
            assert row["direct_count_p50"] == pytest.approx(
                statistics.median(
                    _stated_count(leaf, model, index, status)
                    for model in models
                    for index in range(2)
                )
            )
    assert panel["leaf_b", "public"]["direct_count_p50"] == pytest.approx(5.2125)
    text = write_results_summary(
        results_path=output / "results.json", output=tmp_path / "summary"
    ).read_text()
    # A collected study of two scenarios: the summary prints every limitation
    # of its results.
    assert all(row["text"] in text for row in results["limitations"])
    assert (
        "A respondent whose model does not answer a scenario carries other models' "
        "answers to it, another sample's at each trajectory node (Excluded "
        "samples), scaled by its own category answers where the study asks "
        "category questions. For that scenario the respondents are therefore not "
        "independent, and they tend to lie closer together here than independent "
        "answers would."
    ) in text.split("## Respondent dispersion", 1)[1].split("\n## ", 1)[0]
    assert (
        "How to read it: such a scenario's direct and chain answers are all the "
        "answering models'. Where the study asks category questions, each "
        "borrowing sample's own category answers, and its answers to the "
        "category's other scenarios, still scale the borrowed counts through its "
        "category pools, so the scenario's EAL is not free of the other model's "
        "judgment."
    ) in text
    assert (
        "| `leaf_b` | `fixture-a` | 8 of 8 (100.0%) | 16: 16 validation | "
        "`fixture-b` 50.0% (50.0%); `fixture-c` 50.0% (50.0%) |"
    ) in text
    assert "Scenarios answered by one model: 0 of the 1 listed." in text
    # The summary prints why each standard error is withheld.
    section = text.split("## Respondent sampling", 1)[1].split("\n## ", 1)[0]
    assert f"| `welfare` | — | — | — | {withheld} |" in section
    # It prints why there is no band, and what the unanswered scenario
    # leaves unreported.
    section = text.split("## EAL parameter band", 1)[1].split("\n## ", 1)[0]
    assert section.strip() == f"Unavailable: {no_band}"
    assert (
        "Borrowed answers repeat a donor's, so the samples are not independent. "
        "Results therefore report no respondent-sampling standard error and no "
        "EAL parameter band for an EAL or a change that weights a borrowing "
        "sample, and no standard deviation of the samples' EALs at a trajectory "
        "node that holds one."
    ) in text
    # The scenario table notes the scenario, and only it.
    section = text.split("## Expected annual loss by scenario", 1)[1]
    notes = {
        line.split(" | ")[0]: line.rsplit(" | ", 1)[1]
        for line in section.split("\n## ", 1)[0].splitlines()
        if line.startswith("| `leaf_")
    }
    assert set(notes) == {"| `leaf_a`", "| `leaf_b`"}
    assert notes["| `leaf_b`"].startswith(
        "not answered by `fixture-a`, whose samples use the answers of "
        "`fixture-b`, `fixture-c` (Excluded samples)."
    )
    assert "not answered" not in notes["| `leaf_a`"]
    rows = (tmp_path / "summary" / "summary.csv").read_text().splitlines()
    assert (
        "excluded_samples,leaf_b:fixture-a,,lacking_share,,1.0,share,,available,"
        in rows
    )
    assert (
        "excluded_samples,,,scenarios_answered_by_one_model,,0.0,count,,available,"
        in rows
    )


def test_exactly_the_states_that_read_a_donor_record_the_leaf_they_borrow(
    tmp_path, monkeypatch
):
    package = _import(
        _panel_study(tmp_path, monkeypatch, {"fixture-a"}), tmp_path / "package"
    )
    respondents = sorted(
        (row["respondent_id"], row["model_id"]) for row in PANEL_ROSTER
    )
    marked = {}
    for state in package.joint_states:
        _, model = respondents[int(state.epistemic_group_id.rsplit("-", 1)[1])]
        # A borrowing state keeps its own model's sampling group.
        assert state.sampling_group_id == (
            f"{public_state_node(state.state_id)}:{model}"
        )
        marked.setdefault(model, []).append(state.borrowed_leaves)
        # A state that reads only its own answers serializes no such field.
        assert ("borrowed_leaves" in state.model_dump(mode="json")) == (
            model == "fixture-a"
        )
    # The two samples of fixture-a at each of the four nodes, and no other.
    assert marked == {
        "fixture-a": [("leaf_b",)] * 8,
        "fixture-b": [()] * 8,
        "fixture-c": [()] * 8,
    }


def test_a_scenario_one_model_answers_is_pooled_over_its_samples_in_equal_shares(
    tmp_path, monkeypatch
):
    folder = _panel_study(tmp_path, monkeypatch, {"fixture-a", "fixture-c"})
    package = _import(folder, tmp_path / "package", comparison_families=())
    weights, laws = _panel_laws(package)
    assert len(weights) == 24
    # The four borrowers model by model, fixture-a's two samples and then
    # fixture-c's, take fixture-b's two samples in turn: at a node either
    # model's first sample reads one of them and its second the other, and
    # they change places at the next node.
    first = {"node-00": 0, "node-1-2": 1, "node-1-4": 0, "node-3-4": 1}
    for node in PANEL_NODES:
        read = {0: 0.0, 1: 0.0}
        for model, samples in PANEL.items():
            for index in range(samples):
                donor = index if model == "fixture-b" else (first[node] + index) % 2
                for status in SOURCE_STATUSES:
                    law = laws[node, (model, index), "leaf_b", status]
                    assert law == laws[node, ("fixture-b", donor), "leaf_b", status]
                    assert law.reference_annual_rate == pytest.approx(
                        _stated_count("leaf_b", "fixture-b", donor, status)
                    )
                    own = laws[node, (model, index), "leaf_a", status]
                    assert own.reference_annual_rate == pytest.approx(
                        _stated_count("leaf_a", model, index, status)
                    )
                read[donor] += weights[node, (model, index)]
        # Either sample is borrowed twice, so the scenario's pool at every
        # node is exactly the answering model's two samples in equal shares.
        assert read[0] == read[1]
        assert read[0] + read[1] == pytest.approx(
            sum(weights[node, (model, 0)] * 2 for model in PANEL)
        )
    unanswered = _exhibit(package, "excluded_samples")["unanswered_scenarios"]
    assert [(row["model_id"], row["lacking_share"]) for row in unanswered["pairs"]] == [
        ("fixture-a", 1.0),
        ("fixture-c", 1.0),
    ]
    assert unanswered["leaves_on_one_model"] == 1
    assert unanswered["by_leaf"]["leaf_b"]["answering_models"] == {
        "fixture-b": {"realized_share": pytest.approx(1.0), "equal_share": 1.0}
    }
    (limitation,) = (
        text for text in package.limitations if text.startswith("Unanswered scenarios")
    )
    assert (
        "model fixture-c does not answer scenario leaf_b (unanswered in 8 of its 8 "
        "samples, 100.0%): its samples use the answers of fixture-b."
    ) in limitation
    assert (
        "Scenarios that rest on one model's answers: leaf_b (fixture-b)." in limitation
    )
    # Worker processes compile the same setup: each draw's cases, donors and
    # replayed pools go to a worker and come back the serial import's.
    import catmodel.setup.public_study as service

    mapped = []
    pooled = service.WorkerPool.map

    def recorded(self, function, tasks, **kwargs):
        mapped.append((function.__name__, kwargs.get("concurrency")))
        return pooled(self, function, tasks, **kwargs)

    monkeypatch.setattr(service, "memory_bounded_workers", lambda workers, _: workers)
    monkeypatch.setattr(service.WorkerPool, "map", recorded)
    parallel = _import(folder, tmp_path / "parallel", comparison_families=(), workers=2)
    assert ("_compile_draw", 2) in mapped
    assert parallel.manifest_sha256 == package.manifest_sha256


def test_import_refuses_a_public_study_in_which_no_model_answers_a_scenario(
    tmp_path, monkeypatch
):
    folder = _panel_study(tmp_path, monkeypatch, set(PANEL))
    with pytest.raises(
        ValueError,
        match=r"no model answers leaf 'leaf_b': model fixture-a lacks it in 8 of its "
        r"8 samples \(100\.0%\); model fixture-b lacks it in 8 of its 8 samples "
        r"\(100\.0%\); model fixture-c lacks it in 8 of its 8 samples \(100\.0%\), "
        r"above the frozen control max_excluded_sample_share 0\.1 "
        r"\(POOL-UNANSWERED-SCENARIOS\); recollect the study with a model that "
        r"answers it",
    ):
        _import(folder, tmp_path / "package", comparison_families=())
    # With the control at one no share is above it, so every sample is
    # excluded for its unanswered leaf and no model keeps one.
    with pytest.raises(
        ValueError,
        match=r"model fixture-a keeps 0 of its 2 samples at node 'node-00', fewer "
        r"than half \(unanswered leaves: leaf_b\)",
    ):
        _import(
            folder,
            tmp_path / "package",
            comparison_families=(),
            max_excluded_sample_share=1.0,
        )


# A variant study beside its headline, collected from fake providers

USERS = re.compile(r"About ([0-9,]+) synthetic users")
VARIANT_PANEL = {"fixture-a": 2, "fixture-b": 3}
SAMPLE_FACTORS = {
    ("fixture-a", 0): 3.0,
    ("fixture-a", 1): 4.5,
    ("fixture-b", 0): 1.0,
    ("fixture-b", 1): 1.5,
    ("fixture-b", 2): 2.0,
}
"""Each respondent's multiple of a question's count, by model and sample
index. No two samples of a model share one, so a law's count tells whose
answer it holds."""
B_2 = ("fixture-b-2", "fixture-b", 2)
exposure_variant_state_id = CONTEXT_VARIANT_FAMILIES["exposure_scale"].state_id


def _per_user(request):
    """A responder whose annual count is a hundredth of the users its
    question renders, times its respondent's factor, at USD 100 an incident:
    an exposure variant's answer is the headline's at the variant's scale."""
    _, model, index = _respondent(request)

    def reply(_system_prompt, user_prompt):
        found = USERS.search(user_prompt)
        assert found is not None
        users = float(found.group(1).replace(",", ""))
        return fixture_response(
            user_prompt, counted_answer(users / 100 * SAMPLE_FACTORS[model, index])
        )

    return reply


def _leaf_b_of_fixture_a(request) -> bool:
    """Selects fixture-a's direct questions about leaf_b."""
    return (
        _respondent(request)[1] == "fixture-a"
        and request_kind(request["user_prompt"]) == "direct"
        and _subject(request) == "leaf_b"
    )


def _nothing(_request) -> bool:
    return False


def _headline_study(tmp_path, monkeypatch, when):
    """The variant panel's headline study of two leaves on seven nodes and
    the no-progress node, with ``when``'s requests garbled: its folder, its
    source bundle and its roster."""
    from tests.test_exposure_scale_variants import _ScaledProvider
    from tests.test_public_frontier_study import _collect
    from tests.test_public_frontier_study import _study as public_study

    providers = []
    for model in VARIANT_PANEL:
        provider = _ScaledProvider()
        provider.model_id = model
        providers.append(provider)
    template = ContextualRiskInstrument.model_validate(
        path_linked_payload(
            leaves_instrument(("leaf_a", "leaf_b"), VARIANT_PANEL).model_dump(
                mode="json"
            )
        )
    )
    folder, bundle, _ = public_study(
        tmp_path, monkeypatch, nodes=7, template=template, providers=providers
    )
    with pytest.MonkeyPatch.context() as patch:
        _sending(patch, when=when, respond=_per_user)
        _collect(folder, bundle)
    roster = json.loads((tmp_path / "public.json").read_text())["respondents"]
    return folder, bundle, roster


def _exposure_variant(tmp_path, bundle, roster, variant_id, factor, when):
    """An exposure variant study of the headline's roster on the embedded
    rule's three nodes and the no-progress node, with ``when``'s requests
    garbled."""
    from tests.test_exposure_scale_variants import _variant_study

    with pytest.MonkeyPatch.context() as patch:
        _sending(patch, when=when, respond=_per_user)
        return _variant_study(tmp_path, bundle, variant_id, factor, respondents=roster)


def _users(study, node: str) -> float:
    """The users a study's questions render at one trajectory node."""
    from tests.test_exposure_scale_variants import _rendered

    found = USERS.fullmatch(_rendered(study, node))
    assert found is not None
    return float(found.group(1).replace(",", ""))


def _rates(package, variant: str | None = None):
    """The annual count of each state's laws of a leaf, by trajectory node,
    respondent (a model and sample index) and leaf: the headline study's, or
    the exposure variant ``variant``'s. A state's two source statuses hold
    one count."""
    respondents = sorted(SAMPLE_FACTORS)
    prefix = "" if variant is None else exposure_variant_state_id(variant, "")
    rates: dict[tuple[str, tuple[str, int], str], set[float]] = {}
    for state in package.joint_states:
        if not state.state_id.startswith(prefix) or (
            variant is None and state.state_id.startswith("exposure-")
        ):
            continue
        node = public_state_node(state.state_id.removeprefix(prefix))
        respondent = respondents[int(state.epistemic_group_id.rsplit("-", 1)[1])]
        for law in state.laws:
            rates.setdefault((node, respondent, law.scenario_id), set()).add(
                law.reference_annual_rate
            )
    assert all(len(found) == 1 for found in rates.values())
    return {key: found.pop() for key, found in rates.items()}


def _variant_line(text: str, variant: str) -> str:
    """The summary's line for one exposure variant's excluded samples."""
    (line,) = (
        row
        for row in text.splitlines()
        if row.startswith(f"Exposure-scale variant `{variant}`")
    )
    return line


def test_a_variant_leaves_a_model_out_of_the_scenarios_its_headline_does(
    tmp_path, monkeypatch
):
    # In the headline study fixture-a never answers leaf_b.
    folder, bundle, roster = _headline_study(
        tmp_path, monkeypatch, _leaf_b_of_fixture_a
    )
    headline = folder / "public-study.json"
    # In the half variant every sample answers every leaf.
    half = _exposure_variant(tmp_path, bundle, roster, "half", 0.5, _nothing)
    # In the double variant fixture-a never answers leaf_b either, and one
    # sample of fixture-b lacks leaf_a at one node: one of that model's
    # twelve samples, within the control, so the sample is excluded there.
    double = _exposure_variant(
        tmp_path,
        bundle,
        roster,
        "double",
        2.0,
        lambda request: (
            _leaf_b_of_fixture_a(request)
            or _direct("node-1-2-public", B_2, "leaf_a")(request)
        ),
    )
    package = _importer(tmp_path, headline)(
        "package", (half, double), comparison_families=("exposure_scale",)
    )
    # The headline's eight nodes enter fixture-b's three samples one further
    # on at each. These are its donors at the four nodes the variants share
    # with it, its first, second, third and fifth: for either sample of
    # fixture-a, the sample of fixture-b whose case of leaf_b it reads.
    donors = {
        "node-00": {0: 0, 1: 1},
        "node-1-2": {0: 1, 1: 2},
        "node-1-4": {0: 2, 1: 0},
        "node-3-4": {0: 1, 1: 2},
    }
    assert tuple(donors) == PANEL_NODES

    def sources(donors):
        """Whose answers to leaf_b each respondent carries at each node."""
        return {
            (node, own): ("fixture-b", by_sample[own[1]])
            if own[0] == "fixture-a"
            else own
            for node, by_sample in donors.items()
            for own in SAMPLE_FACTORS
        }

    def stated(study, read, excluded=()):
        """Every state's count of every leaf in ``study``, where each
        respondent carries the answers to leaf_b that ``read`` gives it."""
        return {
            (node, own, leaf): pytest.approx(
                _users(study, node)
                / 100
                * SAMPLE_FACTORS[source if leaf == "leaf_b" else own]
            )
            for (node, own), source in read.items()
            if (node, own) not in excluded
            for leaf in ("leaf_a", "leaf_b")
        }

    followed = sources(donors)
    assert {
        key: rate for key, rate in _rates(package).items() if key[0] in donors
    } == stated(headline, followed)
    # The half variant takes those donors. Its own rotation over its four
    # nodes would give fixture-a's samples the first two of fixture-b at the
    # last node, not the last two.
    assert _rates(package, "half") == stated(half, followed)
    own_rotation = sources({**donors, "node-3-4": {0: 0, 1: 1}})
    assert _rates(package, "half") != stated(half, own_rotation)
    record = _exhibit(package, "excluded_samples")
    (pair,) = record["unanswered_scenarios"]["pairs"]
    assert (pair["model_id"], pair["node_id"], pair["basis"]) == (
        "fixture-a",
        "leaf_b",
        "own",
    )
    assert "matched_control" not in record["unanswered_scenarios"]
    halved = record["exposure_scale_variants"]["half"]
    assert halved["excluded_samples"] == 0 and halved["excluded"] == []
    assert halved["exhausted_calls"]["calls"] == 0
    unanswered = halved["unanswered_scenarios"]
    # The pair follows the headline: none of the model's eight samples in
    # the variant lacks the leaf.
    assert unanswered["pairs"] == [
        {
            "node_id": "leaf_b",
            "scenario_id": "leaf_b",
            "model_id": "fixture-a",
            "basis": "headline",
            "samples": 8,
            "lacking_samples": 0,
            "lacking_share": 0.0,
            "exhausted_direct_calls": {
                "calls": 0,
                "attempts": 0,
                "by_cause": {"validation": 0, "refusal": 0, "output_limit": 0},
            },
        }
    ]
    assert unanswered["matched_control"] == {
        "pairs_not_in_headline": 0,
        "donors": 8,
        "donors_not_headline": 0,
        "differs": False,
    }
    assert unanswered["leaves_on_one_model"] == 1
    limitations = {
        variant: text
        for variant in ("half", "double")
        for text in package.limitations
        if text.startswith(f"Exposure-scale variant {variant}: Unanswered scenarios")
    }
    assert limitations["half"].startswith(
        "Exposure-scale variant half: Unanswered scenarios "
        "(POOL-UNANSWERED-SCENARIOS): model fixture-a does not answer scenario "
        "leaf_b (unanswered in the headline study, which the variant follows; here "
        "unanswered in 0 of its 8 samples, 0.0%): its samples use the answers of "
        "fixture-b."
    )
    # No difference from the matched control is left to disclose.
    assert limitations["half"].endswith(
        "Every pair here is the headline's and every donor the headline's at its "
        "node, so the variant and its matched control, the headline captures "
        "reweighted to the embedded rule, take each scenario from the same models "
        "and each borrowed case from the same sample."
    )
    assert "differ in which models answer" not in limitations["half"]
    # The double variant's own captures make the pair too. It keeps the
    # headline's donors except where it excludes one: at that node the
    # second sample of fixture-a reads, by the variant's rotation over the
    # two samples kept there, the first sample of fixture-b.
    doubled = record["exposure_scale_variants"]["double"]
    assert doubled["excluded"] == [
        {
            "node": "node-1-2",
            "respondent": list(B_2),
            "unanswered": [
                {
                    "context_id": "node-1-2-public",
                    "node_id": "leaf_a",
                }
            ],
        }
    ]
    (pair,) = doubled["unanswered_scenarios"]["pairs"]
    assert (pair["basis"], pair["lacking_samples"], pair["samples"]) == (
        "own_and_headline",
        8,
        8,
    )
    assert doubled["unanswered_scenarios"]["matched_control"] == {
        "pairs_not_in_headline": 0,
        "donors": 8,
        "donors_not_headline": 1,
        "differs": True,
    }
    assert _rates(package, "double") == stated(
        double,
        sources({**donors, "node-1-2": {0: 1, 1: 0}}),
        excluded={("node-1-2", ("fixture-b", 2))},
    )
    assert limitations["double"].endswith(
        "The variant and its matched control, the headline captures reweighted to "
        "the embedded rule, differ in which models answer a scenario or in whose "
        "answers a model's samples borrow (donors other than the headline's, for "
        "the pairs of both studies: 1 of 8), so the variant's effect also carries "
        "that difference."
    )
    assert (
        "(unanswered in 8 of its 8 samples, 100.0%, and unanswered in the headline "
        "study)"
    ) in limitations["double"]
    # The run: the half variant against its matched control, the headline
    # states at the embedded rule's weights. By hand, a leaf's EAL in either
    # is USD 100 an incident in two source statuses, so twice the node's
    # users times the factor of whoever's answer a respondent carries, every
    # model at an equal share.
    quadrature = PublicFrontierQuadrature.model_validate_json(
        (folder / "public-frontier.json").read_text()
    )
    embedded = {
        node.draw_id: weight
        for node, weight in zip(
            quadrature.nodes, quadrature.base.embedded_weights, strict=True
        )
        if weight
    }
    assert set(embedded) == set(PANEL_NODES[1:])

    def eal(study, leaf: str, read=followed) -> float:
        return math.fsum(
            embedded[node]
            * 2
            * _users(study, node)
            * SAMPLE_FACTORS[read[node, own] if leaf == "leaf_b" else own]
            / len(VARIANT_PANEL)
            / VARIANT_PANEL[own[0]]
            for node in embedded
            for own in SAMPLE_FACTORS
        )

    output = tmp_path / "run"
    run_setup(setup_path=package.root, output=output, code_commit="fixture")
    results = load_results(output / "results.json", diagnostics=True)
    views = {
        "scenario::leaf_a::welfare": ("leaf_a",),
        "scenario::leaf_b::welfare": ("leaf_b",),
        "welfare": ("leaf_a", "leaf_b"),
    }
    values = {}
    for arm, study in ((EXPOSURE_CONTROL, headline), ("exposure_scale::half", half)):
        metrics = reduced_arm(results["arms"][arm])["metrics"]
        values[arm] = {
            view: present(available_metric(metrics[view])["expected_annual_loss"])
            for view in views
        }
        assert values[arm] == pytest.approx(
            {
                view: math.fsum(eal(study, leaf) for leaf in leaves)
                for view, leaves in views.items()
            }
        )
    # Every state of the variant is its control's state at half the users,
    # which the study rounds to two significant figures after scaling: each
    # scenario and their total halve, the scenario fixture-a does not answer
    # like the other.
    for view in views:
        assert values["exposure_scale::half"][view] / values[EXPOSURE_CONTROL][
            view
        ] - 1 == pytest.approx(-0.5, abs=0.02)
    # With the donors of its own rotation the variant's leaf_b would be
    # another figure.
    assert eal(half, "leaf_b", own_rotation) != pytest.approx(eal(half, "leaf_b"))
    assert not any(
        "differ in which models answer" in row["text"]
        for row in results["limitations"]
        if row["text"].startswith("Exposure-scale variant half")
    )
    text = write_results_summary(
        results_path=output / "results.json", output=tmp_path / "summary"
    ).read_text()
    line = _variant_line(text, "half")
    assert (
        "In the variant study `fixture-a` does not answer `leaf_b` (following the "
        "headline; in the variant 0 of 8 samples lack it (0.0%)). A variant leaves "
        "a model out of every scenario it does not answer in the headline and gives "
        "a borrowing sample its headline donor at the same node where it can. Here "
        "every pair and every donor is the headline's, so the variant and its "
        "matched control take each scenario from the same models and each borrowed "
        "case from the same sample."
    ) in line
    assert "differ in which models answer" not in line
    line = _variant_line(text, "double")
    assert (
        "`fixture-a` does not answer `leaf_b` (in the variant, where 8 of 8 samples "
        "lack it (100.0%), and in the headline)"
    ) in line
    assert line.endswith(
        "The variant and its matched control differ in which models answer a "
        "scenario or in whose answers a sample borrows (donors other than the "
        "headline's, for the pairs of both studies: 1 of 8), so the variant's "
        "effect also carries that difference."
    )


def test_a_pair_of_a_variant_study_alone_is_disclosed_as_a_difference_from_its_control(
    tmp_path, monkeypatch
):
    folder, bundle, roster = _headline_study(tmp_path, monkeypatch, _nothing)
    # In the variant study alone, one model never answers one leaf.
    variant = _exposure_variant(
        tmp_path, bundle, roster, "half", 0.5, _leaf_b_of_fixture_a
    )
    package = _importer(tmp_path, folder / "public-study.json")(
        "package", (variant,), comparison_families=("exposure_scale",)
    )
    # Eight headline nodes and the variant's four, of five respondents each.
    assert len(package.joint_states) == 5 * (8 + 4)
    record = _exhibit(package, "excluded_samples")
    assert "unanswered_scenarios" not in record
    half = record["exposure_scale_variants"]["half"]
    assert half["excluded_samples"] == 0
    (pair,) = half["unanswered_scenarios"]["pairs"]
    assert (pair["model_id"], pair["node_id"]) == ("fixture-a", "leaf_b")
    assert (pair["basis"], pair["lacking_samples"], pair["samples"]) == ("own", 8, 8)
    assert half["unanswered_scenarios"]["leaves_on_one_model"] == 1
    assert half["unanswered_scenarios"]["matched_control"] == {
        "pairs_not_in_headline": 1,
        "donors": 0,
        "donors_not_headline": 0,
        "differs": True,
    }
    # The headline has no donors to follow: the variant's own rotation over
    # its four nodes gives fixture-a's samples those of fixture-b from one
    # further on at each.
    donors = {
        "node-00": {0: 0, 1: 1},
        "node-1-2": {0: 1, 1: 2},
        "node-1-4": {0: 2, 1: 0},
        "node-3-4": {0: 0, 1: 1},
    }
    assert _rates(package, "half") == {
        (node, own, leaf): pytest.approx(
            _users(variant, node)
            / 100
            * SAMPLE_FACTORS[
                ("fixture-b", by_sample[own[1]])
                if own[0] == "fixture-a" and leaf == "leaf_b"
                else own
            ]
        )
        for node, by_sample in donors.items()
        for own in SAMPLE_FACTORS
        for leaf in ("leaf_a", "leaf_b")
    }
    (limitation,) = (
        text for text in package.limitations if "Unanswered scenarios" in text
    )
    assert limitation.startswith(
        "Exposure-scale variant half: Unanswered scenarios "
        "(POOL-UNANSWERED-SCENARIOS): model fixture-a does not answer scenario leaf_b "
        "(unanswered in 8 of its 8 samples, 100.0%): its samples use the answers of "
        "fixture-b."
    )
    assert limitation.endswith(
        "The variant and its matched control, the headline captures reweighted to "
        "the embedded rule, differ in which models answer a scenario or in whose "
        "answers a model's samples borrow (pairs that the headline study answers: "
        "1 of 1), so the variant's effect also carries that difference."
    )
    # The exhibit's evidence record speaks of borrowing though the headline
    # study has none.
    (evidence,) = (
        row for row in package.evidence.records if row.evidence_id == "excluded_samples"
    )
    assert "Each leaf a model does not answer" in evidence.target
    assert "reads the whole case of a kept sample" in evidence.transport
    assert "a model's own answers to a leaf it does not answer" in evidence.reliance
    # The summary names the variant's pair beside its excluded samples.
    rendered = _Summary(
        {
            "analysis_spec": {"reference_arm": "reference"},
            "arms": {"reference": {"metrics": {}}},
            "sensitivity_summary": {"excluded_samples": record},
        }
    )
    rendered.excluded_samples()
    line = _variant_line("\n".join(rendered.lines), "half")
    assert (
        "In the variant study `fixture-a` does not answer `leaf_b` (in the variant "
        "alone, where 8 of 8 samples lack it (100.0%))."
    ) in line
    assert line.endswith(
        "The variant and its matched control differ in which models answer a "
        "scenario or in whose answers a sample borrows (pairs that the headline "
        "answers: 1 of 1), so the variant's effect also carries that difference."
    )
    assert not any("Unanswered scenarios (`POOL-" in row for row in rendered.lines)
