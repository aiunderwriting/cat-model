"""What each follow-up call of a collection changes, read from a stored capture.

The ``frequency_estimate`` exhibit carries one arm that undoes each follow-up:
the large-loss coherence retry of a direct answer, the chain guardrail retry
and the leaf EAL cross-check. These tests fix the two retries' arms against
values worked by hand or reached by another path;
``tests/test_eal_crosscheck.py`` covers the cross-check's, and
``tests/test_category_pool.py`` the taxonomy parents' pool, which is no call.
"""

import math
from typing import Any, cast

import pytest

from catmodel.contracts.contextual_risk import ContextualRiskArtifact
from catmodel.contracts.elicitation import QUANTILES, Quantiles
from catmodel.contracts.incident import IncidentLaw
from catmodel.elicitation.contextual_risk import contextual_prompts
from catmodel.elicitation.eal_crosscheck import (
    follow_up_counts,
    pooled_answer,
    pooled_answer_before_severity_retry,
)
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.results.results_summary import results_summary
from catmodel.setup.contextual_node import CompileOptions, compile_contextual_node
from catmodel.setup.contextual_setup import (
    compile_contextual_setup,
    contextual_respondent_selections,
)
from catmodel.setup.excluded_samples import pooled_weights
from test_support.contextual_risk import (
    chain_answer,
    chain_artifact,
    contextual_answer,
    contextual_case,
    contextual_instrument,
    crosscheck_answer,
)

# The synthetic node: USD 100 of victim loss and a USD 20 fine per incident,
# and a USD 70 annual business loss in a year with an incident.
VICTIM, FINE, BUSINESS = 100.0, 20.0, 70.0
DIRECT_COUNT = 4.0
# One exposure block, 8,000 opportunities and a first stage of one half: the
# chain counts 1 a year at the original terminal probability, below the
# guardrail's 0.001, and 16 a year at the retry's.
ORIGINAL_TERMINAL, RETRY_TERMINAL = 0.00025, 0.004
ORIGINAL_CHAIN_COUNT, RETRY_CHAIN_COUNT = 1.0, 16.0


def _constant(value: float) -> Quantiles:
    return Quantiles(**dict.fromkeys(QUANTILES, value))


def _direct(prompt):
    # A direct EAL equal to the implied one, so no cross-check is asked.
    return contextual_answer().model_copy(
        update={
            "rate": _constant(DIRECT_COUNT),
            "direct_eal": _constant(DIRECT_COUNT * VICTIM),
        }
    )


def _guardrail_artifact(instrument, retry_terminal: float | None):
    return chain_artifact(
        instrument,
        direct=_direct,
        chain=lambda prompt: chain_answer(8000, (0.5, ORIGINAL_TERMINAL)),
        guardrail_retried=None
        if retry_terminal is None
        else lambda prompt: chain_answer(8000, (0.5, retry_terminal)),
    )


def _exhibit(setup, name: str) -> dict[str, Any]:
    return next(
        row.acquisition[name]
        for row in setup.evidence.records
        if row.acquisition and name in row.acquisition
    )


def _annual(count: float, per_incident: float) -> float:
    """A constant Poisson count's annual mean: incidents times their loss,
    plus the business loss in a year with at least one incident."""
    return count * per_incident + BUSINESS * (1 - math.exp(-count))


def test_the_guardrail_arm_pools_the_chain_answer_as_first_given():
    instrument = contextual_instrument()
    artifact = _guardrail_artifact(instrument, RETRY_TERMINAL)
    record = _exhibit(
        compile_contextual_setup(instrument, artifact, setup_id="guardrail"),
        "frequency_estimate",
    )
    # The pool is the geometric mean of the direct and chain counts.
    before = math.sqrt(DIRECT_COUNT * ORIGINAL_CHAIN_COUNT)
    after = math.sqrt(DIRECT_COUNT * RETRY_CHAIN_COUNT)
    assert (before, after) == (2.0, 8.0)
    lab = f"lab_loss::{instrument.lab.entity_id}"
    for view, expected in (
        # Welfare counts the victims' resource loss alone.
        ("welfare", (before * VICTIM, after * VICTIM)),
        ("gross", (_annual(before, VICTIM + FINE), _annual(after, VICTIM + FINE))),
        (lab, (_annual(before, FINE), _annual(after, FINE))),
    ):
        arms = record["views"][view]["by_arm"]
        assert arms["before_chain_guardrail"]["expected_annual_loss"] == (
            pytest.approx(expected[0], rel=1e-12)
        )
        assert arms["final"]["expected_annual_loss"] == pytest.approx(
            expected[1], rel=1e-12
        )
        effect = record["views"][view]["follow_up_effects"]["chain_guardrail"]
        assert effect == {
            "before_arm": "before_chain_guardrail",
            # The instrument asks no taxonomy parents, so the run's own EAL is
            # the pooled, cross-checked one.
            "after_arm": "final",
            "status": "available",
            "expected_annual_loss_before": pytest.approx(expected[0], rel=1e-12),
            "expected_annual_loss_after": pytest.approx(expected[1], rel=1e-12),
            "relative_change": pytest.approx(expected[1] / expected[0] - 1, rel=1e-12),
        }
    assert record["views"]["welfare"]["follow_up_effects"]["chain_guardrail"][
        "relative_change"
    ] == pytest.approx(3.0, rel=1e-12)
    assert record["follow_ups"] == {
        "severity_retry": {"cases_asked": 0, "answers_replaced": 0},
        "chain_guardrail": {"cases_asked": 3, "answers_replaced": 3},
        "eal_crosscheck": {"cases_asked": 0, "answers_replaced": 0},
    }
    # A follow-up that replaced no answer has no arm of its own to report and
    # no effect.
    welfare = record["views"]["welfare"]
    reason = (
        "No kept case's direct answer was replaced by a large-loss coherence retry."
    )
    assert welfare["by_arm"]["before_severity_retry"] == {
        "status": "inapplicable",
        "reason": reason,
    }
    assert welfare["follow_up_effects"]["severity_retry"] == {
        "before_arm": "before_severity_retry",
        "after_arm": "final",
        "status": "inapplicable",
        "reason": reason,
    }
    assert welfare["follow_up_effects"]["eal_crosscheck"]["status"] == "inapplicable"
    # The taxonomy parents' pool is no follow-up call, and without parents
    # neither end of its weight is an arm.
    assert set(welfare["follow_up_effects"]) == {
        "severity_retry",
        "chain_guardrail",
        "eal_crosscheck",
    }
    for arm in ("before_hierarchy", "hierarchy_parents_only"):
        assert welfare["by_arm"][arm] == {
            "status": "inapplicable",
            "reason": "The instrument asks no taxonomy-parent questions.",
        }


def test_a_retry_that_restates_or_exhausts_replaces_no_chain_answer():
    instrument = contextual_instrument()
    node_ids = {node.node_id for node in instrument.nodes}
    restated = _guardrail_artifact(instrument, ORIGINAL_TERMINAL)
    exhausted = _guardrail_artifact(instrument, None)
    for artifact in (restated, exhausted):
        assert all(case.chain_guardrail is not None for case in artifact.cases)
        assert follow_up_counts(((artifact, node_ids),))["chain_guardrail"] == {
            "cases_asked": 3,
            "answers_replaced": 0,
        }
    record = _exhibit(
        compile_contextual_setup(instrument, restated, setup_id="restated"),
        "frequency_estimate",
    )
    welfare = record["views"]["welfare"]
    assert welfare["by_arm"]["before_chain_guardrail"] == {
        "status": "inapplicable",
        "reason": "No kept case's chain answer was replaced by a guardrail retry.",
    }
    assert welfare["follow_up_effects"]["chain_guardrail"]["status"] == "inapplicable"
    # The pool still uses the chain as first given: the direct count and a
    # chain count of one.
    assert welfare["by_arm"]["final"]["expected_annual_loss"] == pytest.approx(
        math.sqrt(DIRECT_COUNT * ORIGINAL_CHAIN_COUNT) * VICTIM, rel=1e-12
    )


def _retried_artifact(crosscheck=None):
    """A merged leaf whose original large-loss share of 0.06 contradicts a
    p99 of about USD 10 million, and whose retry states 0.002 with the same
    count and loss quantiles."""
    from tests.test_large_loss_reporting import _answer, _instrument

    instrument = _instrument()
    node = instrument.nodes[0]
    original = _answer(node, contradicting=True)
    retried = _answer(node, 2e-3)
    assert original.total_loss == retried.total_loss and original.rate == retried.rate
    artifact = ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(
            contextual_case(
                instrument, prompt, original, retried=retried, crosscheck=crosscheck
            )
            for prompt in contextual_prompts(instrument)
        ),
    )
    return instrument, artifact


def test_the_severity_retry_arm_compiles_the_original_answer_as_unresolved():
    instrument, artifact = _retried_artifact()
    node = instrument.nodes[0]
    assert all(
        case.severity_retry is not None and case.severity_retry.resolved
        for case in artifact.cases
    )
    setup = compile_contextual_setup(
        instrument, artifact, setup_id="retried", years_per_state=1
    )
    record = _exhibit(setup, "frequency_estimate")
    assert record["follow_ups"]["severity_retry"] == {
        "cases_asked": 3,
        "answers_replaced": 3,
    }
    # The original and the retry differ only in their large-loss counts, so
    # putting the original back is ignoring the retry's anchor: the EAL the
    # large_loss_anchor exhibit reaches by compiling the run's answers with
    # the anchor ignored.
    ignored = _exhibit(setup, "large_loss_anchor")["total"]["eal_alternatives"]
    for view in ("welfare", "gross"):
        arms = record["views"][view]["by_arm"]
        assert arms["before_severity_retry"]["expected_annual_loss"] == pytest.approx(
            ignored[view]["anchor_ignored"]["expected_annual_loss"], rel=1e-12
        )
        assert arms["final"]["expected_annual_loss"] == pytest.approx(
            ignored[view]["expected_annual_loss"], rel=1e-12
        )
        effect = record["views"][view]["follow_up_effects"]["severity_retry"]
        assert effect["status"] == "available"
        assert 1 + effect["relative_change"] == pytest.approx(
            1 / (1 + ignored[view]["anchor_ignored"]["relative_effect"]), rel=1e-9
        )
    assert (
        record["views"]["welfare"]["follow_up_effects"]["severity_retry"][
            "relative_change"
        ]
        > 1
    )
    # The arm's law takes the fallback the run gives an unresolved case.
    selection = contextual_respondent_selections(
        instrument, artifact, pooled_weights(instrument.respondents, {})
    )[0]
    answers = {}
    for anchor, arms in selection.frequency_answers_by_node[node.node_id].items():
        answer = arms["before_severity_retry"]
        assert answer is not None
        answers[anchor] = answer
    law = compile_contextual_node(
        instrument,
        node,
        answers,
        instrument.reference_eci,
        options=CompileOptions(unit_count=True),
    )
    assert isinstance(law, IncidentLaw)
    anchor = cast(dict[str, Any], law.conditioning_context["large_loss_anchor"])
    assert anchor["large_loss_role"] == "severity_unresolved_fallback"
    assert anchor["large_loss_share"] == pytest.approx(0.06)


def test_the_severity_retry_arm_keeps_a_later_crosscheck_revision():
    from tests.test_large_loss_reporting import RATE

    halved = dict(zip(QUANTILES, (value / 2 for value in RATE), strict=True))
    _, artifact = _retried_artifact(crosscheck_answer({"rate": halved}))

    def share(answer) -> float:
        assert answer.rate is not None and answer.large_loss_count is not None
        return epistemic_mean(answer.large_loss_count) / epistemic_mean(answer.rate)

    for case in artifact.cases:
        assert case.severity_retry is not None and case.severity_retry.resolved
        before = pooled_answer_before_severity_retry(case)
        after = pooled_answer(case)
        # The cross-check's count, asked about the retried answer, stands in
        # both; the retry's large-loss counts are put back to the original's,
        # so the share against the halved count is 0.06 x 2 and not 0.002 x 2.
        assert before.rate == after.rate
        assert before.total_loss == after.total_loss == case.answer.total_loss
        assert share(before) == pytest.approx(0.12)
        assert share(after) == pytest.approx(0.004)


def test_the_summary_tabulates_each_follow_up(tmp_path):
    from tests.test_contextual_study import _capture

    instrument = contextual_instrument()
    _, _, report, _ = _capture(
        tmp_path, instrument, _guardrail_artifact(instrument, RETRY_TERMINAL)
    )
    markdown, rows = results_summary(report)
    lines = markdown.split("\n")
    start = lines.index("## Effect of follow-up calls")
    table = [line for line in lines[start : start + 12] if line.startswith("| ")]
    assert table == [
        "| Follow-up call | Cases asked | Answers replaced | EAL before | EAL after | Change |",
        "| --- | --- | --- | --- | --- | --- |",
        "| Large-loss coherence retry | 0 | 0 | — | — | — |",
        "| Chain guardrail retry | 3 | 3 | 200 | 800 | 300.0% |",
        "| Leaf EAL cross-check | 0 | 0 | — | — | — |",
    ]
    selected = {
        (row["subject"], row["statistic"]): row
        for row in rows
        if row["section"] == "follow_up_effects"
    }
    assert len(selected) == 3 * 5
    guardrail = {
        statistic: selected["chain_guardrail", statistic]
        for statistic in (
            "cases_asked",
            "answers_replaced",
            "expected_annual_loss_before",
            "expected_annual_loss_after",
            "relative_change",
        )
    }
    assert [row["value"] for row in guardrail.values()] == pytest.approx(
        [3, 3, 200, 800, 3]
    )
    assert [row["unit"] for row in guardrail.values()] == [
        "count",
        "count",
        "USD",
        "USD",
        "ratio",
    ]
    assert {row["status"] for row in guardrail.values()} == {"available"}
    assert guardrail["cases_asked"]["view"] is None
    assert guardrail["relative_change"]["view"] == "welfare"
    retry = selected["severity_retry", "expected_annual_loss_before"]
    assert (retry["value"], retry["status"]) == (None, "unavailable")
    assert retry["reason"] == (
        "No kept case's direct answer was replaced by a large-loss coherence retry."
    )
    # The instrument asks no taxonomy parents, so the summary has no section
    # on their pools.
    assert "## Effect of the taxonomy parents' pools" not in markdown
    assert not [row for row in rows if row["section"] == "hierarchy_pool"]
