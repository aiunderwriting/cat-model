"""The leaf EAL cross-check and the world-output plausibility table.

The dominant case is the failure mode the cross-check targets: count
quantiles spanning more than three orders of magnitude (mean about 12 times
the median) and severity answers whose mean per incident is about 90 times
their median, beside a direct EAL near median count times median severity.
Consistent overestimation of this kind passes the chain pool's count
comparison when the direct and chain counts share a broad law. The direct EAL
cross-check tests the direct answer's count and severity against its own
stated EAL.
"""

import asyncio
import json
import math
import re
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from catmodel.collection.contextual_collection import run_contextual_collection
from catmodel.config import ModelProfile
from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskInstrument,
    EalCrosscheckAnswer,
)
from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.incident import IncidentLaw
from catmodel.elicitation.contextual_frequency import (
    ContextualChainPrompt,
    chain_guardrail_retry_prompt,
    validate_contextual_artifact,
)
from catmodel.elicitation.contextual_risk import (
    contextual_prompts,
)
from catmodel.elicitation.eal_crosscheck import (
    EAL_CROSSCHECK_RATIO,
    answer_before_crosscheck,
    case_crosscheck_prompt,
    case_triggers,
    eal_crosscheck_status,
    eal_crosscheck_summary,
    final_answer,
    pooled_answer,
)
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import ProviderResponse
from catmodel.results.plausibility import (
    plausibility_reference,
    plausibility_table,
)
from catmodel.setup.contextual_inputs import load_contextual_collection
from catmodel.setup.contextual_node import CompileOptions, leaf_eal
from catmodel.setup.contextual_setup import (
    compile_contextual_setup,
    contextual_respondent_selections,
)
from catmodel.setup.excluded_samples import pooled_weights
from catmodel.setup.implied_eal import validate_eal_crosschecks
from test_support.capability_sources import write_source_bundle
from test_support.contextual_risk import (
    CHAIN_MARKER,
    CROSSCHECK_MARKER,
    contextual_case,
    crosscheck_answer,
    declined_crosscheck,
    fixture_response,
    governed_contextual_answer,
    governed_contextual_instrument,
    hierarchy_instrument,
    matched_chain_answer,
    parent_answer,
    with_crosscheck,
)

KEYS = ("p5", "p25", "p50", "p75", "p95")
DOMINANT_COUNT = (2_165, 25_400, 137_600, 715_600, 6_876_000)
DOMINANT_SEVERITY = {
    "p5": 2e3,
    "p25": 1e4,
    "p50": 5e4,
    "p75": 4e5,
    "p95": 1e7,
    "p99": 1e8,
    "p999": 1e9,
}
# Near median count times median severity: 137,600 x USD 50,000.
DOMINANT_DIRECT_EAL = (1e9, 3e9, 6.9e9, 1.5e10, 4e10)
REVISED_COUNT = (1_000, 2_000, 3_000, 4_000, 6_000)


def _quantiles(values) -> dict[str, float]:
    return dict(zip(KEYS, values, strict=True))


def _constant(value: float) -> dict[str, float]:
    return dict.fromkeys(KEYS, value)


def _dominant_instrument() -> ContextualRiskInstrument:
    """A synthetic copy of the repository's other-reliance leaf without its
    admission floor, so the reconstructed severity median below today's floor
    is admissible; it names no repository scenario."""
    payload = governed_contextual_instrument(("output_reliance_other",)).model_dump(
        mode="json"
    )
    node = payload["nodes"][0]
    node["node_id"] = "synthetic_reliance"
    node["population"]["scenario_id"] = "synthetic_reliance"
    node["population"]["threshold"] = None
    node["population"]["threshold_comparator"] = None
    node["total"]["floor"] = 0
    return ContextualRiskInstrument.model_validate(payload)


def _dominant_answer(instrument) -> ContextualRiskAnswer:
    data = governed_contextual_answer(instrument.nodes[0]).model_dump(mode="json")
    data["rate"] = _quantiles(DOMINANT_COUNT)
    data["total_loss"] = DOMINANT_SEVERITY
    data["direct_eal"] = _quantiles(DOMINANT_DIRECT_EAL)
    return ContextualRiskAnswer.model_validate(data)


def _case(instrument, prompt, answer, reply=None):
    chain_prompt = ContextualChainPrompt(
        instrument, prompt.node, prompt.respondent, prompt.anchor_score
    )
    chain = matched_chain_answer(prompt.node, answer, chain_prompt.exposure.count)
    guardrail_retry = (
        chain if chain_guardrail_retry_prompt(chain_prompt, chain) is not None else None
    )
    return contextual_case(
        instrument,
        prompt,
        answer,
        chain=chain,
        guardrail_retried=guardrail_retry,
        crosscheck=reply,
    )


def _artifact(instrument, answer, reply=None) -> ContextualRiskArtifact:
    return ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(
            _case(instrument, prompt, answer, reply)
            for prompt in contextual_prompts(instrument)
        ),
    )


def _reference(instrument):
    """What the exhibit needs of a capture besides its cases: each leaf's
    scenario, and the weight of the states that read each case at the
    reference score, here each respondent's own at an equal share."""
    respondents = instrument.respondents
    return (
        {node.node_id: node.population.scenario_id for node in instrument.nodes},
        {
            (node.node_id, instrument.reference_eci, row.key()): 1 / len(respondents)
            for node in instrument.nodes
            for row in respondents
        },
    )


def test_the_dominant_case_triggers_and_its_question_shows_what_drives_it():
    instrument = _dominant_instrument()
    answer = _dominant_answer(instrument)
    prompt = contextual_prompts(instrument)[0]
    case = _case(instrument, prompt, answer)
    check = case.eal_crosscheck
    assert check is not None and case_triggers(case) == ("eal_gap",)
    implied = check.implied
    assert implied.count_mean is not None and implied.eal_mean is not None
    assert implied.harm_per_incident_mean is not None
    assert check.direct_eal_mean is not None
    # The count mean is about twelve times its median, over half of it from
    # beyond the stated p95; the mean harm per incident is far above the
    # stated severity median.
    assert 11 < implied.count_mean / implied.count_median < 13
    assert implied.count_share_beyond_p95 == pytest.approx(0.547, abs=0.005)
    assert implied.harm_per_incident_mean / DOMINANT_SEVERITY["p50"] > 50
    # The implied EAL is the leaf mean EAL the taxonomy parents' pools read.
    _, mean = leaf_eal(instrument, case.node_id, answer, case.anchor_score)
    assert implied.eal_mean == pytest.approx(mean, rel=5e-4)
    assert implied.eal_mean / check.direct_eal_mean > 100
    question = case_crosscheck_prompt(instrument, case)
    assert question is not None
    text = question.render()
    ratio = implied.count_mean / implied.count_median
    assert f"{ratio:,.3g} times its median" in text
    assert f"{implied.count_share_beyond_p95:.0%} of that mean" in text
    assert CROSSCHECK_MARKER in text
    # The direct call asks for the direct EAL and never shows an implied one.
    direct = prompt.render()
    assert (
        "direct_eal: uncertainty quantiles of the expected annual societal harm"
        in direct
    )
    assert f"{implied.eal_mean:,.4g}" not in direct


def test_a_crosschecked_direct_answer_is_pooled_and_the_exhibit_isolates_it():
    instrument = _dominant_instrument()
    answer = _dominant_answer(instrument)
    reply = crosscheck_answer({"rate": _quantiles(REVISED_COUNT)})
    artifact = _artifact(instrument, answer, reply)
    validate_contextual_artifact(artifact, instrument)
    validate_eal_crosschecks(artifact, instrument)
    node_id = instrument.nodes[0].node_id
    for case in artifact.cases:
        assert eal_crosscheck_status(case) == "reconciled"
        assert final_answer(case).rate == Quantiles(**_quantiles(REVISED_COUNT))
        assert answer_before_crosscheck(case).rate == answer.rate
    selection = contextual_respondent_selections(
        instrument, artifact, pooled_weights(instrument.respondents, {})
    )[0]
    for case in artifact.cases:
        assert selection.answers_by_node[node_id][case.anchor_score].rate == (
            pooled_answer(case).rate
        )
    setup = compile_contextual_setup(
        instrument,
        artifact,
        setup_id="crosscheck",
    )
    reference = next(
        state for state in setup.joint_states if state.weight > 0 and state.laws
    )
    pooled_rate = pooled_answer(artifact.cases[0]).rate
    assert pooled_rate is not None
    assert reference.laws[0].reference_annual_rate == pytest.approx(
        epistemic_mean(pooled_rate)
    )
    record = cast(
        dict[str, Any],
        next(
            row.acquisition["frequency_estimate"]
            for row in setup.evidence.records
            if row.acquisition and "frequency_estimate" in row.acquisition
        ),
    )
    arms = record["views"]["welfare"]["by_arm"]
    before = arms["before_eal_crosscheck"]["expected_annual_loss"]
    # Without taxonomy parents the run's own EAL is the post-cross-check one.
    assert arms["before_hierarchy"]["status"] == "inapplicable"
    after = arms["final"]["expected_annual_loss"]
    check = artifact.cases[0].eal_crosscheck
    assert check is not None and check.reconciliation is not None
    implied_before = check.implied.eal_mean
    assert check.reconciliation.implied_after is not None
    implied_after = check.reconciliation.implied_after.eal_mean
    assert implied_before is not None and implied_after is not None
    # The pooled arms show the cross-check's effect on the run's frequency
    # estimate; the direct arm is the cross-checked direct answer's implied EAL.
    assert before != pytest.approx(after)
    assert arms["direct"]["expected_annual_loss"] == pytest.approx(
        implied_after, rel=2e-3
    )
    assert record["eal_crosscheck"]["cases_by_status"]["reconciled"] == 3
    assert record["eal_crosscheck"]["revised_by_judgment"] == {"count": 3}


def test_a_declined_or_still_distant_case_is_reported_unresolved():
    instrument = _dominant_instrument()
    answer = _dominant_answer(instrument)
    node_id = instrument.nodes[0].node_id
    declined = _artifact(instrument, answer)
    validate_eal_crosschecks(declined, instrument)
    for case in declined.cases:
        assert eal_crosscheck_status(case) == "declined"
        assert final_answer(case) == answer_before_crosscheck(case)
    summary = eal_crosscheck_summary(((declined, *_reference(instrument)),))
    assert summary["eal_crosscheck_unresolved"] == 3
    row = summary["by_leaf"][node_id]["cases"][0]
    assert row["eal_crosscheck_unresolved"] and row["revised"] == []
    assert row["implied_to_direct_before"] == pytest.approx(
        row["implied_eal_before"] / row["direct_eal"]
    )
    assert row["implied_to_direct_before"] > 100
    # Revising only the direct EAL to a value still more than three times
    # from the implied EAL leaves the case unresolved; the run is unchanged.
    reply = crosscheck_answer({"direct_eal": _constant(1e11)})
    distant = _artifact(instrument, answer, reply)
    validate_eal_crosschecks(distant, instrument)
    for case in distant.cases:
        assert eal_crosscheck_status(case) == "unresolved"
        assert final_answer(case).rate == answer.rate
    row = eal_crosscheck_summary(((distant, *_reference(instrument)),))["by_leaf"][
        node_id
    ]["cases"][0]
    assert row["eal_crosscheck_unresolved"] and row["revised"] == ["direct_eal"]
    assert row["implied_to_direct_after"] > EAL_CROSSCHECK_RATIO


def test_the_exhibit_gives_each_leaf_its_direct_and_implied_eal_at_the_run_s_weights():
    instrument = _dominant_instrument()
    node = instrument.nodes[0]
    artifact = _artifact(
        instrument,
        _dominant_answer(instrument),
        crosscheck_answer({"rate": _quantiles(REVISED_COUNT)}),
    )
    scenarios, weights = _reference(instrument)
    capture = (artifact, scenarios, weights)
    leaf = eal_crosscheck_summary((capture,))["by_leaf"][node.node_id]
    assert leaf["scenario_id"] == node.population.scenario_id
    # Only the cases at the reference score carry weight, as in the run, and
    # the figures are the means after the revision.
    (reference,) = (
        row for row in leaf["cases"] if row["anchor_score"] == instrument.reference_eci
    )
    assert len(leaf["cases"]) == len(instrument.anchor_scores) > 1
    assert leaf["weighted_direct_eal"] == pytest.approx(reference["direct_eal_after"])
    assert leaf["weighted_implied_eal"] == pytest.approx(reference["implied_eal_after"])
    assert reference["implied_eal_after"] != reference["implied_eal_before"]
    # A leaf's captures add, as a study's two source statuses do.
    twice = eal_crosscheck_summary((capture, capture))["by_leaf"][node.node_id]
    assert twice["weighted_direct_eal"] == pytest.approx(
        2 * leaf["weighted_direct_eal"]
    )
    assert twice["weighted_implied_eal"] == pytest.approx(
        2 * leaf["weighted_implied_eal"]
    )
    # Cases the run gives no weight count for nothing.
    unweighted = eal_crosscheck_summary(((artifact, scenarios, {}),))
    assert unweighted["by_leaf"][node.node_id]["weighted_direct_eal"] == 0
    # A weighted case without a mean leaves the leaf's figures unknown.
    unchecked = artifact.model_copy(
        update={
            "cases": tuple(
                case.model_copy(update={"eal_crosscheck": None})
                for case in artifact.cases
            )
        }
    )
    unknown = eal_crosscheck_summary(((unchecked, scenarios, weights),))
    assert unknown["by_leaf"][node.node_id]["weighted_direct_eal"] is None
    assert unknown["by_leaf"][node.node_id]["weighted_implied_eal"] is None
    json.dumps(unknown, allow_nan=False)
    with pytest.raises(ValueError, match="one scenario in every capture"):
        eal_crosscheck_summary(
            (capture, (artifact, {node.node_id: "another"}, weights))
        )


def test_a_study_weights_its_leaf_figures_as_its_run_weights_its_states():
    from tests.test_public_frontier_comparisons import _compile

    setup = _compile()
    record = next(
        row for row in setup.evidence.records if row.evidence_id == "frequency_estimate"
    )
    frequency = record.acquisition["frequency_estimate"]
    by_scenario = frequency["views"]["welfare"]["by_scenario"]
    leaves = frequency["eal_crosscheck"]["by_leaf"]
    assert leaves
    for leaf in leaves.values():
        # Over every draw, respondent and source status, the implied EAL at
        # the run's weights is the EAL under the direct count, to the four
        # significant figures the cases store.
        assert leaf["weighted_implied_eal"] == pytest.approx(
            by_scenario[leaf["scenario_id"]]["direct"], rel=1e-3
        )
        # A leaf's source statuses add, each a weighted mean of its cases.
        statuses = len(leaf["cases"]) / len(setup.joint_states)
        assert statuses == 2
        direct = [row["direct_eal_after"] for row in leaf["cases"]]
        assert (
            statuses * min(direct)
            < leaf["weighted_direct_eal"]
            < statuses * max(direct)
        )


def test_keeping_every_answer_is_a_first_class_reply_and_revisions_name_their_judgment():
    instrument = _dominant_instrument()
    case = _case(
        instrument, contextual_prompts(instrument)[0], _dominant_answer(instrument)
    )
    question = case_crosscheck_prompt(instrument, case)
    assert question is not None
    text = question.render()
    # The question offers no change as an equal reply and never presumes an
    # error.
    assert "Two replies are equally acceptable" in text
    assert "return status unchanged with your unchanged_reason" in text
    assert (
        "supply unchanged_reason only with status unchanged; with status revised, "
        "set unchanged_reason to null or omit it" in text
    )
    assert "Name at least one" not in text and "was wrong" not in text
    kept = declined_crosscheck()
    assert not kept.revises and kept.unchanged_reason
    with pytest.raises(
        ValidationError, match="unchanged_reason is required when status is unchanged"
    ):
        EalCrosscheckAnswer.model_validate(
            {"status": "unchanged", "reasoning": _REASONING}
        )
    with pytest.raises(
        ValidationError, match="direct_eal is not allowed when status is unchanged"
    ):
        EalCrosscheckAnswer.model_validate(
            {
                **kept.model_dump(mode="json"),
                "direct_eal": _constant(1e10),
            }
        )
    with pytest.raises(
        ValidationError, match="revised is required when status is revised"
    ):
        EalCrosscheckAnswer.model_validate(
            {
                "status": "revised",
                "direct_eal": _constant(1e10),
                "reasoning": _REASONING,
            }
        )
    # Every supplied value belongs to a judgment the answer names.
    with pytest.raises(ValidationError, match="exactly the judgments it names"):
        crosscheck_answer(
            {"rate": _quantiles(REVISED_COUNT), "direct_eal": _constant(1e10)},
            judgments=("count",),
        )


def test_stored_crosscheck_values_and_calls_are_checked():
    instrument = _dominant_instrument()
    answer = _dominant_answer(instrument)
    artifact = _artifact(
        instrument, answer, crosscheck_answer({"rate": _quantiles(REVISED_COUNT)})
    )
    case = artifact.cases[0]
    check = case.eal_crosscheck
    assert check is not None and check.reconciliation is not None

    def replaced(**update):
        changed = case.model_copy(
            update={"eal_crosscheck": check.model_copy(update=update)}
        )
        return artifact.model_copy(update={"cases": (changed, *artifact.cases[1:])})

    assert check.implied.eal_mean is not None
    implied = check.implied.model_copy(
        update={"eal_mean": check.implied.eal_mean * 1.1}
    )
    with pytest.raises(ValueError, match="stored cross-check EALs differ"):
        validate_eal_crosschecks(replaced(implied=implied), instrument)
    with pytest.raises(ValueError, match="cross-check call exactly when"):
        validate_eal_crosschecks(replaced(reconciliation=None), instrument)
    after = check.reconciliation.model_copy(update={"direct_eal_mean_after": 1.0})
    with pytest.raises(ValueError, match="after revision differ"):
        validate_eal_crosschecks(replaced(reconciliation=after), instrument)
    missing = artifact.model_copy(
        update={
            "cases": (
                case.model_copy(update={"eal_crosscheck": None}),
                *artifact.cases[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="exactly when its direct answer"):
        validate_eal_crosschecks(missing, instrument)


class _Provider:
    """Answers every question of a leaf-only or hierarchy collection."""

    model_id = "fixture-model"
    provider_name = "openai"
    reasoning_effort = "low"

    def __init__(self, direct, reply, instrument=None):
        self.direct = direct
        self.reply = reply
        self.parents = (
            {}
            if instrument is None
            else {row.parent_id: row for row in instrument.parents}
        )
        self.calls = []

    async def complete(self, *, user_prompt, **_kwargs):
        await asyncio.sleep(0.1)
        node = re.search(r"^## Scenario ([a-z_]+)$", user_prompt, re.MULTILINE)
        if "societal harm from the whole category above" in user_prompt:
            category = re.search(r"^## Category ([a-z_]+):", user_prompt, re.MULTILINE)
            assert category is not None
            children = self.parents[category.group(1)].children
            kind = "parent"
            text = parent_answer(1e9, 1e8 if children else None).model_dump_json()
        elif CROSSCHECK_MARKER in user_prompt:
            assert node is not None
            kind, text = "crosscheck", self.reply(node.group(1)).model_dump_json()
        else:
            assert node is not None
            answer = self.direct(node.group(1))
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


_REASONING = {
    "median_reason": "Explicitly synthetic software fixture",
    "low_case": "Synthetic",
    "high_case": "Synthetic",
    "method": "direct_judgment",
    "evidence_used": [],
}


def _collect(tmp_path, monkeypatch, base, provider):
    bundle = write_source_bundle(tmp_path / "source")
    payload = base.model_dump(mode="json")
    payload["source_bundle_manifest_sha256"] = capture_source_bundle_snapshot(
        bundle
    ).manifest_sha256
    for node in payload["nodes"]:
        node["source_cards"] = []
    instrument = ContextualRiskInstrument.model_validate(payload)
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
    return instrument, artifact, run, bundle


def test_collection_freezes_the_crosscheck_and_replay_round_trips(
    tmp_path, monkeypatch
):
    base = _dominant_instrument()
    answer = _dominant_answer(base)
    provider = _Provider(
        lambda _node: answer,
        lambda _node: crosscheck_answer({"rate": _quantiles(REVISED_COUNT)}),
    )
    instrument, artifact, run, bundle = _collect(tmp_path, monkeypatch, base, provider)
    anchors = len(instrument.anchor_scores)
    assert provider.calls.count("crosscheck") == anchors
    assert {eal_crosscheck_status(case) for case in artifact.cases} == {"reconciled"}
    _, replayed, _ = load_contextual_collection(run, bundle)
    assert replayed == artifact
    # A capture whose stored cross-check answer differs from its raw response
    # is refused.
    case = artifact.cases[0]
    check = case.eal_crosscheck
    assert check is not None and check.reconciliation is not None
    other = crosscheck_answer({"rate": _quantiles(tuple(2 * v for v in REVISED_COUNT))})
    from catmodel.setup.implied_eal import crosscheck_reconciliation

    question = case_crosscheck_prompt(instrument, case)
    assert question is not None
    tampered = artifact.model_copy(
        update={
            "cases": (
                case.model_copy(
                    update={
                        "eal_crosscheck": check.model_copy(
                            update={
                                "reconciliation": crosscheck_reconciliation(
                                    question, other, None
                                )
                            }
                        )
                    }
                ),
                *artifact.cases[1:],
            )
        }
    )
    (run / "contextual-risk.json").write_text(
        json.dumps(tampered.model_dump(mode="json"))
    )
    with pytest.raises(ValueError, match="cross-check answer differs"):
        load_contextual_collection(run, bundle)


LEAF = "output_reliance_other"
SIBLING = "output_reliance_legal"


def test_a_parent_s_pool_reads_pooled_leaves_after_their_crosscheck(
    tmp_path, monkeypatch
):
    base = hierarchy_instrument((LEAF, SIBLING))
    nodes = {node.node_id: node for node in base.nodes}

    def direct(node_id):
        answer = governed_contextual_answer(nodes[node_id])
        if node_id != LEAF:
            return answer
        # A direct EAL a hundred times below what the answers imply.
        return answer.model_copy(update={"direct_eal": Quantiles(**_constant(1.7e8))})

    rate = governed_contextual_answer(nodes[LEAF]).rate
    assert rate is not None
    revised = {key: value / 100 for key, value in rate.model_dump().items()}
    provider = _Provider(
        direct, lambda _node: crosscheck_answer({"rate": revised}), base
    )
    instrument, artifact, _, _ = _collect(tmp_path, monkeypatch, base, provider)
    assert provider.calls.count("crosscheck") == len(instrument.anchor_scores)
    for case in artifact.cases:
        if case.node_id != LEAF:
            continue
        assert eal_crosscheck_status(case) == "reconciled"
        check = case.eal_crosscheck
        assert check is not None and check.reconciliation is not None
        after = check.reconciliation.implied_after
        assert after is not None and after.eal_mean is not None
        assert check.implied.eal_mean is not None
        assert after.eal_mean == pytest.approx(check.implied.eal_mean / 100, rel=1e-3)
        parents = [
            row
            for row in artifact.parent_cases
            if row.anchor_score == case.anchor_score and LEAF in row.children_mean_eal
        ]
        assert parents
        for parent in parents:
            stored = parent.children_mean_eal[LEAF]
            assert stored is not None
            _, pooled_mean = leaf_eal(
                instrument, LEAF, pooled_answer(case), case.anchor_score
            )
            assert stored == pytest.approx(pooled_mean, rel=1e-3)
            assert parent.pool is not None and parent.pool.status == "pooled"
    # No call follows a parent's answer.
    assert set(provider.calls) == {"direct", "chain", "crosscheck", "parent"}


def test_plausibility_table_flags_large_shares_without_changing_results():
    reference = plausibility_reference()
    output = reference.world_output.value
    assert reference.world_output.status == "OK"
    assert reference.flags.status == "INITIAL_ESTIMATE"
    assert reference.flags.leaf_share == 0.001
    assert reference.flags.total_share == 0.01
    package = SimpleNamespace(
        scope=SimpleNamespace(
            period_start=SimpleNamespace(year=reference.world_output.year),
            populations=(
                SimpleNamespace(scenario_id="large", currency="USD"),
                SimpleNamespace(scenario_id="small", currency="USD"),
                SimpleNamespace(scenario_id="missing", currency="USD"),
            ),
        )
    )
    metrics = {
        "welfare": {"status": "available", "expected_annual_loss": 0.02 * output},
        "scenario::large::welfare": {
            "status": "available",
            "expected_annual_loss": 0.0015 * output,
            "eal_estimator": "native_analytic_mean",
        },
        "scenario::small::welfare": {
            "status": "available",
            "expected_annual_loss": 0.0005 * output,
        },
    }
    table = plausibility_table(package, metrics)
    assert table["status"] == "available"
    assert table["total"]["flagged"]
    assert table["total"]["share_of_world_output"] == pytest.approx(0.02)
    assert table["by_scenario"]["large"]["flagged"]
    assert not table["by_scenario"]["small"]["flagged"]
    assert table["by_scenario"]["missing"]["status"] == "unavailable"
    assert table["flagged_scenarios"] == ["large"]
    assert table["reference_year_note"] is None
    assert not math.isnan(table["by_scenario"]["large"]["share_of_world_output"])
    other = SimpleNamespace(
        scope=SimpleNamespace(
            period_start=SimpleNamespace(year=reference.world_output.year),
            populations=(SimpleNamespace(scenario_id="large", currency="EUR"),),
        )
    )
    assert plausibility_table(other, metrics)["status"] == "unavailable"


def test_a_run_reports_the_crosscheck_and_the_plausibility_table(tmp_path):
    from catmodel.results.results_summary import results_summary
    from tests.test_contextual_study import _capture

    instrument = _dominant_instrument()
    answer = _dominant_answer(instrument)
    artifact = _artifact(
        instrument, answer, crosscheck_answer({"rate": _quantiles(REVISED_COUNT)})
    )
    _, _, report, _ = _capture(
        tmp_path,
        instrument,
        artifact,
    )
    summary = report["sensitivity_summary"]
    check = summary["frequency_estimate"]["eal_crosscheck"]
    assert check["cases_by_status"]["reconciled"] == len(instrument.anchor_scores)
    table = summary["plausibility"]
    welfare = report["arms"]["reference"]["metrics"]["welfare"]
    output = plausibility_reference().world_output.value
    assert table["total"]["share_of_world_output"] == pytest.approx(
        welfare["expected_annual_loss"] / output
    )
    markdown, rows = results_summary(report)
    assert "## Leaf EAL cross-check" in markdown
    # The summary sets the leaf's three EAL figures side by side and leaves
    # the cases to the CSV.
    node = instrument.nodes[0]
    leaf = check["by_leaf"][node.node_id]
    assert leaf["weighted_direct_eal"] > 0 and leaf["weighted_implied_eal"] > 0
    by_scenario = summary["frequency_estimate"]["views"]["welfare"]["by_scenario"]
    final = by_scenario[leaf["scenario_id"]]["final"]
    assert final == pytest.approx(
        report["arms"]["reference"]["metrics"][
            f"scenario::{leaf['scenario_id']}::welfare"
        ]["expected_annual_loss"]
    )
    assert (
        "| Leaf | Direct EAL | Implied EAL | Run EAL | Direct ÷ implied | "
        "Direct ÷ run | Cases asked | Still apart |"
    ) in markdown
    assert f"| `{node.node_id}` | " in markdown and "| **All leaves** | " in markdown
    assert "| Leaf | Case | Triggers |" not in markdown
    crosscheck_rows = [row for row in rows if row["section"] == "eal_crosscheck"]
    weighted = {
        row["statistic"]: row["value"]
        for row in crosscheck_rows
        if row["subject"] == node.node_id
    }
    assert weighted == {
        "weighted_direct_eal": leaf["weighted_direct_eal"],
        "weighted_implied_eal": leaf["weighted_implied_eal"],
    }
    assert sum(
        row["statistic"] == "implied_to_direct_ratio" for row in crosscheck_rows
    ) == len(instrument.anchor_scores)
    # Results written before the exhibit held the leaf figures still render.
    older = json.loads(json.dumps(report))
    for row in older["sensitivity_summary"]["frequency_estimate"]["eal_crosscheck"][
        "by_leaf"
    ].values():
        del row["weighted_direct_eal"], row["weighted_implied_eal"], row["scenario_id"]
    assert f"| `{node.node_id}` | — | — | " in results_summary(older)[0]
    # The follow-up table sets the cross-check's arm beside the run's own EAL,
    # since this instrument asks no taxonomy parents.
    cases = len(instrument.anchor_scores)
    frequency = summary["frequency_estimate"]
    assert frequency["follow_ups"]["eal_crosscheck"] == {
        "cases_asked": cases,
        "answers_replaced": cases,
    }
    arms = frequency["views"]["welfare"]["by_arm"]
    effect = frequency["views"]["welfare"]["follow_up_effects"]["eal_crosscheck"]
    assert effect["after_arm"] == "final"
    assert (
        effect["expected_annual_loss_before"]
        == (arms["before_eal_crosscheck"]["expected_annual_loss"])
    )
    assert (
        effect["expected_annual_loss_after"] == (arms["final"]["expected_annual_loss"])
    )
    assert effect["relative_change"] < 0
    assert f"| Leaf EAL cross-check | {cases} | {cases} | " in markdown
    assert "## Plausibility checks" in markdown
    assert any(row["section"] == "plausibility" for row in rows)


# The large-loss anchor inside the cross-check


def _anchored(direct_eal: float, large=None):
    """A merged leaf whose USD 1 billion anchor shapes its law (share about
    1e-3, far above a stated p99 of about USD 10 million), with a direct EAL
    far below what the answers imply, so the cross-check triggers."""
    from tests.test_large_loss_elicitation import COHERENT, LEAF
    from tests.test_large_loss_elicitation import _answer as anchored_answer

    instrument = governed_contextual_instrument((LEAF,))
    answer = anchored_answer(
        instrument.nodes[0], COHERENT if large is None else large
    ).model_copy(update={"direct_eal": Quantiles(**_constant(direct_eal))})
    return instrument, answer


def test_the_implied_eal_includes_the_large_loss_anchor():
    from catmodel.setup.contextual_node import compile_contextual_node
    from catmodel.setup.implied_eal import implied_eal
    from catmodel.simulation.incident_moments import single_period_law_means

    instrument, answer = _anchored(1e5)
    node = instrument.nodes[0]

    def welfare_per_count(candidate):
        law = compile_contextual_node(
            instrument,
            node,
            {0.5: candidate},
            0.5,
            options=CompileOptions(unit_count=True),
        )
        assert isinstance(law, IncidentLaw)
        return single_period_law_means(law, ("welfare",))["welfare"]

    implied = implied_eal(instrument, node, answer, 0.5)
    assert implied is not None and implied.eal_mean is not None
    assert answer.rate is not None
    anchored = epistemic_mean(answer.rate) * welfare_per_count(answer)
    unanchored = epistemic_mean(answer.rate) * welfare_per_count(
        answer.model_copy(
            update={
                "large_loss_count": Quantiles(**_constant(0)),
                "very_large_loss_count": Quantiles(**_constant(0)),
            }
        )
    )
    # The anchor adds a tail above the stated p99, so it raises the EAL.
    assert anchored > 1.1 * unanchored
    assert implied.eal_mean == pytest.approx(anchored, rel=5e-4)


def test_a_revision_that_breaks_the_band_rule_gets_its_one_coherence_retry():
    from catmodel.elicitation.eal_crosscheck import large_loss_status

    instrument, answer = _anchored(1e5)
    prompt = next(p for p in contextual_prompts(instrument) if p.anchor_score == 0.5)
    assert answer.rate is not None and answer.large_loss_count is not None
    # A count ten times lower beside the same large-loss count implies that
    # about 1 in 100 units reach USD 1 billion, against a p99 far below it.
    lower = {key: value / 10 for key, value in answer.rate.model_dump().items()}
    reply = crosscheck_answer({"rate": lower})
    retried = crosscheck_answer(
        {
            "rate": lower,
            "large_loss_count": {
                key: value / 10
                for key, value in answer.large_loss_count.model_dump().items()
            },
            "very_large_loss_count": _constant(0),
        }
    )
    case = contextual_case(instrument, prompt, answer, crosscheck=reply)
    check = case.eal_crosscheck
    assert check is not None and check.reconciliation is not None
    # Without a resolving retry the revised count stands and its anchor is
    # left severity-unresolved.
    retry = check.reconciliation.severity_retry
    assert retry is not None and retry.answer is None and not retry.resolved
    assert "1 in 100" in retry.contradiction
    assert large_loss_status(case) == "severity_unresolved"
    assert final_answer(case).rate == Quantiles(**lower)
    # A resolving retry is adopted with the judgments it names.
    case = with_crosscheck(instrument, case, reply, retried)
    check = case.eal_crosscheck
    assert check is not None and check.reconciliation is not None
    retry = check.reconciliation.severity_retry
    assert retry is not None and retry.resolved
    assert large_loss_status(case) == "resolved"
    assert final_answer(case).large_loss_count == retried.large_loss_count
    validate_eal_crosschecks(
        ContextualRiskArtifact(
            instrument_sha256=instrument.digest(),
            evidence_status="fixture",
            cases=tuple(
                with_crosscheck(
                    instrument,
                    contextual_case(instrument, row, answer),
                    reply,
                    retried,
                )
                for row in contextual_prompts(instrument)
            ),
        ),
        instrument,
    )


def test_unresolved_counts_are_not_drivers_and_a_direct_eal_revision_is_not_retried():
    from catmodel.elicitation.eal_crosscheck import large_loss_status
    from tests.test_large_loss_elicitation import CONTRADICTING

    instrument, answer = _anchored(1e5, CONTRADICTING)
    prompt = next(p for p in contextual_prompts(instrument) if p.anchor_score == 0.5)
    reply = crosscheck_answer({"direct_eal": _constant(2e5)})
    # The direct answer's own coherence retry gave no valid answer.
    case = contextual_case(instrument, prompt, answer, crosscheck=reply)
    assert case.severity_retry is not None and not case.severity_retry.resolved
    question = case_crosscheck_prompt(instrument, case)
    assert question is not None
    text = question.render()
    assert (
        "contradict your total_loss quantiles, so they do not shape this implied EAL"
        in text
    )
    assert "Large-loss count (judgment_id large_loss_count):" not in text
    check = case.eal_crosscheck
    assert check is not None and check.reconciliation is not None
    assert check.reconciliation.answer is not None
    assert check.reconciliation.answer.revises
    # Revising only the direct EAL asks no retry of the old contradiction.
    assert check.reconciliation.severity_retry is None
    assert large_loss_status(case) == "severity_unresolved"


def test_the_severity_drivers_compare_like_with_like():
    instrument, answer = _anchored(1e5)
    case = contextual_case(
        instrument,
        next(p for p in contextual_prompts(instrument) if p.anchor_score == 0.5),
        answer,
    )
    check = case.eal_crosscheck
    assert check is not None
    implied = check.implied
    assert implied.total_mean is not None and implied.severity_median is not None
    assert implied.harm_per_incident_mean is not None
    assert implied.gross_per_incident_mean
    question = case_crosscheck_prompt(instrument, case)
    assert question is not None
    text = question.render()
    # The gross mean beside the gross median, and the welfare share apart.
    ratio = implied.total_mean / implied.severity_median
    assert f"{ratio:,.3g} times your total_loss median" in text
    assert "the mean gross loss per qualified unit of total_loss is" in text
    share = implied.harm_per_incident_mean / implied.gross_per_incident_mean
    assert f"{share:.0%} of the mean gross loss per unit" in text
    assert "of that mean comes from units at or above USD 1,000,000,000" in text


def test_the_exhibit_lists_loss_level_counts_at_the_stored_count_means():
    from catmodel.elicitation.eal_crosscheck import eal_crosscheck_limitation

    instrument = _dominant_instrument()
    answer = _dominant_answer(instrument)
    node_id = instrument.nodes[0].node_id
    artifact = _artifact(
        instrument, answer, crosscheck_answer({"rate": _quantiles(REVISED_COUNT)})
    )
    summary = eal_crosscheck_summary(((artifact, *_reference(instrument)),))
    json.dumps(summary, allow_nan=False)
    case = artifact.cases[0]
    check = case.eal_crosscheck
    assert check is not None and check.reconciliation is not None
    assert check.reconciliation.implied_after is not None
    row = summary["by_leaf"][node_id]["cases"][0]
    counts = row["loss_level_counts"]
    assert counts["before_crosscheck_count_mean"] == check.implied.count_mean
    mean = check.reconciliation.implied_after.count_mean
    assert mean is not None and counts["final_count_mean"] == mean
    assert [level["basis"] for level in counts["final"]] == [
        "p50",
        "p75",
        "fixed",
        "p95",
        "p99",
    ]
    final = {level["level"]: level for level in counts["final"]}
    # One value at a stated quantile, bounds at a reporting level between two.
    at_p99 = final[DOMINANT_SEVERITY["p99"]]
    assert at_p99["annual_count_low"] == at_p99["annual_count_high"]
    assert at_p99["annual_count_low"] == pytest.approx(0.01 * mean, rel=1e-3)
    assert final[1e6]["annual_count_low"] <= 0.05 * mean
    assert final[1e6]["annual_count_high"] >= 0.25 * mean
    before = {level["level"]: level for level in counts["before_crosscheck"]}
    before_mean = counts["before_crosscheck_count_mean"]
    assert before_mean is not None
    assert before[DOMINANT_SEVERITY["p99"]]["annual_count_low"] == pytest.approx(
        0.01 * before_mean, rel=1e-3
    )
    assert "count_moved" not in row
    assert "count_moved" not in summary
    assert "Count moved:" not in eal_crosscheck_limitation((artifact,))
