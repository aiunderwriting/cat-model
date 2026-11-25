"""The large-loss anchor's elicitation, coherence retry and compiled law.

A leaf's direct question asks the expected annual number of admitted incidents
at or above USD 1 billion (the anchor) and USD 100 billion (a diagnostic), each
only where it lies below the leaf's cap. The anchor's share, its count mean over
the count mean, shapes the tail above the law's own p99 where the level lies
above it. A share that contradicts the stated quantiles gets one coherence retry; if
that does not resolve it the original answers stand, the case is
``severity_unresolved`` and the law falls back to the p99-only tail. Collection
always completes.
"""

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest

from catmodel.collection.contextual_collection import collect_contextual_risk
from catmodel.config import ExecutionSettings
from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskInstrument,
    ContextualRiskNode,
    SharedTotal,
)
from catmodel.contracts.elicitation import QUANTILES, Quantiles
from catmodel.contracts.incident import IncidentLaw
from catmodel.contracts.native_severity import native_severity_sampler
from catmodel.elicitation.contextual_frequency import direct_answer
from catmodel.elicitation.contextual_risk import (
    ContextualPrompt,
    contextual_prompts,
    direct_severity_retry_prompt,
    parse_contextual_answer,
    stored_large_loss_share,
)
from catmodel.elicitation.eal_crosscheck import (
    final_answer,
    large_loss_status,
    pooled_answer,
)
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.llm.transport import LLMProvider, ProviderResponse
from catmodel.probability.distributions import Distribution
from catmodel.scenarios.hierarchy_contract import (
    cached_hierarchy_contract,
    large_loss_levels,
)
from catmodel.setup.contextual_inputs import (
    read_contextual_capture_files,
    validate_contextual_capture,
)
from catmodel.setup.contextual_node import CompileOptions, compile_contextual_node
from catmodel.setup.contextual_setup import contextual_respondent_selections
from catmodel.setup.excluded_samples import pooled_weights
from catmodel.simulation.incident_moments import single_period_law_means
from catmodel.validation_errors import EvidenceValidationError
from test_support.contextual_risk import (
    CHAIN_MARKER,
    contextual_case,
    governed_contextual_instrument,
    scenario_node_answer,
)

LEAF = "compromise_organizations"
LEVEL, HIGHER = 1e9, 1e11
RATE = (6.0, 9.0, 12.0, 16.0, 24.0)
COHERENT = (0.006, 0.009, 0.012, 0.016, 0.03)  # share about 1e-3, below 1 in 100
CONTRADICTING = (0.3, 0.45, 0.6, 0.8, 1.5)  # share about 0.05, at least 1 in 20
VERY = (0.0, 0.0001, 0.0002, 0.0004, 0.001)


def _ladder(values) -> dict[str, float]:
    return dict(zip(QUANTILES, values, strict=True))


def _instrument() -> ContextualRiskInstrument:
    return governed_contextual_instrument((LEAF,))


def _node(instrument=None) -> ContextualRiskNode:
    return (instrument or _instrument()).nodes[0]


def _answer(
    node: ContextualRiskNode,
    large=COHERENT,
    *,
    rate=RATE,
    very=VERY,
    p99: float | None = None,
) -> ContextualRiskAnswer:
    """A synthetic merged-leaf answer: p95 about USD 6.1 million, p99 about
    USD 10.1 million, so USD 1 billion lies far above p99."""
    payload = scenario_node_answer(node).model_dump(mode="json")
    payload["rate"] = _ladder(rate)
    payload["large_loss_count"] = _ladder(large)
    payload["very_large_loss_count"] = _ladder(very)
    if p99 is not None:
        payload["total_loss"]["p99"] = p99
        payload["total_loss"]["p999"] = 10 * p99
    return ContextualRiskAnswer.model_validate(payload)


def _prompt(instrument, score=0.5) -> ContextualPrompt:
    return next(
        prompt
        for prompt in contextual_prompts(instrument)
        if prompt.anchor_score == score
    )


def _total(law: IncidentLaw) -> Distribution:
    severity = next(
        component.severity
        for component in law.segments[0].kernel.member_outcomes[0].components
        if component.severity is not None
    )
    sampler = native_severity_sampler(severity)
    assert isinstance(sampler, Distribution)
    return sampler


def _modelled(law: IncidentLaw, level: float = 1e9) -> float:
    """The law's expected annual number of incidents at or above ``level``
    under the count law's mean over the rank."""
    count = law.reference_annual_rate
    return count * _total(law).survival_probability(level)


def _anchor_record(law: IncidentLaw) -> dict:
    record = law.conditioning_context["large_loss_anchor"]
    assert isinstance(record, dict)
    return record


def _law(instrument, answer, *, severity=1.0, unit_count=False) -> IncidentLaw:
    law = compile_contextual_node(
        instrument,
        _node(instrument),
        dict.fromkeys(instrument.anchor_scores, answer),
        0.5,
        severity_factors=dict.fromkeys(instrument.anchor_scores, severity),
        options=CompileOptions(unit_count=unit_count),
    )
    assert isinstance(law, IncidentLaw), law
    return law


# Authoring and the question


def test_levels_are_authored_once_and_reach_every_elicited_total():
    levels = large_loss_levels()
    assert (levels.large_loss, levels.very_large_loss) == (LEVEL, HIGHER)
    assert cached_hierarchy_contract().admission.threshold < levels.large_loss
    total = _node().total
    assert total is not None
    assert (total.large_loss_level, total.very_large_loss_level) == (LEVEL, HIGHER)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"large_loss_level": None}, "needs the large-loss level"),
        ({"large_loss_level": 2e11}, "increasing order"),
        ({"large_loss_level": 50.0}, "increasing order"),
        ({"cap": 5e10}, "below the cap"),
    ],
)
def test_shared_total_refuses_inconsistent_levels(changes, message):
    payload = {
        "floor": 100.0,
        "gpd_shape": 0.7,
        "large_loss_level": LEVEL,
        "very_large_loss_level": HIGHER,
    } | changes
    with pytest.raises(ValueError, match=message):
        SharedTotal.model_validate(payload)


def test_question_asks_both_counts_with_the_band_rule():
    text = _prompt(_instrument()).render()
    assert "large_loss_count: uncertainty quantiles" in text
    assert "USD 1,000,000,000 (exactly USD 1,000,000,000 counts)" in text
    assert "very_large_loss_count: the same at USD 100,000,000,000" in text
    assert "below 1 in 100" in text and "at least 1 in 20" in text
    # How code uses the counts is not the respondent's business.
    assert "loss law" not in text and "diagnostic" not in text


def test_nodes_without_levels_ask_no_large_loss_count():
    from test_support.contextual_risk import contextual_instrument

    text = _prompt(contextual_instrument()).render()
    assert "large_loss_count" not in text


@pytest.mark.parametrize(
    ("large", "very", "code", "location"),
    [
        ((0.006, 0.009, 13.0, 16.0, 24.0), VERY, "large_loss_count", "p50"),
        (COHERENT, (0.0, 0.0001, 0.0002, 0.0004, 0.05), "very_large_loss_count", "p95"),
    ],
)
def test_nested_counts_must_be_ordered_quantile_by_quantile(
    large, very, code, location
):
    instrument = _instrument()
    with pytest.raises(EvidenceValidationError) as error:
        _answer(_node(instrument), large, very=very).validate_against(_node(instrument))
    assert error.value.code == f"{code}_exceeds_its_superset"
    assert error.value.location == (code, location)


def test_a_zero_count_forces_zero_large_loss_counts():
    node = _node()
    with pytest.raises(EvidenceValidationError):
        _answer(node, COHERENT, rate=(0, 0, 0, 0, 0)).validate_against(node)
    zero = _answer(node, (0, 0, 0, 0, 0), rate=(0, 0, 0, 0, 0), very=(0, 0, 0, 0, 0))
    zero.validate_against(node)
    assert stored_large_loss_share(zero.rate, zero.large_loss_count) is None


def test_counts_are_required_exactly_with_levels():
    node = _node()
    payload = _answer(node).model_dump(mode="json")
    del payload["large_loss_count"]
    with pytest.raises(EvidenceValidationError, match="required exactly"):
        ContextualRiskAnswer.model_validate(payload).validate_against(node)


def test_share_is_the_ratio_of_epistemic_means():
    answer = _answer(_node())
    assert answer.rate is not None and answer.large_loss_count is not None
    share = stored_large_loss_share(answer.rate, answer.large_loss_count)
    assert share == epistemic_mean(answer.large_loss_count) / epistemic_mean(
        answer.rate
    )
    assert share is not None and 5e-4 < share < 2e-3


def test_contradiction_retry_shows_the_numbers_neutrally():
    instrument = _instrument()
    prompt = _prompt(instrument)
    answer = _answer(prompt.node, CONTRADICTING)
    share = stored_large_loss_share(answer.rate, answer.large_loss_count)
    retry = direct_severity_retry_prompt(prompt, answer, share)
    assert retry is not None
    text = retry.render()
    assert text.startswith(prompt.render())
    assert "fewer than 1 in 100" in retry.contradiction
    assert f"share {share:.4g}" in retry.contradiction
    assert "Any of rate, large_loss_count and total_loss" in text
    assert retry.call_id != prompt.call_id
    coherent = _answer(prompt.node)
    assert (
        direct_severity_retry_prompt(
            prompt,
            coherent,
            stored_large_loss_share(coherent.rate, coherent.large_loss_count),
        )
        is None
    )


# Collection: the coherence retry never stops it


class _Provider:
    provider_name = "offline-fixture"
    model_id = "fixture-model"
    reasoning_effort = "low"

    def __init__(self, direct: ContextualRiskAnswer, retry: str | None):
        self.direct, self.retry, self.calls = direct, retry, []

    async def complete(self, **kwargs):
        prompt = kwargs["user_prompt"]
        if CHAIN_MARKER in prompt:
            kind, text = "chain", _matched_chain(prompt, self.direct)
        elif "Your answer to this question contradicted itself" in prompt:
            kind, text = "retry", self.retry or "not json"
        else:
            kind, text = "direct", self.direct.model_dump_json()
        self.calls.append(kind)
        return ProviderResponse(
            text=text, model_id=self.model_id, provider_name=self.provider_name
        )


def _matched_chain(prompt: str, answer: ContextualRiskAnswer) -> str:
    from test_support.contextual_risk import fixture_response

    return fixture_response(prompt, answer)


def _collect(tmp_path: Path, direct, retry):
    instrument = _instrument()
    provider = _Provider(direct, retry)
    run = tmp_path / "run"
    artifact = asyncio.run(
        collect_contextual_risk(
            instrument=instrument,
            run_directory=run,
            providers=[cast(LLMProvider, provider)],
            config=ExecutionSettings(
                samples_per_question=1,
                max_output_tokens_by_model={provider.model_id: 100},
                request_timeout_seconds=60,
                concurrency_by_model={provider.model_id: 1},
            ),
        )
    )
    validate_contextual_capture(
        instrument, artifact, read_contextual_capture_files(run)
    )
    return instrument, artifact, provider


def test_coherent_answers_need_no_retry(tmp_path):
    node = _node()
    _, artifact, provider = _collect(tmp_path, _answer(node), None)
    assert "retry" not in provider.calls
    assert {large_loss_status(case) for case in artifact.cases} == {"coherent"}
    assert all(case.severity_retry is None for case in artifact.cases)


def test_a_retry_that_resolves_the_contradiction_is_adopted(tmp_path):
    node = _node()
    resolved = _answer(node)
    instrument, artifact, provider = _collect(
        tmp_path, _answer(node, CONTRADICTING), resolved.model_dump_json()
    )
    assert provider.calls.count("retry") == len(artifact.cases)
    for case in artifact.cases:
        assert case.severity_retry is not None and case.severity_retry.resolved
        assert large_loss_status(case) == "resolved"
        assert direct_answer(case) == resolved
        assert final_answer(case).large_loss_count == resolved.large_loss_count
    assert _total(_law(instrument, final_answer(artifact.cases[0]))).large_loss_amount


@pytest.mark.parametrize("retry", ["contradicting", "invalid"])
def test_an_unresolved_contradiction_keeps_the_originals_and_completes(tmp_path, retry):
    node = _node()
    original = _answer(node, CONTRADICTING)
    instrument, artifact, provider = _collect(
        tmp_path,
        original,
        original.model_dump_json() if retry == "contradicting" else None,
    )
    # One direct, one chain and one retry call per case; an invalid retry
    # uses its own three attempts and never the direct call's budget.
    retries = 1 if retry == "contradicting" else 3
    assert provider.calls.count("retry") == retries * len(artifact.cases)
    assert provider.calls.count("direct") == len(artifact.cases)
    for case in artifact.cases:
        assert case.severity_retry is not None and not case.severity_retry.resolved
        assert (case.severity_retry.answer is None) == (retry == "invalid")
        assert large_loss_status(case) == "severity_unresolved"
        assert case.answer == original == direct_answer(case)
        fallback = final_answer(case)
        # The contradicting counts stay; compilation finds the contradiction.
        assert fallback.large_loss_count == original.large_loss_count
    law = _law(instrument, final_answer(artifact.cases[0]))
    assert _total(law).large_loss_amount is None
    assert _anchor_record(law)["large_loss_role"] == "severity_unresolved_fallback"


def test_validation_refuses_a_missing_or_mislabelled_retry():
    instrument = _instrument()
    prompt = _prompt(instrument)
    contradicting = _answer(prompt.node, CONTRADICTING)
    case = contextual_case(
        instrument, prompt, contradicting, retried=_answer(prompt.node)
    )
    from catmodel.elicitation.contextual_frequency import validate_large_loss_coherence

    validate_large_loss_coherence(instrument, prompt.node, case)
    with pytest.raises(ValueError, match="exactly when"):
        validate_large_loss_coherence(
            instrument, prompt.node, case.model_copy(update={"severity_retry": None})
        )
    assert case.severity_retry is not None
    mislabelled = case.severity_retry.model_copy(update={"resolved": False})
    with pytest.raises(ValueError, match="resolution differs"):
        validate_large_loss_coherence(
            instrument,
            prompt.node,
            case.model_copy(update={"severity_retry": mislabelled}),
        )
    with pytest.raises(ValueError, match="share differs"):
        validate_large_loss_coherence(
            instrument, prompt.node, case.model_copy(update={"large_loss_share": 0.5})
        )


# The compiled law reproduces the adopted count


def test_compiled_law_reproduces_the_anchor_under_the_count_laws_mean():
    instrument = _instrument()
    answer = _answer(_node(instrument))
    law = _law(instrument, answer)
    assert answer.large_loss_count is not None and answer.rate is not None
    expected = epistemic_mean(answer.large_loss_count)
    total = _total(law)
    assert total.large_loss_amount == LEVEL
    assert law.reference_annual_rate == pytest.approx(epistemic_mean(answer.rate))
    modelled = _modelled(law)
    assert modelled == pytest.approx(expected, rel=1e-12)
    record = _anchor_record(law)
    assert record["large_loss_role"] == "shapes_tail_above_p99"
    # The unit-count law, used for means per expected count, is the same law.
    unit = _law(instrument, answer, unit_count=True)
    assert unit.reference_annual_rate == 1.0
    assert _total(unit) == total


def test_gross_eal_is_at_least_the_level_times_the_large_loss_count():
    instrument = _instrument()
    law = _law(instrument, _answer(_node(instrument)))
    gross = single_period_law_means(law, ("gross",))["gross"]
    count = law.reference_annual_rate * _total(law).survival_probability(LEVEL)
    assert gross >= LEVEL * count > 0


def test_an_anchor_at_or_below_p99_is_a_diagnostic():
    instrument = _instrument()
    # USD 1 billion lies between p95 and p99 and the share is 0.02: coherent.
    answer = _answer(_node(instrument), tuple(0.02 * v for v in RATE), p99=2e9)
    law = _law(instrument, answer)
    assert _total(law).large_loss_amount is None
    assert (
        _anchor_record(law)["large_loss_role"] == "diagnostic_within_stated_quantiles"
    )


def test_a_contradicting_answer_compiles_the_provisional_p99_law():
    """A share below 1 in 100 with the level at or below p99 is a
    contradiction; an answer that reaches compilation still contradicting
    (an unresolved case) gets the p99-only tail, recorded as the fallback."""
    instrument = _instrument()
    answer = _answer(_node(instrument), COHERENT, p99=2e9)
    law = _law(instrument, answer)
    assert _total(law).large_loss_amount is None
    assert _anchor_record(law)["large_loss_role"] == "severity_unresolved_fallback"


@pytest.mark.parametrize(("frequency", "severity"), [(2.0, 1.0), (0.5, 3.0)])
def test_count_and_severity_factors_follow_one_rule(frequency, severity):
    """A frequency factor scales both counts, so the share and law are
    unchanged and the modelled USD 1 billion count scales by the factor; a
    severity factor maps the whole law above the floor, which moves it."""
    instrument = _instrument()
    answer = _answer(_node(instrument))
    assert answer.rate is not None and answer.large_loss_count is not None
    scaled = answer.model_copy(
        update={
            key: Quantiles(
                **{q: frequency * getattr(getattr(answer, key), q) for q in QUANTILES}
            )
            for key in ("rate", "large_loss_count")
        }
    )
    base = _total(_law(instrument, answer))
    law = _law(instrument, scaled, severity=severity)
    floor = 1e5
    expected = (
        frequency
        * epistemic_mean(answer.rate)
        * base.survival_probability(floor + (LEVEL - floor) / severity)
    )
    modelled = _modelled(law)
    assert modelled == pytest.approx(expected, rel=1e-9)
    if severity == 1.0:
        assert modelled == pytest.approx(
            frequency * epistemic_mean(answer.large_loss_count), rel=1e-12
        )


def test_final_model_reproduces_the_anchor_count_under_its_parent_s_pool():
    """A taxonomy parent's pool scales the pooled leaf count and its anchor
    alike, so the share and the law are unchanged: the modelled USD 1 billion
    count is the stated one times the pool's factor."""
    from catmodel.setup.contextual_node import leaf_eal
    from test_support.contextual_risk import (
        fixture_artifact,
        hierarchy_artifact,
        hierarchy_instrument,
    )
    from tests.test_category_pool import _Answers, _outcome, _owning_parent

    instrument = hierarchy_instrument((LEAF,))
    node = instrument.nodes[0]
    answer = _answer(node)
    assert answer.rate is not None and answer.large_loss_count is not None

    def leaf(prompt):
        return _answer(prompt.node)

    unpooled = next(
        row
        for row in fixture_artifact(instrument, leaf).cases
        if row.anchor_score == 0.5
    )
    _, mean = leaf_eal(instrument, node.node_id, pooled_answer(unpooled), 0.5)
    assert mean is not None
    # The leaf's own parent, over leaves alone, answers four times what is
    # beneath it: its pool's factor is 4 ** (1 / 2).
    branch = _owning_parent(instrument, node.node_id)
    answers = _Answers(instrument, {node.node_id: mean}, scales={branch.parent_id: 4})
    artifact = hierarchy_artifact(instrument, leaf, answers)
    factors = _outcome(instrument, artifact).factors[node.node_id]
    assert factors.frequency == pytest.approx(2.0, rel=1e-9)
    assert factors.severity == 1.0
    case = next(row for row in artifact.cases if row.anchor_score == 0.5)
    pooled = pooled_answer(case)
    assert pooled.rate is not None and pooled.large_loss_count is not None
    selection = contextual_respondent_selections(
        instrument, artifact, pooled_weights(instrument.respondents, {})
    )[0]
    final = selection.answers_by_node[node.node_id][0.5]
    assert final.large_loss_count is not None
    assert final.large_loss_count.p50 == pytest.approx(2 * pooled.large_loss_count.p50)
    law = compile_contextual_node(
        instrument,
        node,
        selection.answers_by_node[node.node_id],
        0.5,
        severity_factors=selection.severity_factors_by_node[node.node_id],
    )
    assert isinstance(law, IncidentLaw)
    base = _total(_law(instrument, answer))
    expected = 2.0 * epistemic_mean(pooled.rate) * base.survival_probability(LEVEL)
    modelled = _modelled(law)
    assert modelled == pytest.approx(expected, rel=1e-9)
    assert modelled == pytest.approx(
        2.0 * epistemic_mean(pooled.large_loss_count), rel=1e-9
    )


def test_parse_contextual_answer_accepts_the_counts():
    instrument = _instrument()
    prompt = _prompt(instrument)
    answer = _answer(prompt.node)
    assert parse_contextual_answer(answer.model_dump_json(), prompt) == answer


def test_a_resolving_retry_replaces_only_the_counts_and_loss_quantiles():
    instrument = _instrument()
    prompt = _prompt(instrument)
    original = _answer(prompt.node, CONTRADICTING)
    payload = _answer(prompt.node).model_dump(mode="json")
    payload["lab_annual_business_loss"] = {
        "status": "available",
        "positive_probability": 0,
    }
    payload["lab_compensation"] = {"status": "available", "share": 0.9}
    payload["total_loss"]["p50"] = payload["total_loss"]["p50"] * 1.01
    retried = ContextualRiskAnswer.model_validate(payload)
    case = contextual_case(instrument, prompt, original, retried=retried)
    adopted = direct_answer(case)
    assert adopted.large_loss_count == retried.large_loss_count
    assert adopted.total_loss == retried.total_loss
    assert adopted.lab_compensation == original.lab_compensation
    assert adopted.lab_annual_business_loss == original.lab_annual_business_loss
    assert adopted.branches == original.branches
    assert adopted.reasoning == original.reasoning


def test_a_large_loss_quantile_may_equal_the_count_quantile():
    node = _node()
    _answer(node, RATE, very=(0.0, 0.0, 0.0, 0.0, 0.0)).validate_against(node)


def test_levels_at_or_above_a_cap_are_skipped_with_a_reason():
    from catmodel.elicitation.contextual_risk import large_loss_levels_below

    assert large_loss_levels_below(None) == (LEVEL, HIGHER, ())
    assert large_loss_levels_below(1e14) == (LEVEL, HIGHER, ())
    large, very, reasons = large_loss_levels_below(HIGHER)
    assert (large, very) == (LEVEL, None)
    assert len(reasons) == 1 and "at or above the total's cap" in reasons[0]
    large, very, reasons = large_loss_levels_below(5e8)
    assert (large, very) == (None, None) and len(reasons) == 2
    total = SharedTotal(
        floor=1e5,
        cap=HIGHER,
        gpd_shape=0.7,
        large_loss_level=large_loss_levels_below(HIGHER)[0],
        large_loss_level_exclusions=large_loss_levels_below(HIGHER)[2],
    )
    payload = _instrument().model_dump(mode="json")
    payload["nodes"][0]["total"] = total.model_dump(mode="json")
    instrument = ContextualRiskInstrument.model_validate(payload)
    node = _node(instrument)
    # Only the anchor is asked; the skipped diagnostic level is refused.
    text = _prompt(instrument).render()
    assert "large_loss_count: uncertainty quantiles" in text
    assert "very_large_loss_count" not in text
    answer = _answer(node).model_copy(update={"very_large_loss_count": None})
    answer.validate_against(node)
    with pytest.raises(EvidenceValidationError, match="very_large_loss_count"):
        _answer(node).validate_against(node)
    # No question offers the skipped field, so a respondent following the
    # schema cannot exhaust its validation attempts on it.
    from catmodel.contracts.contextual_risk import EalCrosscheckAnswer

    schemas = (
        ContextualRiskAnswer.json_schema_for(node),
        EalCrosscheckAnswer.json_schema_for(node),
    )
    for schema in schemas:
        assert "very_large_loss_count" not in schema["properties"]
        assert "large_loss_count" in schema["properties"]
    assert '"very_large_loss_count"' not in text
    # A node with no level offers neither count.
    unasked = _instrument().model_dump(mode="json")
    unasked["nodes"][0]["total"] = SharedTotal(
        floor=1e5, cap=5e8, gpd_shape=0.7
    ).model_dump(mode="json")
    bare = _node(ContextualRiskInstrument.model_validate(unasked))
    for field in ("large_loss_count", "very_large_loss_count"):
        assert field not in ContextualRiskAnswer.json_schema_for(bare)["properties"]
        assert field not in EalCrosscheckAnswer.json_schema_for(bare)["properties"]


def test_interpolated_shares_outside_their_band_are_flagged():
    """Both anchors are coherent (USD 1 billion above p99 with share 5e-3, and
    between p95 and p99 with share 0.02); at fraction 0.4 the interpolated
    p99 lies below USD 1 billion while the share is 0.011."""
    instrument = _instrument()
    node = _node(instrument)
    left = _answer(node, tuple(0.005 * value for value in RATE), p99=2e8)
    payload = _answer(node, tuple(0.02 * value for value in RATE)).model_dump(
        mode="json"
    )
    payload["total_loss"].update(p95=5e8, p99=2e9, p999=2e10)
    right = ContextualRiskAnswer.model_validate(payload)
    for answer in (left, right):
        answer.validate_against(node)
    law = compile_contextual_node(
        instrument,
        node,
        {0.5: left, 2: right},
        1.1,
    )
    assert isinstance(law, IncidentLaw)
    assert _anchor_record(law)["large_loss_role"] == "interpolated_band_conflict"
    assert _total(law).large_loss_amount is None


def test_the_pin_is_decided_against_the_laws_computed_p99():
    from catmodel.setup.contextual_node import _large_loss_anchor

    node = _node()
    answer = _answer(node)
    assert node.total is not None and answer.total_loss is not None
    for computed, role in (
        (answer.total_loss.p99, "shapes_tail_above_p99"),
        (1.5e9, "skipped_at_or_below_computed_p99"),
    ):
        anchor = _large_loss_anchor(
            node.total, answer, answer, 0.0, answer.total_loss, computed
        )
        assert anchor is not None and anchor.role == role


def test_a_tiny_share_shapes_the_tail_and_its_record():
    instrument = _instrument()
    law = _law(instrument, _answer(_node(instrument), tuple(1e-7 * v for v in RATE)))
    record = _anchor_record(law)
    assert record == {
        "large_loss_level": LEVEL,
        "large_loss_share": pytest.approx(1e-7, rel=1e-12),
        "stated_p99": record["stated_p99"],
        "computed_p99": pytest.approx(record["stated_p99"], rel=1e-12),
        "large_loss_role": "shapes_tail_above_p99",
    }
    assert _total(law).survival_probability(LEVEL) == pytest.approx(1e-7, rel=1e-12)


def test_replay_tells_an_exhausted_retry_from_an_unfinished_one(tmp_path):
    node = _node()
    original = _answer(node, CONTRADICTING)
    instrument, artifact, _ = _collect(tmp_path, original, None)
    run = tmp_path / "run"
    retry_calls = [
        request.parent
        for request in (run / "calls").glob("*/request.json")
        if "contradicted itself" in json.loads(request.read_text())["user_prompt"]
    ]
    assert retry_calls
    attempts = sorted(retry_calls[0].glob("attempt-*.json"))
    assert len(attempts) == 3
    attempts[-1].unlink()
    with pytest.raises(ValueError, match="coherence retry"):
        validate_contextual_capture(
            instrument, artifact, read_contextual_capture_files(run)
        )
