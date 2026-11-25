"""Collection cross-checks direct frequency before pooling the chain law."""

from typing import cast

import pytest

from catmodel.collection.contextual_collection import collect_contextual_risk
from catmodel.config import ExecutionSettings
from catmodel.contracts.contextual_risk import (
    ContextualChainAnswer,
    ContextualRiskArtifact,
    ContextualRiskCase,
    ContextualRiskInstrument,
)
from catmodel.contracts.elicitation import QUANTILES, Quantiles, Respondent
from catmodel.contracts.incident import IncidentLaw
from catmodel.elicitation.chain_law import pooled_count
from catmodel.elicitation.contextual_frequency import (
    ContextualChainPrompt,
    chain_count_law_for_answer,
    driving_chain_answer,
    frequency_collection_summary,
)
from catmodel.elicitation.contextual_risk import contextual_prompts
from catmodel.elicitation.eal_crosscheck import (
    answer_before_crosscheck,
    final_answer,
    pooled_answer,
    pooled_answer_before_crosscheck,
)
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.llm.transport import LLMProvider
from catmodel.setup.contextual_compilation import (
    frequency_contribution,
    frequency_estimate_views,
)
from catmodel.setup.contextual_node import CompileOptions, compile_contextual_node
from catmodel.setup.contextual_setup import contextual_respondent_selections
from catmodel.setup.excluded_samples import pooled_weights
from catmodel.simulation.incident_moments import single_period_law_means
from test_support.collection_trace import (
    CollectionTrace,
    LatencyProvider,
    ModelLatency,
    StudyResponder,
    TriggerRates,
    run_warped,
)
from test_support.contextual_risk import (
    chain_answer,
    contextual_answer,
    contextual_case,
    governed_contextual_instrument,
    hierarchy_instrument,
    scenario_node_answer,
)


def _quantiles(values):
    return Quantiles(**dict(zip(QUANTILES, values, strict=True)))


def _scale(law, factor):
    return Quantiles(**{key: getattr(law, key) * factor for key in QUANTILES})


@pytest.fixture
def collected_pool(tmp_path):
    instrument = hierarchy_instrument(("output_reliance_legal",))
    responder = StudyResponder(
        [instrument],
        TriggerRates(
            direct_retry=0,
            chain_guardrail_retry=1,
            crosscheck=1,
            crosscheck_count_revision=1,
            crosscheck_retry=0,
            parent_gap=0,
        ),
    )
    provider = LatencyProvider(
        model_id=instrument.respondents[0].model_id,
        provider_name="offline-fixture",
        latency=ModelLatency(
            output_tokens=100,
            tokens_per_second=1000,
            first_token_seconds=0.01,
        ),
        responder=responder,
        trace=CollectionTrace(),
        max_concurrent=4,
    )
    artifact, _ = run_warped(
        collect_contextual_risk(
            instrument=instrument,
            run_directory=tmp_path / "run",
            providers=[cast(LLMProvider, provider)],
            config=ExecutionSettings(
                samples_per_question=1,
                max_output_tokens_by_model={provider.model_id: 1000},
                request_timeout_seconds=60,
                concurrency_by_model={provider.model_id: 4},
            ),
        )
    )
    return instrument, artifact


def test_collection_crosschecks_direct_then_pools_the_driving_chain(collected_pool):
    instrument, artifact = collected_pool
    node = instrument.collectible_nodes[0]
    assert node.frequency_chain is not None
    exposure = instrument.chain_exposure(node.frequency_chain).count
    for case in artifact.cases:
        assert case.eal_crosscheck is not None
        assert case.eal_crosscheck.reconciliation is not None
        crosscheck = case.eal_crosscheck.reconciliation.answer
        assert crosscheck is not None and crosscheck.revised[0].judgment_id == "count"
        assert case.chain_guardrail is not None and case.chain_guardrail.resolved
        assert case.chain_guardrail.answer is not None
        assert driving_chain_answer(case) == case.chain_guardrail.answer
        assert case.chain_count == chain_count_law_for_answer(
            case.chain_guardrail.answer, exposure
        )

        before = answer_before_crosscheck(case)
        direct = final_answer(case)
        assert before.rate is not None and direct.rate is not None
        assert direct.rate.p50 == pytest.approx(100 * before.rate.p50, rel=0.01)
        assert pooled_answer_before_crosscheck(case).rate == pooled_count(
            before.rate, case.chain_count
        )
        assert pooled_answer(case).rate == pooled_count(direct.rate, case.chain_count)
        assert pooled_answer(case).rate != direct.rate

    summary = frequency_collection_summary(
        ((artifact, {row.node_id for row in instrument.collectible_nodes}),)
    )
    assert summary["chain_guardrail"] == {
        "chains_checked": len(artifact.cases),
        "chains_flagged": len(artifact.cases),
        "retries_resolved": len(artifact.cases),
        "retries_still_violating": 0,
        "retries_exhausted": 0,
        "retries_unavailable": 0,
    }
    assert "log10_chain_to_direct_median" in summary
    assert "log10_chain_to_direct_central_width" in summary


def test_pooling_scales_large_loss_counts_by_the_pooled_to_direct_mean():
    direct_rate = _quantiles((1, 2, 4, 8, 16))
    chain_response = chain_answer()
    chain_count = chain_count_law_for_answer(chain_response, 1)
    assert chain_count is not None
    large = _scale(direct_rate, 0.02)
    case = ContextualRiskCase(
        node_id="node",
        anchor_score=0,
        respondent=Respondent(
            respondent_id="fixture", model_id="fixture-model", sample_index=0
        ),
        answer=contextual_answer().model_copy(
            update={"rate": direct_rate, "large_loss_count": large}
        ),
        prompt_sha256="a" * 64,
        chain_answer=chain_response,
        chain_prompt_sha256="b" * 64,
        chain_count=chain_count,
        large_loss_share=0.02,
    )
    pooled = pooled_answer(case)
    assert pooled.rate is not None
    assert pooled.rate == pooled_count(direct_rate, chain_count)
    assert pooled.large_loss_count is not None
    assert epistemic_mean(pooled.large_loss_count) / epistemic_mean(
        pooled.rate
    ) == pytest.approx(0.02)


def test_chain_only_arm_keeps_a_wholly_zero_chain():
    """A chain whose count law is wholly zero is set aside by the pool, and
    reported as zero by the chain-only arm, with both large-loss counts."""
    direct_rate = _quantiles((1, 2, 4, 8, 16))
    zero = _quantiles((0, 0, 0, 0, 0))
    case = ContextualRiskCase(
        node_id="node",
        anchor_score=0,
        respondent=Respondent(
            respondent_id="fixture", model_id="fixture-model", sample_index=0
        ),
        answer=contextual_answer().model_copy(
            update={"rate": direct_rate, "large_loss_count": _scale(direct_rate, 0.02)}
        ),
        prompt_sha256="a" * 64,
        chain_answer=chain_answer(),
        chain_prompt_sha256="b" * 64,
        chain_count=zero,
        large_loss_share=0.02,
    )
    assert pooled_answer(case).rate == direct_rate
    chain_only = pooled_answer(case, weight=1.0)
    assert chain_only.rate == zero
    assert chain_only.large_loss_count == zero


def test_direct_frequency_arm_compiles_the_direct_large_loss_shares_at_midpoint():
    instrument = governed_contextual_instrument(("output_reliance_legal",))
    payload = instrument.model_dump(mode="json")
    payload["anchor_scores"] = [0.0, 2.0]
    payload["reference_eci"] = 0.0
    instrument = ContextualRiskInstrument.model_validate(payload)
    (node,) = instrument.nodes
    chain_definition = node.frequency_chain
    assert chain_definition is not None
    direct_inputs: dict[float, tuple[float, float]] = {
        0.0: (1.0, 0.001),
        2.0: (100.0, 0.009),
    }
    chain_counts: dict[float, float] = {0.0: 10_000.0, 2.0: 0.01}
    cases = []

    for prompt in contextual_prompts(instrument):
        count, large_loss_share = direct_inputs[prompt.anchor_score]
        answer = scenario_node_answer(node, count=count).model_copy(
            update={
                "large_loss_count": _quantiles([count * large_loss_share] * 5),
                "very_large_loss_count": _quantiles([0.0] * 5),
            }
        )
        chain_prompt = ContextualChainPrompt(
            instrument, node, prompt.respondent, prompt.anchor_score
        )
        chain_count = chain_counts[prompt.anchor_score]
        opportunity_rate = chain_count / (chain_prompt.exposure.count * 0.25)
        template = chain_answer()
        chain = ContextualChainAnswer.model_validate(
            {
                "status": "available",
                "opportunity_rate": _quantiles([opportunity_rate] * 5),
                "stages": [
                    {
                        "stage_id": stage.stage_id,
                        "probability": dict.fromkeys(
                            QUANTILES,
                            0.25 if index == len(chain_definition.stages) - 1 else 1,
                        ),
                    }
                    for index, stage in enumerate(chain_definition.stages)
                ],
                "reasoning": template.reasoning.model_dump(),
            }
        )
        cases.append(
            contextual_case(
                instrument,
                prompt,
                answer,
                chain=chain,
            )
        )

    artifact = ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(cases),
    )
    for case in artifact.cases:
        expected_count, expected_share = direct_inputs[case.anchor_score]
        direct = final_answer(case)
        assert direct.rate is not None and direct.large_loss_count is not None
        assert epistemic_mean(direct.rate) == pytest.approx(expected_count)
        assert epistemic_mean(direct.large_loss_count) / epistemic_mean(
            direct.rate
        ) == pytest.approx(expected_share)
        assert case.chain_count is not None
        assert epistemic_mean(case.chain_count) == pytest.approx(
            chain_counts[case.anchor_score]
        )
    selection = contextual_respondent_selections(
        instrument, artifact, pooled_weights(instrument.respondents, {})
    )[0]
    midpoint = 1.0

    def compile_law(answers, severity_factors, **options):
        return compile_contextual_node(
            instrument,
            node,
            answers,
            midpoint,
            severity_factors=severity_factors,
            options=CompileOptions(**options),
        )

    answers = selection.answers_by_node[node.node_id]
    run_law = compile_law(answers, selection.severity_factors_by_node[node.node_id])
    contribution = frequency_contribution(
        instrument,
        node,
        selection,
        midpoint,
        run_law,
        compile_law=compile_law,
    )

    direct_answers = {case.anchor_score: final_answer(case) for case in artifact.cases}
    direct_unit_law = compile_law(direct_answers, None, unit_count=True)
    views = frequency_estimate_views(instrument)
    direct_per_count = single_period_law_means(direct_unit_law, views, annual=False)[
        "gross"
    ]
    direct_midpoint_count = _quantiles([50.5] * 5)
    # Gross adds the lab's annual business loss once in a year with an
    # incident, under the arm's own (constant) count.
    assert isinstance(direct_unit_law, IncidentLaw)
    annual = single_period_law_means(
        direct_unit_law.model_copy(
            update={"reference_annual_rate": 50.5, "parameter_rank": None}
        ),
        views,
        episodes=False,
    )["gross"]
    expected_direct_gross = (
        epistemic_mean(direct_midpoint_count) * direct_per_count + annual
    )

    assert contribution["gross"]["direct"] == pytest.approx(
        expected_direct_gross, rel=1e-9
    )
