"""Synthetic contextual judgments for tests."""

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from catmodel.contracts.contextual_risk import (
    ChainGuardrailRetry,
    ContextualChainAnswer,
    ContextualParentCase,
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskCase,
    ContextualRiskInstrument,
    ContextualRiskNode,
    EalCrosscheckAnswer,
    ParentAnswer,
)
from catmodel.elicitation.contextual_risk import ContextualPrompt, contextual_prompts
from test_support.study import study_payload

FIXTURE_CHAIN: dict[str, Any] = {
    "authoring_status": "authored",
    "opportunity": {
        "unit": "One synthetic misuse attempt against the fixed deployment",
        "counting_rule": "Each distinct attempt once; retries within one attempt are one attempt",
        "cohort": "Attempts initiated during the study year",
    },
    "exposure": {
        "unit": "block of 1,000 synthetic users",
        "basis": "normalisation",
        "count": {
            "kind": "derived",
            "scale_id": "users",
            "factor": 1e-3,
            "rationale": "Synthetic base; software fixture only.",
            "source": "Software fixture only.",
        },
        "derivation": "The stipulated synthetic users divided by 1,000.",
    },
    "stages": [
        {
            "stage_id": "bypasses_controls",
            "event": "The attempt bypasses the fixed synthetic controls",
        },
        {
            "stage_id": "qualified_terminal",
            "event": "Qualified synthetic initiating incidents; no additional probability gates",
        },
    ],
}
"""A synthetic two-stage chain.

Its rate is per block of 1,000 synthetic users, and the fixture instrument
stipulates 1,000 users, one block, so its opportunity rate equals its
opportunity count."""

FIXTURE_EXPOSURE: tuple[dict[str, Any], ...] = (
    {"scale_id": "users", "unit": "synthetic users", "level": 1000},
    {
        "scale_id": "consumer_weekly_users",
        "unit": "weekly active users of the consumer assistant",
        "level": 6e8,
    },
    {
        "scale_id": "paying_business_customers",
        "unit": "paying business customers",
        "level": 1.6e6,
    },
    {
        "scale_id": "api_tokens_per_day",
        "unit": "tokens per day processed through the API",
        "level": 1e11,
    },
    {"scale_id": "lab_staff", "unit": "lab staff", "level": 8000},
)
"""The fixture instrument's fixed scales: the synthetic chain's and every
scale a repository scenario's chain names, at the hypothetical lab's 2027
median-path levels except API tokens, a thousandth of it so that no fixture
question states a trillion."""


def contextual_instrument() -> ContextualRiskInstrument:
    original = study_payload()
    model = {**original["entities"][0], "related_entity_ids": ["lab"]}
    return ContextualRiskInstrument.model_validate(
        {
            "context_id": "synthetic-lab-public",
            "source_bundle_manifest_sha256": "a" * 64,
            "period": {
                "policy_period_start_utc": original["period_start"],
                "policy_period_end_utc": original["period_end"],
                "forecast_as_of": original["forecast_as_of"],
                "data_cutoff_utc": original["data_cutoff"],
                "availability_cutoff_utc": original["availability_cutoff"],
                "information_date_utc": original["information_date"],
            },
            "lab": {
                "entity_id": "lab",
                "kind": "lab",
                "definition": "One synthetic consolidated lab before recoveries",
                "display_name": "Synthetic lab",
                "status": "hypothetical",
            },
            "model": model,
            "deployment": {
                "entity_id": "deployment",
                "kind": "deployment",
                "definition": "A fixed synthetic public deployment",
                "display_name": "Public deployment",
                "status": "hypothetical",
            },
            "deployment_binding": {
                "deployment_id": "deployment",
                "model_id": "model",
                "use_population": "Synthetic fixed use: 1,000 synthetic users; "
                "600 million weekly active users of the consumer assistant; 1.6 "
                "million paying business customers; 100 billion tokens per day "
                "processed through the API; 8,000 lab staff",
                "tools_and_permissions": "Synthetic fixed permissions",
                "human_oversight": "Synthetic fixed oversight",
            },
            "capability_scale": original["capability_scales"][0],
            "capability_domain_id": "general",
            "capability_access": "public",
            "reference_eci": 0.5,
            "integer_min": 0,
            "integer_max": 2,
            "anchor_scores": [0, 0.5, 2],
            "exposure": "Fixed synthetic exposure at every score",
            "fixed_exposure": list(FIXTURE_EXPOSURE),
            "fixed_controls": "Fixed synthetic controls at every score",
            "fixed_adoption": "Fixed synthetic adoption at every score",
            "locality_assumption": "For software tests only, apply constant-score instantaneous rates and unchanged marks to each path segment.",
            "nodes": [
                {
                    "node_id": "synthetic-node",
                    "population": {
                        **original["populations"][0],
                        "deployment_ids": ["deployment"],
                    },
                    "qualified_count_definition": "Qualified synthetic initiating incidents; no additional probability gates",
                    "count_family": "poisson",
                    "lab_loss_gpd_shape": 0.25,
                    "frequency_chain": FIXTURE_CHAIN,
                    "branches": [
                        {
                            "branch_id": "one",
                            "definition": "The single synthetic event type",
                        }
                    ],
                    "components": [
                        {
                            "component_id": "victim",
                            "definition": "Unique victim resource loss",
                            "nature": "resource_cost",
                            "bearer": "third_party",
                            "rank_group_id": "severity",
                            "gpd_shape": 0.25,
                        },
                        {
                            "component_id": "fine",
                            "definition": "Separate fine paid by the lab",
                            "nature": "transfer",
                            "bearer": "provider",
                            "rank_group_id": "severity",
                        },
                    ],
                }
            ],
            "respondents": [
                {
                    "respondent_id": "fixture",
                    "model_id": "fixture-model",
                    "sample_index": 0,
                }
            ],
        }
    )


def contextual_answer() -> ContextualRiskAnswer:
    return ContextualRiskAnswer.model_validate(
        {
            "status": "available",
            "rate": {"p5": 1, "p25": 1, "p50": 1, "p75": 1, "p95": 1},
            "lab_compensation": {"status": "available", "share": 0},
            "lab_annual_business_loss": {
                "status": "available",
                "positive_probability": 1,
                "amount": 70,
            },
            "branches": [
                {
                    "branch_id": "one",
                    "probability": 1,
                    "components": [
                        {
                            "component_id": "victim",
                            "status": "available",
                            "amount": 100,
                        },
                        {
                            "component_id": "fine",
                            "status": "available",
                            "amount": 20,
                        },
                    ],
                }
            ],
            "direct_eal": _constant(100),
            "reasoning": {
                "median_reason": "Explicitly synthetic software fixture",
                "low_case": "Fixed for exact arithmetic",
                "high_case": "Fixed for exact arithmetic",
                "method": "direct_judgment",
                "evidence_used": [],
            },
        }
    )


def matched_chain_answer(
    node: ContextualRiskNode, answer: ContextualRiskAnswer, exposure: float
) -> ContextualChainAnswer:
    """A synthetic chain whose opportunity law matches the direct count law."""
    chain = node.frequency_chain
    assert chain is not None
    opportunity_rate = (
        _constant(1.0 / exposure)
        if answer.rate is None
        else {key: value / exposure for key, value in answer.rate.model_dump().items()}
    )
    return ContextualChainAnswer.model_validate(
        {
            "status": "available",
            "opportunity_rate": opportunity_rate,
            "stages": [
                {"stage_id": stage.stage_id, "probability": _constant(1)}
                for stage in chain.stages
            ],
            "reasoning": _REASONING,
        }
    )


def _share(
    answer: ContextualRiskAnswer | None,
) -> float | None:
    from catmodel.elicitation.contextual_risk import stored_large_loss_share

    if answer is None or answer.status != "available":
        return None
    return stored_large_loss_share(answer.rate, answer.large_loss_count)


def declined_crosscheck() -> EalCrosscheckAnswer:
    """A fixture respondent that keeps every answer in its EAL cross-check,
    recorded as declined."""
    return EalCrosscheckAnswer.model_validate(
        {
            "status": "unchanged",
            "unchanged_reason": "Synthetic fixture keeps its answers",
            "reasoning": _REASONING,
        }
    )


def crosscheck_answer(
    revised: dict[str, Any], *, judgments: tuple[str, ...] | None = None
) -> EalCrosscheckAnswer:
    """A synthetic cross-check revision: ``revised`` maps each revised field
    to its new value; ``judgments`` defaults to the judgments those fields
    belong to."""
    from catmodel.contracts.contextual_risk import EAL_CROSSCHECK_FIELDS

    named = judgments or tuple(
        judgment
        for judgment, fields in EAL_CROSSCHECK_FIELDS.items()
        if set(fields) & set(revised)
    )
    return EalCrosscheckAnswer.model_validate(
        {
            "status": "revised",
            "revised": [
                {"judgment_id": judgment, "explanation": "Synthetic revision"}
                for judgment in named
            ],
            **revised,
            "reasoning": _REASONING,
        }
    )


def with_crosscheck(
    instrument: ContextualRiskInstrument,
    case: ContextualRiskCase,
    reply: EalCrosscheckAnswer | None = None,
    retried: EalCrosscheckAnswer | None = None,
) -> ContextualRiskCase:
    """``case`` with its EAL cross-check as collection records it: a
    triggered case answers with ``reply`` (a decline by default), and a
    revised large-loss count that contradicts its loss quantiles gets its one
    coherence retry with ``retried`` as the retry's answer."""
    from catmodel.elicitation.eal_crosscheck import (
        answer_before_crosscheck,
        crosscheck_prompt,
        crosscheck_retry_prompt,
        revised_share,
    )
    from catmodel.setup.implied_eal import (
        crosscheck_reconciliation,
        recorded_crosscheck,
    )

    node = next(
        row for row in instrument.collectible_nodes if row.node_id == case.node_id
    )
    recorded = recorded_crosscheck(
        instrument, node, answer_before_crosscheck(case), case.anchor_score
    )
    question = (
        None
        if recorded is None
        else crosscheck_prompt(
            instrument,
            node,
            case.respondent,
            case.anchor_score,
            answer_before_crosscheck(case),
            recorded,
        )
    )
    if recorded is not None and question is not None:
        answer = declined_crosscheck() if reply is None else reply
        coherence = crosscheck_retry_prompt(
            question, answer, revised_share(question, answer)
        )
        recorded = recorded.model_copy(
            update={
                "reconciliation": crosscheck_reconciliation(
                    question,
                    answer,
                    None
                    if coherence is None
                    else (coherence.prompt_sha256(), coherence.contradiction, retried),
                )
            }
        )
    return case.model_copy(update={"eal_crosscheck": recorded})


def contextual_case(
    instrument: ContextualRiskInstrument,
    prompt: ContextualPrompt,
    answer: ContextualRiskAnswer,
    *,
    chain: ContextualChainAnswer | None = None,
    retried: ContextualRiskAnswer | None = None,
    guardrail_retried: ContextualChainAnswer | None = None,
    crosscheck: EalCrosscheckAnswer | None = None,
) -> ContextualRiskCase:
    """One valid fixture case with a chain, retries and direct EAL cross-check.

    A large-loss count that contradicts its loss quantiles gets its one
    coherence retry with ``retried`` as its answer. A chain guardrail finding
    gets its one retry with ``guardrail_retried`` as its answer. Either missing
    retry answer records an exhausted call. A triggered EAL cross-check uses
    ``crosscheck``, a decline by default.
    """
    from catmodel.contracts.contextual_risk import (
        DirectSeverityRetry,
    )
    from catmodel.elicitation.contextual_frequency import (
        ContextualChainPrompt,
        chain_count_law_for_answer,
        chain_guardrail_retry_prompt,
    )
    from catmodel.elicitation.contextual_risk import (
        direct_severity_retry_prompt,
        large_loss_contradiction,
    )

    share = _share(answer)
    coherence = direct_severity_retry_prompt(prompt, answer, share)
    severity_retry = None
    adopted = answer
    if coherence is not None:
        retry_share = _share(retried)
        resolved = (
            retried is not None
            and retried.status == "available"
            and large_loss_contradiction(
                prompt.node.total, retried.total_loss, retry_share
            )
            is None
        )
        severity_retry = DirectSeverityRetry(
            prompt_sha256=coherence.prompt_sha256(),
            contradiction=coherence.contradiction,
            answer=retried,
            large_loss_share=retry_share,
            resolved=resolved,
        )
        if resolved and retried is not None:
            from catmodel.elicitation.contextual_frequency import adopt_retry

            adopted = adopt_retry(answer, retried)
    chain_prompt = ContextualChainPrompt(
        instrument, prompt.node, prompt.respondent, prompt.anchor_score
    )
    exposure = chain_prompt.exposure.count
    chained = (
        matched_chain_answer(prompt.node, adopted, exposure) if chain is None else chain
    )
    guardrail_prompt = chain_guardrail_retry_prompt(chain_prompt, chained)
    chain_guardrail = None
    driving_chain = chained
    if guardrail_prompt is not None:
        resolved = (
            guardrail_retried is not None
            and guardrail_retried.status == "available"
            and chain_guardrail_retry_prompt(chain_prompt, guardrail_retried) is None
        )
        chain_guardrail = ChainGuardrailRetry(
            prompt_sha256=guardrail_prompt.prompt_sha256(),
            findings=guardrail_prompt.findings,
            answer=guardrail_retried,
            resolved=resolved,
        )
        if guardrail_retried is not None and guardrail_retried.status == "available":
            driving_chain = guardrail_retried
    elif guardrail_retried is not None:
        raise ValueError("a chain guardrail answer needs a triggering chain")
    chain_count = chain_count_law_for_answer(driving_chain, exposure)
    case = ContextualRiskCase(
        node_id=prompt.node.node_id,
        anchor_score=prompt.anchor_score,
        respondent=prompt.respondent,
        answer=answer,
        prompt_sha256=prompt.prompt_sha256(),
        chain_answer=chained,
        chain_prompt_sha256=chain_prompt.prompt_sha256(),
        chain_guardrail=chain_guardrail,
        chain_count=chain_count,
        large_loss_share=share,
        severity_retry=severity_retry,
    )
    return with_crosscheck(instrument, case, crosscheck)


def fixture_artifact(
    instrument: ContextualRiskInstrument,
    answer_for: Callable[[ContextualPrompt], ContextualRiskAnswer],
) -> ContextualRiskArtifact:
    """Capture answers with matched chains and an accepted retry if flagged."""
    from catmodel.elicitation.contextual_frequency import (
        ContextualChainPrompt,
        chain_guardrail_retry_prompt,
    )

    def case_for(prompt: ContextualPrompt) -> ContextualRiskCase:
        answer = answer_for(prompt)
        chain_prompt = ContextualChainPrompt(
            instrument, prompt.node, prompt.respondent, prompt.anchor_score
        )
        chain = matched_chain_answer(prompt.node, answer, chain_prompt.exposure.count)
        retry = (
            chain
            if chain_guardrail_retry_prompt(chain_prompt, chain) is not None
            else None
        )
        return contextual_case(
            instrument,
            prompt,
            answer,
            chain=chain,
            guardrail_retried=retry,
        )

    return ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(case_for(prompt) for prompt in contextual_prompts(instrument)),
    )


CROSSCHECK_MARKER = "(judgment_id direct_eal)"
"""Text only an EAL cross-check question contains."""

CHAIN_MARKER = "builds the expected annual number of qualified units from the bottom up"
"""Text only a frequency-chain question, and its guardrail retry, contains."""

_STAGE_LINE = re.compile(r"^- ([a-z][a-z0-9_]*): ", re.MULTILINE)
_BASE_COUNT = re.compile(
    r"the stipulated count is ([\d.,]+)(?: (million|billion|trillion))? base units"
)
_SCALE = {None: 1.0, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def chain_question_terms(user_prompt: str) -> tuple[list[str], float]:
    """A chain question's stage IDs, in order, and its stipulated base count,
    read from its text as a respondent reads them."""
    stages = user_prompt.split("Stages, in order (stage_id: event):\n", 1)[1]
    stages = stages.split("\n\n", 1)[0]
    match = _BASE_COUNT.search(user_prompt)
    assert match is not None
    count = float(match.group(1).replace(",", "")) * _SCALE[match.group(2)]
    return _STAGE_LINE.findall(stages), count


def fixture_response(user_prompt: str, answer: ContextualRiskAnswer) -> str:
    """A fake transport's reply: ``answer``, its matched chain for a chain
    question, or a decline for an EAL cross-check."""
    if CROSSCHECK_MARKER in user_prompt:
        return declined_crosscheck().model_dump_json()
    if CHAIN_MARKER not in user_prompt:
        return answer.model_dump_json()
    stages, exposure = chain_question_terms(user_prompt)
    opportunity_rate = (
        _constant(1.0 / exposure)
        if answer.rate is None
        else {key: value / exposure for key, value in answer.rate.model_dump().items()}
    )
    return ContextualChainAnswer.model_validate(
        {
            "status": "available",
            "opportunity_rate": opportunity_rate,
            "stages": [
                {"stage_id": stage, "probability": _constant(1)} for stage in stages
            ],
            "reasoning": _REASONING,
        }
    ).model_dump_json()


def contextual_artifact(
    instrument: ContextualRiskInstrument | None = None,
) -> ContextualRiskArtifact:
    instrument = instrument or contextual_instrument()
    return fixture_artifact(instrument, lambda _prompt: contextual_answer())


def scenario_node_answer(
    node: ContextualRiskNode, *, count: float = 12
) -> ContextualRiskAnswer:
    """A valid synthetic answer for any scenario-derived node with a shared
    total."""
    total = node.total
    assert total is not None
    partition_count = len([row for row in node.components if row.role == "partition"])
    answer = contextual_answer().model_dump(mode="json")
    answer["rate"] = dict.fromkeys(("p5", "p25", "p50", "p75", "p95"), count)
    # Near the EAL its other answers imply, so the cross-check agrees.
    answer["direct_eal"] = _constant(count * 2.7e6)
    if total.large_loss_level is not None:
        # Exact zero means no anchor: the stated quantiles govern the law.
        answer["large_loss_count"] = _constant(0)
        answer["very_large_loss_count"] = _constant(0)
    answer["total_loss"] = dict(
        zip(
            ("p5", "p25", "p50", "p75", "p95", "p99", "p999"),
            [total.floor + value for value in (2e6, 3e6, 4e6, 5e6, 6e6, 10e6, 20e6)],
            strict=True,
        )
    )
    answer["branches"] = [
        {
            "branch_id": node.branches[0].branch_id,
            "probability": 1,
            "components": [
                {
                    "component_id": row.component_id,
                    "status": "available",
                    **(
                        {"share": 1 / partition_count}
                        if row.role == "partition"
                        else {"amount": 0}
                    ),
                }
                for row in node.components
            ],
        }
    ]
    return ContextualRiskAnswer.model_validate(answer)


GOVERNED_SCENARIOS = ("compromise_critical_infrastructure", "output_reliance_legal")
"""A catastrophe-class node with the USD 100 trillion endpoint and an uncapped
liability node."""


def governed_contextual_instrument(
    scenario_ids: tuple[str, ...] = GOVERNED_SCENARIOS,
    model_ids: tuple[str, ...] = ("fixture-model",),
) -> ContextualRiskInstrument:
    """The synthetic lab instrument with repository scenario nodes and governance.

    Nodes come from the repository scenarios, so their governed tail shapes and
    caps match the authored severity governance; the respondents are synthetic.
    """
    from catmodel.elicitation.contextual_risk import contextual_node_from_scenario
    from catmodel.resources import scenarios_path
    from catmodel.scenarios.loader import load_scenarios

    scenarios = {row.id: row for row in load_scenarios(scenarios_path())}
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"] = [
        contextual_node_from_scenario(
            scenarios[scenario_id], "deployment", "public"
        ).model_dump(mode="json")
        for scenario_id in scenario_ids
    ]
    if len(scenario_ids) > 1:
        payload["independent_node_origins_rationale"] = (
            "Synthetic disjoint initiating incidents with unique costs for software tests."
        )
    payload["respondents"] = [
        {"respondent_id": f"fixture-{index}", "model_id": model, "sample_index": 0}
        for index, model in enumerate(model_ids)
    ]
    return ContextualRiskInstrument.model_validate(payload)


GOVERNED_DIRECT_EAL = 2e10
"""The governed fixture answer's direct EAL at unit scale where its first
component carries welfare: within the cross-check factor of the USD 17 to 22
billion its other answers imply for every repository scenario."""


def governed_contextual_answer(node, *, scale: float = 1.0) -> ContextualRiskAnswer:
    """A synthetic stochastic answer: uncertain rate, elicited total and lab
    quantiles, and exact-zero large-loss counts (no anchor)."""
    levels = node.total is not None and node.total.large_loss_level is not None
    return ContextualRiskAnswer.model_validate(
        {
            "status": "available",
            **(
                {
                    "large_loss_count": _constant(0),
                    "very_large_loss_count": _constant(0),
                }
                if levels
                else {}
            ),
            "rate": {
                key: value * scale
                for key, value in zip(
                    ("p5", "p25", "p50", "p75", "p95"),
                    (0.05, 0.1, 0.2, 0.4, 1.0),
                    strict=True,
                )
            },
            "total_loss": {
                "p5": 1e8,
                "p25": 1e9,
                "p50": 5e9,
                "p75": 2e10,
                "p95": 1e11,
                "p99": 1e12,
                "p999": 1e13,
            },
            "lab_compensation": {"status": "available", "share": 0},
            "lab_annual_business_loss": {
                "status": "available",
                "positive_probability": 0.5,
                "loss": {
                    "p5": 1e6,
                    "p25": 1e7,
                    "p50": 5e7,
                    "p75": 2e8,
                    "p95": 1e9,
                    "p99": 1e10,
                    "p999": 1e11,
                },
            },
            "branches": [
                {
                    "branch_id": "event",
                    "probability": 1,
                    "components": [
                        {
                            "component_id": component.component_id,
                            "status": "available",
                            "share": 1.0 if index == 0 else 0.0,
                        }
                        for index, component in enumerate(node.components)
                    ],
                }
            ],
            # The whole total falls on the first component, so the societal
            # harm is zero where that component is an ordinary transfer.
            "direct_eal": _constant(
                0
                if node.components and node.components[0].nature == "transfer"
                else GOVERNED_DIRECT_EAL * scale
            ),
            "reasoning": {
                "median_reason": "Explicitly synthetic software fixture",
                "low_case": "Synthetic",
                "high_case": "Synthetic",
                "method": "direct_judgment",
                "evidence_used": [],
            },
        }
    )


def _constant(value: float) -> dict[str, float]:
    return dict.fromkeys(("p5", "p25", "p50", "p75", "p95"), value)


_REASONING = {
    "median_reason": "Explicitly synthetic software fixture",
    "low_case": "Fixed for exact arithmetic",
    "high_case": "Fixed for exact arithmetic",
    "method": "direct_judgment",
    "evidence_used": [],
}


def chain_answer(
    opportunity: float = 8, probabilities: tuple[float, ...] = (0.5, 0.25)
) -> ContextualChainAnswer:
    """Constant synthetic chain quantiles, so the implied count is their
    product times the chain's exposure; ``opportunity`` is the opportunity
    rate, which is the opportunity count at the fixture chain's exposure of
    one unit."""
    return ContextualChainAnswer.model_validate(
        {
            "status": "available",
            "opportunity_rate": _constant(opportunity),
            "stages": [
                {"stage_id": stage["stage_id"], "probability": _constant(value)}
                for stage, value in zip(
                    FIXTURE_CHAIN["stages"], probabilities, strict=True
                )
            ],
            "reasoning": _REASONING,
        }
    )


def chain_artifact(
    instrument: ContextualRiskInstrument,
    *,
    direct: Callable[[ContextualPrompt], ContextualRiskAnswer] | None = None,
    chain: Callable[[ContextualPrompt], ContextualChainAnswer] | None = None,
    guardrail_retried: Callable[[ContextualPrompt], ContextualChainAnswer]
    | None = None,
) -> ContextualRiskArtifact:
    """A fixture capture with optional direct, chain and guardrail answers."""
    return ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(
            contextual_case(
                instrument,
                prompt,
                contextual_answer() if direct is None else direct(prompt),
                chain=chain_answer() if chain is None else chain(prompt),
                guardrail_retried=(
                    None if guardrail_retried is None else guardrail_retried(prompt)
                ),
            )
            for prompt in contextual_prompts(instrument)
        ),
    )


def hierarchy_instrument(scenario_ids: tuple[str, ...]) -> ContextualRiskInstrument:
    """Repository scenario nodes with every taxonomy parent over them."""
    from catmodel.elicitation.contextual_hierarchy import hierarchy_parents
    from catmodel.resources import scenarios_path
    from catmodel.scenarios.loader import load_scenarios

    scenarios = {row.id: row for row in load_scenarios(scenarios_path())}
    instrument = governed_contextual_instrument(scenario_ids)
    payload = instrument.model_dump(mode="json")
    payload["parents"] = [
        row.model_dump(mode="json")
        for row in hierarchy_parents(instrument.nodes, scenarios)
    ]
    return ContextualRiskInstrument.model_validate(payload)


def parent_answer(category: float, remainder: float | None) -> ParentAnswer:
    """Constant synthetic parent quantiles; no remainder for a parent with no
    modeled child."""
    return ParentAnswer.model_validate(
        {
            "status": "available",
            "category_eal": _constant(category),
            "remainder_eal": None if remainder is None else _constant(remainder),
            "reasoning": _REASONING,
        }
    )


def hierarchy_artifact(
    instrument: ContextualRiskInstrument,
    leaf_answer: Callable[[ContextualPrompt], ContextualRiskAnswer],
    parent_for: Callable[[str, float], ParentAnswer | None],
) -> ContextualRiskArtifact:
    """A fixture capture whose parent cases store their child EALs and pools
    as collection does; ``parent_for`` gives each parent's answer, None for
    an exhausted question."""
    from catmodel.elicitation.contextual_hierarchy import ContextualParentPrompt
    from catmodel.elicitation.eal_crosscheck import pooled_answer
    from catmodel.setup.contextual_hierarchy import (
        hierarchy_outcome,
        recorded_parents,
    )

    leaves = fixture_artifact(instrument, leaf_answer)
    parent_cases = []
    for score in instrument.anchor_scores:
        for respondent in instrument.respondents:
            finals = {
                case.node_id: pooled_answer(case)
                for case in leaves.cases
                if case.anchor_score == score and case.respondent == respondent
            }
            answers = {
                row.parent_id: parent_for(row.parent_id, score)
                for row in instrument.parents
            }
            stored = recorded_parents(
                hierarchy_outcome(instrument, finals, answers, score)
            )
            parent_cases.extend(
                ContextualParentCase(
                    parent_id=row.parent_id,
                    anchor_score=score,
                    respondent=respondent,
                    answer=answers[row.parent_id],
                    prompt_sha256=ContextualParentPrompt(
                        instrument, row, respondent, score
                    ).prompt_sha256(),
                    children_median_eal=stored[row.parent_id].children_median_eal,
                    children_mean_eal=stored[row.parent_id].children_mean_eal,
                    pool=stored[row.parent_id].pool,
                )
                for row in instrument.parents
            )
    return leaves.model_copy(update={"parent_cases": tuple(parent_cases)})


POOL_FIXTURE_COUNTS = {"leaf_a": 1, "leaf_b": 2, "leaf_c": 1}
"""Each leaf's annual count in the pool fixture. A leaf's welfare is USD 100
per incident, so the leaves' EALs are 100, 200 and 100."""


def leaves_instrument(
    leaf_ids: Sequence[str], models: Mapping[str, int] | None = None
) -> ContextualRiskInstrument:
    """The synthetic fixture node as each of ``leaf_ids``, each its own
    scenario, with no taxonomy parent. ``models`` gives each respondent
    model's number of samples, in place of the fixture's one respondent;
    a sample's respondent ID is its model and index joined by a hyphen."""
    import json

    payload = contextual_instrument().model_dump(mode="json")
    payload["independent_node_origins_rationale"] = (
        "Synthetic disjoint initiating incidents with unique costs for software tests."
    )
    (node,) = payload["nodes"]
    payload["nodes"] = []
    for node_id in leaf_ids:
        row = json.loads(json.dumps(node))
        row["node_id"] = node_id
        row["population"]["scenario_id"] = node_id
        payload["nodes"].append(row)
    if models is not None:
        payload["respondents"] = [
            {
                "respondent_id": f"{model}-{index}",
                "model_id": model,
                "sample_index": index,
            }
            for model, count in models.items()
            for index in range(count)
        ]
    return ContextualRiskInstrument.model_validate(payload)


def pool_instrument(
    models: Mapping[str, int] | None = None,
) -> ContextualRiskInstrument:
    """Three synthetic leaves under two hand-declared parents, for exact pool
    arithmetic: ``mid`` over ``leaf_a`` and ``leaf_b``, and the root ``top``
    over ``mid`` and ``leaf_c``, so their heights are one and two. Software
    fixture only: the parents follow no taxonomy. ``models`` are
    ``leaves_instrument``'s."""
    payload = leaves_instrument(tuple(POOL_FIXTURE_COUNTS), models).model_dump(
        mode="json"
    )
    parent = {"definition": "A synthetic category.", "threshold": "Synthetic."}
    payload["parents"] = [
        {
            **parent,
            "parent_id": "top",
            "level": "root",
            "label": "Every synthetic incident",
            "children": ["mid", "leaf_c"],
            "unmodeled_children": ["Other synthetic incidents"],
        },
        {
            **parent,
            "parent_id": "mid",
            "level": "branch",
            "label": "Middle synthetic category",
            "children": ["leaf_a", "leaf_b"],
            "unmodeled_children": [],
        },
    ]
    return ContextualRiskInstrument.model_validate(payload)


def counted_answer(count: float) -> ContextualRiskAnswer:
    """The fixture answer at an annual count, with a direct EAL equal to the
    EAL its count implies: a leaf's welfare is USD 100 per incident."""
    payload = contextual_answer().model_dump(mode="json")
    payload["rate"] = _constant(count)
    payload["direct_eal"] = _constant(100 * count)
    return ContextualRiskAnswer.model_validate(payload)


def pool_leaf_answer(prompt: ContextualPrompt) -> ContextualRiskAnswer:
    """The fixture answer at the leaf's count (``POOL_FIXTURE_COUNTS``), with
    a direct EAL equal to the EAL its count implies."""
    return counted_answer(POOL_FIXTURE_COUNTS[prompt.node.node_id])


def pool_artifact(
    instrument: ContextualRiskInstrument,
    mid: ParentAnswer | None,
    top: ParentAnswer | None,
) -> ContextualRiskArtifact:
    """The pool fixture's capture with the two parents' answers, the same at
    every coordinate. With ``parent_answer(1600, 100)`` for ``mid`` and
    ``parent_answer(8000, 100)`` for ``top`` each pool's factor is two."""
    answers = {"mid": mid, "top": top}
    return hierarchy_artifact(
        instrument, pool_leaf_answer, lambda parent_id, _score: answers[parent_id]
    )


def path_linked_payload(payload: dict[str, Any], *scale_ids: str) -> dict[str, Any]:
    """``payload`` with path-linked scales in place of its fixed ones.

    The synthetic chain's ``users`` and each of ``scale_ids`` (the scales the
    payload's scenario chains name) grow from 100 to 200, and
    ``use_population`` names each, so ``users`` alone renders ``About
    {users} synthetic users``.
    """
    import json

    linked = json.loads(json.dumps(payload))
    linked.pop("fixed_exposure", None)
    ids = ("users", *scale_ids)
    linked["deployment_binding"]["use_population"] = "About " + ", ".join(
        "{" + scale_id + "} synthetic " + scale_id for scale_id in ids
    )
    linked["path_linked_exposure"] = {
        "max_growth_multiple": 3,
        "scales": [
            {
                "scale_id": scale_id,
                "unit": f"synthetic {scale_id}",
                "as_of": 100,
                "study_year_median": 200,
            }
            for scale_id in ids
        ],
    }
    return linked


def with_access_bases(nodes: list[dict[str, Any]], access: str) -> None:
    """Give each repository scenario node in ``nodes`` its chain's exposure
    base for ``access`` (``public`` or ``internal``), in place, as a
    scenario-derived node of that access has."""
    from catmodel.resources import scenarios_path
    from catmodel.scenarios.loader import load_scenarios

    scenarios = {row.id: row for row in load_scenarios(scenarios_path())}
    for node in nodes:
        scenario = scenarios.get(node["population"]["scenario_id"])
        if scenario is not None and node.get("frequency_chain"):
            node["frequency_chain"]["exposure"] = scenario.event_chain.exposure.base(
                "public" if access == "public" else "internal"
            ).model_dump(mode="json")
