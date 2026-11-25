"""Every answer check that collection runs names its rule and locates its field.

A response that fails validation is retried with feedback built only from
each error's code and location (``validation_feedback``): a check raising a
plain ``ValueError`` gave the retry ``value_error`` with no location, so the
respondent could only guess what to change, and three such failures exhaust
the call. Each check below fails one field of an otherwise valid answer
through the parser collection uses; the stored feedback names the check's
code and the field, the retry carries it, and replay derives it again from
the raw response.
"""

import asyncio
import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from types import FunctionType
from typing import Any, cast

import pytest
from pydantic import BaseModel

from catmodel.collection.frozen_call_collection import collect_frozen_call
from catmodel.config import ExecutionSettings
from catmodel.contracts.contextual_risk import (
    ContextualChainAnswer,
    ContextualRiskAnswer,
    ContextualRiskInstrument,
    EalCrosscheckAnswer,
    ParentAnswer,
)
from catmodel.elicitation.contextual_frequency import (
    ContextualChainPrompt,
    parse_chain_answer,
)
from catmodel.elicitation.contextual_hierarchy import (
    ContextualParentPrompt,
    parse_parent_answer,
)
from catmodel.elicitation.contextual_risk import (
    contextual_prompts,
    parse_contextual_answer,
)
from catmodel.elicitation.eal_crosscheck import (
    case_crosscheck_prompt,
    parse_eal_crosscheck_answer,
)
from catmodel.frozen_calls import replay_call, validation_feedback
from catmodel.llm.transport import LLMProvider, ProviderResponse
from catmodel.setup.contextual_inputs import contextual_request
from test_support.contextual_risk import (
    chain_answer,
    contextual_answer,
    contextual_case,
    contextual_instrument,
    crosscheck_answer,
    governed_contextual_answer,
    governed_contextual_instrument,
    hierarchy_instrument,
    parent_answer,
)

LEAF = "compromise_organizations"
OWNER = "n_compromise"  # LEAF's own parent
CHILDLESS = "n_fraud"  # a parent with no modeled child
COUNT = dict.fromkeys(("p5", "p25", "p50", "p75", "p95"), 2.0)
LOSS = {
    "p5": 1e6,
    "p25": 2e6,
    "p50": 5e6,
    "p75": 1e7,
    "p95": 5e7,
    "p99": 1e8,
    "p999": 1e9,
}
BRANCH = contextual_answer().model_dump(mode="json")["branches"][0]
LAB = "lab_annual_business_loss"
_CONFIG = ExecutionSettings(
    samples_per_question=1,
    max_output_tokens_by_model={"fixture-model": 100},
    request_timeout_seconds=60,
    concurrency_by_model={"fixture-model": 1},
)


@dataclass(frozen=True)
class Question:
    """A question as collection asks it: its prompt, parser and response
    model, and a valid answer."""

    prompt: Any
    parser: Callable[..., BaseModel]
    model: type[BaseModel]
    valid: BaseModel


@cache
def asked(builder: FunctionType) -> Question:
    """The question ``builder`` builds, built once."""
    return builder()


def _fixture(change: Callable[[dict[str, Any]], None]) -> ContextualRiskInstrument:
    """The synthetic instrument with its one node changed."""
    payload = contextual_instrument().model_dump(mode="json")
    change(payload["nodes"][0])
    return ContextualRiskInstrument.model_validate(payload)


def _direct_question(instrument, answer) -> Question:
    prompt = next(
        row for row in contextual_prompts(instrument) if row.anchor_score == 0.5
    )
    return Question(prompt, parse_contextual_answer, ContextualRiskAnswer, answer)


def direct() -> Question:
    """No shared total, two independent components (only the first has a
    tail shape) and one joint branch."""
    return _direct_question(contextual_instrument(), contextual_answer())


def unshaped() -> Question:
    """The same node without a governed lab business-loss tail shape."""
    instrument = _fixture(lambda node: node.update(lab_loss_gpd_shape=None))
    return _direct_question(instrument, contextual_answer())


def two_branches() -> Question:
    instrument = _fixture(
        lambda node: node["branches"].append(
            {"branch_id": "two", "definition": "A second synthetic event type"}
        )
    )
    payload = contextual_answer().model_dump(mode="json")
    payload["branches"] = [
        {**BRANCH, "probability": 0.5},
        {**BRANCH, "branch_id": "two", "probability": 0.5},
    ]
    return _direct_question(instrument, ContextualRiskAnswer.model_validate(payload))


def governed() -> Question:
    """A repository node: an elicited shared total with a USD 100,000
    qualifying floor, large-loss levels and partition components."""
    instrument = governed_contextual_instrument(("compromise_critical_infrastructure",))
    node = instrument.nodes[0]
    return _direct_question(instrument, governed_contextual_answer(node))


def chain() -> Question:
    direct_prompt = asked(direct).prompt
    prompt = ContextualChainPrompt(
        direct_prompt.instrument,
        direct_prompt.node,
        direct_prompt.respondent,
        direct_prompt.anchor_score,
    )
    return Question(prompt, parse_chain_answer, ContextualChainAnswer, chain_answer())


def crosscheck() -> Question:
    """The synthetic node, without large-loss levels, whose direct EAL is far
    above the EAL its other answers imply."""
    instrument = contextual_instrument()
    payload = contextual_answer().model_dump(mode="json")
    payload["direct_eal"] = dict.fromkeys(COUNT, 1e6)
    answer = ContextualRiskAnswer.model_validate(payload)
    prompt = asked(direct).prompt
    question = case_crosscheck_prompt(
        instrument, contextual_case(instrument, prompt, answer)
    )
    assert question is not None
    return Question(
        question,
        parse_eal_crosscheck_answer,
        EalCrosscheckAnswer,
        crosscheck_answer({"rate": COUNT}),
    )


@cache
def _hierarchy() -> ContextualRiskInstrument:
    return hierarchy_instrument((LEAF,))


def _parent_question(parent_id: str, answer: ParentAnswer) -> Question:
    instrument = _hierarchy()
    parent = next(row for row in instrument.parents if row.parent_id == parent_id)
    prompt = ContextualParentPrompt(instrument, parent, instrument.respondents[0], 0.5)
    return Question(prompt, parse_parent_answer, ParentAnswer, answer)


def parent() -> Question:
    return _parent_question(OWNER, parent_answer(1e9, 1e6))


def childless_parent() -> Question:
    return _parent_question(CHILDLESS, parent_answer(1e9, None))


type Change = Callable[[dict[str, Any]], None]


def _at(payload: Any, path: tuple[str | int, ...]) -> Any:
    for key in path:
        payload = payload[key]
    return payload


def _set(*path: str | int, value: object) -> Change:
    def change(payload: dict[str, Any]) -> None:
        _at(payload, path[:-1])[path[-1]] = copy.deepcopy(value)

    return change


def _drop(*path: str | int) -> Change:
    def change(payload: dict[str, Any]) -> None:
        del _at(payload, path[:-1])[path[-1]]

    return change


def _repeat(name: str) -> Change:
    """Name the first entry of the list ``name`` twice."""

    def change(payload: dict[str, Any]) -> None:
        payload[name].append(copy.deepcopy(payload[name][0]))

    return change


def _all(*changes: Change) -> Change:
    def change(payload: dict[str, Any]) -> None:
        for item in changes:
            item(payload)

    return change


def _reverse_stages(payload: dict[str, Any]) -> None:
    payload["stages"].reverse()


COMPONENT = ("branches", 0, "components")
CASES: list[tuple[FunctionType, Change, str, list[str | int]]] = [
    # Direct answers: the answer's own rules.
    (
        direct,
        _set("branches", value=[{**BRANCH, "probability": 0.5}] * 2),
        "branch_ids_must_be_unique",
        ["branches", 1, "branch_id"],
    ),
    (
        direct,
        _set("branches", 0, "probability", value=0.5),
        "branch_probabilities_must_sum_to_one",
        ["branches"],
    ),
    (
        direct,
        _set(*COMPONENT, 1, "component_id", value="victim"),
        "component_ids_must_be_unique_within_a_branch",
        [*COMPONENT, 1, "component_id"],
    ),
    (
        direct,
        _set(*COMPONENT, 0, "amount", value=1e25),
        "amount_outside_admitted_monetary_domain",
        [*COMPONENT, 0, "amount"],
    ),
    (
        direct,
        _set(LAB, "positive_probability", value=0),
        "amount_not_allowed_when_positive_probability_is_zero",
        [LAB, "amount"],
    ),
    (
        direct,
        _set(LAB, "amount", value=None),
        "exactly_one_of_amount_loss_required_when_positive_probability_is_positive",
        [LAB],
    ),
    (
        direct,
        _set(LAB, "amount", value=0),
        "fixed_lab_business_loss_amount_must_be_positive",
        [LAB, "amount"],
    ),
    (
        direct,
        _all(
            _set(LAB, "amount", value=None), _set(LAB, "loss", value={**LOSS, "p5": 0})
        ),
        "lab_business_loss_quantiles_must_be_positive",
        [LAB, "loss", "p5"],
    ),
    (
        direct,
        _set(LAB, "amount", value=1e25),
        "amount_outside_admitted_monetary_domain",
        [LAB, "amount"],
    ),
    # Direct answers against their node.
    (
        direct,
        _drop("lab_compensation"),
        "lab_compensation_required_when_available",
        ["lab_compensation"],
    ),
    (direct, _drop(LAB), "lab_annual_business_loss_required_when_available", [LAB]),
    (
        unshaped,
        _all(_set(LAB, "amount", value=None), _set(LAB, "loss", value=LOSS)),
        "lab_business_loss_quantiles_require_governed_tail_shape",
        [LAB, "loss"],
    ),
    (
        direct,
        _set("total_loss", value=LOSS),
        "total_loss_not_asked_for_this_node",
        ["total_loss"],
    ),
    (
        governed,
        _drop("total_loss"),
        "total_loss_required_for_this_node",
        ["total_loss"],
    ),
    (
        governed,
        _set("total_loss", "p5", value=5e4),
        "total_loss_must_preserve_qualifying_floor",
        ["total_loss", "p5"],
    ),
    (
        direct,
        _set("branches", 0, "branch_id", value="two"),
        "branch_id_not_in_authored_roster",
        ["branches", 0, "branch_id"],
    ),
    (
        two_branches,
        _set("branches", value=[BRANCH]),
        "branches_must_cover_every_authored_branch",
        ["branches"],
    ),
    (
        direct,
        _set(*COMPONENT, 1, "component_id", value="penalty"),
        "component_id_not_in_authored_roster",
        [*COMPONENT, 1, "component_id"],
    ),
    (
        direct,
        _drop(*COMPONENT, 1),
        "components_must_cover_every_authored_component",
        [*COMPONENT],
    ),
    (
        direct,
        _all(
            _set(*COMPONENT, 1, "amount", value=None),
            _set(*COMPONENT, 1, "share", value=0.5),
        ),
        "share_not_allowed_for_independent_component",
        [*COMPONENT, 1, "share"],
    ),
    (
        governed,
        _all(
            _set(*COMPONENT, 0, "share", value=None),
            _set(*COMPONENT, 0, "amount", value=1e6),
        ),
        "share_required_for_partition_component",
        [*COMPONENT, 0, "share"],
    ),
    (
        direct,
        _all(
            _set(*COMPONENT, 1, "amount", value=None),
            _set(*COMPONENT, 1, "loss", value=LOSS),
        ),
        "component_loss_requires_authored_tail_shape",
        [*COMPONENT, 1, "loss"],
    ),
    (
        governed,
        _set(*COMPONENT, 0, "share", value=0.5),
        "partition_shares_must_sum_to_one",
        [*COMPONENT],
    ),
    # Chain answers.
    (
        chain,
        _set("stages", 0, "probability", "p95", value=1.5),
        "probability_quantiles_cannot_exceed_one",
        ["stages", 0, "probability", "p95"],
    ),
    (
        chain,
        _reverse_stages,
        "stage_must_follow_authored_order",
        ["stages", 0, "stage_id"],
    ),
    (
        chain,
        _drop("stages", 1),
        "stages_must_cover_every_authored_stage",
        ["stages"],
    ),
    # EAL cross-check answers.
    (
        crosscheck,
        _repeat("revised"),
        "revised_judgments_must_be_unique",
        ["revised", 1, "judgment_id"],
    ),
    (
        crosscheck,
        _set("revised", 0, "judgment_id", value="frequency"),
        "unknown_revised_judgment",
        ["revised", 0, "judgment_id"],
    ),
    (crosscheck, _drop("rate"), "rate_required_when_count_revised", ["rate"]),
    (
        crosscheck,
        _all(_drop("rate"), _set("revised", 0, "judgment_id", value="severity")),
        "total_loss_or_branches_required_when_severity_revised",
        [],
    ),
    (
        crosscheck,
        _all(
            _drop("rate"),
            _set("revised", 0, "judgment_id", value="large_loss_count"),
            _set("very_large_loss_count", value=COUNT),
        ),
        "large_loss_count_required_when_large_loss_count_revised",
        ["large_loss_count"],
    ),
    (
        crosscheck,
        _set("direct_eal", value=COUNT),
        "direct_eal_not_allowed_when_direct_eal_not_revised",
        ["direct_eal"],
    ),
    (
        crosscheck,
        _all(
            _drop("rate"),
            _set("revised", 0, "judgment_id", value="large_loss_count"),
            _set("large_loss_count", value=COUNT),
        ),
        "revised_judgment_not_asked_for_this_node",
        ["revised", 0, "judgment_id"],
    ),
    # Taxonomy parents.
    (
        parent,
        _set("remainder_eal", value=None),
        "remainder_eal_required_with_modeled_children",
        ["remainder_eal"],
    ),
    (
        childless_parent,
        _set("remainder_eal", value=COUNT),
        "remainder_eal_not_allowed_without_modeled_children",
        ["remainder_eal"],
    ),
]


class _Scripted:
    """Returns each scripted response text in turn."""

    provider_name = "fixture"
    model_id = "fixture-model"
    reasoning_effort = "low"

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)

    async def complete(self, **_kwargs) -> ProviderResponse:
        return ProviderResponse(text=self.texts.pop(0), model_id=self.model_id)


@pytest.mark.parametrize(
    ("question", "change", "code", "location"),
    CASES,
    ids=[f"{question.__name__}-{code}" for question, _, code, _ in CASES],
)
def test_a_failed_check_names_its_rule_and_field_and_replays(
    tmp_path, question, change, code, location
):
    current = asked(question)
    invalid = current.valid.model_dump(mode="json")
    change(invalid)
    request = contextual_request(current.prompt)
    parse = partial(current.parser, prompt=current.prompt)
    answer = asyncio.run(
        collect_frozen_call(
            run_directory=tmp_path,
            request=request,
            provider=cast(
                LLMProvider,
                _Scripted(json.dumps(invalid), current.valid.model_dump_json()),
            ),
            config=_CONFIG,
            parse_answer=parse,
            response_model=current.model,
        )
    )
    assert answer == current.valid
    first, second = (
        json.loads(path.read_bytes())
        for path in sorted(tmp_path.glob("calls/*/attempt-*.json"))
    )
    assert json.loads(first["error"]) == {
        "schema_version": 1,
        "validation_errors": [{"type": code, "location": location}],
    }
    # The retry carries the feedback, and replay derives it again from the
    # raw response.
    assert first["error"] in second["rendered_user_prompt"]
    files = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*.json")
    }
    replayed = replay_call(request=request, files=files, parse_answer=parse)
    assert replayed.answer == answer and replayed.responses == 2
    # The location-free feedback of a plain ValueError does not replay.
    path = next(name for name in files if name.endswith("attempt-001.json"))
    generic = {**first, "error": validation_feedback(ValueError(first["error"]))}
    with pytest.raises(ValueError, match="stored validation feedback differs"):
        replay_call(
            request=request,
            files={**files, path: json.dumps(generic).encode()},
            parse_answer=parse,
        )
