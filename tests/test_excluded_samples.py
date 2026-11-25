"""A call that exhausts its validation attempts never stops its context.

It is recorded with answer None, replayed and disclosed. A sample whose leaf's
direct question is exhausted is excluded at import, its weight kept within its
model, inside the import guard (``POOL-EXCLUDED-SAMPLES``), unless its model
does not answer that leaf over the study (``POOL-UNANSWERED-SCENARIOS``;
tests/test_unanswered_scenarios.py). A direct answer that declares the lab's
loss unavailable excludes no sample and changes no weight: its sample is left
out of the views that count the lab's loss alone, the lab-loss views and the
gross views, where the excluded and left-out samples stay inside the guard
together, and otherwise those views are withheld.
"""

import asyncio
import json
import math
import statistics
from typing import Any, cast

import pytest

from catmodel.collection import contextual_collection
from catmodel.collection.contextual_collection import (
    collect_contextual_risk,
    collect_study,
    run_study_collection,
)
from catmodel.config import ExecutionSettings, ModelProfile
from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskInstrument,
)
from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.incident import counts_lab_loss
from catmodel.contracts.run_setup import RunSetup
from catmodel.elicitation.contextual_frequency import (
    direct_answer,
)
from catmodel.elicitation.eal_crosscheck import (
    answer_before_crosscheck,
    eal_crosscheck_status,
    eal_crosscheck_summary,
    final_answer,
)
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.frozen_calls import call_limit_usage, exhaustion_causes
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import LLMProvider, ProviderResponse
from catmodel.results.execution import load_run_capture, run_setup
from catmodel.results.results_file import load_results
from catmodel.results.results_summary import write_results_summary
from catmodel.setup.contextual_hierarchy import hierarchy_outcomes
from catmodel.setup.contextual_inputs import (
    read_contextual_capture_files,
    validate_contextual_capture,
)
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.excluded_samples import _calls as _exhausted_calls_by_cause
from catmodel.setup.excluded_samples import pooled_weights
from catmodel.setup.public_frontier import (
    PublicFrontierRiskInput,
    public_frontier_setup_fields,
    public_state_node,
)
from catmodel.setup.public_study import (
    import_public_frontier_study,
    write_public_frontier_study,
)
from catmodel.simulation.incident_runtime import state_key
from test_support.capability_sources import write_source_bundle
from test_support.collection_trace import (
    StudyResponder,
    TriggerRates,
    question_identity,
    request_kind,
)
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    fixture_response,
    hierarchy_instrument,
    leaves_instrument,
)
from test_support.public_frontier import synthetic_forecast
from test_support.results import available_metric, present, reduced_arm

SCENARIOS = ("output_reliance_legal", "fraud_organizations")
MODELS = ("fixture-a", "fixture-b")
ROSTER = [
    {"respondent_id": f"{model}-{index}", "model_id": model, "sample_index": index}
    for model in MODELS
    for index in (0, 1)
]
EVERY_TURN = TriggerRates(
    direct_retry=1.0,
    chain_guardrail_retry=1.0,
    crosscheck=1.0,
    crosscheck_retry=1.0,
    parent_gap=1.0,
)
CONFIG = ExecutionSettings(
    samples_per_question=1,
    max_output_tokens_by_model=dict.fromkeys(MODELS, 100),
    request_timeout_seconds=7200,
    concurrency_by_model=dict.fromkeys(MODELS, 8),
)
A_1 = ("fixture-a-1", "fixture-a", 1)
NOT_AN_ANSWER = "not an answer"


def _identified(kwargs) -> dict:
    """A frozen call's request with its question's kind, context, subject and
    coordinate under ``question``: a question's text, as a respondent reads
    it, names neither its context nor, on a public trajectory, a coordinate."""
    return {
        **kwargs["request"],
        "question": question_identity(kwargs["parse_answer"].keywords["prompt"]),
    }


def _context(request) -> str:
    return request["question"][1]


def _subject(request) -> str:
    """The scenario or taxonomy category a request asks about."""
    return request["question"][2]


def _anchor(request) -> float:
    return request["question"][3]


def _respondent(request) -> tuple[str, str, int]:
    row = request["respondent"]
    return row["respondent_id"], row["model_id"], row["sample_index"]


class _Immediate:
    """A fake provider that answers at once through ``responder``."""

    provider_name = "openai"
    reasoning_effort = "max"

    def __init__(self, model_id, responder) -> None:
        self.model_id = model_id
        self.responder = responder

    async def complete(self, *, system_prompt, user_prompt, **_options):
        return ProviderResponse(
            text=self.responder(system_prompt, user_prompt),
            model_id=self.model_id,
            provider_name=self.provider_name,
            latency_ms=0,
            input_tokens=None,
            output_tokens=None,
        )


class _Garbled:
    """A provider whose every response fails validation."""

    def __init__(self, inner) -> None:
        self._inner = inner

    model_id = property(lambda self: self._inner.model_id)
    provider_name = property(lambda self: self._inner.provider_name)
    reasoning_effort = property(lambda self: self._inner.reasoning_effort)

    async def complete(self, **request) -> ProviderResponse:
        response = await self._inner.complete(**request)
        return response.model_copy(update={"text": NOT_AN_ANSWER})


def _sending(monkeypatch, when, respond=None) -> list[dict]:
    """Garble the responses to each request ``when`` selects; record every
    request. ``respond``, given a request, replaces its provider's
    responder."""
    original = contextual_collection.collect_frozen_call
    sent: list[dict] = []

    async def patched(**kwargs):
        request = _identified(kwargs)
        sent.append(request)
        if respond is not None:
            kwargs["provider"] = _Immediate(
                kwargs["provider"].model_id, respond(request)
            )
        if when(request):
            kwargs["provider"] = _Garbled(kwargs["provider"])
        return await original(**kwargs)

    monkeypatch.setattr(contextual_collection, "collect_frozen_call", patched)
    return sent


def _instrument() -> ContextualRiskInstrument:
    payload = hierarchy_instrument(SCENARIOS).model_dump(mode="json")
    payload["respondents"] = ROSTER
    return ContextualRiskInstrument.model_validate(payload)


INSTRUMENT = _instrument()


def _providers(instrument=INSTRUMENT) -> list[LLMProvider]:
    responder = StudyResponder([instrument], EVERY_TURN)
    return cast(list[LLMProvider], [_Immediate(model, responder) for model in MODELS])


def _collect(root, instrument=INSTRUMENT) -> ContextualRiskArtifact:
    (result,) = asyncio.run(
        collect_study(
            [(instrument, root)], providers=_providers(instrument), config=CONFIG
        )
    )
    assert isinstance(result, ContextualRiskArtifact)
    validate_contextual_capture(instrument, result, read_contextual_capture_files(root))
    return result


def _exhibit(setup, name: str):
    """A setup's exhibit as results report it, in JSON types."""
    record = next(
        row.acquisition[name]
        for row in setup.evidence.records
        if row.acquisition and name in row.acquisition
    )
    return json.loads(json.dumps(record))


def _calls(artifact):
    return [
        (row.kind, row.subject_id, row.anchor_score, row.respondent)
        for row in artifact.exhausted_calls()
    ]


def _group(rows, anchor, respondent):
    return [
        row
        for row in rows
        if row.anchor_score == anchor and row.respondent.key() == respondent
    ]


def test_an_exhausted_direct_question_leaves_its_sample_unanswered(
    tmp_path, monkeypatch
):
    sent = _sending(
        monkeypatch,
        when=lambda request: (
            _respondent(request) == A_1
            and request_kind(request["user_prompt"]) == "direct"
            and _subject(request) == "fraud_organizations"
            and _anchor(request) == 0.5
        ),
    )
    artifact = _collect(tmp_path)
    # The context completed; its one exhausted call is recorded, not raised.
    assert _calls(artifact) == [("direct", "fraud_organizations", 0.5, A_1)]
    (unanswered,) = artifact.unanswered_cases
    assert unanswered.key() == ("fraud_organizations", 0.5, A_1)
    assert unanswered.chain_answer is not None
    assert artifact.incomplete_groups() == {(0.5, A_1)}
    # The group's other leaf completes; its parents are answered but never
    # pooled, while every complete group's are.
    assert [case.node_id for case in _group(artifact.cases, 0.5, A_1)] == [
        "output_reliance_legal"
    ]
    parents = _group(artifact.parent_cases, 0.5, A_1)
    assert len(parents) == len(INSTRUMENT.parents)
    assert all(
        case.answer is not None and case.children_mean_eal is None and case.pool is None
        for case in parents
    )
    pooled = {
        (case.anchor_score, case.respondent.key())
        for case in artifact.parent_cases
        if case.pool is not None
    }
    assert (0.5, A_1) not in pooled
    assert len(pooled) == len(INSTRUMENT.anchor_scores) * len(ROSTER) - 1
    assert set(hierarchy_outcomes(INSTRUMENT, artifact)) == pooled
    # The exhausted question was one call of three attempts, sent once.
    exhausted = [
        request
        for request in sent
        if _respondent(request) == A_1
        and request_kind(request["user_prompt"]) == "direct"
        and _subject(request) == "fraud_organizations"
        and _anchor(request) == 0.5
    ]
    assert len(exhausted) == 1
    # A resumed collection replays the capture and sends nothing more.
    stored = sorted(tmp_path.glob("calls/*/attempt-*.json"))
    resumed = asyncio.run(
        collect_contextual_risk(
            instrument=INSTRUMENT,
            run_directory=tmp_path,
            providers=_providers(),
            config=CONFIG,
        )
    )
    assert resumed == artifact
    assert sorted(tmp_path.glob("calls/*/attempt-*.json")) == stored
    # One of the model's two samples lacks the leaf, a half. With the control
    # at a half that is not above it, so the fixed-anchor import excludes the
    # respondent at every coordinate and gives its weight to the model's
    # other sample; at the default control the model does not answer the
    # leaf instead (tests/test_unanswered_scenarios.py).
    files = read_contextual_capture_files(tmp_path)
    setup = compile_contextual_setup(
        INSTRUMENT,
        artifact,
        setup_id="excluded",
        max_excluded_sample_share=0.5,
        files=files,
    )
    weights = {
        state.state_id: state.weight for state in setup.joint_states if state.weight
    }
    assert len(weights) == 3
    assert sorted(weights.values()) == pytest.approx([0.25, 0.25, 0.5])
    record = _exhibit(setup, "excluded_samples")
    assert record["excluded_samples"] == 1 and record["samples"] == 4
    assert record["excluded"] == [
        {
            "node": INSTRUMENT.context_id,
            "respondent": list(A_1),
            "unanswered": [
                {
                    "context_id": INSTRUMENT.context_id,
                    "node_id": "fraud_organizations",
                }
            ],
        }
    ]
    assert record["renormalized_weights"] == {
        INSTRUMENT.context_id: {
            "fixture-a": {
                "samples": 2,
                "kept": 1,
                "model_weight": 0.5,
                "weight_per_sample": 0.25,
                "weight_per_kept_sample": 0.5,
            }
        }
    }
    assert record["exhausted_calls"]["by_subject"] == {
        "fraud_organizations": {
            "direct": {"calls": 1, "attempts": 3, "by_cause": {"validation": 1}}
        }
    }
    assert record["by_model"]["fixture-a"] == {
        "samples": 2,
        "excluded": 1,
        "excluded_share": 0.5,
    }
    assert record["excluded_share"] == record["excluded_share_weighted"] == 0.25
    assert any(
        text.startswith("Excluded samples (POOL-EXCLUDED-SAMPLES): 1 of 4 samples")
        and "refusals are content-linked" in text
        for text in setup.limitations
    )
    # The exhibits describe only the kept respondents' cases: three of four
    # respondents, two leaves, three anchors.
    frequency = _exhibit(setup, "frequency_estimate")
    assert frequency["frequency_collection"]["cases"] == 18
    assert frequency["eal_crosscheck"]["cases"] == 18
    assert all(len(row["case_results"]) == 9 for row in frequency["hierarchy"].values())


def _exhausted_attempts(root, kind: str) -> list:
    rows = []
    for call in sorted((root / "calls").iterdir()):
        request = json.loads((call / "request.json").read_text())
        attempts = sorted(call.glob("attempt-*.json"))
        if request_kind(request["user_prompt"]) == kind and all(
            json.loads(path.read_text())["raw_response"] == NOT_AN_ANSWER
            for path in attempts
        ):
            rows.append(attempts)
    return rows


def test_replay_refuses_an_inconsistent_exhausted_record(tmp_path, monkeypatch):
    _sending(
        monkeypatch,
        when=lambda request: (
            _respondent(request) == A_1
            and request_kind(request["user_prompt"]) == "direct"
            and _subject(request) == "fraud_organizations"
            and _anchor(request) == 0.5
        ),
    )
    artifact = _collect(tmp_path)
    (attempts,) = _exhausted_attempts(tmp_path, "direct")
    assert len(attempts) == 3

    def refused(capture: ContextualRiskArtifact, files=None) -> None:
        with pytest.raises(ValueError):
            validate_contextual_capture(
                INSTRUMENT, capture, files or read_contextual_capture_files(tmp_path)
            )

    # An answered case recorded as unanswered: its direct question answered.
    answered = artifact.cases[0]
    claimed = artifact.model_copy(
        update={
            "cases": artifact.cases[1:],
            "unanswered_cases": (
                *artifact.unanswered_cases,
                artifact.unanswered_cases[0].model_copy(
                    update={
                        "node_id": answered.node_id,
                        "anchor_score": answered.anchor_score,
                        "respondent": answered.respondent,
                        "prompt_sha256": answered.prompt_sha256,
                        "chain_answer": answered.chain_answer,
                        "chain_prompt_sha256": answered.chain_prompt_sha256,
                    }
                ),
            ),
        }
    )
    refused(ContextualRiskArtifact.model_validate(claimed.model_dump(mode="json")))
    files = dict(read_contextual_capture_files(tmp_path))
    root = tmp_path.as_posix() + "/"
    names = [path.as_posix().removeprefix(root) for path in attempts]
    # An unfinished call is not exhausted.
    refused(artifact, {name: data for name, data in files.items() if name != names[-1]})
    # Nor is one whose last attempt failed at the provider.
    row = json.loads(files[names[-1]])
    row.update(
        raw_response=None,
        error="transport unavailable",
        diagnostic_error="transport unavailable",
        response_model_id=None,
    )
    refused(
        artifact,
        {
            **files,
            names[-1]: (json.dumps(row, sort_keys=True, indent=2) + "\n").encode(),
        },
    )


def test_every_other_exhausted_leaf_call_leaves_the_answers_before_it(
    tmp_path, monkeypatch
):
    b_0 = ("fixture-b-0", "fixture-b", 0)
    garbled = {
        ("chain", "output_reliance_legal", 0.0, A_1),
        ("direct_retry", "output_reliance_legal", 0.5, b_0),
        ("chain_guardrail_retry", "fraud_organizations", 0.0, b_0),
        ("crosscheck", "fraud_organizations", 2.0, A_1),
        ("crosscheck_retry", "output_reliance_legal", 2.0, b_0),
    }
    _sending(
        monkeypatch,
        when=lambda request: (
            (
                request_kind(request["user_prompt"]),
                _subject(request),
                _anchor(request),
                _respondent(request),
            )
            in garbled
        ),
    )
    artifact = _collect(tmp_path)
    assert not artifact.unanswered_cases
    assert {
        (row.kind, row.subject_id, row.anchor_score, row.respondent)
        for row in artifact.exhausted_calls()
    } == {
        ("chain", "output_reliance_legal", 0.0, A_1),
        ("direct_severity_retry", "output_reliance_legal", 0.5, b_0),
        (
            "chain_guardrail_retry",
            "fraud_organizations",
            0.0,
            b_0,
        ),
        ("eal_crosscheck", "fraud_organizations", 2.0, A_1),
        ("eal_crosscheck_severity_retry", "output_reliance_legal", 2.0, b_0),
    }
    cases = {case.key(): case for case in artifact.cases}
    # An exhausted chain question leaves no chain law, so the direct count
    # supplies the pool.
    chainless = cases["output_reliance_legal", 0.0, A_1]
    assert chainless.chain_answer is None and chainless.chain_count is None
    assert final_answer(chainless).rate == direct_answer(chainless).rate
    # An exhausted coherence retry leaves its answer standing, unresolved.
    retried = cases["output_reliance_legal", 0.5, b_0]
    assert retried.severity_retry is not None
    assert retried.severity_retry.answer is None
    assert not retried.severity_retry.resolved
    # An exhausted cross-check keeps every answer.
    unchecked = cases["fraud_organizations", 2.0, A_1]
    check = unchecked.eal_crosscheck
    assert check is not None and check.reconciliation is not None
    assert check.reconciliation.answer is None
    assert eal_crosscheck_status(unchecked) == "exhausted"
    assert final_answer(unchecked) == answer_before_crosscheck(unchecked)
    summary = eal_crosscheck_summary(
        (
            (
                artifact,
                {
                    node.node_id: node.population.scenario_id
                    for node in INSTRUMENT.collectible_nodes
                },
                {},
            ),
        )
    )
    assert summary["eal_crosscheck_exhausted"] == 1


def test_exhausted_parent_calls_leave_the_leaves_as_they_are(tmp_path, monkeypatch):
    b_0 = ("fixture-b-0", "fixture-b", 0)

    def garbled(request) -> bool:
        return (
            request_kind(request["user_prompt"]) == "parent"
            and _anchor(request) == 0.5
            and (
                _respondent(request) == A_1
                or (_respondent(request) == b_0 and _subject(request) == "n_fraud")
            )
        )

    _sending(monkeypatch, when=garbled)
    artifact = _collect(tmp_path)
    calls = artifact.exhausted_calls()
    assert {row.kind for row in calls} == {"parent"}
    assert {row.subject_id for row in calls if row.respondent == A_1} == {
        row.parent_id for row in INSTRUMENT.parents
    }
    assert [row for row in _calls(artifact) if row[3] == b_0] == [
        ("parent", "n_fraud", 0.5, b_0)
    ]
    outcomes = hierarchy_outcomes(INSTRUMENT, artifact)
    # With no parent answered, nothing is pooled and no leaf moves.
    unanswered = outcomes[0.5, A_1]
    assert {pool.status for pool in unanswered.pools.values()} == {"parent_exhausted"}
    assert all(factors.frequency == 1 for factors in unanswered.factors.values())
    # One exhausted parent is not pooled, and neither is any parent above it,
    # which cannot add up a child without a value; the leaf beneath it keeps
    # its own count, while a parent elsewhere in the tree is pooled as usual.
    partial = outcomes[0.5, b_0]
    assert partial.pools["n_fraud"].status == "parent_exhausted"
    assert {
        partial.pools[parent].status
        for parent in ("n_misuse_assisted", "n_misuse", "omega")
    } == {"comparison_unavailable"}
    assert partial.factors["fraud_organizations"].frequency == 1
    reliance, failure = (
        partial.pools[parent] for parent in ("n_failure_output_reliance", "n_failure")
    )
    assert reliance.status == failure.status == "pooled" and reliance.factor > 1.5
    assert partial.factors["output_reliance_legal"].frequency == pytest.approx(
        reliance.factor * failure.factor
    )
    # Elsewhere every parent with a modeled child is pooled.
    complete = outcomes[0.0, b_0]
    assert complete.pools["n_fraud"].status == "pooled"
    assert complete.factors["fraud_organizations"].frequency > 1.5
    setup = compile_contextual_setup(
        INSTRUMENT,
        artifact,
        setup_id="parents",
        files=read_contextual_capture_files(tmp_path),
    )
    assert len([state for state in setup.joint_states if state.weight]) == 4
    frequency = _exhibit(setup, "frequency_estimate")
    statuses = frequency["hierarchy"]["n_fraud"]["cases_by_status"]
    assert statuses["parent_exhausted"] == 2
    assert statuses["pooled"] == len(INSTRUMENT.anchor_scores) * len(ROSTER) - 2


def test_a_provider_failure_is_never_an_exhausted_call(tmp_path):
    responder = StudyResponder([INSTRUMENT], EVERY_TURN)

    def failing(system_prompt, user_prompt):
        if (
            request_kind(user_prompt) == "direct"
            and "\n## Scenario fraud_organizations\n" in user_prompt
        ):
            raise ConnectionError("provider outage")
        return responder(system_prompt, user_prompt)

    down = cast(
        list[LLMProvider],
        [_Immediate("fixture-a", failing), _Immediate("fixture-b", responder)],
    )
    # A bound on real time: a failure mistaken for an exhausted call, or a
    # group waiting on its failed call, would leave requests queued forever.
    (failed,) = asyncio.run(
        asyncio.wait_for(
            collect_study([(INSTRUMENT, tmp_path)], providers=down, config=CONFIG),
            120,
        )
    )
    assert isinstance(failed, ValueError)
    assert "failed at the provider" in str(failed)
    assert not (tmp_path / "contextual-risk.json").exists()
    # The failed calls are sent again on resume, never recorded as exhausted.
    (result,) = asyncio.run(
        collect_study([(INSTRUMENT, tmp_path)], providers=_providers(), config=CONFIG)
    )
    assert isinstance(result, ContextualRiskArtifact)
    assert not result.unanswered_cases


def test_a_model_without_samples_at_a_coordinate_is_refused(tmp_path, monkeypatch):
    _sending(
        monkeypatch,
        when=lambda request: (
            _respondent(request)[1] == "fixture-a"
            and request_kind(request["user_prompt"]) == "direct"
            and _subject(request) == "fraud_organizations"
            and _anchor(request) == 0.0
        ),
    )
    artifact = _collect(tmp_path)
    assert len(artifact.unanswered_cases) == 2
    with pytest.raises(
        ValueError, match=r"model fixture-a keeps 0 of its 2 samples at node .*fewer"
    ):
        compile_contextual_setup(
            INSTRUMENT,
            artifact,
            setup_id="refused",
            max_excluded_sample_share=1.0,
            files=read_contextual_capture_files(tmp_path),
        )


# Public study: import, guard, results

PUBLIC_ROSTER = [
    {"respondent_id": f"{model}-{index}", "model_id": model, "sample_index": index}
    for model in MODELS
    for index in (0, 1, 2)
]
"""Three samples per model, so a model may lose one at a node."""
A_0, A_2 = ("fixture-a-0", "fixture-a", 0), ("fixture-a-2", "fixture-a", 2)


def _public_study(
    tmp_path, monkeypatch, when, respond=None, *, instrument=None, roster=None
):
    """A collected public study of three interior nodes and the no-progress
    node, with ``when``'s requests garbled. It asks ``instrument``'s leaves,
    the fixture's one by default, of ``roster``, ``PUBLIC_ROSTER`` by
    default; ``respond``, given a request, replaces its provider's
    responder."""
    roster = PUBLIC_ROSTER if roster is None else roster
    models = tuple(dict.fromkeys(row["model_id"] for row in roster))
    bundle = write_source_bundle(tmp_path / "source")
    payload = (instrument or contextual_instrument()).model_dump(mode="json")
    payload["source_bundle_manifest_sha256"] = capture_source_bundle_snapshot(
        bundle
    ).manifest_sha256
    payload["respondents"] = roster
    payload["deployment"]["definition"] = "Synthetic internal use location"
    templates = {}
    for status in ("public", "not_yet_public"):
        candidate = json.loads(json.dumps(payload))
        candidate["context_id"] = status
        candidate["model"]["entity_id"] = f"model_{status}"
        candidate["lab"]["related_entity_ids"] = [f"model_{status}"]
        candidate["deployment"]["entity_id"] = f"deployment_{status}"
        candidate["deployment_binding"]["model_id"] = f"model_{status}"
        candidate["deployment_binding"]["deployment_id"] = f"deployment_{status}"
        for node in candidate["nodes"]:
            node["population"]["deployment_ids"] = [f"deployment_{status}"]
        candidate["capability_access"] = "public" if status == "public" else "internal"
        path = tmp_path / f"{status}.json"
        path.write_text(
            ContextualRiskInstrument.model_validate(candidate).model_dump_json()
        )
        templates[status] = path
    forecast = tmp_path / "forecast-input.json"
    forecast.write_text(synthetic_forecast().model_dump_json())
    assumptions = tmp_path / "assumptions.json"
    assumptions.write_text(
        json.dumps(
            {
                "exposure_rationale": "Disjoint synthetic incidents by source model status at origin.",
                "residual_independence_rationale": "Conditional independence given the common public path, respondent and rank.",
                "parameter_coupling_rationale": "Matched respondent identity across the synthetic conditional judgments.",
            }
        )
    )
    folder = tmp_path / "study"
    write_public_frontier_study(
        forecast_path=forecast,
        public_instrument=templates["public"],
        not_yet_public_instrument=templates["not_yet_public"],
        assumptions_path=assumptions,
        output=folder,
        nodes=3,
        seed=42,
    )

    def answer(_system_prompt, user_prompt):
        return fixture_response(user_prompt, contextual_answer())

    providers = [_Immediate(model, answer) for model in models]
    profile = ModelProfile(
        models=tuple(
            ModelSpec(
                provider="openai",
                model_id=model,
                reasoning_effort="low",
                max_output_tokens=100_000,
            )
            for model in models
        )
    )
    monkeypatch.setattr("catmodel.config.load_model_profile", lambda **_kwargs: profile)
    monkeypatch.setattr(
        "catmodel.llm.factory.build_providers", lambda *_args, **_kwargs: providers
    )
    _sending(monkeypatch, when=when, respond=respond)
    artifacts = run_study_collection(
        study=folder / "public-study.json",
        source_bundle=bundle,
        profile="test",
    )
    assert artifacts is not None and len(artifacts) == 8
    return folder


def _direct(context_id: str, respondent: tuple[str, str, int], leaf: str | None = None):
    """Selects ``respondent``'s direct questions in one context: every
    leaf's, or ``leaf``'s alone."""
    return lambda request: (
        _context(request) == context_id
        and _respondent(request) == respondent
        and request_kind(request["user_prompt"]) == "direct"
        and leaf in (None, _subject(request))
    )


def _import(folder, destination, **options):
    return import_public_frontier_study(
        source=folder / "public-study.json",
        destination=destination,
        setup_id="excluded-samples",
        years_per_state=4,
        max_reference_state_years=4 * 4096,
        max_comparison_state_years=4 * 4096,
        seed=3,
        code_commit="fixture",
        **options,
    )


def test_a_public_study_excludes_the_unanswered_sample_and_discloses_it(
    tmp_path, monkeypatch
):
    folder = _public_study(tmp_path, monkeypatch, _direct("node-1-4-public", A_1))
    package = _import(folder, tmp_path / "package")
    states = {state.state_id: state.weight for state in package.joint_states}
    # 4 nodes x 6 respondents, less the one excluded sample.
    assert len(states) == 23
    by_node: dict[str, list[float]] = {}
    for state_id, weight in states.items():
        by_node.setdefault(public_state_node(state_id), []).append(weight)
    plan = json.loads((folder / "public-frontier.json").read_text())
    path_weights = dict(
        zip(
            (row["draw_id"] for row in plan["nodes"]),
            plan["base"]["weights"],
            strict=True,
        )
    )
    for node, weights in by_node.items():
        assert sum(weights) == pytest.approx(path_weights[node])
    # At the node, fixture-a's two kept samples carry the model's whole share.
    node_one = sorted(
        weight
        for state_id, weight in states.items()
        if state_id.startswith("node-1-4:")
    )
    share = path_weights["node-1-4"]
    assert node_one == pytest.approx([share / 6] * 3 + [share / 4] * 2)
    assert not any(
        state_id.startswith("node-1-4:respondent-1") for state_id in states
    ), "the excluded sample keeps no state"
    record = _exhibit(package, "excluded_samples")
    assert (record["excluded_samples"], record["samples"]) == (1, 24)
    assert record["excluded_share"] == pytest.approx(1 / 24)
    assert record["excluded_share_weighted"] == pytest.approx(share / 6)
    assert record["by_model"]["fixture-a"]["excluded_share"] == pytest.approx(1 / 12)
    assert record["exhausted_calls"]["by_kind"]["direct"] == 1
    assert record["exhausted_calls"]["by_cause"]["validation"] == 1
    assert record["renormalized_weights"]["node-1-4"]["fixture-a"]["kept"] == 2
    # One of the model's twelve samples lacks the leaf, within the control:
    # the model answers it, and nothing of the setup speaks of a model that
    # does not.
    assert "unanswered_scenarios" not in record
    assert "unanswered_scenarios" not in record["interpretation"]
    assert not any("POOL-UNANSWERED-SCENARIOS" in text for text in package.limitations)
    output = tmp_path / "run"
    run_setup(setup_path=package.root, output=output, code_commit="fixture")
    results = load_results(output / "results.json", diagnostics=True)
    assert results["sensitivity_summary"]["excluded_samples"] == record
    assert any(
        "Excluded samples (POOL-EXCLUDED-SAMPLES): 1 of 24 samples" in row["text"]
        for row in results["limitations"]
    )
    # The respondent excluded at one node is an incomplete epistemic group.
    band = results["sensitivity_summary"]["eal_parameter_band"]
    assert band["status"] == "available"
    assert band["incomplete_epistemic_groups"] == ["respondent-1"]
    summary = write_results_summary(
        results_path=output / "results.json", output=tmp_path / "summary"
    )
    text = summary.read_text()
    assert "## Excluded samples" in text
    assert (
        "1 call was exhausted (last attempt by cause: 1 validation). A sample (a "
        "respondent at one trajectory node, or at every coordinate of a "
        "fixed-anchor setup) whose leaf's direct question was exhausted has no "
        "law for that leaf and is excluded: 1 of 24 samples" in text
    )
    assert "| `synthetic-node` | `direct` | 1 | 3 | 1 validation |" in text
    assert "| `node-1-4` | `fixture-a` | 2 of 3 | 0.25 | 0.1667 |" in text
    assert "Incomplete respondents: `respondent-1`." in text
    # The study's evidence is elicited, so the summary prints the limitations
    # that describe a collected study, items 1 and 8 of docs/limitations.md.
    # Of its results' limitations it leaves out only item 5, the independence
    # of scenarios, of which the study has one.
    references = {
        row["reference"].partition("#")[2].split("-")[0]: row["text"]
        for row in results["limitations"]
        if row["source"] == "model"
    }
    assert references["1"] in text and references["8"] in text
    assert [
        row["text"] for row in results["limitations"] if row["text"] not in text
    ] == [references["5"]]


def test_import_refuses_a_model_keeping_under_half_its_samples_at_a_node(
    tmp_path, monkeypatch
):
    folder = _public_study(
        tmp_path,
        monkeypatch,
        lambda request: (
            _direct("node-1-2-not_yet_public", A_1)(request)
            or _direct("node-1-2-public", A_0)(request)
        ),
    )
    with pytest.raises(
        ValueError,
        match="model fixture-a keeps 1 of its 3 samples at node 'node-1-2', fewer than half",
    ):
        _import(folder, tmp_path / "package", max_excluded_sample_share=1.0)


def test_import_refuses_a_models_excluded_share_above_the_control(
    tmp_path, monkeypatch
):
    # Two samples of the model, each lacking another leaf: one of its twelve
    # samples lacks either leaf, within the control, so the model answers
    # both and each sample is excluded.
    folder = _public_study(
        tmp_path,
        monkeypatch,
        lambda request: (
            _direct("node-1-4-public", A_1, "leaf_a")(request)
            or _direct("node-3-4-public", A_2, "leaf_b")(request)
        ),
        instrument=leaves_instrument(("leaf_a", "leaf_b")),
    )
    # 2 of fixture-a's 12 samples, though 2 of 24 overall is within 10%.
    with pytest.raises(
        ValueError, match=r"model fixture-a has 2 of its 12 samples \(16.7%\) excluded"
    ):
        _import(folder, tmp_path / "refused")
    # Exactly at the control, import proceeds.
    package = _import(folder, tmp_path / "package", max_excluded_sample_share=2 / 12)
    assert len(package.joint_states) == 22
    assert "unanswered_scenarios" not in _exhibit(package, "excluded_samples")


def _declaring(*parts: str) -> ContextualRiskAnswer:
    """The fixture's direct answer with ``parts`` of the lab's loss declared
    unavailable."""
    payload = contextual_answer().model_dump(mode="json")
    for part in parts:
        payload[part] = {
            "status": "unavailable",
            "unavailable_reason": "No basis separates it from the components",
        }
    return ContextualRiskAnswer.model_validate(payload)


def _declares(parts_of):
    """A ``respond`` for ``_public_study``: ``parts_of`` gives, for each
    request, the parts of the lab's loss its direct answer declares
    unavailable."""

    def respond(request):
        answer = _declaring(*parts_of(request))
        return lambda _system_prompt, user_prompt: fixture_response(user_prompt, answer)

    return respond


def _own_answers(parts_of=lambda _request: ()):
    """A ``respond`` for ``_public_study`` whose samples differ, so that a
    pool changes with the samples it holds. Sample ``i`` of a model gives the
    fixture's direct answer with the victim's loss and the direct EAL
    multiplied by ``1 + i``, tenfold again for fixture-a's sample 1, the
    provider's fine by ``1 + i`` and the lab's annual business loss by
    ``(1 + i) ** 2``. ``parts_of`` gives, for each request, the parts of the
    lab's loss its direct answer declares unavailable."""

    def respond(request):
        _, model, index = _respondent(request)
        harm = (1 + index) * (10 if (model, index) == ("fixture-a", 1) else 1)
        payload = _declaring(*parts_of(request)).model_dump(mode="json")
        victim, fine = payload["branches"][0]["components"]
        victim["amount"] *= harm
        fine["amount"] *= 1 + index
        payload["direct_eal"] = {
            level: value * harm for level, value in payload["direct_eal"].items()
        }
        if payload["lab_annual_business_loss"]["status"] == "available":
            payload["lab_annual_business_loss"]["amount"] *= (1 + index) ** 2
        answer = ContextualRiskAnswer.model_validate(payload)
        return lambda _system_prompt, user_prompt: fixture_response(user_prompt, answer)

    return respond


def _studied(tmp_path, monkeypatch, name: str, respond, **options):
    """A public study under ``tmp_path / name``, collected with ``respond``,
    imported with ``options`` and run: its setup, its results and its run
    directory."""
    root = tmp_path / name
    root.mkdir()
    with monkeypatch.context() as patch:
        folder = _public_study(root, patch, lambda _request: False, respond=respond)
    package = _import(folder, root / "package", **options)
    run_setup(setup_path=package.root, output=root / "run", code_commit="fixture")
    results = load_results(root / "run" / "results.json", diagnostics=True)
    return package, results, root / "run"


def _state_eals(run, package, view: str) -> dict[str, float]:
    """Each reference state's analytic EAL in ``view``, by state ID, from the
    run's capture; a state that did not retain the view has none."""
    draws, _ = load_run_capture(run / "capture")
    found = {}
    for state in package.joint_states:
        draw = draws["reference"].get(state_key(state.state_id))
        if draw is not None and view in draw.views:
            found[state.state_id] = present(draw.views[view].native_mean).value
    return found


def _hand_standard_error(package, weights, eals) -> float:
    """The respondent-sampling standard error of ``sum(weight x EAL)`` by
    hand: with n_g weighted samples in sampling group g and y their weighted
    EALs, the square root of the sum over groups of n_g times the unbiased
    sample variance of y."""
    groups: dict[str, list[float]] = {}
    for state in package.joint_states:
        weight = weights[state.state_id]
        if weight > 0:
            groups.setdefault(present(state.sampling_group_id), []).append(
                weight * eals[state.state_id]
            )
    return math.sqrt(
        math.fsum(len(rows) * statistics.variance(rows) for rows in groups.values())
    )


LEFT_OUT = "node-1-4:respondent-1"
"""The state of fixture-a's sample 1 at node-1-4."""
SOURCE_STATUSES = ("public", "not_yet_public")


def _eal(arm, view: str) -> float:
    """A reduced arm's reported EAL in ``view``."""
    return present(available_metric(arm["metrics"][view])["expected_annual_loss"])


def test_a_lab_loss_declared_unavailable_changes_only_the_views_that_count_it(
    tmp_path, monkeypatch
):
    """Two studies differ only in that one sample's direct answer declares
    the lab's annual business loss unavailable in one context. Import keeps
    the same samples at the same weights and marks that sample's state. The
    run reports every view that counts no part of the lab's loss identically.
    The views that count it, the lab-loss views and the gross views, are
    computed on the other samples, the marked one's weight split between the
    two others of its model at its node."""
    declares = _direct("node-1-4-public", A_1)
    given, given_results, given_run = _studied(
        tmp_path, monkeypatch, "given", _own_answers()
    )
    declared, results, run = _studied(
        tmp_path,
        monkeypatch,
        "declared",
        _own_answers(
            lambda request: ("lab_annual_business_loss",) if declares(request) else ()
        ),
    )
    # Import: 4 nodes x 6 respondents in both, the same weights and groups.
    assert len(given.joint_states) == 24
    for before, after in zip(given.joint_states, declared.joint_states, strict=True):
        assert after.state_id == before.state_id
        assert after.lab_loss_left_out == (after.state_id == LEFT_OUT)
        assert after.model_copy(
            update={"lab_loss_left_out": False, "laws": before.laws}
        ) == (before)
        if after.state_id != LEFT_OUT:
            assert after == before
        # A state that is not left out serializes without the field.
        assert "lab_loss_left_out" not in before.model_dump(mode="json")
    assert not any(state.borrowed_leaves for state in declared.joint_states)
    # The stored setup of the study without such an answer names the field
    # nowhere, and its exhibit's record of lab losses is empty.
    assert not any(b"lab_loss_left_out" in data for data in given.files.values())
    assert any(b"lab_loss_left_out" in data for data in declared.files.values())
    before, record = (
        _exhibit(package, "excluded_samples") for package in (given, declared)
    )
    assert before["lab_loss_unavailable"] == {
        "cases": 0,
        "by_part": {"lab_compensation": 0, "lab_annual_business_loss": 0},
        "by_subject": {},
        "disposition": "none",
    }
    by_part = {"lab_compensation": 0, "lab_annual_business_loss": 1}
    assert record["lab_loss_unavailable"] == {
        "cases": 1,
        "by_part": by_part,
        "by_subject": {"synthetic-node": {"cases": 1, "by_part": by_part}},
        "disposition": "samples_left_out",
        "left_out_samples": 1,
        "left_out_by_model": {"fixture-a": 1, "fixture-b": 0},
        "left_out": [
            {
                "node": "node-1-4",
                "respondent": list(A_1),
                "reads": [
                    {
                        "context_id": "node-1-4-public",
                        "node_id": "synthetic-node",
                        "respondent": list(A_1),
                    }
                ],
            }
        ],
    }
    # Nothing else of the disclosure differs: no sample is excluded, no
    # weight renormalized, no call exhausted, no scenario unanswered.
    assert {
        key: row for key, row in record.items() if key != "lab_loss_unavailable"
    } == {key: row for key, row in before.items() if key != "lab_loss_unavailable"}
    assert (record["excluded_samples"], record["excluded"]) == (0, [])
    assert record["renormalized_weights"] == {}
    assert record["exhausted_calls"]["calls"] == 0
    assert "unanswered_scenarios" not in record
    limitation = next(
        text for text in declared.limitations if "POOL-EXCLUDED-SAMPLES" in text
    )
    assert limitation.startswith(
        "Excluded samples (POOL-EXCLUDED-SAMPLES): 0 of 24 samples (each a "
        "respondent at one trajectory node, or at every coordinate of a "
        "fixed-anchor setup) are excluded because a leaf's direct question "
        "exhausted its validation attempts, 0.0% of the reference weight; 0 calls "
        "exhausted theirs in all. 1 direct answer declared the lab's own loss "
        "unavailable (1 lab_annual_business_loss). Such an answer excludes no "
        "sample and changes no weight, so the welfare views, which count no part "
        "of the lab's loss, are as if it had been given. The views that count "
        "the lab's own loss are computed on the samples that give it: the "
        "lab-loss views and the gross views, since gross adds the lab's annual "
        "business loss, which a state that reads such an answer does not hold. "
        "1 sample whose states read such an answer, its own or that of a sample "
        "whose answers it uses, is left out of those views (fixture-a 1, "
        "fixture-b 0)"
    )
    assert limitation.endswith("Every model keeps every sample at every node.")
    assert results["sensitivity_summary"]["excluded_samples"] == record

    # The run: every view that counts no part of the lab's loss is the same
    # in every arm, statistics, statuses and diagnostics alike.
    reference = reduced_arm(results["arms"]["reference"])
    lab = next(view for view in reference["metrics"] if view.startswith("lab_loss::"))
    by_status = [f"source_status::{status}::{lab}" for status in SOURCE_STATUSES]
    gross_views = ["gross", "scenario::scenario::gross"]
    assert {view for view in reference["metrics"] if counts_lab_loss(view)} == {
        lab,
        *by_status,
        *gross_views,
    }
    assert results["arms"].keys() == given_results["arms"].keys()
    for arm, row in given_results["arms"].items():
        metrics = reduced_arm(row)["metrics"]
        changed = reduced_arm(results["arms"][arm])["metrics"]
        assert metrics.keys() == changed.keys()
        compared = [view for view in metrics if not counts_lab_loss(view)]
        assert set(compared) == {
            "welfare",
            "scenario::scenario::welfare",
            "source_status::public::welfare",
            "source_status::not_yet_public::welfare",
        }
        for view in compared:
            assert changed[view] == metrics[view], (arm, view)
            for comparisons in ("comparisons", "reference_comparisons"):
                assert results[comparisons].get(arm, {}).get(view) == (
                    given_results[comparisons].get(arm, {}).get(view)
                ), (arm, view)
            assert (
                results["diagnostics"]["arms"][arm]["views"][view]
                == (given_results["diagnostics"]["arms"][arm]["views"][view])
            ), (arm, view)
    assert results["scenario_contributions"] == given_results["scenario_contributions"]
    assert results["scenario_contributions"]["measure"] == "welfare"
    assert results["tail_gate"]["rounds"] == given_results["tail_gate"]["rounds"]
    for exhibit in ("dependence_bracket", "trajectory_quadrature"):
        assert (
            results["sensitivity_summary"][exhibit]
            == (given_results["sensitivity_summary"][exhibit])
        ), exhibit
    # The welfare EAL holds the sample the other views leave out, whose
    # victim loss is twenty times its model's first sample's.
    weights = {state.state_id: state.weight for state in declared.joint_states}
    welfare = _state_eals(run, declared, "welfare")
    # The reference simulates the 18 states it weighs: the three interior nodes'.
    assert welfare.keys() == {state for state, weight in weights.items() if weight}
    assert len(welfare) == 18
    assert welfare[LEFT_OUT] == pytest.approx(20 * welfare["node-1-4:respondent-0"])
    assert _eal(reference, "welfare") == pytest.approx(
        math.fsum(weights[state] * eal for state, eal in welfare.items()), rel=1e-12
    )

    # The views that count the lab's loss weigh the marked state zero and
    # split its group's weight, its model's at its node, between the group's
    # two other states.
    states = {state.state_id: state for state in declared.joint_states}
    group = [
        state.state_id
        for state in declared.joint_states
        if state.sampling_group_id == states[LEFT_OUT].sampling_group_id
    ]
    assert group == [f"node-1-4:respondent-{index}" for index in range(3)]
    lab_weights = dict(weights)
    lab_weights[LEFT_OUT] = 0.0
    for state in group[0], group[2]:
        lab_weights[state] = 1.5 * weights[state]
    assert reference["lab_loss_state_weights"] == pytest.approx(
        {state_key(state): weight for state, weight in lab_weights.items()}, rel=1e-12
    )
    given_reference = reduced_arm(given_results["arms"]["reference"])
    assert "lab_loss_state_weights" not in given_reference
    # The marked state holds no lab-loss law, and its gross loss lacks the
    # annual business loss; every other state's laws are the same.
    eals = {view: _state_eals(run, declared, view) for view in (lab, *gross_views)}
    given_eals = {
        view: _state_eals(given_run, given, view) for view in (lab, *gross_views)
    }
    assert eals[lab].keys() == welfare.keys() - {LEFT_OUT}
    assert welfare[LEFT_OUT] < eals["gross"][LEFT_OUT] < given_eals["gross"][LEFT_OUT]
    for view in (lab, *gross_views):
        assert given_eals[view].keys() == welfare.keys()
        assert {
            state: eal for state, eal in eals[view].items() if state != LEFT_OUT
        } == {
            state: eal for state, eal in given_eals[view].items() if state != LEFT_OUT
        }
        # Each is the mean over the samples that give the lab's loss, with
        # the marked sample's weight split between its group's two others,
        own = _eal(reference, view)
        assert own == pytest.approx(
            math.fsum(
                lab_weights[state] * eal
                for state, eal in eals[view].items()
                if lab_weights[state]
            ),
            rel=1e-12,
        ), view
        # and the same number from the study in which the lab's loss was
        # given, with that sample's weight moved to its group's other samples.
        assert own == pytest.approx(
            math.fsum(
                lab_weights[state] * eal for state, eal in given_eals[view].items()
            ),
            rel=1e-12,
        ), view
        # That study's own EAL, which holds the sample, is another number.
        theirs = _eal(given_reference, view)
        assert theirs == pytest.approx(
            math.fsum(weights[state] * eal for state, eal in given_eals[view].items()),
            rel=1e-12,
        ), view
        assert abs(theirs - own) > 1e-3 * own, view
    # Every view that counts the lab's loss holds the same samples, so the
    # source statuses' lab-loss EALs add up to the lab-loss EAL and the
    # scenario's gross EAL is the gross EAL, as where no sample is left out.
    for arm in (given_reference, reference):
        assert math.fsum(_eal(arm, view) for view in by_status) == pytest.approx(
            _eal(arm, lab), rel=1e-12
        )
        assert _eal(arm, "scenario::scenario::gross") == pytest.approx(
            _eal(arm, "gross"), rel=1e-12
        )
    # Each respondent's own EAL mixes its states by its view's weights.
    for view in (lab, "gross"):
        for respondent, own in reference["conditional_eal_by_epistemic_group"][
            view
        ].items():
            mine = [
                state.state_id
                for state in declared.joint_states
                if state.epistemic_group_id == respondent
                and lab_weights[state.state_id] > 0
            ]
            assert own == pytest.approx(
                math.fsum(lab_weights[state] * eals[view][state] for state in mine)
                / math.fsum(lab_weights[state] for state in mine),
                rel=1e-12,
            )

    # Sampling error: the groups of a view that counts the lab's loss are
    # the samples it holds, two of fixture-a at node-1-4; the welfare view's
    # are every sample.
    sampling = results["sensitivity_summary"]["respondent_sampling"]
    given_sampling = given_results["sensitivity_summary"]["respondent_sampling"]
    for view in (lab, "gross"):
        assert sampling["views"][view]["expected_annual_loss"] == pytest.approx(
            _eal(reference, view)
        )
        assert sampling["views"][view]["sampling_standard_error"] == pytest.approx(
            _hand_standard_error(declared, lab_weights, eals[view]), rel=1e-12
        )
        assert sampling["views"][view]["sampling_standard_error"] != pytest.approx(
            given_sampling["views"][view]["sampling_standard_error"], rel=1e-6
        )
    assert sampling["views"]["welfare"]["sampling_standard_error"] == pytest.approx(
        _hand_standard_error(declared, weights, welfare), rel=1e-12
    )
    assert sampling["views"]["welfare"] == given_sampling["views"]["welfare"]
    # The exhibit's counts are the arm's samples in both studies.
    assert (sampling["samples"], sampling["groups"]) == (18, 6)
    assert (given_sampling["samples"], given_sampling["groups"]) == (18, 6)
    # The parameter band of a view that counts the lab's loss is computed at
    # its weights and says so; the welfare view's is the same in both studies.
    band = results["sensitivity_summary"]["eal_parameter_band"]
    given_band = given_results["sensitivity_summary"]["eal_parameter_band"]
    assert band["status"] == "available"
    assert band["views"]["welfare"] == given_band["views"]["welfare"]
    assert "incomplete_epistemic_groups" not in band
    for view in (lab, "gross"):
        row = band["views"][view]
        assert row["expected_annual_loss"] == pytest.approx(_eal(reference, view))
        assert row["states"] == band["states"] - 1 == 17
        assert row["incomplete_epistemic_groups"] == ["respondent-1"]
        assert "states" not in given_band["views"][view]
    # The setup's frequency exhibit weighs each view as the run does: its
    # welfare view is the same in both studies, and the others' EALs the run's.
    frequency = results["sensitivity_summary"]["frequency_estimate"]["views"]
    assert (
        frequency["welfare"]
        == (
            given_results["sensitivity_summary"]["frequency_estimate"]["views"][
                "welfare"
            ]
        )
    )
    for view in (lab, "gross"):
        assert frequency[view]["by_arm"]["final"] == {
            "status": "available",
            "expected_annual_loss": pytest.approx(_eal(reference, view), rel=1e-12),
        }

    text = write_results_summary(
        results_path=run / "results.json", output=tmp_path / "summary"
    ).read_text()
    assert (
        "0 calls were exhausted. A sample (a respondent at one trajectory node, "
        "or at every coordinate of a fixed-anchor setup) whose leaf's direct "
        "question was exhausted has no law for that leaf and is excluded: 0 of "
        "24 samples" in text
    )
    assert (
        "1 direct answer declared the lab's own loss unavailable "
        "(1 `lab_annual_business_loss`). Such a leaf has every law but the "
        "lab's. The answer excludes no sample and changes no weight, so the "
        "welfare views, which count no part of the lab's loss, are as if it had "
        "been given. The views that count the lab's own loss are computed on "
        "the samples that give it: the lab-loss views and the gross views, "
        "since gross adds the lab's annual business loss. 1 sample whose states "
        "read such an answer, its own or that of a sample whose answers it "
        "uses, is left out of those views (`fixture-a` 1; `fixture-b` 0)" in text
    )
    assert (
        "| Node | Respondent left out of the lab-loss and gross views | Model | "
        "Context | Leaf | Answer read |" in text
    )
    assert (
        "| `node-1-4` | `fixture-a-1` | `fixture-a` | `node-1-4-public` | "
        "`synthetic-node` | its own |" in text
    )
    assert "| `synthetic-node` | 1 | 0 | 1 |" in text
    assert "Excluded respondent" not in text
    for view in (lab, "gross"):
        assert (
            f"`{view}` is computed at its own weights, from the 17 states that "
            "are not left out of the views that count the lab's own loss "
            "(Excluded samples). Respondents left out at some trajectory node "
            "hold fewer of them than the others, and each integrates only its "
            "own: `respondent-1`." in text
        )


def test_either_part_of_the_lab_loss_declared_unavailable_leaves_the_sample_out(
    tmp_path, monkeypatch
):
    """A sample that declares the compensation share unavailable in one of
    its node's contexts and both parts in the other is one sample left out of
    the views that count the lab's loss, reading two such cases; the tally
    counts cases and parts."""
    public = _direct("node-1-4-public", A_1)
    internal = _direct("node-1-4-not_yet_public", A_1)

    def parts_of(request) -> tuple[str, ...]:
        if public(request):
            return ("lab_compensation",)
        if internal(request):
            return ("lab_compensation", "lab_annual_business_loss")
        return ()

    folder = _public_study(
        tmp_path, monkeypatch, lambda _request: False, respond=_declares(parts_of)
    )
    package = _import(folder, tmp_path / "package")
    assert [
        state.state_id for state in package.joint_states if state.lab_loss_left_out
    ] == [LEFT_OUT]
    assert len(package.joint_states) == 24
    record = _exhibit(package, "excluded_samples")
    assert (record["excluded_samples"], record["excluded"]) == (0, [])
    by_part = {"lab_compensation": 2, "lab_annual_business_loss": 1}
    assert record["lab_loss_unavailable"] == {
        "cases": 2,
        "by_part": by_part,
        "by_subject": {"synthetic-node": {"cases": 2, "by_part": by_part}},
        "disposition": "samples_left_out",
        "left_out_samples": 1,
        "left_out_by_model": {"fixture-a": 1, "fixture-b": 0},
        "left_out": [
            {
                "node": "node-1-4",
                "respondent": list(A_1),
                "reads": [
                    {
                        "context_id": context,
                        "node_id": "synthetic-node",
                        "respondent": list(A_1),
                    }
                    for context in ("node-1-4-not_yet_public", "node-1-4-public")
                ],
            }
        ],
    }


def test_lab_losses_declared_unavailable_too_often_withhold_the_views_that_count_them(
    tmp_path, monkeypatch
):
    """Two of a model's twelve samples declare one leaf's lab loss
    unavailable. Leaving both out of the views that count the lab's loss
    would take the model's share of excluded or left-out samples above the
    control, so import leaves neither out and refuses nothing. No state is
    marked, and the run withholds both families: the lab-loss views, which
    those states do not hold, and the gross views, which would omit their
    business loss. The record names the guard, and the welfare views are as
    in the study in which the lab's loss was given. With the control at their
    share, both are left out and both families are reported."""
    first = _direct("node-1-4-public", A_1, "leaf_b")
    second = _direct("node-3-4-public", A_2, "leaf_b")
    instrument = leaves_instrument(("leaf_a", "leaf_b"))
    studies = {}
    for name, respond in (
        ("given", _declares(lambda _request: ())),
        (
            "declared",
            _declares(
                lambda request: (
                    ("lab_annual_business_loss",)
                    if first(request) or second(request)
                    else ()
                )
            ),
        ),
    ):
        root = tmp_path / name
        root.mkdir()
        with monkeypatch.context() as patch:
            studies[name] = _public_study(
                root,
                patch,
                lambda _request: False,
                respond=respond,
                instrument=instrument,
            )
    folder = studies["declared"]
    package = _import(folder, tmp_path / "package")
    given = _import(studies["given"], tmp_path / "given-package")
    assert len(package.joint_states) == 24
    assert not any(state.lab_loss_left_out for state in package.joint_states)
    assert [(state.state_id, state.weight) for state in package.joint_states] == [
        (state.state_id, state.weight) for state in given.joint_states
    ]
    record = _exhibit(package, "excluded_samples")
    assert (record["excluded_samples"], record["excluded"]) == (0, [])
    assert "unanswered_scenarios" not in record
    guard = (
        "model fixture-a would have 2 of its 12 samples (16.7%) excluded or left "
        "out of the views that count the lab's own loss, above the frozen control "
        "max_excluded_sample_share 0.1"
    )
    assert record["lab_loss_unavailable"] == {
        "cases": 2,
        "by_part": {"lab_compensation": 0, "lab_annual_business_loss": 2},
        "by_subject": {
            "leaf_b": {
                "cases": 2,
                "by_part": {"lab_compensation": 0, "lab_annual_business_loss": 2},
            }
        },
        "disposition": "views_withheld",
        "withheld_because": guard,
    }
    limitation = next(
        text for text in package.limitations if "POOL-EXCLUDED-SAMPLES" in text
    )
    assert limitation.startswith(
        "Excluded samples (POOL-EXCLUDED-SAMPLES): 0 of 24 samples"
    )
    assert "2 direct answers declared the lab's own loss unavailable" in limitation
    assert (
        "No sample is left out of those views, because leaving out the samples "
        f"whose states read such an answer would breach an import guard ({guard}); "
        "their states hold every other law, and the lab-loss views and the gross "
        "views are unavailable." in limitation
    )
    assert limitation.endswith("Every model keeps every sample at every node.")
    outcomes = {}
    for name, setup in (("declared", package), ("given", given)):
        output = tmp_path / f"{name}-run"
        run_setup(setup_path=setup.root, output=output, code_commit="fixture")
        outcomes[name] = load_results(output / "results.json", diagnostics=True)
    results = outcomes["declared"]
    metrics = reduced_arm(results["arms"]["reference"])["metrics"]
    unchanged = reduced_arm(outcomes["given"]["arms"]["reference"])["metrics"]
    # The views that count no part of the lab's loss are the same.
    welfare_views = [view for view in metrics if not counts_lab_loss(view)]
    assert len(welfare_views) == 5
    for view in welfare_views:
        assert metrics[view] == unchanged[view], view
    assert _eal(reduced_arm(results["arms"]["reference"]), "welfare") > 0
    # The states that declared it keep their weight and hold no lab-loss law,
    # so the lab-loss views are not retained in them,
    lab = next(view for view in metrics if view.startswith("lab_loss::"))
    withheld = cast(dict[str, Any], metrics[lab])
    assert withheld["status"] == "unavailable"
    assert withheld["reason"] == (
        "Requested view was not retained in every positive-weight state."
    )
    assert len(withheld["state_reasons"]) == 2
    # and the gross views, which would omit their business loss, are
    # withheld with the reason, for both scenarios alike.
    for view in ("gross", "scenario::leaf_a::gross", "scenario::leaf_b::gross"):
        assert metrics[view] == {
            "status": "unavailable",
            "reason": "State 'node-1-4:respondent-1' has weight and holds no law of "
            "the lab's own loss, so its gross loss would omit the lab's annual "
            "business loss, and it is not left out of the views that count that "
            "loss (POOL-EXCLUDED-SAMPLES).",
            "state_reasons": {},
        }, view
        assert unchanged[view]["status"] == "available"
    assert "lab_loss_state_weights" not in results["arms"]["reference"]
    frequency = cast(
        dict[str, Any], results["sensitivity_summary"]["frequency_estimate"]
    )["views"]
    assert frequency["gross"]["by_arm"]["final"]["status"] == "unavailable"
    assert frequency["welfare"]["by_arm"]["final"]["status"] == "available"
    text = write_results_summary(
        results_path=tmp_path / "declared-run" / "results.json",
        output=tmp_path / "summary",
    ).read_text()
    assert (
        "No sample is left out of those views, because leaving out the samples "
        "whose states read such an answer would breach an import guard" in text
    )
    # Exactly at the control both are left out, and both families are
    # reported from the other samples.
    fitted = _import(folder, tmp_path / "fitted", max_excluded_sample_share=2 / 12)
    assert len(fitted.joint_states) == 24
    assert [
        state.state_id for state in fitted.joint_states if state.lab_loss_left_out
    ] == [LEFT_OUT, "node-3-4:respondent-2"]
    record = _exhibit(fitted, "excluded_samples")
    assert record["excluded_samples"] == 0
    assert record["lab_loss_unavailable"]["disposition"] == "samples_left_out"
    assert record["lab_loss_unavailable"]["left_out_samples"] == 2
    assert "withheld_because" not in record["lab_loss_unavailable"]
    assert "unanswered_scenarios" not in record
    run_setup(
        setup_path=fitted.root, output=tmp_path / "fitted-run", code_commit="fixture"
    )
    reported = reduced_arm(
        load_results(tmp_path / "fitted-run" / "results.json")["arms"]["reference"]
    )
    assert reported["status"] == "reduced" and reported["missing_views"] == []
    for view in welfare_views:
        assert _eal(reported, view) == _eal(
            reduced_arm(outcomes["given"]["arms"]["reference"]), view
        )
    # The scenarios' gross EALs add up to gross, each on the same samples.
    assert math.fsum(
        _eal(reported, f"scenario::{leaf}::gross") for leaf in ("leaf_a", "leaf_b")
    ) == pytest.approx(_eal(reported, "gross"), rel=1e-12)
    assert _eal(reported, "gross") > _eal(reported, "welfare") > 0
    assert _eal(reported, lab) > 0


def test_pooled_weights_keep_each_models_share():
    roster = list(INSTRUMENT.respondents)
    weights = pooled_weights(roster, {A_1: ()})
    assert weights == pytest.approx(
        {
            ("fixture-a-0", "fixture-a", 0): 0.5,
            ("fixture-b-0", "fixture-b", 0): 0.25,
            ("fixture-b-1", "fixture-b", 1): 0.25,
        }
    )
    assert pooled_weights(roster, {}) == pytest.approx(
        {row.key(): 0.25 for row in roster}
    )


def test_each_exhausted_call_takes_its_last_attempts_cause(tmp_path, monkeypatch):
    _sending(
        monkeypatch,
        when=lambda request: (
            _respondent(request) == A_1
            and request_kind(request["user_prompt"]) == "direct"
            and _subject(request) == "fraud_organizations"
            and _anchor(request) == 0.5
        ),
    )
    artifact = _collect(tmp_path)
    (call,) = artifact.exhausted_calls()
    identity = (call.prompt_sha256, call.respondent)
    files = dict(read_contextual_capture_files(tmp_path))
    (attempts,) = _exhausted_attempts(tmp_path, "direct")
    last = attempts[-1].as_posix().removeprefix(tmp_path.as_posix() + "/")
    assert exhaustion_causes(files, [identity]) == {identity: "validation"}
    row = json.loads(files[last])
    for stop, cause in (
        ("refusal", "refusal"),
        ("output_limit", "output_limit"),
        ("completed", "validation"),
    ):
        stopped = {**files, last: json.dumps({**row, "stop_reason": stop}).encode()}
        assert exhaustion_causes(stopped, [identity]) == {identity: cause}


class _StoppedAtOutputLimit(_Garbled):
    """A provider that stops every response at its output limit, as the
    transport reports it: a truncated text with ``stop_reason``."""

    async def complete(self, **request) -> ProviderResponse:
        response = await self._inner.complete(**request)
        return response.model_copy(
            update={
                "text": response.text[: len(response.text) // 2],
                "stop_reason": "output_limit",
            }
        )


def test_a_call_exhausted_at_the_output_limit_is_classified_end_to_end(
    tmp_path, monkeypatch
):
    original = contextual_collection.collect_frozen_call

    async def patched(**kwargs):
        request = _identified(kwargs)
        if (
            _respondent(request) == A_1
            and request_kind(request["user_prompt"]) == "direct"
            and _subject(request) == "fraud_organizations"
            and _anchor(request) == 0.5
        ):
            kwargs["provider"] = _StoppedAtOutputLimit(kwargs["provider"])
        return await original(**kwargs)

    monkeypatch.setattr(contextual_collection, "collect_frozen_call", patched)
    # Collection records each stop as a failed attempt named for it, and the
    # capture replays.
    artifact = _collect(tmp_path)
    (call,) = artifact.exhausted_calls()
    assert call.kind == "direct"
    (attempts,) = (
        sorted(path.parent.glob("attempt-*.json"))
        for path in (tmp_path / "calls").glob("*/request.json")
        if json.loads(path.read_bytes())["prompt_sha256"] == call.prompt_sha256
        and _respondent(json.loads(path.read_bytes())) == call.respondent
    )
    rows = [json.loads(path.read_bytes()) for path in attempts]
    assert [row["stop_reason"] for row in rows] == ["output_limit"] * 3
    assert all('"type":"output_limit"' in row["error"] for row in rows)
    identity = (call.prompt_sha256, call.respondent)
    files = dict(read_contextual_capture_files(tmp_path))
    assert exhaustion_causes(files, [identity]) == {identity: "output_limit"}
    calls = _exhausted_calls_by_cause([(artifact, files)])
    assert calls["by_cause"] == {"validation": 0, "refusal": 0, "output_limit": 1}


def test_an_exposure_variant_is_guarded_like_the_headline(tmp_path, monkeypatch):
    from tests.test_exposure_scale_variants import _headline, _importer, _variant_study

    folder, bundle = _headline(tmp_path, monkeypatch)
    roster = json.loads((tmp_path / "public.json").read_text())["respondents"]
    sample = (roster[1]["respondent_id"], roster[1]["model_id"], 1)
    with pytest.MonkeyPatch.context() as patch:
        _sending(patch, when=_direct("node-1-4-public", sample))
        variant = _variant_study(tmp_path, bundle, "half", 0.5, respondents=roster)
    imported = _importer(tmp_path, folder / "public-study.json")
    # One of the variant model's eight samples (three embedded-rule nodes and the
    # no-progress node), against the headline's none. That is above the
    # control, so the variant's one model does not answer its one leaf.
    with pytest.raises(
        ValueError,
        match=r"no model answers leaf 'synthetic-node': model fixture-model lacks "
        r"it in 1 of its 8 samples \(12.5%\)",
    ):
        imported("refused", (variant,))
    # With the control at that share the sample is excluded, in the variant
    # study alone.
    package = imported("excluded", (variant,), max_excluded_sample_share=1 / 8)
    record = _exhibit(package, "excluded_samples")
    assert record["excluded_samples"] == 0
    assert record["exposure_scale_variants"]["half"]["excluded_samples"] == 1


def _public_hierarchy_inputs(tmp_path, monkeypatch, when):
    """Both source statuses of two trajectory nodes of the hierarchy
    instrument, collected with ``when``'s requests garbled."""
    instruments, weights = [], {}
    for draw, score, weight in (("low", 1, 0.25), ("high", 3, 0.75)):
        for status in ("public", "not_yet_public"):
            payload = INSTRUMENT.model_dump(mode="json")
            payload.update(
                context_id=f"{draw}-{status}",
                coordinate_kind="public_trajectory",
                reference_eci=0,
                anchor_scores=[0],
                integer_min=0,
                integer_max=0,
                capability_access="internal",
            )
            payload["capability_scale"]["scale_id"] = "eci"
            payload["model"]["entity_id"] = f"model-{status}"
            payload["deployment"]["entity_id"] = f"deployment-{status}"
            payload["lab"]["related_entity_ids"] = [f"model-{status}"]
            payload["deployment_binding"].update(
                model_id=f"model-{status}", deployment_id=f"deployment-{status}"
            )
            for node in payload["nodes"]:
                node["population"]["deployment_ids"] = [f"deployment-{status}"]
            payload["public_trajectory"] = {
                "trajectory_sha256": "e" * 64,
                "draw_id": draw,
                "observed_at": "2026-12-01T00:00:00Z",
                "observed_values": [score, 100 + score],
                "earlier_frontier": None,
                "times": [f"2027-{month:02d}-01T00:00:00Z" for month in range(1, 13)]
                + ["2028-01-01T00:00:00Z"],
                "coordinates": [
                    {
                        "coordinate_id": lab,
                        "lab_id": lab,
                        "domain_id": "general",
                        "access": "public",
                        "scale_id": "eci",
                        "definition": f"Synthetic {lab} general ECI",
                    }
                    for lab in ("lab_a", "lab_b")
                ],
                "values": [[score, 100 + score]] * 13,
                "interpolation": "linear",
                "target_lab": "public_frontier",
                "source_model_status": status,
                "source_references": ["urn:fixture:public-frontier"],
                "rationale": "Synthetic joint public forecast condition.",
                "limitations": ["Synthetic values validate software only."],
            }
            instrument = ContextualRiskInstrument.model_validate(payload)
            instruments.append(instrument)
            weights[instrument.context_id] = weight
    _sending(monkeypatch, when=when)
    results = asyncio.run(
        collect_study(
            [(row, tmp_path / row.context_id) for row in instruments],
            providers=cast(
                list[LLMProvider],
                [
                    _Immediate(model, StudyResponder(instruments, EVERY_TURN))
                    for model in MODELS
                ],
            ),
            config=CONFIG,
        )
    )
    assert all(isinstance(row, ContextualRiskArtifact) for row in results)
    return [
        PublicFrontierRiskInput(
            row,
            cast(ContextualRiskArtifact, artifact),
            weights[row.context_id],
            quadrature_sha256="q" * 64,
            files=read_contextual_capture_files(tmp_path / row.context_id),
        )
        for row, artifact in zip(instruments, results, strict=True)
    ]


def test_a_public_remainder_line_weights_only_the_kept_samples(tmp_path, monkeypatch):
    inputs = _public_hierarchy_inputs(
        tmp_path,
        monkeypatch,
        lambda request: (
            _context(request) == "high-public"
            and _respondent(request) == A_1
            and request_kind(request["user_prompt"]) == "direct"
            and _subject(request) == "fraud_organizations"
        ),
    )
    setup = RunSetup(
        **public_frontier_setup_fields(
            tuple(inputs),
            setup_id="hierarchy",
            years_per_state=4,
            seed=4,
            epistemic_tail_scale=1.0,
            comparison_families=(),
            max_excluded_sample_share=0.25,
            exposure_rationale="Synthetic disjoint source-status populations.",
            residual_independence_rationale="Synthetic conditional independence.",
            parameter_coupling_rationale="Synthetic matched respondents.",
        )
    )
    assert len(setup.joint_states) == 7
    # Every respondent gives the same answers in a context, so its pooled
    # remainder is one law there and the line is that law's mean in each
    # context weighted by its node: the kept samples' weights at the high
    # node still add to one.
    b_0 = ("fixture-b-0", "fixture-b", 0)
    laws = [
        hierarchy_outcomes(row.instrument, row.artifact)[0.0, b_0].remainder_laws[
            "n_fraud"
        ]
        for row in inputs
    ]
    assert all(law is not None for law in laws)
    expected = math.fsum(
        row.path_weight * epistemic_mean(cast(Quantiles, law))
        for row, law in zip(inputs, laws, strict=True)
    )
    line = _exhibit(setup, "frequency_estimate")["hierarchy"]["n_fraud"][
        "remainder_line"
    ]
    assert line["expected_annual_loss"] == pytest.approx(expected)


def _by_sample(request):
    """Answers whose rate grows with the respondent's sample, so samples of
    one model disagree."""
    respondent = request["respondent"]
    factor = (1 + respondent["sample_index"] / 2) * (
        1.25 if respondent["model_id"] == "fixture-b" else 1
    )
    answer = contextual_answer()
    assert answer.rate is not None
    rate = Quantiles(
        **{key: value * factor for key, value in answer.rate.model_dump().items()}
    )
    scaled = answer.model_copy(update={"rate": rate})
    return lambda _system_prompt, user_prompt: fixture_response(user_prompt, scaled)


def test_a_public_study_reports_its_respondent_sampling_error(tmp_path, monkeypatch):
    """The exhibit ties to the run's own EAL and weights, exclusion included,
    and to a recomputation from each state's analytic EAL grouped by node and
    model; replay and export reproduce it."""
    from catmodel.results.execution import replay_capture
    from catmodel.results.results_export import export_results
    from catmodel.simulation.incident import compile_incident_state
    from catmodel.simulation.incident_moments import analytic_linear_eals

    folder = _public_study(
        tmp_path, monkeypatch, _direct("node-1-4-public", A_1), respond=_by_sample
    )
    package = _import(folder, tmp_path / "package")
    output = tmp_path / "run"
    run_setup(setup_path=package.root, output=output, code_commit="fixture")
    results = load_results(output / "results.json", diagnostics=True)
    summary = results["sensitivity_summary"]
    exhibit = summary["respondent_sampling"]
    assert exhibit["status"] == "available"
    # Every model answers the leaf, so no state reads a donor's answers and
    # none serializes the field that would name them.
    for state in package.joint_states:
        assert state.borrowed_leaves == ()
        assert "borrowed_leaves" not in state.model_dump(mode="json")
    # Three positive nodes and two models; fixture-a keeps two of its three
    # samples at node-1-4.
    assert (exhibit["groups"], exhibit["samples"], exhibit["smallest_group"]) == (
        6,
        17,
        2,
    )
    # Recompute from each state's analytic EAL. The roster sorts fixture-a's
    # respondents first, so respondent-0 to -2 are its samples.
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for state in package.joint_states:
        if state.weight > 0:
            index = int(state.state_id.rsplit("-", 1)[1])
            model = "fixture-a" if index < 3 else "fixture-b"
            value = analytic_linear_eals(
                compile_incident_state(package, state.state_id), ("welfare",)
            ).values["welfare"]
            groups.setdefault((public_state_node(state.state_id), model), []).append(
                (state.weight, value)
            )
    assert len(groups[("node-1-4", "fixture-a")]) == 2
    variance = 0.0
    for rows in groups.values():
        weight = math.fsum(weight for weight, _ in rows)
        values = [value for _, value in rows]
        mean = sum(values) / len(values)
        spread = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        variance += weight**2 * spread / len(values)
    welfare = exhibit["views"]["welfare"]
    assert welfare["status"] == "available"
    reported = available_metric(
        reduced_arm(results["arms"]["reference"])["metrics"]["welfare"]
    )
    assert welfare["expected_annual_loss"] == reported["expected_annual_loss"]
    assert welfare["sampling_standard_error"] == pytest.approx(
        math.sqrt(variance), rel=1e-9
    )
    assert welfare["sampling_standard_error"] > 0
    assert set(exhibit["views"]) >= {
        "gross",
        "welfare",
        "source_status::public::welfare",
    }
    # Each sensitivity row carries its change's standard error and z.
    rows = summary["tornado"]["rows"]
    assert rows
    for row in rows:
        change = exhibit["comparisons"][row["arm"]][present(summary["tornado"]["view"])]
        assert change["status"] == "available"
        assert change["control_arm"] == row["control_arm"]
        assert row["sampling_standard_error"] == change["sampling_standard_error"]
        assert change["absolute_difference"] == pytest.approx(
            row["absolute_difference"], rel=1e-9, abs=1e-9
        )
    no_progress = next(
        row for row in rows if row["arm"] == "public_forecast::no_progress"
    )
    assert no_progress["sampling_z"] is not None
    trajectory = present(summary.get("trajectory_quadrature"))
    assert trajectory["status"] == "available"
    quadrature = trajectory["by_arm"]["reference"]
    assert present(quadrature["sampling_standard_error"]) > 0
    assert quadrature["by_node"]["node-1-4"]["samples"] == 5
    # No state reads a donor's answers, so every node reports the standard
    # deviation of its state EALs and the results report the band.
    for row in quadrature["by_node"].values():
        assert present(row["state_eal_sd"]) >= 0
        assert row["state_eal_sd_reason"] is None
    band = summary["eal_parameter_band"]
    assert band["status"] == "available"
    assert band["views"]["welfare"]["status"] == "available"
    assert math.fsum(
        row["difference_contribution"] for row in quadrature["by_node"].values()
    ) == pytest.approx(quadrature["absolute_difference"], rel=1e-9, abs=1e-12)
    # Replay and export reproduce the exhibit.
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=output / "capture",
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
    assert replay["sensitivity_summary"]["respondent_sampling"] == exhibit
    export_results(
        setup_path=package.root,
        capture_directory=output / "capture",
        results_path=output / "results.json",
        output=tmp_path / "export",
        code_commit="fixture",
    )
    text = write_results_summary(
        results_path=output / "results.json", output=tmp_path / "summary"
    ).read_text()
    assert "## Respondent sampling" in text
    assert "| `welfare` |" in text


class _Refused(_Garbled):
    """A provider that refuses every response, as the transport reports a
    refusal or content filter: ``stop_reason`` ``refusal``."""

    async def complete(self, **request) -> ProviderResponse:
        response = await self._inner.complete(**request)
        return response.model_copy(update={"text": "", "stop_reason": "refusal"})


def test_a_model_refusing_one_scenario_everywhere_is_left_out_of_that_scenario(
    tmp_path, monkeypatch
):
    """One respondent model refuses one scenario's direct and chain questions
    in every context: collection goes on and discloses each refusal, and
    import leaves the model out of that scenario alone, giving its samples
    the other model's answers to it (``POOL-UNANSWERED-SCENARIOS``)."""
    original = contextual_collection.collect_frozen_call

    def refused(request) -> bool:
        return (
            _respondent(request)[1] == "fixture-a"
            and request_kind(request["user_prompt"]) in {"direct", "chain"}
            and _subject(request) == "fraud_organizations"
        )

    async def patched(**kwargs):
        if refused(_identified(kwargs)):
            kwargs["provider"] = _Refused(kwargs["provider"])
        return await original(**kwargs)

    monkeypatch.setattr(contextual_collection, "collect_frozen_call", patched)
    inputs = _public_hierarchy_inputs(tmp_path, monkeypatch, when=lambda _: False)
    # Every context completes: each of fixture-a's samples has an unanswered
    # fraud_organizations leaf whose direct and chain calls are exhausted by
    # refusals, and no other leaf or model is unanswered.
    for row in inputs:
        files = dict(row.files)
        validate_contextual_capture(row.instrument, row.artifact, row.files)
        # Two samples' direct and chain calls, three refused attempts each.
        usage = call_limit_usage(files)
        assert (usage["fixture-a"]["refusals"], usage["fixture-b"]["refusals"]) == (
            12,
            0,
        )
        exhausted = row.artifact.exhausted_calls()
        assert {(call.kind, call.subject_id) for call in exhausted} >= {
            ("direct", "fraud_organizations"),
            ("chain", "fraud_organizations"),
        }
        assert {call.respondent[1] for call in exhausted} == {"fixture-a"}
        identities = [(call.prompt_sha256, call.respondent) for call in exhausted]
        assert set(exhaustion_causes(files, identities).values()) == {"refusal"}
        assert {
            (case.respondent.model_id, case.node_id)
            for case in row.artifact.unanswered_cases
        } == {("fixture-a", "fraud_organizations")}
        assert len(row.artifact.unanswered_cases) == 2
    options: dict[str, Any] = {
        "setup_id": "refusals",
        "years_per_state": 4,
        "seed": 4,
        "epistemic_tail_scale": 1.0,
        "comparison_families": (),
        "exposure_rationale": "Synthetic disjoint source-status populations.",
        "residual_independence_rationale": "Synthetic conditional independence.",
        "parameter_coupling_rationale": "Synthetic matched respondents.",
    }
    setup = RunSetup(**public_frontier_setup_fields(tuple(inputs), **options))
    # No sample is excluded: both nodes keep their four respondents.
    assert len(setup.joint_states) == 8
    record = _exhibit(setup, "excluded_samples")
    assert (record["excluded_samples"], record["samples"]) == (0, 8)
    assert record["exhausted_calls"]["by_cause"] == {
        "validation": 0,
        "refusal": 16,
        "output_limit": 0,
    }
    unanswered = record["unanswered_scenarios"]
    # The model's four samples, two at either node, each lack the leaf in
    # both of their contexts.
    assert unanswered["pairs"] == [
        {
            "node_id": "fraud_organizations",
            "scenario_id": "fraud_organizations",
            "model_id": "fixture-a",
            "basis": "own",
            "samples": 4,
            "lacking_samples": 4,
            "lacking_share": 1.0,
            "exhausted_direct_calls": {
                "calls": 8,
                "attempts": 24,
                "by_cause": {"validation": 0, "refusal": 8, "output_limit": 0},
            },
        }
    ]
    assert unanswered["by_leaf"]["fraud_organizations"]["answering_models"] == {
        "fixture-b": {"realized_share": pytest.approx(1.0), "equal_share": 1.0}
    }
    assert unanswered["leaves_on_one_model"] == 1
    assert any(
        text.startswith(
            "Unanswered scenarios (POOL-UNANSWERED-SCENARIOS): model fixture-a does "
            "not answer scenario fraud_organizations (unanswered in 4 of its 4 "
            "samples, 100.0%): its samples use the answers of fixture-b."
        )
        for text in setup.limitations
    )
    # The exhibits count the kept stored cases, each once: two leaves of two
    # samples of the answering model and one leaf of the other's, in four
    # contexts. Every kept respondent's taxonomy parents are pooled, the
    # borrowers' with the borrowed leaf.
    frequency = _exhibit(setup, "frequency_estimate")
    assert frequency["frequency_collection"]["cases"] == 24
    assert frequency["eal_crosscheck"]["cases"] == 24
    for row in frequency["hierarchy"].values():
        assert row["cases"] == 16
        assert row["cases_by_status"]["parent_exhausted"] == 0
    # The large-loss anchor's statuses count a borrowed case as its donor's:
    # the leaf has the answering model's two samples in four contexts.
    anchor = _exhibit(setup, "large_loss_anchor")["by_scenario"]
    assert {
        scenario: sum(row["case_statuses"].values()) for scenario, row in anchor.items()
    } == {"fraud_organizations": 8, "output_reliance_legal": 16}
    # With the control at one no share is above it, so the leaf excludes
    # every sample of the model and the half-of-samples floor refuses.
    with pytest.raises(
        ValueError,
        match=r"model fixture-a keeps 0 of its 2 samples at node 'high', fewer "
        r"than half \(unanswered leaves: fraud_organizations\), because a leaf's "
        r"direct question exhausted its validation attempts "
        r"\(POOL-EXCLUDED-SAMPLES\)",
    ):
        RunSetup(
            **public_frontier_setup_fields(
                tuple(inputs), max_excluded_sample_share=1.0, **options
            )
        )
