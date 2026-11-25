"""The answer table keeps responses apart from the collections that asked for
them: collection reads it before every request and adds every response, so a
changed prompt, parser, roster or sample count sends only what the table
cannot answer."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner
from pydantic import BaseModel

from catmodel.cli import cli
from catmodel.collection.answer_table import (
    AddedRows,
    AnswerKey,
    AnswerTable,
    prompt_digest,
)
from catmodel.collection.contextual_collection import (
    run_contextual_collection,
)
from catmodel.collection.frozen_call_collection import (
    ProviderFailed,
    collect_frozen_call,
)
from catmodel.config import ExecutionSettings, ModelProfile
from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from catmodel.evidence.source_bundle import capture_source_bundle_snapshot
from catmodel.frozen_calls import frozen_call_key, replay_call
from catmodel.json_io import strict_json_loads
from catmodel.llm.models import ModelSpec
from catmodel.llm.transport import LLMProvider, ProviderResponse
from catmodel.setup.public_study import public_study_collections
from test_support.capability_sources import authored_source_files, write_source_bundle
from test_support.contextual_risk import (
    contextual_answer,
    contextual_instrument,
    fixture_response,
)
from test_support.public_frontier import collect_study_nodes
from tests.test_public_frontier_study import _study

_RESPONDENT = {"respondent_id": "fixture", "model_id": "fixture", "sample_index": 0}
_REQUEST = {
    "call_id": "fixture-call",
    "respondent": _RESPONDENT,
    "prompt_sha256": "a" * 64,
    "system_prompt": "fixture system",
    "user_prompt": "fixture question",
}
_CONFIG = ExecutionSettings(
    samples_per_question=1,
    max_output_tokens_by_model={"fixture": 100},
    request_timeout_seconds=60,
    concurrency_by_model={"fixture": 1},
)


class Answer(BaseModel):
    value: int


def parse(text: str) -> Answer:
    return Answer.model_validate_json(text)


def strict_parse(text: str) -> Answer:
    return Answer.model_validate_json(text, strict=True)


class Provider:
    """Returns each scripted text in turn; ``None`` fails at the provider."""

    provider_name = "fixture"
    model_id = "fixture"
    reasoning_effort = "low"

    def __init__(self, *texts: str | None):
        self.texts = list(texts)
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    async def complete(self, *, user_prompt: str, **_kwargs):
        self.prompts.append(user_prompt)
        text = self.texts.pop(0)
        if text is None:
            raise RuntimeError("transport unavailable")
        return ProviderResponse(
            text=text,
            model_id="fixture-2026-11",
            provider_name=self.provider_name,
            provider_request_id=f"request-{self.calls}",
            latency_ms=7,
            input_tokens=11,
            output_tokens=5,
            evidence={"mechanism": "fixture", "sent": self.calls},
        )


class Unreachable(Provider):
    """A provider no request may reach."""

    async def complete(self, **_kwargs):
        pytest.fail("the provider was called although the table had the answer")


def _collect(run, provider, *, table=None, parser=parse, request=_REQUEST):
    return asyncio.run(
        collect_frozen_call(
            run_directory=run,
            request=request,
            provider=cast(LLMProvider, provider),
            config=_CONFIG,
            parse_answer=parser,
            response_model=Answer,
            answers=table,
        )
    )


def _attempts(run: Path, request=_REQUEST) -> list[dict]:
    call = run / "calls" / frozen_call_key(request)
    return [
        strict_json_loads(path.read_bytes())
        for path in sorted(call.glob("attempt-*.json"))
    ]


_KEY = AnswerKey(
    provider_name="fixture",
    model_id="fixture",
    reasoning_effort="low",
    max_output_tokens=100,
    sample_index=0,
    prompt_sha256=prompt_digest("fixture system", "fixture question"),
)


def _text(table: AnswerTable, key: AnswerKey) -> str:
    response = table.lookup(key)
    assert response is not None
    return response.text


def test_a_second_collection_is_answered_from_the_table(tmp_path):
    table = AnswerTable(tmp_path / "answers")
    first = Provider('{"value": 3}')
    assert _collect(tmp_path / "a", first, table=table).value == 3
    assert first.calls == 1

    # Another run directory, as a re-planned study has: nothing is sent.
    assert _collect(tmp_path / "b", Unreachable(), table=table).value == 3
    (live,), (reused,) = _attempts(tmp_path / "a"), _attempts(tmp_path / "b")
    assert "reused_from_answer_table" not in live
    assert reused.pop("reused_from_answer_table") is True
    # The reused attempt records what the provider returned the first time.
    assert reused == live
    assert reused["provider_request_id"] == "request-1"
    assert reused["provider_evidence"] == {"mechanism": "fixture", "sent": 1}

    # The reused capture replays like any other.
    call = f"calls/{frozen_call_key(_REQUEST)}"
    files = {
        f"{call}/{path.name}": path.read_bytes()
        for path in (tmp_path / "b" / call).iterdir()
    }
    replayed = replay_call(request=_REQUEST, files=files, parse_answer=parse)
    assert replayed.answer == Answer(value=3)


def test_only_a_changed_prompt_setting_or_sample_is_sent(tmp_path):
    table = AnswerTable(tmp_path / "answers")
    _collect(tmp_path / "a", Provider('{"value": 3}'), table=table)

    reworded = {**_REQUEST, "call_id": "reworded", "user_prompt": "another question"}
    provider = Provider('{"value": 4}')
    assert _collect(tmp_path / "b", provider, table=table, request=reworded).value == 4
    assert provider.prompts == ["another question"]

    second_sample = {**_REQUEST, "respondent": {**_RESPONDENT, "sample_index": 1}}
    provider = Provider('{"value": 5}')
    result = _collect(tmp_path / "c", provider, table=table, request=second_sample)
    assert result.value == 5 and provider.calls == 1

    # The same prompt under another reasoning effort is another question.
    harder = Provider('{"value": 6}')
    harder.reasoning_effort = "high"
    assert _collect(tmp_path / "d", harder, table=table).value == 6
    assert harder.calls == 1

    # Each is now in the table beside the first.
    assert table.verify() == 4
    assert _collect(tmp_path / "e", Unreachable(), table=table).value == 3
    assert (
        _collect(tmp_path / "f", Unreachable(), table=table, request=reworded).value
        == 4
    )


def test_a_changed_parser_reads_the_stored_response_again(tmp_path):
    table = AnswerTable(tmp_path / "answers")
    strict = Provider('{"value": "3"}', '{"value": 4}')
    # The strict parser refuses a quoted number, so the call retries.
    assert _collect(tmp_path / "a", strict, table=table, parser=strict_parse).value == 4
    assert strict.calls == 2 and table.verify() == 2

    # A lenient parser accepts the first response: one attempt, nothing sent.
    assert _collect(tmp_path / "b", Unreachable(), table=table).value == 3
    assert len(_attempts(tmp_path / "b")) == 1


def test_a_retry_that_repeats_its_prompt_keeps_each_response(tmp_path):
    table = AnswerTable(tmp_path / "answers")
    provider = Provider("not json", "still not json", '{"value": 5}')
    assert _collect(tmp_path / "a", provider, table=table).value == 5
    # Two failures with the same feedback give the third attempt the second
    # one's prompt.
    assert provider.prompts[1] == provider.prompts[2] != provider.prompts[0]
    draws = sorted(
        (row["prompt_sha256"], row["draw"], row["raw_response"]) for row in table.rows()
    )
    retry = prompt_digest("fixture system", provider.prompts[1])
    assert [(digest, draw) for digest, draw, _ in draws if digest == retry] == [
        (retry, 0),
        (retry, 1),
    ]
    assert {text for _, _, text in draws} == {
        "not json",
        "still not json",
        '{"value": 5}',
    }

    assert _collect(tmp_path / "b", Unreachable(), table=table).value == 5
    assert [row["raw_response"] for row in _attempts(tmp_path / "b")] == [
        "not json",
        "still not json",
        '{"value": 5}',
    ]


def test_a_provider_failure_adds_nothing_and_its_resend_reads_the_table(tmp_path):
    table = AnswerTable(tmp_path / "answers")
    with pytest.raises(ProviderFailed):
        _collect(tmp_path / "a", Provider(None), table=table)
    assert table.verify() == 0

    # Another collection then answers the question; the failed call's resend
    # takes that answer and sends nothing.
    _collect(tmp_path / "other", Provider('{"value": 8}'), table=table)
    assert _collect(tmp_path / "a", Unreachable(), table=table).value == 8
    failed, reused = _attempts(tmp_path / "a")
    assert failed["raw_response"] is None
    assert reused["reused_from_answer_table"] is True


def test_resuming_with_a_table_adds_the_stored_responses(tmp_path):
    # Collected without a table, then resumed with one.
    _collect(tmp_path / "a", Provider('{"value": 3}'))
    table = AnswerTable(tmp_path / "answers")
    assert _collect(tmp_path / "a", Unreachable(), table=table).value == 3
    assert _text(table, _KEY) == '{"value": 3}'


def test_a_collection_directory_is_added_whole(tmp_path):
    _collect(tmp_path / "a", Provider("not json", '{"value": 2}'))
    reworded = {**_REQUEST, "call_id": "reworded", "user_prompt": "another question"}
    _collect(tmp_path / "a", Provider('{"value": 4}'), request=reworded)

    table = AnswerTable(tmp_path / "answers")
    assert table.add_collection(tmp_path / "a") == AddedRows(added=3, present=0)
    assert table.add_collection(tmp_path / "a") == AddedRows(added=0, present=3)
    assert table.verify() == 3
    assert _collect(tmp_path / "b", Unreachable(), table=table).value == 2

    # Each row names by digest the prompt it answers, which the table keeps.
    prompts = {
        row["prompt_sha256"]: table.prompt(row["prompt_sha256"]) for row in table.rows()
    }
    assert len(prompts) == 3
    assert {prompt["user_prompt"] for prompt in prompts.values()} >= {
        "fixture question",
        "another question",
    }
    assert all(prompt_digest(**prompt) == digest for digest, prompt in prompts.items())

    with pytest.raises(ValueError, match="holds no calls/ directory"):
        table.add_collection(tmp_path / "answers")


def test_the_first_response_under_a_key_stays(tmp_path):
    table = AnswerTable(tmp_path / "answers")

    def record(text: str, user_prompt: str = "fixture question") -> bool:
        return table.record(
            _KEY,
            system_prompt="fixture system",
            user_prompt=user_prompt,
            response=ProviderResponse(text=text, model_id="fixture"),
        )

    assert record("first") is True
    assert record("second") is False
    assert _text(table, _KEY) == "first"
    assert table.lookup(replace(_KEY, sample_index=1)) is None
    assert table.lookup(replace(_KEY, max_output_tokens=200)) is None
    assert table.holds(_KEY) and not table.holds(replace(_KEY, draw=1))

    with pytest.raises(ValueError, match="differs from the prompt it answers"):
        record("first", user_prompt="not the keyed question")


def test_a_directory_that_is_not_a_table_is_refused(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("not a table")
    with pytest.raises(ValueError, match="is not an answer table"):
        AnswerTable(other)
    with pytest.raises(ValueError, match="is not a directory"):
        AnswerTable(other / "notes.txt")

    table = tmp_path / "answers"
    AnswerTable(table)
    # Reopening an existing table, or starting one in an empty directory.
    AnswerTable(table)
    (tmp_path / "empty").mkdir()
    AnswerTable(tmp_path / "empty")
    (table / "answer-table.json").write_text('{"kind": "something else"}')
    with pytest.raises(ValueError, match="answer table marker"):
        AnswerTable(table)


def test_verification_finds_a_changed_row_or_prompt(tmp_path):
    table = AnswerTable(tmp_path / "answers")
    _collect(tmp_path / "a", Provider('{"value": 3}'), table=table)
    assert table.verify() == 1
    (row,) = (tmp_path / "answers" / "rows").glob("*/*/*.json")
    (prompt,) = (tmp_path / "answers" / "prompts").glob("*/*.json")

    original = row.read_bytes()
    changed = strict_json_loads(original)
    changed["sample_index"] = 4
    row.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs from its path"):
        table.verify()
    with pytest.raises(ValueError, match="differs from its key"):
        table.lookup(_KEY)
    row.write_bytes(original)

    prompt.write_text(json.dumps({"system_prompt": "s", "user_prompt": "edited"}))
    with pytest.raises(ValueError, match="prompt differs from its name"):
        table.verify()


# A whole contextual collection


class ContextualProvider:
    provider_name = "openai"
    model_id = "fixture-model"
    reasoning_effort = "low"

    def __init__(self):
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        return ProviderResponse(
            text=fixture_response(kwargs["user_prompt"], contextual_answer()),
            model_id=self.model_id,
            provider_name=self.provider_name,
        )


@pytest.fixture
def study(tmp_path, monkeypatch):
    """A source bundle, a one-respondent instrument bound to it and a
    function that collects an instrument offline, returning the number of
    requests that reached the provider."""
    import catmodel.config

    source = write_source_bundle(tmp_path / "source", files=authored_source_files())
    data = contextual_instrument().model_dump(mode="json")
    data["source_bundle_manifest_sha256"] = capture_source_bundle_snapshot(
        source
    ).manifest_sha256
    monkeypatch.setattr(
        catmodel.config,
        "load_model_profile",
        lambda **_kwargs: ModelProfile(
            models=(
                ModelSpec(
                    provider="openai",
                    model_id="fixture-model",
                    reasoning_effort="low",
                    max_output_tokens=100_000,
                ),
            )
        ),
    )
    monkeypatch.setattr("catmodel.config.Settings.provider_api_keys", lambda self: {})

    def collect(payload: dict, run: Path, **options) -> int:
        provider = ContextualProvider()
        monkeypatch.setattr(
            "catmodel.llm.factory.build_providers", lambda *_a, **_k: [provider]
        )
        run_contextual_collection(
            instrument=ContextualRiskInstrument.model_validate(payload),
            source_bundle=source,
            run_directory=run,
            profile="test",
            **options,
        )
        return provider.calls

    return source, data, collect


def test_a_replanned_study_sends_only_what_the_table_lacks(tmp_path, study):
    _source, data, collect = study
    table = tmp_path / "answers"
    # Three anchors, each with a direct and a chain question.
    assert collect(data, tmp_path / "first", answers=table) == 6
    assert AnswerTable(table).verify() == 6

    # The same instrument in a new directory sends nothing.
    assert collect(data, tmp_path / "again", answers=table) == 0
    first = (tmp_path / "first" / "contextual-risk.json").read_bytes()
    assert (tmp_path / "again" / "contextual-risk.json").read_bytes() == first

    # A second sample changes the instrument and its digest, which a
    # collection directory cannot absorb; only the new sample is asked.
    grown = {
        **data,
        "respondents": [
            *data["respondents"],
            {
                "respondent_id": "fixture-1",
                "model_id": "fixture-model",
                "sample_index": 1,
            },
        ],
    }
    assert collect(grown, tmp_path / "grown", answers=table) == 6
    assert AnswerTable(table).verify() == 12

    # A reworded control changes every prompt, so everything is asked again.
    reworded = {**data, "fixed_controls": "Reworded synthetic controls"}
    assert collect(reworded, tmp_path / "reworded", answers=table) == 6


def test_a_planned_collection_reports_what_the_table_answers(tmp_path, study):
    source, data, collect = study
    table = tmp_path / "answers"
    assert collect(data, tmp_path / "planned", dry_run=True) == 0
    assert AnswerTable(table).coverage(tmp_path / "planned") == (0, 6)

    # Collected without a table, then added to it.
    assert collect(data, tmp_path / "collected") == 6
    runner = CliRunner()
    added = runner.invoke(
        cli, ["answers", "add", "--answers", str(table), str(tmp_path / "collected")]
    )
    assert added.exit_code == 0, added.output
    assert json.loads(added.output) == {
        "added": 6,
        "already_present": 0,
        "collections": 1,
        "responses": 6,
    }
    assert AnswerTable(table).coverage(tmp_path / "planned") == (6, 6)
    # A complete collection resumed with a table adds what it stores.
    other = tmp_path / "other-table"
    assert collect(data, tmp_path / "collected", answers=other) == 0
    assert AnswerTable(other).verify() == 6

    def dry_run(payload: dict, name: str, *options: str) -> str:
        """The last line the collect command prints for a dry run of
        ``payload`` in the run directory ``name``."""
        instrument = tmp_path / f"{name}.json"
        instrument.write_text(json.dumps(payload))
        result = runner.invoke(
            cli,
            [
                "capability-risk",
                "collect",
                "--instrument",
                str(instrument),
                "--source-bundle",
                str(source),
                "--run-directory",
                str(tmp_path / name),
                "--dry-run",
                *options,
            ],
        )
        assert result.exit_code == 0, result.output
        return result.stdout.splitlines()[-1]

    # A dry run with a table ends by saying how many of its frozen requests
    # the table holds, which a collection with the table would not send: all
    # of the collected instrument's, and half of one with a second sample.
    answers = ("--answers", str(table))
    assert dry_run(data, "same", *answers) == (
        "Validated frozen collection plan; the answer table holds 6 of 6 "
        "frozen requests"
    )
    second = {
        "respondent_id": "fixture-1",
        "model_id": "fixture-model",
        "sample_index": 1,
    }
    grown = {**data, "respondents": [*data["respondents"], second]}
    assert dry_run(grown, "grown", *answers) == (
        "Validated frozen collection plan; the answer table holds 6 of 12 "
        "frozen requests"
    )
    # A new table holds none of them, and a dry run without one counts nothing.
    assert dry_run(grown, "grown", "--answers", str(tmp_path / "new-table")) == (
        "Validated frozen collection plan; the answer table holds 0 of 12 "
        "frozen requests"
    )
    assert dry_run(grown, "grown") == "Validated frozen collection plan"
    assert AnswerTable(table).verify() == 6

    not_a_table = runner.invoke(
        cli,
        [
            "answers",
            "add",
            "--answers",
            str(tmp_path / "collected"),
            str(tmp_path / "collected"),
        ],
    )
    assert not_a_table.exit_code != 0
    assert "is not an answer table" in not_a_table.output


def test_a_planned_study_reports_what_the_table_answers(tmp_path, monkeypatch):
    """A dry run of a study with a table counts the frozen requests of every
    capture directory the plan names."""
    folder, bundle, _ = _study(tmp_path, monkeypatch)
    plan = folder / "public-study.json"
    table = tmp_path / "answers"
    captures = [capture for _, capture in public_study_collections(plan)]

    def frozen(directories) -> int:
        return sum(
            len(list(capture.glob("calls/*/request.json"))) for capture in directories
        )

    def dry_run() -> str:
        result = CliRunner().invoke(
            cli,
            [
                "--profile",
                "test",
                "capability-risk",
                "collect",
                "--study",
                str(plan),
                "--source-bundle",
                str(bundle),
                "--dry-run",
                "--answers",
                str(table),
            ],
        )
        assert result.exit_code == 0, result.output
        return result.stdout.splitlines()[-1]

    # Each of the eight contexts freezes one respondent's direct and chain
    # question.
    assert dry_run() == (
        "Validated frozen collection plans; the answer table holds 0 of 16 "
        "frozen requests"
    )
    assert frozen(captures) == 16
    # One trajectory node's two contexts are collected into the table.
    first = strict_json_loads(plan.read_bytes())["contexts"][0]["draw_id"]
    collected = [capture for _, capture in public_study_collections(plan, [first])]
    assert len(collected) == 2
    collect_study_nodes(
        plan, [first], source_bundle=bundle, profile="test", answers=table
    )
    held, total = frozen(collected), frozen(captures)
    assert 4 <= held < total
    assert dry_run() == (
        f"Validated frozen collection plans; the answer table holds {held} of "
        f"{total} frozen requests"
    )
