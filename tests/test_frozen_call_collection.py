"""One frozen provider call fails closed, keeps its retry budget and resumes."""

import asyncio
import errno
import hashlib
import json
import os
from pathlib import Path
from typing import cast

import litellm
import pytest
from pydantic import BaseModel

from catmodel.artifacts import remove_temporary_files
from catmodel.collection.frozen_call_collection import (
    ProviderFailed,
    ValidationAttemptsExhausted,
    collect_frozen_call,
    failed_calls,
    provider_failure_report,
)
from catmodel.config import ExecutionSettings
from catmodel.frozen_calls import (
    PARSE_ATTEMPTS,
    PROVIDER_FAILURES_PER_CALL,
    ResponseStopped,
    frozen_call_key,
    replay_call,
    retry_prompt,
    unfenced,
    validation_feedback,
)
from catmodel.json_io import strict_json_loads
from catmodel.llm import litellm_provider
from catmodel.llm.litellm_provider import LiteLLMProvider
from catmodel.llm.transport import LLMProvider, ProviderError, ProviderResponse

_REQUEST = {
    "call_id": "fixture-call",
    "respondent": {
        "respondent_id": "fixture",
        "model_id": "fixture",
        "sample_index": 0,
    },
    "prompt_sha256": "a" * 64,
    "system_prompt": "fixture",
    "user_prompt": "fixture",
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


class Provider:
    provider_name = "fixture"
    model_id = "fixture"
    reasoning_effort = "low"

    def __init__(self, text: str | None):
        self.text = text
        self.calls = 0

    async def complete(self, **_kwargs):
        self.calls += 1
        if self.text is None:
            raise RuntimeError("transport unavailable")
        return ProviderResponse(text=self.text, model_id=self.model_id)


def _collect(tmp_path, provider, parser=parse, request=_REQUEST):
    return asyncio.run(
        collect_frozen_call(
            run_directory=tmp_path,
            request=request,
            provider=cast(LLMProvider, provider),
            config=_CONFIG,
            parse_answer=parser,
            response_model=Answer,
        )
    )


def test_valid_answer_is_collected_once_and_replayed(tmp_path):
    provider = Provider('{"value": 3}')
    assert _collect(tmp_path, provider).value == 3
    assert _collect(tmp_path, provider).value == 3
    assert provider.calls == 1


def test_a_call_is_written_under_the_key_replay_reads_it_by(tmp_path):
    request = {
        **_REQUEST,
        "call_id": "appel-\u00e9-\u547c",
        "respondent": {
            "respondent_id": "r\u00e9pondant-\u03b2",
            "model_id": "fixture",
            "sample_index": 3,
        },
    }
    assert _collect(tmp_path, Provider('{"value": 3}'), request=request).value == 3
    key = frozen_call_key(request)
    # The call ID and the respondent's identity, as compact JSON in UTF-8.
    identity = '["appel-\u00e9-\u547c",["r\u00e9pondant-\u03b2","fixture",3]]'
    assert key == hashlib.sha256(identity.encode()).hexdigest()
    assert [path.name for path in (tmp_path / "calls").iterdir()] == [key]
    assert [path.name for path in (tmp_path / "acquisition").iterdir()] == [
        f"{key}.json"
    ]
    replayed = replay_call(request=request, files=_files(tmp_path), parse_answer=parse)
    assert replayed.answer == Answer(value=3)


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"value": 3}\n```',
        '```\n{"value": 3}\n```',
        '\n ```JSON \r\n{"value": 3}\r\n```\n',
    ],
)
def test_an_answer_inside_one_code_fence_is_read_without_it(tmp_path, text):
    assert unfenced(text) == '{"value": 3}'
    provider = Provider(text)
    assert _collect(tmp_path, provider).value == 3
    assert provider.calls == 1
    # The response is stored as the provider gave it, and replays.
    (row,) = _rows(tmp_path)
    assert row["raw_response"] == text and row["parse_success"]
    replayed = replay_call(request=_REQUEST, files=_files(tmp_path), parse_answer=parse)
    assert replayed.answer == Answer(value=3)


@pytest.mark.parametrize(
    "text",
    [
        '{"value": 3}',
        'My answer:\n```json\n{"value": 3}\n```',
        '```json\n{"value": 3}\n```\nThat is my answer.',
        '```json\n{"value": 3}```',
        '```json {"value": 3}\n```',
    ],
)
def test_only_a_fence_around_the_whole_response_is_removed(text):
    assert unfenced(text) == text


def test_a_fence_inside_the_answer_stays_in_it():
    # Two fenced answers are not one answer, and a fence quoted in a string
    # is part of that string.
    twice = '```json\n{"value": 3}\n```\n```json\n{"value": 4}\n```'
    assert unfenced(twice) == '{"value": 3}\n```\n```json\n{"value": 4}'
    with pytest.raises(ValueError):
        parse(unfenced(twice))
    quoted = '```\n{"text": "a\\n```\\nb"}\n```'
    assert unfenced(quoted) == '{"text": "a\\n```\\nb"}'


def test_a_provider_failure_is_frozen_and_sent_again_only_on_resume(tmp_path):
    provider = _Scripted(
        ProviderError("submission failed 5 times: 502", transient=True),
        ProviderResponse(text='{"value": 3}', model_id="fixture"),
    )
    with pytest.raises(ProviderFailed, match="502") as caught:
        _collect(tmp_path, provider)
    assert caught.value.transient and provider.calls == 1
    # Resume sends it again as a new attempt with the original prompt.
    assert _collect(tmp_path, provider).value == 3
    failed, answered = _rows(tmp_path)
    assert (failed["attempt"], answered["attempt"]) == (1, 2)
    assert failed["raw_response"] is None and "502" in failed["error"]
    assert (
        failed["rendered_user_prompt"]
        == answered["rendered_user_prompt"]
        == _REQUEST["user_prompt"]
    )
    # The capture replays: a provider failure, then an answer.
    replayed = replay_call(request=_REQUEST, files=_files(tmp_path), parse_answer=parse)
    assert replayed.answer == Answer(value=3)
    assert (replayed.responses, replayed.provider_failures) == (1, 1)
    assert _collect(tmp_path, provider).value == 3 and provider.calls == 2


def test_a_systematic_provider_failure_is_not_transient(tmp_path):
    provider = _Scripted(ProviderError("AuthenticationError: 401 invalid key"))
    with pytest.raises(ProviderFailed, match="401") as caught:
        _collect(tmp_path, provider)
    assert not caught.value.transient


def test_validation_retry_budget_does_not_reset_on_resume(tmp_path):
    provider = Provider('{"value": "not a number"}')
    with pytest.raises(ValueError, match="Collection incomplete"):
        _collect(tmp_path, provider)
    attempts = provider.calls
    assert attempts > 1
    with pytest.raises(ValueError, match="exhausted"):
        _collect(tmp_path, provider)
    assert provider.calls == attempts


class _FailsLast(Provider):
    """Invalid answers, then a provider failure on the third attempt, then
    ``outcome``."""

    outcome = None

    async def complete(self, **_kwargs):
        self.calls += 1
        if self.calls == 3:
            raise ProviderError("transport unavailable", transient=True)
        if self.outcome is not None:
            return self.outcome
        return ProviderResponse(text='{"value": "x"}', model_id=self.model_id)


def test_a_provider_failure_on_the_last_attempt_uses_no_validation_attempt(
    tmp_path,
):
    provider = _FailsLast(None)
    with pytest.raises(ProviderFailed, match="transport unavailable"):
        _collect(tmp_path, provider)
    assert provider.calls == 3
    # Two validation attempts are used; the failed third is sent again with
    # the retry its attempt used, built from the second answer's feedback.
    provider.outcome = ProviderResponse(text='{"value": 3}', model_id="fixture")
    assert _collect(tmp_path, provider).value == 3
    _, second, failed, answered = _rows(tmp_path)
    assert answered["attempt"] == 4
    expected = retry_prompt(_REQUEST, 3, second["error"])
    assert failed["rendered_user_prompt"] == answered["rendered_user_prompt"]
    assert answered["rendered_user_prompt"] == expected
    assert "transport unavailable" not in answered["rendered_user_prompt"]


def test_a_resent_retry_keeps_the_last_validation_feedback(tmp_path):
    provider = _Scripted(
        ProviderResponse(text="not json", model_id="fixture"),
        ProviderError("stream cut 3 times", transient=True),
    )
    with pytest.raises(ProviderFailed):
        _collect(tmp_path, provider)
    provider.outcomes.append(ProviderResponse(text='{"value": 3}', model_id="fixture"))
    assert _collect(tmp_path, provider).value == 3
    first, failed, answered = _rows(tmp_path)
    expected = retry_prompt(_REQUEST, 2, first["error"])
    assert failed["rendered_user_prompt"] == expected
    assert answered["rendered_user_prompt"] == expected
    # A retry built from the provider error instead does not replay.
    files = _files(tmp_path)
    path = next(name for name in files if name.endswith("attempt-003.json"))
    tampered = {
        **answered,
        "rendered_user_prompt": retry_prompt(_REQUEST, 2, failed["error"]),
    }
    with pytest.raises(ValueError, match="actual prompt differs"):
        replay_call(
            request=_REQUEST,
            files={**files, path: json.dumps(tampered).encode()},
            parse_answer=parse,
        )


def test_resume_ignores_the_temporary_files_an_interrupted_write_left(tmp_path):
    provider = _Scripted(
        ProviderError("submission failed 5 times: 502", transient=True),
        ProviderResponse(text='{"value": 3}', model_id="fixture"),
    )
    with pytest.raises(ProviderFailed):
        _collect(tmp_path, provider)
    (call,) = (tmp_path / "calls").iterdir()
    first = call / "attempt-001.json"
    # A crash while writing the second attempt leaves a truncated temporary
    # file and no record; one after publishing the first, but before removing
    # its temporary name, leaves a whole copy.
    leftovers = [
        call / ".attempt-002.json.0123456789abcdef.tmp",
        call / ".attempt-001.json.fedcba9876543210.tmp",
    ]
    leftovers[0].write_bytes(first.read_bytes()[:40])
    leftovers[1].write_bytes(first.read_bytes())
    assert failed_calls(tmp_path) == []
    assert _collect(tmp_path, provider).value == 3
    assert provider.calls == 2
    assert [row["attempt"] for row in _rows(tmp_path)] == [1, 2]
    replayed = replay_call(request=_REQUEST, files=_files(tmp_path), parse_answer=parse)
    assert replayed.answer == Answer(value=3)
    remove_temporary_files(tmp_path)
    assert not any(path.exists() for path in leftovers)


def test_a_record_that_fails_to_publish_is_absent_and_sent_again(tmp_path, monkeypatch):
    link = os.link

    def failing(source, target):
        if Path(target).name == "attempt-001.json":
            raise OSError(errno.ENOSPC, "no space left on device")
        link(source, target)

    monkeypatch.setattr(os, "link", failing)
    provider = Provider('{"value": 3}')
    with pytest.raises(OSError, match="no space left"):
        _collect(tmp_path, provider)
    (call,) = (tmp_path / "calls").iterdir()
    assert sorted(path.name for path in call.iterdir()) == ["request.json"]
    monkeypatch.setattr(os, "link", link)
    assert _collect(tmp_path, provider).value == 3
    assert provider.calls == 2
    assert [row["attempt"] for row in _rows(tmp_path)] == [1]


def test_provider_failures_are_bounded_per_call(tmp_path):
    provider = Provider(None)
    for _ in range(PROVIDER_FAILURES_PER_CALL):
        with pytest.raises(ProviderFailed):
            _collect(tmp_path, provider)
    assert provider.calls == PROVIDER_FAILURES_PER_CALL
    (key,) = failed_calls(tmp_path)
    # The run's report names it as needing a new run, not as sent again.
    report = provider_failure_report(tmp_path, [key])
    assert f"need a new run: {key}" in report and "sends" not in report
    # A call with its last allowed failure is refused before sending.
    with pytest.raises(ValueError, match="most one call may store"):
        _collect(tmp_path, provider)
    assert provider.calls == PROVIDER_FAILURES_PER_CALL
    replayed = replay_call(request=_REQUEST, files=_files(tmp_path), parse_answer=parse)
    assert replayed.failed and replayed.answer is None and not replayed.exhausted
    # Nothing may follow the last allowed failure.
    files = _files(tmp_path)
    last = json.loads(
        next(data for name, data in files.items() if name.endswith("attempt-003.json"))
    )
    path = next(name for name in files if name.endswith("attempt-003.json"))
    extra = {**last, "attempt": 4}
    with pytest.raises(ValueError, match="cannot have later attempts"):
        replay_call(
            request=_REQUEST,
            files={**files, path.replace("003", "004"): json.dumps(extra).encode()},
            parse_answer=parse,
        )


def test_provider_failures_use_no_validation_attempts(tmp_path):
    provider = _Scripted(
        ProviderError("down", transient=True),
        ProviderError("down", transient=True),
        *[ProviderResponse(text="not json", model_id="fixture")] * PARSE_ATTEMPTS,
    )
    for _ in range(2):
        with pytest.raises(ProviderFailed):
            _collect(tmp_path, provider)
    with pytest.raises(ValidationAttemptsExhausted):
        _collect(tmp_path, provider)
    rows = _rows(tmp_path)
    assert len(rows) == 2 + PARSE_ATTEMPTS
    replayed = replay_call(request=_REQUEST, files=_files(tmp_path), parse_answer=parse)
    assert replayed.exhausted and replayed.provider_failures == 2
    with pytest.raises(ValidationAttemptsExhausted):
        _collect(tmp_path, provider)


class _Scripted(Provider):
    """Each call returns the next scripted response or raises the next error."""

    def __init__(self, *outcomes):
        super().__init__(None)
        self.outcomes = list(outcomes)

    async def complete(self, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _rows(tmp_path):
    return [
        json.loads(path.read_bytes())
        for path in sorted(tmp_path.glob("calls/*/attempt-*.json"))
    ]


def _files(tmp_path):
    return {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*.json")
    }


def test_output_limit_and_refusal_are_named_validation_attempts(tmp_path):
    # Even a stopped text that would parse is never parsed.
    provider = _Scripted(
        ProviderResponse(
            text='{"value": 1}', model_id="fixture", stop_reason="output_limit"
        ),
        ProviderResponse(text="", model_id="fixture", stop_reason="refusal"),
        ProviderResponse(text='{"value": 3}', model_id="fixture"),
    )
    assert _collect(tmp_path, provider).value == 3
    first, second, third = _rows(tmp_path)
    assert first["stop_reason"] == "output_limit"
    assert first["error"] == validation_feedback(ResponseStopped("output_limit"))
    assert '"type":"output_limit"' in first["error"]
    assert first["raw_response"] == '{"value": 1}' and not first["parse_success"]
    assert second["stop_reason"] == "refusal"
    assert '"type":"refusal"' in second["error"]
    assert first["error"] in second["rendered_user_prompt"]
    assert third["stop_reason"] == "completed" and third["parse_success"]

    # A resumed collection replays the stops without calling again.
    assert _collect(tmp_path, provider).value == 3
    assert provider.calls == 3
    files = _files(tmp_path)
    replayed = replay_call(request=_REQUEST, files=files, parse_answer=parse).answer
    assert isinstance(replayed, Answer) and replayed.value == 3
    first_path = next(name for name in files if name.endswith("attempt-001.json"))
    for tampered in (
        {**first, "stop_reason": "completed"},
        {**first, "stop_reason": None},
    ):
        with pytest.raises(ValueError):
            replay_call(
                request=_REQUEST,
                files={**files, first_path: json.dumps(tampered).encode()},
                parse_answer=parse,
            )


_XAI_SAFETY_CHECK = json.dumps(
    {
        "code": "The caller does not have permission to execute the specified "
        "operation",
        "error": "Content violates usage guidelines. Failed check: "
        "SAFETY_CHECK_TYPE_BIO",
    }
)


@pytest.mark.parametrize(
    ("provider_name", "rejection", "classed"),
    [
        (
            "anthropic",
            lambda: litellm.ContentPolicyViolationError(
                "Output blocked by content filtering policy", "fixture", "anthropic"
            ),
            "ContentPolicyViolationError",
        ),
        # xAI's safety checks, as reported: a 403 naming its usage guidelines.
        (
            "xai",
            lambda: litellm.APIError(
                403, f"XaiException - {_XAI_SAFETY_CHECK}", "xai", "fixture"
            ),
            "APIError",
        ),
    ],
)
def test_a_policy_rejection_is_a_refusal_and_three_exhaust_the_call(
    tmp_path, monkeypatch, provider_name, rejection, classed
):
    # The transport's own provider: the provider rejects the prompt on
    # content-policy grounds each time it is sent.
    rejections = []

    async def rejected(**_kwargs):
        rejections.append(None)
        raise rejection()

    monkeypatch.setattr(litellm_provider, "acompletion", rejected)
    provider = LiteLLMProvider(
        api_key="fixture-key",
        provider_name=provider_name,
        model="fixture",
        max_concurrent=1,
        reasoning_effort="low",
    )
    with pytest.raises(ValidationAttemptsExhausted):
        _collect(tmp_path, provider)
    rows = _rows(tmp_path)
    assert len(rows) == len(rejections) == PARSE_ATTEMPTS
    refusal = validation_feedback(ResponseStopped("refusal"))
    for row in rows:
        assert (row["raw_response"], row["stop_reason"]) == ("", "refusal")
        assert row["error"] == refusal and not row["parse_success"]
        record = row["provider_evidence"]["policy_rejection"]
        assert record["error"].startswith(classed)
    # Each retry carries the refusal as its feedback, and the capture replays
    # as an exhausted call that a resumed collection does not send again.
    assert all(refusal in row["rendered_user_prompt"] for row in rows[1:])
    replayed = replay_call(request=_REQUEST, files=_files(tmp_path), parse_answer=parse)
    assert replayed.exhausted and replayed.provider_failures == 0
    with pytest.raises(ValidationAttemptsExhausted):
        _collect(tmp_path, provider)
    assert len(rejections) == PARSE_ATTEMPTS


def test_lost_streams_are_recorded_beside_the_attempt_not_as_attempts(tmp_path):
    evidence = {
        "mechanism": "stream",
        "connection_lost": [{"elapsed_ms": 1200, "chunks": 4, "error": "cut"}],
        "response": {"choices": []},
    }
    provider = _Scripted(
        ProviderResponse(text='{"value": 3}', model_id="fixture", evidence=evidence)
    )
    assert _collect(tmp_path, provider).value == 3
    [row] = _rows(tmp_path)
    assert row["attempt"] == 1
    assert row["provider_evidence"] == evidence


def test_a_failed_request_keeps_its_evidence(tmp_path):
    evidence = {"mechanism": "stream", "connection_lost": [{"error": "cut"}] * 3}
    provider = _Scripted(ProviderError("stream cut 3 times", evidence=evidence))
    with pytest.raises(ValueError, match="stream cut 3 times"):
        _collect(tmp_path, provider)
    [row] = _rows(tmp_path)
    assert row["raw_response"] is None and row["stop_reason"] is None
    assert row["provider_evidence"] == evidence


def test_replay_requires_a_stop_reason_exactly_on_responses(tmp_path):
    def tampered(directory, attempt, **fields):
        files = _files(directory)
        path = next(
            name for name in files if name.endswith(f"attempt-00{attempt}.json")
        )
        row = {**json.loads(files[path]), **fields}
        return {**files, path: json.dumps(row).encode()}

    # A parse failure, then an answer.
    answered = tmp_path / "answered"
    provider = _Scripted(
        ProviderResponse(text="not json", model_id="fixture"),
        ProviderResponse(text='{"value": 3}', model_id="fixture"),
    )
    assert _collect(answered, provider).value == 3
    for stop_reason in (None, 7):
        with pytest.raises(ValueError, match="response and stop reason differ"):
            replay_call(
                request=_REQUEST,
                files=tampered(answered, 1, stop_reason=stop_reason),
                parse_answer=parse,
            )

    # A provider failure has no response, so no stop reason.
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="unavailable"):
        _collect(failed, _Scripted(ProviderError("unavailable")))
    assert (
        replay_call(request=_REQUEST, files=_files(failed), parse_answer=parse).answer
        is None
    )
    with pytest.raises(ValueError, match="response and stop reason differ"):
        replay_call(
            request=_REQUEST,
            files=tampered(failed, 1, stop_reason="completed"),
            parse_answer=parse,
        )


def strict_parse(text: str) -> Answer:
    """Decodes an answer as the package's parsers do."""
    return Answer.model_validate(strict_json_loads(text))


# More levels than any interpreter's JSON decoder recurses through.
_OVER_DEEP = "[" * 200_000 + "]" * 200_000


def test_an_over_deep_response_is_a_failed_attempt_that_replays_and_resumes(tmp_path):
    provider = _Scripted(
        ProviderResponse(text=_OVER_DEEP, model_id="fixture"),
        ProviderResponse(text='{"value": 3}', model_id="fixture"),
    )
    assert _collect(tmp_path, provider, strict_parse).value == 3
    first, second = _rows(tmp_path)
    assert first["raw_response"] == _OVER_DEEP and not first["parse_success"]
    # A value error on every interpreter, never the decoder's RecursionError,
    # which strikes at a depth the Python build and the caller's stack decide.
    assert first["error"] == validation_feedback(ValueError())
    assert second["rendered_user_prompt"] == retry_prompt(_REQUEST, 2, first["error"])
    # Replay, as validation and import run it, re-derives both attempts.
    replayed = replay_call(
        request=_REQUEST, files=_files(tmp_path), parse_answer=strict_parse
    )
    assert replayed.answer == Answer(value=3) and replayed.responses == 2
    # So does resume, which sends nothing more.
    assert _collect(tmp_path, provider, strict_parse).value == 3
    assert provider.calls == 2


def keyed_parse(text: str) -> Answer:
    """A parser with a defect: an answer without ``value`` raises KeyError."""
    return Answer(value=json.loads(text)["value"])


def test_a_parser_defect_stops_the_call_and_freezes_no_attempt(tmp_path):
    provider = _Scripted(
        ProviderResponse(text="{}", model_id="fixture"),
        ProviderResponse(text="not json", model_id="fixture"),
        ProviderResponse(text='{"value": 3}', model_id="fixture"),
    )
    # Neither a provider failure nor a failed validation attempt, which replay
    # could not re-derive: the parser's error is raised as itself.
    with pytest.raises(KeyError):
        _collect(tmp_path, provider, keyed_parse)
    assert _rows(tmp_path) == [] and provider.calls == 1
    replayed = replay_call(
        request=_REQUEST, files=_files(tmp_path), parse_answer=keyed_parse
    )
    assert (replayed.responses, replayed.provider_failures) == (0, 0)
    # A rerun sends the call again from its first attempt, and a malformed
    # answer is still a failed validation attempt.
    assert _collect(tmp_path, provider, keyed_parse).value == 3
    first, second = _rows(tmp_path)
    assert first["attempt"] == 1
    assert first["rendered_user_prompt"] == _REQUEST["user_prompt"]
    assert first["raw_response"] == "not json" and not first["parse_success"]
    assert second["parse_success"]


class Other(BaseModel):
    value: int


def other_parse(text: str) -> BaseModel:
    """A parser wired to another response model."""
    return Other.model_validate_json(text)


def test_an_answer_of_another_model_is_a_defect_not_a_failed_attempt(tmp_path):
    provider = Provider('{"value": 3}')
    with pytest.raises(TypeError, match="declared response schema"):
        _collect(tmp_path, provider, other_parse)
    assert _rows(tmp_path) == [] and provider.calls == 1
