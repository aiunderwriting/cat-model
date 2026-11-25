"""Collect one frozen provider call with strict parsing, retries and replay."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel

from catmodel.artifacts import freeze_json
from catmodel.collection.answer_table import answer_key
from catmodel.frozen_calls import (
    FROZEN_CALL_SCHEMA_VERSION,
    PARSE_ATTEMPTS,
    PROVIDER_FAILURES_PER_CALL,
    AcquisitionTreatment,
    frozen_call_key,
    parser_protocol,
    replay_call,
    runtime_versions,
    validated_response,
    validation_feedback,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import strict_json_loads
from catmodel.llm import transport

if TYPE_CHECKING:
    from catmodel.collection.answer_table import AnswerKey, AnswerTable
    from catmodel.config import ExecutionSettings
    from catmodel.llm.transport import LLMProvider, ProviderResponse

logger = structlog.get_logger(__name__)


class ValidationAttemptsExhausted(ValueError):
    """Every validation attempt of one frozen call failed; no transport failure."""


class ProviderFailed(ValueError):
    """One frozen call's request failed at the provider; its attempt is frozen
    with no response and a resumed collection sends it again
    (``PROVIDER_FAILURES_PER_CALL``). ``transient`` is the transport's
    classification (``transport.is_transient``): a transient failure fails
    only the call, a systematic one stops the study."""

    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient

    def __reduce__(self):
        # Pickled whole, so a collector process can hand it to its parent.
        return partial(ProviderFailed, transient=self.transient), (str(self),)


def provider_failure_report(run_directory: Path, failed: list[str]) -> str:
    """``failed`` call keys of a run, split into those a rerun sends again and
    those that stored their last allowed provider failure."""
    final = set(failed_calls(run_directory))
    again = sorted(set(failed) - final)
    report = f"{len(failed)} calls failed at the provider"
    if again:
        report += f"; a rerun sends {len(again)} again: " + ", ".join(again)
    if final & set(failed):
        report += (
            f"; {len(final & set(failed))} stored their last allowed failure and "
            "need a new run: " + ", ".join(sorted(final & set(failed)))
        )
    return report


def failed_calls(run_directory: Path) -> list[str]:
    """The stored calls of a run that failed at the provider
    ``PROVIDER_FAILURES_PER_CALL`` times: none can complete in this run."""
    calls = run_directory / "calls"
    return [
        call.name
        for call in (sorted(calls.iterdir()) if calls.is_dir() else ())
        if sum(
            strict_json_loads(row.read_bytes()).get("raw_response") is None
            for row in call.glob("attempt-*.json")
        )
        >= PROVIDER_FAILURES_PER_CALL
    ]


def _record(
    answers: AnswerTable,
    key: AnswerKey,
    request: dict[str, Any],
    user_prompt: str,
    response: ProviderResponse,
) -> None:
    """Add a provider's response to the answer table. A table that cannot
    take it never costs the collection the response: the attempt record keeps
    it, and ``AnswerTable.add_collection`` adds it later."""
    try:
        answers.record(
            key,
            system_prompt=request["system_prompt"],
            user_prompt=user_prompt,
            response=response,
        )
    except (OSError, ValueError) as error:
        logger.warning(
            "could not add a response to the answer table; its attempt record keeps it",
            call=request["call_id"],
            error=str(error),
        )


async def collect_frozen_call(
    *,
    run_directory: Path,
    request: dict[str, Any],
    provider: LLMProvider,
    config: ExecutionSettings,
    parse_answer: Callable[[str], BaseModel],
    response_model: type[BaseModel],
    answers: AnswerTable | None = None,
) -> BaseModel:
    """Collect one frozen provider call, resuming from its stored attempts.

    The caller owns the exact prompt contract and parser. A retry supplies validator
    feedback only; it never repairs or changes an answer. A response is a
    failed validation attempt exactly when replay finds it one
    (``validated_response``): the provider stopped it at its output limit or
    by refusing, which names the attempt for its stop, or the parser raised
    ``ValueError`` or ``TypeError``. Any other exception from the parser, and a
    parser answer that is not a ``response_model``, is a defect: it is raised
    and its response is not frozen, so a resumed collection sends the call
    again. A request that fails at the provider is frozen as an
    attempt with no response and raises ``ProviderFailed``; a resumed
    collection sends it again as a new attempt with the same prompt, the
    original or the retry built from the last validation feedback, using no
    validation attempt. A call that has stored ``PROVIDER_FAILURES_PER_CALL``
    failures raises at once without sending.

    Each record is frozen whole or not at all (``freeze_json``), so an
    interrupted write loses at most the attempt it was storing, which resume
    sends again. Resume requires the call's acquisition record unchanged:
    the same provider settings, request and parser protocol
    (``parser_protocol``); each attempt also records the ``runtime_versions``
    it ran under.

    With an answer table (``answers``), each attempt first looks for a
    response the table already holds to its exact prompt, under this
    provider's settings and this sample: a response found is used as if the
    provider had just returned it, sends nothing and is marked
    ``reused_from_answer_table`` in its attempt record. Every response a
    provider does return is added to the table before it is parsed, so a
    parser defect loses no response, and so are the responses of the call's
    stored attempts. Parsing and validation never read the table: they run on
    the raw response, as they do for a live one.
    """
    respondent = request["respondent"]
    if provider.model_id != respondent["model_id"]:
        raise ValueError("request and provider identities differ")
    key = frozen_call_key(request)
    path = run_directory / "calls" / key
    if (
        (run_directory / "calls").is_symlink()
        or path.is_symlink()
        or not path.resolve().is_relative_to(run_directory.resolve())
        or (path / "request.json").is_symlink()
    ):
        raise ValueError(
            "frozen call paths cannot follow symbolic links or escape their run"
        )
    path.mkdir(parents=True, exist_ok=True)
    freeze_json(path / "request.json", request)
    rows = sorted(path.glob("attempt-*.json"))
    if any(row.is_symlink() for row in rows):
        raise ValueError("frozen response records cannot follow symbolic links")
    if len(rows) > PARSE_ATTEMPTS + PROVIDER_FAILURES_PER_CALL:
        raise ValueError("stored request exceeds its validation retry budget")
    acquisition_path = run_directory / "acquisition" / f"{key}.json"
    if rows and not acquisition_path.exists():
        raise ValueError("stored call has no acquisition treatment; start a new run")
    protocol = parser_protocol(parse_answer, response_model)
    protocol_sha256 = canonical_json_sha256(protocol.model_dump(mode="json"))
    treatment = AcquisitionTreatment(
        model_id=provider.model_id,
        provider_name=provider.provider_name,
        reasoning_effort=provider.reasoning_effort,
        max_output_tokens=config.max_output_tokens_by_model[provider.model_id],
        request_timeout_seconds=config.request_timeout_seconds,
        request_sha256=canonical_json_sha256(request),
        parser_protocol_sha256=protocol_sha256,
        retry_protocol=FROZEN_CALL_SCHEMA_VERSION,
        parse_attempts=PARSE_ATTEMPTS,
    )
    protocol_path = run_directory / "parser-protocols" / f"{protocol_sha256}.json"
    for frozen_path in (acquisition_path, protocol_path):
        if (
            frozen_path.is_symlink()
            or frozen_path.parent.is_symlink()
            or not frozen_path.resolve().is_relative_to(run_directory.resolve())
        ):
            raise ValueError(
                "frozen acquisition paths cannot follow symlinks or escape their run"
            )
    acquisition_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_json(acquisition_path, treatment.model_dump(mode="json"))
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_json(
        protocol_path,
        protocol.model_dump(mode="json"),
    )

    def declared(answer: BaseModel) -> BaseModel:
        if not isinstance(answer, response_model):
            raise TypeError("parser answer differs from its declared response schema")
        return answer

    captured_files = {
        str(row_path.relative_to(run_directory)): row_path.read_bytes()
        for row_path in (path / "request.json", *rows)
    }
    stored = replay_call(
        request=request, files=captured_files, parse_answer=parse_answer
    )
    # The prompts this call's responses answered, in order: the table keys
    # a repeated prompt's responses by how many came before (``AnswerKey``).
    asked: list[str] = []
    if answers is not None:
        attempts = [
            strict_json_loads(captured_files[str(row.relative_to(run_directory))])
            for row in rows
        ]
        answers.add_call(
            request=request,
            treatment=treatment.model_dump(mode="json"),
            attempts=attempts,
        )
        asked = [
            row["rendered_user_prompt"]
            for row in attempts
            if row.get("raw_response") is not None
        ]
        del attempts
    # A call can wait a long time for its lane and then for its answer;
    # thousands wait at once, so none keeps its stored records' bytes.
    del captured_files
    if stored.answer is not None:
        return declared(stored.answer)
    if stored.exhausted:
        raise ValidationAttemptsExhausted(
            "call exhausted validation attempts; start a new run"
        )
    if stored.failed:
        raise ValueError(
            f"call failed at the provider {stored.provider_failures} times, the "
            "most one call may store; start a new run"
        )
    attempt = len(rows)
    while not stored.exhausted:
        attempt += 1
        user_prompt = stored.next_prompt(request)
        response = None
        answer = None
        error = None
        diagnostic_error = None
        failure_evidence = None
        failure = None
        key = None
        if answers is not None:
            key = answer_key(
                provider,
                max_output_tokens=treatment.max_output_tokens,
                sample_index=respondent["sample_index"],
                system_prompt=request["system_prompt"],
                user_prompt=user_prompt,
                draw=asked.count(user_prompt),
            )
            response = answers.lookup(key)
        reused = response is not None
        if response is None:
            try:
                response = await transport.complete(
                    provider,
                    system_prompt=request["system_prompt"],
                    user_prompt=user_prompt,
                    max_tokens=treatment.max_output_tokens,
                    request_timeout_seconds=config.request_timeout_seconds,
                )
            except Exception as exc:
                # Whatever the request raised, it left no response.
                failure = exc
                error = diagnostic_error = str(exc)
                failure_evidence = getattr(exc, "evidence", None)
            else:
                if answers is not None and key is not None:
                    _record(answers, key, request, user_prompt, response)
        if response is not None:
            asked.append(user_prompt)
            answer, invalid = validated_response(
                response.text, response.stop_reason, parse_answer
            )
            if answer is not None:
                answer = declared(answer)
            if invalid is not None:
                diagnostic_error = str(invalid)
                error = validation_feedback(invalid)
        row = {
            "schema_version": FROZEN_CALL_SCHEMA_VERSION,
            "call_id": request["call_id"],
            "model_id": provider.model_id,
            "respondent_id": respondent["respondent_id"],
            "sample_index": respondent["sample_index"],
            "answer": None if answer is None else answer.model_dump(mode="json"),
            "parse_success": answer is not None,
            "error": error,
            "diagnostic_error": diagnostic_error,
            "attempt": attempt,
            "raw_response": response.text if response else None,
            "rendered_system_prompt": request["system_prompt"],
            "rendered_user_prompt": user_prompt,
            "prompt_sha256": request["prompt_sha256"],
            "provider_name": provider.provider_name,
            "response_model_id": response.model_id if response else None,
            "provider_request_id": response.provider_request_id if response else None,
            "latency_ms": response.latency_ms if response else None,
            "input_tokens": response.input_tokens if response else None,
            "response_tokens": response.output_tokens if response else None,
            "stop_reason": response.stop_reason if response else None,
            "provider_evidence": response.evidence if response else failure_evidence,
            "runtime_versions": runtime_versions(),
        }
        if reused:
            row["reused_from_answer_table"] = True
        freeze_json(path / f"attempt-{attempt:03}.json", row)
        if answer is not None:
            return answer
        if failure is not None:
            raise ProviderFailed(
                f"Collection incomplete: {request['call_id']}: {error}",
                transient=transport.is_transient(failure),
            ) from failure
        stored = replace(stored, responses=stored.responses + 1, feedback=error)
    raise ValidationAttemptsExhausted(
        f"Collection incomplete: {request['call_id']}: {error}"
    )
