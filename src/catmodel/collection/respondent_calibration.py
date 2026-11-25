"""Freeze, collect, replay and fit respondent calibration runs."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from catmodel.artifacts import freeze_json, remove_temporary_files
from catmodel.calibration.respondent_questions import (
    CalibrationQuestion,
    calibration_questions,
    calibration_request,
    parse_calibration_answer,
)
from catmodel.calibration.respondent_spread import (
    calibration_treatments,
    fit_respondent_spread,
)
from catmodel.collection.answer_table import AnswerTable
from catmodel.collection.collection_scheduler import CollectionScheduler
from catmodel.collection.frozen_call_collection import (
    ProviderFailed,
    ValidationAttemptsExhausted,
    collect_frozen_call,
    failed_calls,
    provider_failure_report,
)
from catmodel.contracts.contextual_risk import CONTEXTUAL_SCHEMA_VERSION
from catmodel.contracts.elicitation import Respondent
from catmodel.contracts.respondent_calibration import (
    CalibrationAnswer,
    CalibrationArtifact,
    CalibrationCase,
    CalibrationInstrument,
    CalibrationSeedSet,
    CalibrationTreatments,
    RespondentSpreadFit,
    load_calibration_seeds,
    load_seed_resolutions,
    resolution_script_digests,
)
from catmodel.frozen_calls import (
    FROZEN_CALL_SCHEMA_VERSION,
    PARSE_ATTEMPTS,
    PROVIDER_FAILURES_PER_CALL,
    CaptureFiles,
    frozen_call_key,
    replay_call,
    validate_acquisition_closure,
)
from catmodel.hashing import canonical_json_sha256, sha256_file
from catmodel.json_io import require_marker, strict_json_loads

if TYPE_CHECKING:
    from catmodel.config import ExecutionSettings, ProfilesFile
    from catmodel.llm.models import ModelSpec
    from catmodel.llm.transport import LLMProvider


_INSTRUMENT_FILE = "calibration-instrument.json"
_ARTIFACT_FILE = "calibration-answers.json"
_PLAN_FILE = "collection-plan.json"


def build_calibration_instrument(
    seeds_path: Path | str, respondents: tuple[Respondent, ...]
) -> CalibrationInstrument:
    """Build an immutable instrument without reading any seed resolutions."""
    seed_set = load_calibration_seeds(seeds_path)
    return CalibrationInstrument(
        seeds=seed_set.seeds,
        respondents=respondents,
        latest_training_cutoff=seed_set.latest_training_cutoff,
        resolution_protocol=seed_set.resolution_protocol,
        super_blocks=seed_set.super_blocks,
        curation_sensitive_seed_ids=seed_set.curation_sensitive_seed_ids,
        risk_contract_version=CONTEXTUAL_SCHEMA_VERSION,
    )


def _first(error: BaseException) -> BaseException:
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error


def _model_execution(
    instrument: CalibrationInstrument, profile: str, profiles: Path | None
) -> tuple[list[ModelSpec], ExecutionSettings, ProfilesFile | None]:
    from catmodel.config import load_model_profile, settings

    model_profile = load_model_profile(profile=profile, path=profiles)
    requested = {row.model_id for row in instrument.respondents}
    selected = [row for row in model_profile.models if row.model_id in requested]
    if {row.model_id for row in selected} != requested:
        raise ValueError("instrument models must belong to the selected profile")
    execution = model_profile.execution_settings(
        selected, max_output_tokens=settings.ELICITATION_MAX_TOKENS
    )
    return selected, execution, model_profile.profiles_file


def _requests_and_plan(
    instrument: CalibrationInstrument,
    selected: list[ModelSpec],
    execution: ExecutionSettings,
    profile: str,
    profiles_file: ProfilesFile | None,
) -> tuple[tuple[CalibrationQuestion, ...], list[dict[str, Any]], dict[str, object]]:
    questions = calibration_questions(instrument)
    requests = [calibration_request(question) for question in questions]
    plan: dict[str, object] = {
        "instrument_sha256": instrument.digest(),
        "profile": profile,
        "parse_attempts": PARSE_ATTEMPTS,
        "execution": {
            "models": [row.model_dump(mode="json") for row in selected],
            "settings": execution.model_dump(
                mode="json", exclude={"concurrency_by_model"}
            ),
        },
        "requests": [
            {
                "key": frozen_call_key(request),
                "prompt_sha256": request["prompt_sha256"],
            }
            for request in requests
        ],
    }
    if profiles_file is not None:
        plan["profiles_file"] = profiles_file.model_dump(mode="json")
    return questions, requests, plan


def _freeze_collection(
    *,
    instrument: CalibrationInstrument,
    run_directory: Path,
    requests: list[dict[str, Any]],
    plan: dict[str, object],
) -> None:
    if run_directory.is_symlink() or (
        run_directory.exists()
        and any(path.is_symlink() for path in run_directory.rglob("*"))
    ):
        raise ValueError("calibration capture cannot follow symbolic links")
    remove_temporary_files(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    freeze_json(run_directory / _INSTRUMENT_FILE, instrument.model_dump(mode="json"))
    freeze_json(run_directory / _PLAN_FILE, plan)
    for request in requests:
        path = run_directory / "calls" / frozen_call_key(request) / "request.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        freeze_json(path, request)


async def _collect_requests(
    *,
    questions: tuple[CalibrationQuestion, ...],
    requests: list[dict[str, Any]],
    run_directory: Path,
    providers: list[LLMProvider],
    execution: ExecutionSettings,
    answers: AnswerTable | None = None,
) -> list[CalibrationAnswer | None]:
    provider_by_model = {row.model_id: row for row in providers}
    expected_models = {question.respondent.model_id for question in questions}
    if (
        len(provider_by_model) != len(providers)
        or set(provider_by_model) != expected_models
    ):
        raise ValueError("providers must match the exact calibration model roster")
    scheduler = CollectionScheduler(
        {model: execution.concurrency_by_model[model] for model in expected_models}
    )
    provider_failed: list[str] = []

    async def one(
        index: int, question: CalibrationQuestion, request: dict[str, Any]
    ) -> CalibrationAnswer | None:
        provider = scheduler.provider(
            provider_by_model[question.respondent.model_id], (index,)
        )
        try:
            answer = await collect_frozen_call(
                run_directory=run_directory,
                request=request,
                provider=provider,
                config=execution,
                parse_answer=partial(parse_calibration_answer, question=question),
                response_model=CalibrationAnswer,
                answers=answers,
            )
        except ValidationAttemptsExhausted:
            return None
        except ProviderFailed as exc:
            if not exc.transient:
                raise ValueError(
                    f"Collection incomplete: {question.seed.seed_id}: {exc}"
                ) from exc
            provider_failed.append(frozen_call_key(request))
            return None
        except Exception as exc:
            raise ValueError(
                f"Collection incomplete: {question.seed.seed_id}: {exc}"
            ) from exc
        if not isinstance(answer, CalibrationAnswer):
            raise TypeError("calibration parser returned the wrong response model")
        return answer

    tasks = []
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(one(index, question, request))
                for index, (question, request) in enumerate(
                    zip(questions, requests, strict=True)
                )
            ]
    except BaseExceptionGroup as failure:
        raise _first(failure) from None
    if provider_failed:
        raise ValueError(
            "Collection incomplete: "
            + provider_failure_report(run_directory, provider_failed)
        )
    return [task.result() for task in tasks]


def run_calibration_collection(
    *,
    instrument: CalibrationInstrument,
    run_directory: Path,
    profile: str,
    profiles: Path | None = None,
    dry_run: bool = False,
    answers: Path | None = None,
) -> CalibrationArtifact | None:
    """Freeze all requests, then collect them through the shared frozen-call path.

    A call that fails transiently at the provider, once the transport's own
    retries ran out, stops no other call: the collection raises naming every
    failed call once the rest finish, and a rerun sends each again
    (``PROVIDER_FAILURES_PER_CALL``). A systematic provider failure, such as
    a 400 or 401, stops the collection at once.

    ``profile`` names a profile of the model profiles file ``profiles``
    (``load_model_profile`` gives the default file); the plan records the
    file as a contextual collection's plan does. ``answers`` names an answer
    table that every call reads and feeds (``collect_frozen_call``); a
    collection that was already complete adds its stored responses to it."""
    table = None if answers is None else AnswerTable(answers)
    selected, execution, profiles_file = _model_execution(instrument, profile, profiles)
    questions, requests, plan = _requests_and_plan(
        instrument, selected, execution, profile, profiles_file
    )
    _freeze_collection(
        instrument=instrument,
        run_directory=run_directory,
        requests=requests,
        plan=plan,
    )
    if dry_run:
        return None
    artifact_path = run_directory / _ARTIFACT_FILE
    if artifact_path.exists():
        if table is not None:
            table.add_collection(run_directory)
        return validate_calibration_capture(run_directory)[1]

    failed = failed_calls(run_directory)
    if failed:
        raise ValueError(
            f"{len(failed)} stored calls failed at the provider "
            f"{PROVIDER_FAILURES_PER_CALL} times, the most one call may store; "
            f"start a new calibration run: {', '.join(failed)}"
        )

    from catmodel.config import settings
    from catmodel.llm.factory import build_providers
    from catmodel.llm.transport import with_open_runs_cancelled

    providers = build_providers(
        selected,
        api_keys=settings.provider_api_keys(),
        concurrency_limits=execution.concurrency_by_model,
    )
    collected = asyncio.run(
        with_open_runs_cancelled(
            _collect_requests(
                questions=questions,
                requests=requests,
                run_directory=run_directory,
                providers=providers,
                execution=execution,
                answers=table,
            )
        )
    )
    result = CalibrationArtifact(
        instrument_sha256=instrument.digest(),
        collection_plan_sha256=canonical_json_sha256(plan),
        cases=tuple(
            CalibrationCase(
                seed_id=question.seed.seed_id,
                respondent=question.respondent,
                answer=answer,
            )
            for question, answer in zip(questions, collected, strict=True)
        ),
    )
    freeze_json(artifact_path, result.model_dump(mode="json"))
    return result


def _capture_files(run_directory: Path) -> CaptureFiles:
    if run_directory.is_symlink():
        raise ValueError("calibration capture cannot follow symbolic links")
    root = run_directory.expanduser().resolve()
    files: dict[str, bytes] = {}
    for path in run_directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("calibration capture cannot follow symbolic links")
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("calibration capture path escapes its run directory")
        files[path.relative_to(run_directory).as_posix()] = path.read_bytes()
    return CaptureFiles(files)


def _validate_calibration_capture(
    run_directory: Path,
) -> tuple[
    CalibrationInstrument,
    CalibrationArtifact,
    dict[str, object],
    CalibrationTreatments,
]:
    """Re-render and replay a calibration capture without contacting providers."""
    files = _capture_files(run_directory)
    required = {_INSTRUMENT_FILE, _ARTIFACT_FILE, _PLAN_FILE}
    if not required <= files.keys():
        raise ValueError("calibration capture lacks its instrument, plan or answers")
    saved = {
        name: strict_json_loads(files[name])
        for name in (_INSTRUMENT_FILE, _ARTIFACT_FILE)
    }
    for name, version in ((_INSTRUMENT_FILE, 1), (_ARTIFACT_FILE, 1)):
        require_marker(saved[name], "schema_version", version, f"{name} schema version")
    instrument = CalibrationInstrument.model_validate(saved[_INSTRUMENT_FILE])
    artifact = CalibrationArtifact.model_validate(saved[_ARTIFACT_FILE])
    plan = strict_json_loads(files[_PLAN_FILE])
    if not isinstance(plan, dict):
        raise ValueError("calibration collection plan must be a JSON object")
    if plan.get("instrument_sha256") != instrument.digest():
        raise ValueError("calibration plan names a different instrument")
    questions = calibration_questions(instrument)
    requests = [calibration_request(question) for question in questions]
    expected_plan_requests = [
        {"key": frozen_call_key(request), "prompt_sha256": request["prompt_sha256"]}
        for request in requests
    ]
    if plan.get("requests") != expected_plan_requests:
        raise ValueError("calibration plan requests differ from the frozen instrument")
    plan_digest = canonical_json_sha256(plan)
    if artifact.instrument_sha256 != instrument.digest():
        raise ValueError("calibration artifact names a different instrument")
    if artifact.collection_plan_sha256 != plan_digest:
        raise ValueError("calibration artifact names a different collection plan")
    treatments = calibration_treatments(instrument, plan)
    if treatments.parse_attempts != PARSE_ATTEMPTS:
        raise ValueError("calibration plan uses a different validation retry budget")

    expected_keys = {
        f"calls/{frozen_call_key(request)}/request.json" for request in requests
    }
    actual_request_paths = {
        name
        for name in files
        if name.startswith("calls/") and name.endswith("/request.json")
    }
    if expected_keys != actual_request_paths:
        raise ValueError(
            "calibration frozen request files differ from the request roster"
        )
    expected_call_ids = {frozen_call_key(request) for request in requests}
    actual_call_ids = set()
    for name in files:
        if not name.startswith("calls/"):
            continue
        parts = name.split("/")
        if len(parts) != 3 or not parts[1]:
            raise ValueError("calibration call path has an invalid shape")
        actual_call_ids.add(parts[1])
        filename = parts[2]
        if filename != "request.json" and not (
            filename.startswith("attempt-") and filename.endswith(".json")
        ):
            raise ValueError("calibration calls tree contains an unexpected file")
    if actual_call_ids != expected_call_ids:
        raise ValueError("calibration calls tree differs from the frozen requests")

    acquisition_evidence = validate_acquisition_closure(requests, files)
    execution = plan.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("calibration plan lacks its frozen execution settings")
    model_rows = execution.get("models")
    settings = execution.get("settings")
    if not isinstance(model_rows, list) or not isinstance(settings, dict):
        raise ValueError("calibration plan lacks its frozen execution settings")
    planned_models = {
        row["model_id"]: row
        for row in model_rows
        if isinstance(row, dict) and isinstance(row.get("model_id"), str)
    }
    output_budgets = settings.get("max_output_tokens_by_model")
    if not isinstance(output_budgets, dict):
        raise ValueError("calibration plan lacks per-model output-token budgets")
    timeout = settings.get("request_timeout_seconds")
    for row in acquisition_evidence:
        planned = planned_models.get(row.model_id)
        if planned is None:
            raise ValueError("calibration acquisition model is absent from the plan")
        if (
            row.provider_name != planned.get("provider")
            or row.reasoning_effort != planned.get("reasoning_effort")
            or row.max_output_tokens != output_budgets.get(row.model_id)
            or row.parse_attempts != treatments.parse_attempts
            or row.retry_protocol != FROZEN_CALL_SCHEMA_VERSION
            or row.request_timeout_seconds != timeout
        ):
            raise ValueError(
                "calibration acquisition treatment differs from the frozen plan"
            )
    validated_treatments = calibration_treatments(
        instrument, plan, acquisition_treatments=acquisition_evidence
    )

    replayed_cases = []
    for question, request in zip(questions, requests, strict=True):
        replayed = replay_call(
            request=request,
            files=files,
            parse_answer=partial(parse_calibration_answer, question=question),
        )
        answer = replayed.answer
        if answer is None:
            if not replayed.exhausted:
                raise ValueError(
                    "calibration capture contains an unfinished call or one the "
                    "provider failed"
                )
        elif not isinstance(answer, CalibrationAnswer):
            raise ValueError("calibration replay returned the wrong response schema")
        replayed_cases.append(
            CalibrationCase(
                seed_id=question.seed.seed_id,
                respondent=question.respondent,
                answer=answer,
            )
        )
    replayed = CalibrationArtifact(
        instrument_sha256=instrument.digest(),
        collection_plan_sha256=plan_digest,
        cases=tuple(replayed_cases),
    )
    if replayed != artifact:
        raise ValueError("calibration artifact differs from strict frozen-call replay")
    return instrument, artifact, plan, validated_treatments


def validate_calibration_capture(
    run_directory: Path,
) -> tuple[CalibrationInstrument, CalibrationArtifact, dict[str, object]]:
    """Re-render and replay a capture, validating its acquisition closure."""
    instrument, artifact, plan, _ = _validate_calibration_capture(run_directory)
    return instrument, artifact, plan


def fit_calibration(
    run_directory: Path,
    resolutions_path: Path,
    output: Path,
    *,
    scripts_root: Path | None = None,
    bootstrap: int = 2000,
    seed: int = 42,
    reference_replicates: int = 200,
    workers: int = 1,
) -> RespondentSpreadFit:
    """Verify resolver scripts, fit a validated capture and write its artifact."""
    from catmodel.evidence.source_bundle import MANIFEST_PATH

    calls_root = (run_directory / "calls").resolve()
    output_resolved = output.expanduser().resolve()
    if output_resolved == calls_root or output_resolved.is_relative_to(calls_root):
        raise ValueError(
            "fit artifacts cannot be written inside the capture's calls tree"
        )
    if any(
        (ancestor / MANIFEST_PATH).is_file() for ancestor in output_resolved.parents
    ):
        raise ValueError("fit artifacts cannot be written inside a source bundle")
    instrument, artifact, plan, validated_treatments = _validate_calibration_capture(
        run_directory
    )
    if scripts_root is None:
        from catmodel.resources import source_checkout_root

        scripts_root = source_checkout_root()
        if scripts_root is None:
            raise ValueError(
                "resolution scripts cannot be verified outside a source checkout; "
                "fit from the checkout that holds them, or pass their root as "
                "scripts_root"
            )
    seed_set = CalibrationSeedSet(
        seeds=instrument.seeds,
        latest_training_cutoff=instrument.latest_training_cutoff,
        resolution_protocol=instrument.resolution_protocol,
        super_blocks=instrument.super_blocks,
        curation_sensitive_seed_ids=instrument.curation_sensitive_seed_ids,
    )
    script_mismatches = resolution_script_digests(seed_set, scripts_root)
    if script_mismatches:
        details = "; ".join(
            f"{path} (expected {seed_set.resolution_protocol.scripts[path]}, "
            f"actual {actual if actual is not None else 'missing/non-file/symlink'})"
            for path, actual in sorted(script_mismatches.items())
        )
        raise ValueError(f"frozen resolution script digest mismatch: {details}")
    resolutions = load_seed_resolutions(resolutions_path)
    fit = fit_respondent_spread(
        instrument,
        artifact,
        plan,
        resolutions,
        instrument_sha256=instrument.digest(),
        collection_plan_sha256=canonical_json_sha256(plan),
        artifact_sha256=canonical_json_sha256(artifact.model_dump(mode="json")),
        resolutions_sha256=sha256_file(resolutions_path),
        treatments=validated_treatments,
        bootstrap=bootstrap,
        seed=seed,
        reference_replicates=reference_replicates,
        workers=workers,
    )
    with output_resolved.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(fit.model_dump(mode="json"), indent=2, allow_nan=False) + "\n"
        )
    return fit


__all__ = [
    "build_calibration_instrument",
    "fit_calibration",
    "run_calibration_collection",
    "validate_calibration_capture",
]
