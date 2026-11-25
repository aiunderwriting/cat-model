"""Strict replay of frozen provider calls and the parser identity they record."""

import ast
import inspect
import json
import platform
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from importlib.metadata import version
from pathlib import Path
from types import FunctionType
from typing import Annotated, Any, Literal, TypeAliasType, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import require_marker, strict_json_loads
from catmodel.stored_files import Stored, StoredFiles
from catmodel.validation_errors import EvidenceValidationError

PARSE_ATTEMPTS = 3
FROZEN_CALL_SCHEMA_VERSION = 1
PROVIDER_FAILURES_PER_CALL = 3
"""The most attempts one call may store that failed at the provider, an error
rather than a response (``NUM-PROVIDER-FAILURES``). Resume sends such a call
again with the prompt the failed attempt used, using no validation attempt; a
call with this many failures needs a new run."""


class ResponseStopped(ValueError):
    """A response the provider ended at its output limit or by refusing.

    It is a failed validation attempt like a malformed answer, but its
    feedback names the stop (``output_limit`` or ``refusal``) rather than the
    parse error its text would give."""

    def __init__(self, stop_reason: str) -> None:
        super().__init__(f"provider stopped the response: {stop_reason}")
        self.stop_reason = stop_reason


type _Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ParserProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1] = 1
    callable_name: str
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_definitions: dict[str, _Digest] = Field(min_length=1)


class _NoOpStatements(ast.NodeTransformer):
    """Drops the statements that do nothing when run: bare constants, which
    include every docstring, and ``pass``."""

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        return None if isinstance(node.value, ast.Constant) else node

    def visit_Pass(self, node: ast.Pass) -> None:
        return None


def _defined_names(statement: ast.stmt) -> tuple[str, ...]:
    """The module names a top-level statement defines."""
    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return (statement.name,)
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign | ast.AugAssign):
        targets = [statement.target]
    elif isinstance(statement, ast.TypeAlias):
        targets = [statement.name]
    else:
        return ()
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


class _NamesRead(ast.NodeVisitor):
    """The names a definition reads when it runs or declares a model field.

    The type hints of a function's parameters, return value and locals run
    nothing, so their names are left out; a class's annotations declare its
    fields.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()
        self._in_function = False

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        defaults = [*node.args.defaults, *node.args.kw_defaults]
        for child in (*node.decorator_list, *defaults):
            if child is not None:
                self.visit(child)
        self._visit_body(node.body, in_function=True)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for child in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(child)
        self._visit_body(node.body, in_function=False)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self._in_function:
            self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)

    def _visit_body(self, body: list[ast.stmt], *, in_function: bool) -> None:
        outer, self._in_function = self._in_function, in_function
        for statement in body:
            self.visit(statement)
        self._in_function = outer


@lru_cache(maxsize=64)
def _definitions_digest(source: bytes, roots: frozenset[str]) -> str:
    """The digest of the top-level definitions of a module that ``roots``
    reach by name, transitively, in syntax trees without their no-op
    statements: formatting, comments, docstrings, the order of definitions and
    every definition not reached leave it unchanged. It follows the
    interpreter's tree format, which a Python minor release can change."""
    tree = _NoOpStatements().visit(ast.parse(source))
    defining: dict[str, list[ast.stmt]] = {}
    for statement in tree.body:
        for name in _defined_names(statement):
            defining.setdefault(name, []).append(statement)
    if not roots <= set(defining):
        raise ValueError(
            "parser protocol requires top-level parser and response model definitions"
        )
    reached: dict[int, ast.stmt] = {}
    pending = list(roots)
    while pending:
        for statement in defining.get(pending.pop(), ()):
            if id(statement) not in reached:
                reached[id(statement)] = statement
                names = _NamesRead()
                names.visit(statement)
                pending.extend(names.names)
    return canonical_json_sha256(
        sorted(
            ast.dump(statement, include_attributes=False)
            for statement in reached.values()
        )
    )


def _schema_models(model: type[BaseModel]) -> tuple[type[BaseModel], ...]:
    """The response model and every model its fields hold, at any depth."""
    found: dict[type[BaseModel], None] = {}
    pending = [model]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found[current] = None
        annotations = [field.annotation for field in current.model_fields.values()]
        while annotations:
            annotation = annotations.pop()
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                pending.append(annotation)
            elif isinstance(annotation, TypeAliasType):
                annotations.append(annotation.__value__)
            else:
                annotations.extend(get_args(annotation))
    return tuple(found)


def parser_protocol(
    parse_answer: Callable, response_model: type[BaseModel]
) -> ParserProtocol:
    """Identify a parser by name, response JSON schema and the source parsing reaches.

    Partials and decorators are unwrapped to the named parser function;
    partial-bound arguments are not part of the identity. Source digests
    cover, in each module that defines the parser or a model of the response
    schema (the response model and the models its fields hold, at any depth),
    the syntax trees of those definitions and of the module's functions,
    classes and assignments they reach by name, transitively: through bodies,
    decorators, base classes, default values and model field annotations, but
    not through the type hints of a function's parameters, return value or
    locals. Formatting, comments, docstrings, the order of definitions and the
    rest of those modules, such as prompt text or reporting, do not change
    them, and a docstring a response schema shows changes its schema digest
    instead. Definitions imported from other
    modules and the Python and library versions are not part of the identity:
    collection replays every stored attempt before sending anything for its
    call, and capture validation replays them all, so a change there that
    alters how a stored response parses fails both, and one that alters none
    leaves the capture exactly as the changed code would have collected it.
    Each attempt records ``runtime_versions``.
    """
    parser = parse_answer
    while isinstance(parser, partial):
        parser = parser.func
    parser = inspect.unwrap(parser)
    if not inspect.isfunction(parser) or parser.__name__ == "<lambda>":
        raise ValueError("collection requires a named parser or a partial of it")
    return _parser_protocol(parser, response_model)


@lru_cache(maxsize=256)
def _parser_protocol(
    parser: FunctionType, response_model: type[BaseModel]
) -> ParserProtocol:
    """The protocol of one parser function and response model, computed once
    per pair of loaded objects: the source those objects were loaded from
    cannot change under them, and redefining either makes new objects."""
    roots: dict[str, set[str]] = {}
    paths: dict[str, str] = {}
    for value in (parser, *_schema_models(response_model)):
        module = inspect.getmodule(value)
        path = None if module is None else inspect.getsourcefile(module)
        if module is None or path is None:
            raise ValueError(
                "parser protocol requires inspectable parser and response model source"
            )
        # A nested definition is reached through the top-level one holding it.
        roots.setdefault(module.__name__, set()).add(value.__qualname__.split(".")[0])
        paths[module.__name__] = path
    return ParserProtocol(
        callable_name=f"{parser.__module__}.{parser.__qualname__}",
        response_schema_sha256=canonical_json_sha256(
            response_model.model_json_schema()
        ),
        source_definitions={
            name: _definitions_digest(
                Path(paths[name]).read_bytes(), frozenset(roots[name])
            )
            for name in sorted(roots)
        },
    )


@lru_cache(maxsize=1)
def _runtime_versions() -> tuple[tuple[str, str], ...]:
    return (
        ("python", platform.python_version()),
        *((name, version(name)) for name in ("pydantic", "numpy", "scipy")),
    )


def runtime_versions() -> dict[str, str]:
    """The Python, pydantic, NumPy and SciPy versions this process runs,
    which each frozen attempt records; they are not part of the parser
    identity (``parser_protocol``)."""
    return dict(_runtime_versions())


class CaptureFiles(StoredFiles):
    """A capture's files by relative path, read-only, with each call's
    records indexed once: replaying a capture of many calls then reads each
    call's records without scanning every file. A file may stay on disk
    (``StoredFiles``), so a large capture is read one file at a time."""

    def __init__(
        self, files: Mapping[str, Stored] | Iterable[tuple[str, Stored]]
    ) -> None:
        super().__init__(files)
        self._calls: dict[str, tuple[str, ...]] | None = None

    def call_records(self, key: str) -> tuple[str, ...]:
        """Every path under ``calls/<key>/``, sorted."""
        if self._calls is None:
            grouped: dict[str, list[str]] = {}
            for name in self:
                parts = name.split("/", 2)
                if len(parts) == 3 and parts[0] == "calls":
                    grouped.setdefault(parts[1], []).append(name)
            self._calls = {key: tuple(sorted(names)) for key, names in grouped.items()}
        return self._calls.get(key, ())


def call_attempt_paths(files: Mapping[str, bytes], key: str) -> tuple[str, ...]:
    """The sorted paths beginning ``calls/<key>/attempt-``."""
    prefix = f"calls/{key}/attempt-"
    names = files.call_records(key) if isinstance(files, CaptureFiles) else files
    return tuple(sorted(name for name in names if name.startswith(prefix)))


class AcquisitionTreatment(BaseModel):
    """How one frozen call was acquired: the provider settings, the digest of
    the exact request and the digest of the captured parser protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    model_id: str = Field(min_length=1)
    provider_name: str = Field(min_length=1)
    reasoning_effort: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0, strict=True)
    request_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    retry_protocol: Literal[1]
    parse_attempts: Literal[3]
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_protocol_sha256: _Digest


def validate_acquisition_capture(
    request: Mapping[str, Any], files: Mapping[str, bytes]
) -> AcquisitionTreatment:
    key = frozen_call_key(request)
    path = f"acquisition/{key}.json"
    if path not in files:
        raise ValueError("forecast capture lacks its original acquisition treatment")
    payload = strict_json_loads(files[path])
    if not isinstance(payload, dict) or not {
        "request_sha256",
        "parser_protocol_sha256",
    } <= set(payload):
        raise ValueError(
            "captured acquisition treatment does not bind its request and parser "
            "protocol; a capture written without them is read only by the "
            "revision that wrote it"
        )
    treatment = AcquisitionTreatment.model_validate(payload)
    if treatment.request_sha256 != canonical_json_sha256(dict(request)):
        raise ValueError(
            "captured acquisition request differs from its bound question and conditioning"
        )
    protocol_path = f"parser-protocols/{treatment.parser_protocol_sha256}.json"
    if protocol_path not in files:
        raise ValueError("captured acquisition lacks its parser protocol definition")
    protocol = strict_json_loads(files[protocol_path])
    if canonical_json_sha256(protocol) != treatment.parser_protocol_sha256:
        raise ValueError("captured parser protocol definition checksum differs")
    require_marker(
        protocol, "schema_version", 1, "captured parser protocol schema version"
    )
    ParserProtocol.model_validate(protocol)
    if treatment.model_id != request["respondent"]["model_id"]:
        raise ValueError("captured acquisition model differs from its request")
    for name in call_attempt_paths(files, key):
        row = strict_json_loads(files[name])
        if row.get("provider_name") != treatment.provider_name:
            raise ValueError("captured acquisition provider differs from its attempts")
    return treatment


def validate_acquisition_closure(
    requests: Iterable[Mapping[str, Any]], files: Mapping[str, bytes]
) -> tuple[AcquisitionTreatment, ...]:
    """Each planned request's treatment and each referenced protocol, exactly once."""
    requests = tuple(requests)
    expected = {f"acquisition/{frozen_call_key(request)}.json" for request in requests}
    actual = {name for name in files if name.startswith("acquisition/")}
    if expected != actual:
        raise ValueError(
            "captured acquisition files differ from the complete request closure"
        )
    treatments = tuple(
        validate_acquisition_capture(request, files) for request in requests
    )
    protocols = {
        f"parser-protocols/{row.parser_protocol_sha256}.json" for row in treatments
    }
    if protocols != {name for name in files if name.startswith("parser-protocols/")}:
        raise ValueError(
            "captured parser protocol files differ from their exact reference closure"
        )
    return treatments


def _shares(values: list[float]) -> dict[str, float] | None:
    """Median, 95th percentile (nearest rank) and largest of *values*."""
    if not values:
        return None
    ordered = sorted(values)
    return {
        label: round(ordered[max(0, -(-len(ordered) * rank // 100) - 1)], 4)
        for label, rank in (("p50", 50), ("p95", 95), ("max", 100))
    }


def call_limit_usage(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    """Each model's responses against its call limits (``NUM-CALL-LIMITS``),
    its lost streams (``NUM-CONNECTION-LOST``) and its provider failures
    (``NUM-PROVIDER-FAILURES``).

    For every model with a stored attempt: how many attempts returned a
    response and how many of those the provider stopped at the output limit
    or by refusing; how many failed at the provider instead; how many streams
    were cut or background runs lost and sent again (``connection_lost``,
    counted over every attempt); the largest output; and the median,
    95th-percentile (nearest rank) and largest time of a response as a share
    of the request deadline, where one means a timeout. A server-side run's
    time is read from its own status timeline, from submission to when its
    final status was seen, so a lost earlier run's time is not counted; a
    background run's splits into queueing, until it was first seen in
    progress, and generation after that. A stream's time, cut streams
    included, is all generation.
    """
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(files):
        if not (name.startswith("acquisition/") and name.endswith(".json")):
            continue
        treatment = strict_json_loads(files[name])
        key = name.removeprefix("acquisition/").removesuffix(".json")
        for path in call_attempt_paths(files, key):
            attempt = strict_json_loads(files[path])
            attempt["deadline_s"] = treatment["request_timeout_seconds"]
            rows.setdefault(treatment["model_id"], []).append(attempt)
    report = {}
    for model, attempts in sorted(rows.items()):
        responses = [row for row in attempts if row.get("raw_response") is not None]
        tokens = [
            row["response_tokens"]
            for row in responses
            if row.get("response_tokens") is not None
        ]
        queue, generation = [], []
        for row in responses:
            evidence = row.get("provider_evidence") or {}
            seen = evidence.get("status_first_seen_ms") or {}
            queued_ms = seen.get("in_progress")
            deadline_ms = row["deadline_s"] * 1000
            if queued_ms is not None:
                queue.append(queued_ms / deadline_ms)
            ended_ms = max(seen.values(), default=row.get("latency_ms"))
            if ended_ms is not None:
                generation.append((ended_ms - (queued_ms or 0)) / deadline_ms)
        report[model] = {
            "responses": len(responses),
            "provider_failures": len(attempts) - len(responses),
            "output_limit_stops": sum(
                row.get("stop_reason") == "output_limit" for row in responses
            ),
            "refusals": sum(row.get("stop_reason") == "refusal" for row in responses),
            "connection_lost": sum(
                len((row.get("provider_evidence") or {}).get("connection_lost", ()))
                for row in attempts
            ),
            "largest_output_tokens": max(tokens, default=None),
            "queue_share_of_deadline": _shares(queue),
            "generation_share_of_deadline": _shares(generation),
        }
    return report


EXHAUSTION_CAUSES = ("validation", "refusal", "output_limit")
"""Why an exhausted call's last attempt failed (``exhaustion_causes``)."""


def exhaustion_causes(
    files: Mapping[str, bytes], calls: Iterable[tuple[str, tuple[str, str, int]]]
) -> dict[tuple[str, tuple[str, str, int]], str]:
    """Why each exhausted call's last attempt failed, by (question digest,
    respondent key): ``refusal`` or ``output_limit`` where the attempt
    records that provider stop (``stop_reason``), otherwise ``validation``,
    a response that failed parsing or validation."""
    calls = tuple(calls)
    if not calls:
        return {}
    keys: dict[tuple[str, tuple[str, str, int]], str] = {}
    for name in files:
        parts = name.split("/")
        if len(parts) != 3 or parts[0] != "calls" or parts[2] != "request.json":
            continue
        request = strict_json_loads(files[name])
        row = request["respondent"]
        identity = (
            request["prompt_sha256"],
            (row["respondent_id"], row["model_id"], row["sample_index"]),
        )
        if identity in keys:
            raise ValueError("two captured calls share one question and respondent")
        keys[identity] = parts[1]
    causes = {}
    for call in calls:
        if call not in keys:
            raise ValueError("an exhausted call is missing from its capture")
        attempts = call_attempt_paths(files, keys[call])
        stop = strict_json_loads(files[attempts[-1]]).get("stop_reason")
        causes[call] = stop if stop in EXHAUSTION_CAUSES[1:] else "validation"
    return causes


def validation_feedback(error: Exception) -> str:
    """Retry feedback built only from each error's type and location, so it is
    reproducible from the raw answer; validator messages, inputs and URLs are
    left out of the retry and its replay identity.
    """
    if isinstance(error, ValidationError):
        issues = []
        for item in error.errors(include_url=False, include_input=False):
            domain_error = item.get("ctx", {}).get("error")
            issues.append(
                {
                    "type": domain_error.code
                    if isinstance(domain_error, EvidenceValidationError)
                    else item["type"],
                    "location": [
                        *item["loc"],
                        *(
                            domain_error.location
                            if isinstance(domain_error, EvidenceValidationError)
                            else ()
                        ),
                    ],
                }
            )
    elif isinstance(error, EvidenceValidationError):
        issues = [{"type": error.code, "location": list(error.location)}]
    elif isinstance(error, ResponseStopped):
        issues = [{"type": error.stop_reason, "location": []}]
    else:
        issues = [
            {
                "type": "json_invalid"
                if isinstance(error, json.JSONDecodeError)
                else "type_error"
                if isinstance(error, TypeError)
                else "value_error",
                "location": [],
            }
        ]
    return json.dumps(
        {"schema_version": 1, "validation_errors": issues},
        sort_keys=True,
        separators=(",", ":"),
    )


_CODE_FENCE = re.compile(
    r"\A\s*```[A-Za-z]*[ \t]*\r?\n(?P<body>.*?)\r?\n[ \t]*```\s*\Z", re.DOTALL
)


def unfenced(text: str) -> str:
    """*text* without a Markdown code fence that encloses all of it.

    Some models put a JSON answer between two fence lines, the first with or
    without a language name, though asked for JSON alone. The fence is
    formatting, not part of the answer, so the answer is read without it. Any
    other text, with more around the answer than one fence or with none, is
    returned as it is and passes or fails unchanged.
    """
    match = _CODE_FENCE.match(text)
    return text if match is None else match["body"]


def validated_response(
    text: str, stop_reason: str, parse_answer: Callable[[str], BaseModel]
) -> tuple[BaseModel | None, Exception | None]:
    """A response's answer, or the error that fails its validation attempt.

    Collection and replay both decide here, so a stored attempt replays to
    the outcome collection froze. A response the provider stopped fails
    unparsed (``ResponseStopped``). A completed one is parsed without a code
    fence around it (``unfenced``) and fails when ``parse_answer`` raises
    ``ValueError`` or ``TypeError``, as its JSON, schema and domain checks do.
    Any other exception is a defect in the parser, not a finding about the
    response, and is raised: collection then stops without freezing an
    attempt that replay could not reproduce.
    """
    try:
        if stop_reason != "completed":
            raise ResponseStopped(stop_reason)
        return parse_answer(unfenced(text)), None
    except (TypeError, ValueError) as error:
        return None, error


def frozen_call_key(request: Mapping[str, Any]) -> str:
    respondent = request["respondent"]
    return canonical_json_sha256(
        [
            request["call_id"],
            (
                respondent["respondent_id"],
                respondent["model_id"],
                respondent["sample_index"],
            ),
        ]
    )


def retry_prompt(
    request: Mapping[str, Any], attempt: int, last_error: str | None
) -> str:
    prompt = request["user_prompt"]
    if attempt > 1:
        prompt += "\nReturn a complete JSON replacement for the identical question."
        prompt += (
            "\nThe preceding answer failed validation for the following reasons. Correct these contract errors while answering the same question; return a complete replacement answer, not a patch.\n"
            + str(last_error)[:12000]
        )
    return prompt


@dataclass(frozen=True)
class CallReplay:
    """A frozen call's stored attempts, replayed: its answer, if any, the
    validation attempts its responses used, its provider failures and the
    validation feedback the next validation attempt carries."""

    answer: BaseModel | None
    responses: int
    provider_failures: int
    feedback: str | None

    @property
    def exhausted(self) -> bool:
        """Every validation attempt returned a response that failed."""
        return self.answer is None and self.responses >= PARSE_ATTEMPTS

    @property
    def failed(self) -> bool:
        """The call stored its last allowed provider failure."""
        return (
            self.answer is None
            and not self.exhausted
            and self.provider_failures >= PROVIDER_FAILURES_PER_CALL
        )

    def next_prompt(self, request: Mapping[str, Any]) -> str:
        """The user prompt of the call's next attempt: the original prompt
        before any response, otherwise the retry built from the last
        validation feedback. A provider failure changes neither."""
        return retry_prompt(request, self.responses + 1, self.feedback)


def replay_call(
    *,
    request: Mapping[str, Any],
    files: Mapping[str, bytes],
    parse_answer: Callable[[str], BaseModel],
) -> CallReplay:
    """Replay every stored attempt of one call strictly from its raw responses.

    Attempts are numbered from one without gaps. Each is a response, one
    validation attempt of at most ``PARSE_ATTEMPTS``, or a provider failure,
    at most ``PROVIDER_FAILURES_PER_CALL``, which may come before any response
    and is followed by the attempt that sent the call again. A response's
    answer or failure is ``validated_response``'s, as at collection. Each
    attempt's prompt is ``CallReplay.next_prompt`` of the attempts before it.
    Nothing follows an answer, the last validation attempt or the last allowed
    provider failure.
    """
    prefix = f"calls/{frozen_call_key(request)}/"
    if (
        prefix + "request.json" not in files
        or strict_json_loads(files[prefix + "request.json"]) != request
    ):
        raise ValueError("captured frozen request differs or is missing")
    paths = [
        path
        for path in call_attempt_paths(files, frozen_call_key(request))
        if path.endswith(".json")
    ]
    if len(paths) > PARSE_ATTEMPTS + PROVIDER_FAILURES_PER_CALL:
        raise ValueError("stored request exceeds its validation retry budget")
    respondent = request["respondent"]
    state = CallReplay(answer=None, responses=0, provider_failures=0, feedback=None)
    for index, path in enumerate(paths, 1):
        row = strict_json_loads(files[path])
        if (
            row.get("schema_version") != FROZEN_CALL_SCHEMA_VERSION
            or path != prefix + f"attempt-{index:03}.json"
            or row.get("attempt") != index
            or row.get("prompt_sha256") != request["prompt_sha256"]
            or row.get("model_id") != respondent["model_id"]
            or row.get("call_id") != request["call_id"]
            or row.get("respondent_id") != respondent["respondent_id"]
            or row.get("sample_index") != respondent["sample_index"]
        ):
            raise ValueError("stored response identity or attempt sequence differs")
        if state.answer is not None or state.exhausted or state.failed:
            raise ValueError(
                "complete, exhausted and failed calls cannot have later attempts"
            )
        if (
            row.get("rendered_user_prompt") != state.next_prompt(request)
            or row.get("rendered_system_prompt") != request["system_prompt"]
        ):
            raise ValueError("stored response actual prompt differs")
        raw = row.get("raw_response")
        stop_reason = row.get("stop_reason")
        if (stop_reason is None) == isinstance(raw, str) or not (
            stop_reason is None or isinstance(stop_reason, str)
        ):
            raise ValueError("stored response and stop reason differ")
        parsed, invalid = (
            validated_response(raw, stop_reason, parse_answer)
            if isinstance(raw, str)
            else (None, None)
        )
        derived_error = None if invalid is None else validation_feedback(invalid)
        if (parsed is not None) != (row.get("parse_success") is True):
            raise ValueError(
                "stored parse success differs from strict raw response replay"
            )
        if row.get("answer") != (
            parsed.model_dump(mode="json") if parsed is not None else None
        ):
            raise ValueError("stored answer differs from strict raw response replay")
        if isinstance(raw, str):
            if row.get("error") != derived_error:
                raise ValueError(
                    "stored validation feedback differs from re-derived raw response error"
                )
            state = CallReplay(
                answer=parsed,
                responses=state.responses + 1,
                provider_failures=state.provider_failures,
                feedback=derived_error,
            )
        else:
            state = replace(state, provider_failures=state.provider_failures + 1)
    return state
