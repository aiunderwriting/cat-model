"""An append-only table of provider responses, keyed by the exact prompt sent.

A collection directory ties each response to one instrument, one roster and
one code revision: its replay refuses a capture whose request, rendered prompt
or parsed answer the current code does not reproduce. The table keeps what
outlives those: the prompt a model was sent and the response it returned.
``collect_frozen_call`` reads it before every provider request and adds every
response a provider gives. A change to one question's wording, to a parser, to
the roster or to the number of samples then sends only the requests the table
cannot answer.

A row answers one question: under this provider, model, reasoning effort and
output budget, what did sample ``sample_index`` return to exactly this system
and user prompt? It holds the raw response and what the provider reported with
it, never a parsed answer: the current parser reads the raw response each
time, as replay does. A row is valid on its own, whatever instrument, setup or
code revision first asked its question. Rows are never changed or removed; the
first response recorded under a key is the one the table keeps. A refusal or
an output-limit stop is a response like any other, so a call that exhausted
its validation attempts exhausts them again from the table; a fresh draw
needs another sample index or a collection without the table.

Layout, under the table's directory::

    answer-table.json                        the table's marker and version
    prompts/<aa>/<prompt_sha256>.json        one system and user prompt
    rows/<aa>/<prompt_sha256>/<row_id>.json  one response to it

``<aa>`` is the first two hexadecimal digits of the prompt's digest. A prompt
file is named by the digest of its content, and a row by the digest of the
rest of its key, so a lookup opens one path and a table can be checked without
anything outside it (``AnswerTable.verify``).

A provider can change the model behind an identifier. The table cannot detect
that: a row records the model identifier the provider reported
(``response_model_id``), and a study that must not mix revisions collects into
a table of its own.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from catmodel.artifacts import freeze_json
from catmodel.frozen_calls import frozen_call_key
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import strict_json_loads
from catmodel.llm.transport import ProviderResponse

if TYPE_CHECKING:
    from catmodel.llm.transport import LLMProvider

ANSWER_TABLE_SCHEMA_VERSION = 1
_MARKER_FILE = "answer-table.json"
_MARKER = {
    "kind": "catmodel-answer-table",
    "schema_version": ANSWER_TABLE_SCHEMA_VERSION,
}
_TREATMENT_FIELDS = (
    "provider_name",
    "model_id",
    "reasoning_effort",
    "max_output_tokens",
)
_KEY_FIELDS = (*_TREATMENT_FIELDS, "sample_index", "prompt_sha256", "draw")


def prompt_digest(system_prompt: str, user_prompt: str) -> str:
    """The digest that names one exact system and user prompt."""
    return canonical_json_sha256(
        {"system_prompt": system_prompt, "user_prompt": user_prompt}
    )


@dataclass(frozen=True)
class AnswerKey:
    """What one row answers.

    ``draw`` counts the earlier responses of the same call to an identical
    prompt: a validation retry repeats its prompt when two attempts in a row
    fail with the same feedback, and each of those responses is its own row.
    """

    provider_name: str
    model_id: str
    reasoning_effort: str
    max_output_tokens: int
    sample_index: int
    prompt_sha256: str
    draw: int = 0

    def fields(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _KEY_FIELDS}

    def row_id(self) -> str:
        """The digest of the key apart from its prompt, which names the
        row's directory."""
        return canonical_json_sha256(
            [
                getattr(self, name)
                for name in (*_TREATMENT_FIELDS, "sample_index", "draw")
            ]
        )


def answer_key(
    provider: LLMProvider,
    *,
    max_output_tokens: int,
    sample_index: int,
    system_prompt: str,
    user_prompt: str,
    draw: int,
) -> AnswerKey:
    """The key of ``provider``'s response to one prompt of one sample."""
    return AnswerKey(
        provider_name=provider.provider_name,
        model_id=provider.model_id,
        reasoning_effort=provider.reasoning_effort,
        max_output_tokens=max_output_tokens,
        sample_index=sample_index,
        prompt_sha256=prompt_digest(system_prompt, user_prompt),
        draw=draw,
    )


@dataclass(frozen=True)
class AddedRows:
    """What adding stored calls to a table did: the responses it added and
    those it already held."""

    added: int = 0
    present: int = 0

    def __add__(self, other: AddedRows) -> AddedRows:
        return AddedRows(self.added + other.added, self.present + other.present)


class AnswerTable:
    """One table directory. Opening a path that does not exist, or an empty
    directory, starts a table there; any other directory must already be one."""

    def __init__(self, root: Path) -> None:
        self.root = root
        if root.is_symlink():
            raise ValueError("an answer table cannot be a symbolic link")
        marker = root / _MARKER_FILE
        if root.exists() and not root.is_dir():
            raise ValueError(f"{root} is not a directory")
        if root.is_dir() and any(root.iterdir()) and not marker.is_file():
            raise ValueError(
                f"{root} is not an answer table: it holds files but no {_MARKER_FILE}"
            )
        if marker.is_file():
            found = strict_json_loads(marker.read_bytes())
            if found != _MARKER:
                raise ValueError(
                    f"{marker} is not a version {ANSWER_TABLE_SCHEMA_VERSION} "
                    "answer table marker"
                )
        else:
            root.mkdir(parents=True, exist_ok=True)
            freeze_json(marker, _MARKER)

    def _prompt_path(self, digest: str) -> Path:
        return self.root / "prompts" / digest[:2] / f"{digest}.json"

    def _row_path(self, key: AnswerKey) -> Path:
        digest = key.prompt_sha256
        return self.root / "rows" / digest[:2] / digest / f"{key.row_id()}.json"

    def holds(self, key: AnswerKey) -> bool:
        """Whether the table has a response under ``key``."""
        return self._row_path(key).is_file()

    def lookup(self, key: AnswerKey) -> ProviderResponse | None:
        """The response recorded under ``key``, as the provider returned it,
        or None."""
        path = self._row_path(key)
        try:
            row = strict_json_loads(path.read_bytes())
        except FileNotFoundError:
            return None
        _require_row(row, key, path)
        return ProviderResponse(
            text=row["raw_response"],
            model_id=row["response_model_id"],
            provider_name=key.provider_name,
            provider_request_id=row["provider_request_id"],
            latency_ms=row["latency_ms"],
            input_tokens=row["input_tokens"],
            output_tokens=row["response_tokens"],
            stop_reason=row["stop_reason"],
            evidence=row["provider_evidence"],
        )

    def record(
        self,
        key: AnswerKey,
        *,
        system_prompt: str,
        user_prompt: str,
        response: ProviderResponse,
    ) -> bool:
        """Add ``response`` under ``key`` unless the table already has a
        response there; return whether it was added.

        The first response recorded under a key stays: a second one, from a
        collection that ran at the same time or that did not read the table,
        is another valid answer to the same question, and its own collection
        keeps it.
        """
        if prompt_digest(system_prompt, user_prompt) != key.prompt_sha256:
            raise ValueError("an answer's key differs from the prompt it answers")
        prompt_path = self._prompt_path(key.prompt_sha256)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_json(
            prompt_path, {"system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        path = self._row_path(key)
        if path.is_file():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": ANSWER_TABLE_SCHEMA_VERSION,
            **key.fields(),
            "raw_response": response.text,
            "stop_reason": response.stop_reason,
            "response_model_id": response.model_id,
            "provider_request_id": response.provider_request_id,
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "response_tokens": response.output_tokens,
            "provider_evidence": response.evidence,
        }
        try:
            freeze_json(path, row)
        except ValueError:
            # Another writer recorded a different response under this key
            # between the check above and the write.
            return False
        return True

    def add_call(
        self,
        *,
        request: Mapping[str, Any],
        treatment: Mapping[str, Any],
        attempts: list[Mapping[str, Any]],
    ) -> AddedRows:
        """Add one stored call's responses: ``attempts`` are its attempt
        records in order and ``treatment`` its acquisition record. An attempt
        that failed at the provider has no response and adds nothing."""
        sample_index = request["respondent"]["sample_index"]
        asked: list[str] = []
        result = AddedRows()
        for attempt in attempts:
            if attempt.get("raw_response") is None:
                continue
            system_prompt = attempt["rendered_system_prompt"]
            user_prompt = attempt["rendered_user_prompt"]
            key = AnswerKey(
                **{name: treatment[name] for name in _TREATMENT_FIELDS},
                sample_index=sample_index,
                prompt_sha256=prompt_digest(system_prompt, user_prompt),
                draw=asked.count(user_prompt),
            )
            asked.append(user_prompt)
            added = self.record(
                key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=ProviderResponse(
                    text=attempt["raw_response"],
                    model_id=attempt["response_model_id"],
                    provider_name=attempt["provider_name"],
                    provider_request_id=attempt.get("provider_request_id"),
                    latency_ms=attempt.get("latency_ms") or 0,
                    input_tokens=attempt.get("input_tokens"),
                    output_tokens=attempt.get("response_tokens"),
                    stop_reason=attempt["stop_reason"],
                    evidence=attempt.get("provider_evidence") or {},
                ),
            )
            result += AddedRows(added=int(added), present=int(not added))
        return result

    def add_collection(self, run_directory: Path) -> AddedRows:
        """Add every response a collection directory stores, whatever code
        revision collected it and whether or not its capture still replays.

        The directory is any frozen-call capture: a contextual collection, a
        calibration collection or one written by an earlier revision, as long
        as its ``calls/`` and ``acquisition/`` records have the current
        layout. Nothing is parsed or validated beyond that layout.
        """
        calls = run_directory / "calls"
        if not calls.is_dir():
            raise ValueError(f"{run_directory} holds no calls/ directory")
        result = AddedRows()
        for call in sorted(calls.iterdir()):
            request_path = call / "request.json"
            if not request_path.is_file():
                continue
            request = strict_json_loads(request_path.read_bytes())
            if frozen_call_key(request) != call.name:
                raise ValueError(
                    f"{request_path}: the call's key differs from its request"
                )
            attempts = [
                strict_json_loads(path.read_bytes())
                for path in sorted(call.glob("attempt-*.json"))
            ]
            if not any(row.get("raw_response") is not None for row in attempts):
                continue
            acquisition = run_directory / "acquisition" / f"{call.name}.json"
            if not acquisition.is_file():
                raise ValueError(
                    f"{call}: its responses have no acquisition record, so their "
                    "model settings are unknown"
                )
            treatment = strict_json_loads(acquisition.read_bytes())
            try:
                result += self.add_call(
                    request=request, treatment=treatment, attempts=attempts
                )
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"{call}: its records lack a field an answer row needs: {error}"
                ) from error
        return result

    def coverage(self, run_directory: Path) -> tuple[int, int]:
        """How many of a collection directory's frozen requests the table
        already answers on their first attempt, and how many requests the
        directory holds.

        A directory that ``--dry-run`` planned holds its initial requests, so
        this says how many of them a collection with this table would not
        send. Follow-up questions quote earlier answers and are frozen only
        once asked, so they are counted only in a directory that has asked
        them.
        """
        plan_path = run_directory / "collection-plan.json"
        if not plan_path.is_file():
            raise ValueError(f"{run_directory} holds no collection-plan.json")
        execution = strict_json_loads(plan_path.read_bytes())["execution"]
        models = {row["model_id"]: row for row in execution["models"]}
        budgets = execution["settings"]["max_output_tokens_by_model"]
        held = total = 0
        for path in sorted((run_directory / "calls").glob("*/request.json")):
            request = strict_json_loads(path.read_bytes())
            respondent = request["respondent"]
            model = models[respondent["model_id"]]
            key = AnswerKey(
                provider_name=model["provider"],
                model_id=model["model_id"],
                reasoning_effort=model["reasoning_effort"],
                max_output_tokens=budgets[model["model_id"]],
                sample_index=respondent["sample_index"],
                prompt_sha256=prompt_digest(
                    request["system_prompt"], request["user_prompt"]
                ),
            )
            total += 1
            held += self.holds(key)
        return held, total

    def rows(self) -> Iterator[dict[str, Any]]:
        """Every row, in the order of its path."""
        for path in sorted((self.root / "rows").glob("*/*/*.json")):
            yield strict_json_loads(path.read_bytes())

    def prompt(self, digest: str) -> dict[str, str]:
        """The system and user prompt a digest names."""
        return strict_json_loads(self._prompt_path(digest).read_bytes())

    def verify(self) -> int:
        """Check every prompt against its name and every row against its
        path; return the number of rows."""
        for path in sorted((self.root / "prompts").glob("*/*.json")):
            prompt = strict_json_loads(path.read_bytes())
            if (
                set(prompt) != {"system_prompt", "user_prompt"}
                or prompt_digest(**prompt) != path.stem
                or path.parent.name != path.stem[:2]
            ):
                raise ValueError(f"{path}: the prompt differs from its name")
        count = 0
        for path in sorted((self.root / "rows").glob("*/*/*.json")):
            row = strict_json_loads(path.read_bytes())
            try:
                key = AnswerKey(**{name: row[name] for name in _KEY_FIELDS})
            except (KeyError, TypeError) as error:
                raise ValueError(f"{path}: the row has no complete key") from error
            if path != self._row_path(key):
                raise ValueError(f"{path}: the row differs from its path")
            _require_row(row, key, path)
            if not self._prompt_path(key.prompt_sha256).is_file():
                raise ValueError(f"{path}: the row's prompt is missing")
            count += 1
        return count


def _require_row(row: object, key: AnswerKey, path: Path) -> None:
    if (
        not isinstance(row, dict)
        or row.get("schema_version") != ANSWER_TABLE_SCHEMA_VERSION
        or any(row.get(name) != value for name, value in key.fields().items())
        or not isinstance(row.get("raw_response"), str)
        or not isinstance(row.get("stop_reason"), str)
        or not isinstance(row.get("response_model_id"), str)
        or not isinstance(row.get("provider_evidence"), dict)
    ):
        raise ValueError(f"{path}: the answer table row differs from its key")
