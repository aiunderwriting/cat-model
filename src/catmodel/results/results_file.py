"""Write and read ``results.json`` and the diagnostics file beside it.

A report is stored in two files. ``<name>.json`` holds its headline sections
(``catmodel.contracts.results.HEADLINE_SECTIONS``), and
``<name>.diagnostics.json`` beside it holds its exhibits
(``EXHIBIT_SECTIONS``) and its ``diagnostics`` section, the per-arm view
diagnostics. In each, every reason is a code resolved by its ``reasons`` table
(``catmodel.contracts.results.intern_reasons``). Reading checks the schema
version, each file against its committed JSON Schema document
(``contracts/results/``) and the analysis identity, and restores each reason's
text. ``load_results`` reads the headline alone, or with ``diagnostics`` the
whole report as a run or a replay returns it.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast, overload

from catmodel.artifacts import freeze_json
from catmodel.contracts.results import (
    EXHIBIT_SECTIONS,
    HEADLINE_SECTIONS,
    RESULTS_SCHEMA_VERSION,
    VIEW_DIAGNOSTICS_SECTION,
    HeadlineResults,
    NativeResults,
    intern_reasons,
    resolve_reasons,
)
from catmodel.contracts.results_schema import (
    DIAGNOSTICS_SCHEMA_FILE,
    RESULTS_SCHEMA_FILE,
    schema_errors,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import require_marker, strict_json_loads

DIAGNOSTICS_SUFFIX = ".diagnostics.json"
# The sections the diagnostics file holds: the exhibits and the view
# diagnostics.
DIAGNOSTICS_FILE_SECTIONS = (*EXHIBIT_SECTIONS, VIEW_DIAGNOSTICS_SECTION)
# How many of a file's departures from its schema an error lists.
_SCHEMA_ERRORS_SHOWN = 3
_REPLAY = "derive current results from the run's capture with `catmodel results replay`"
_HEADLINE_ONLY = (
    "it was read from results.json alone, which holds a report's headline; the "
    "diagnostics file beside it holds the exhibits and the view diagnostics, so "
    "read the results with it: load_results(path, diagnostics=True)"
)


class HeadlineOnly(dict):
    """A report read from ``results.json`` alone. Asking it for a section the
    diagnostics file holds says where the section is, rather than that the
    key is missing."""

    def __missing__(self, key: object) -> Any:
        if key in DIAGNOSTICS_FILE_SECTIONS:
            raise ValueError(f"the report has no `{key}`: {_HEADLINE_ONLY}")
        raise KeyError(key)


def require_exhibits(report: Mapping[str, Any]) -> None:
    """Refuse a report read from ``results.json`` alone, for a reader that
    needs the exhibits."""
    if isinstance(report, HeadlineOnly):
        raise ValueError(f"the report has no exhibits: {_HEADLINE_ONLY}")


def diagnostics_path(results_path: Path) -> Path:
    """The diagnostics file of ``results_path``: ``<name>.diagnostics.json``
    beside ``<name>.json``."""
    return results_path.with_name(results_path.stem + DIAGNOSTICS_SUFFIX)


def _require_sections(report: Mapping[str, Any]) -> None:
    """Refuse a report with a section no file holds."""
    unknown = sorted(set(report) - {*HEADLINE_SECTIONS, *DIAGNOSTICS_FILE_SECTIONS})
    if unknown:
        raise ValueError(
            "a report's sections are its headline, its exhibits and its "
            f"diagnostics; neither results file holds {', '.join(unknown)}"
        )


def results_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    """``results.json``'s content: the report's headline sections, with its
    reasons interned."""
    _require_version(report)
    _require_sections(report)
    return intern_reasons(
        {key: report[key] for key in HEADLINE_SECTIONS if key in report}
    )


def diagnostics_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    """The diagnostics file's content: the report's exhibits and view
    diagnostics, tied to its results by their analysis identity, with its
    reasons interned."""
    _require_version(report)
    _require_sections(report)
    missing = [key for key in DIAGNOSTICS_FILE_SECTIONS if key not in report]
    if missing:
        raise ValueError(
            f"a report without {', '.join(missing)} has no diagnostics file to write"
        )
    return intern_reasons(
        {
            "schema_version": report["schema_version"],
            "analysis_sha256": report["analysis_sha256"],
            **{key: report[key] for key in EXHIBIT_SECTIONS},
            **report[VIEW_DIAGNOSTICS_SECTION],
        }
    )


def write_results(path: Path, report: Mapping[str, Any]) -> None:
    """Freeze ``report``'s headline at ``path`` and its exhibits and view
    diagnostics beside it."""
    freeze_json(diagnostics_path(path), diagnostics_payload(report))
    freeze_json(path, results_payload(report))


def decode_results(payload: object) -> HeadlineResults:
    """The headline a ``results.json`` payload holds, with each reason's text.

    Refuses another schema version, a section that belongs to the diagnostics
    file, an analysis digest that is not its analysis specification's and a
    malformed reasons table. It decodes a payload in memory and does not check
    it against the results schema, which ``load_results`` does for a file.
    """
    payload = _current_object(payload, "results")
    misplaced = [key for key in DIAGNOSTICS_FILE_SECTIONS if key in payload]
    if misplaced:
        raise ValueError(
            f"results.json does not hold {', '.join(misplaced)}, which the "
            "diagnostics file beside it holds"
        )
    report = resolve_reasons(payload)
    spec = report.get("analysis_spec")
    if spec is None or canonical_json_sha256(spec) != report.get("analysis_sha256"):
        raise ValueError(
            "the results' analysis_sha256 is not the digest of their analysis_spec, "
            f"so the file changed after it was derived; {_REPLAY}"
        )
    return cast(HeadlineResults, HeadlineOnly(report))


@overload
def load_results(path: Path) -> HeadlineResults: ...
@overload
def load_results(path: Path, *, diagnostics: Literal[False]) -> HeadlineResults: ...
@overload
def load_results(path: Path, *, diagnostics: Literal[True]) -> NativeResults: ...


def load_results(
    path: Path, *, diagnostics: bool = False
) -> HeadlineResults | NativeResults:
    """Read the headline in ``results.json`` at ``path``, or with
    ``diagnostics`` the whole report: the headline with the exhibits and view
    diagnostics of the diagnostics file beside it, which must carry the same
    analysis identity.

    Each file must have the current schema version and match its committed
    JSON Schema document, so a file with a missing, unknown or retyped field
    is refused with the field's path; the results are then decoded
    (``decode_results``). A headline read alone says so when asked for an
    exhibit or the view diagnostics.
    """
    payload = _current_object(strict_json_loads(path.read_bytes()), "results")
    _require_schema(payload, RESULTS_SCHEMA_FILE, "results")
    headline = decode_results(payload)
    if not diagnostics:
        return headline
    companion = strict_json_loads(read_diagnostics_bytes(path))
    return cast(
        NativeResults,
        {**headline, **_decode_diagnostics(companion, headline["analysis_sha256"])},
    )


def read_diagnostics_bytes(results_path: Path) -> bytes:
    """The bytes of ``results_path``'s diagnostics file; a results file
    without one is refused."""
    companion = diagnostics_path(results_path)
    try:
        return companion.read_bytes()
    except FileNotFoundError:
        raise ValueError(
            f"{results_path.name} has no diagnostics file {companion.name} beside "
            f"it, which holds its exhibits and view diagnostics; {_REPLAY}"
        ) from None


def _decode_diagnostics(payload: object, analysis_sha256: str) -> dict[str, Any]:
    """The exhibits and the ``diagnostics`` section a diagnostics file's
    ``payload`` holds, as a report carries them."""
    payload = _current_object(payload, "diagnostics")
    _require_schema(payload, DIAGNOSTICS_SCHEMA_FILE, "diagnostics")
    if payload.get("analysis_sha256") != analysis_sha256:
        raise ValueError("the diagnostics file belongs to another analysis")
    decoded = resolve_reasons(payload)
    del decoded["schema_version"], decoded["analysis_sha256"]
    exhibits = {key: decoded.pop(key) for key in EXHIBIT_SECTIONS}
    return {**exhibits, VIEW_DIAGNOSTICS_SECTION: decoded}


def _current_object(payload: object, file: str) -> Mapping[str, Any]:
    """``payload``, the JSON object of a ``file`` (results or diagnostics)
    file at the current schema version."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"a {file} file must hold a JSON object")
    _require_version(payload)
    return payload


def _require_schema(payload: Mapping[str, Any], name: str, file: str) -> None:
    """Refuse a ``file`` (results or diagnostics) file's ``payload`` that
    departs from the committed document ``name``, naming where."""
    errors = schema_errors(payload, name)
    if errors:
        shown = errors[:_SCHEMA_ERRORS_SHOWN]
        if len(errors) > len(shown):
            shown.append(f"and {len(errors) - len(shown)} more")
        raise ValueError(
            f"the {file} file does not match its schema (contracts/results/{name}): "
            f"{'; '.join(shown)}; {_REPLAY}"
        )


def _require_version(payload: Mapping[str, Any]) -> None:
    require_marker(
        payload,
        "schema_version",
        RESULTS_SCHEMA_VERSION,
        "results schema version",
        _REPLAY,
    )
