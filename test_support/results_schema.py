"""Check results files against their committed JSON Schema documents."""

import json
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from jsonschema.protocols import Validator

from catmodel.contracts.results_schema import (
    DIAGNOSTICS_SCHEMA_FILE,
    RESULTS_SCHEMA_FILE,
    schema_path,
)
from catmodel.results.results_file import diagnostics_path


@cache
def _validator(name: str) -> Validator:
    return Draft202012Validator(json.loads(schema_path(name).read_bytes()))


def payload_violations(payload: object, name: str = RESULTS_SCHEMA_FILE) -> list[str]:
    """Each way ``payload`` departs from the committed document ``name``, as
    ``<instance path>: <message>``; an error in a union of records is the
    deepest one in the member it best matches."""
    rows = []
    for error in _validator(name).iter_errors(payload):
        error = best_match([error])
        path = "/".join(map(str, error.absolute_path))
        rows.append(f"/{path}: {error.message[:300]}")
    return rows


def schema_violations(results_path: Path, *, diagnostics: bool = True) -> list[str]:
    """Each way the results file at ``results_path``, and with ``diagnostics``
    the diagnostics file beside it, depart from their committed JSON Schema
    documents; empty when both conform."""
    files = [(results_path, RESULTS_SCHEMA_FILE)]
    if diagnostics:
        files.append((diagnostics_path(results_path), DIAGNOSTICS_SCHEMA_FILE))
    return [
        f"{path.name}: {row}"
        for path, name in files
        for row in payload_violations(json.loads(path.read_bytes()), name)
    ]
