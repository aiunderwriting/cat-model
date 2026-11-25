"""The JSON Schema documents of results.json and its diagnostics file.

They are generated from the results contracts and committed under
``contracts/results/``. Real results files are checked against the committed
documents where a fixture already writes them: the golden Poisson run and
replay, with a misspelt and a retyped field (``test_golden_poisson.py``), the
stipulated campaign (``test_workflow_cli.py``) and a synthetic public study
and its export (``test_capability_scenarios.py``). Reading a results file checks
it against its document, and ``conftest.py`` reads back each one a test's run or
replay writes.
"""

import json
import re
import typing
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import pytest
from jsonschema import Draft202012Validator

from catmodel.contracts import results_schema
from catmodel.contracts.incident import ComparisonRequirement, Evidence, ExecutionPlan
from catmodel.contracts.results import (
    EXHIBIT_SECTIONS,
    HEADLINE_SECTIONS,
    RESULTS_SCHEMA_VERSION,
    VIEW_DIAGNOSTICS_SECTION,
    AuthoredComparisonRequirement,
    CodeProvenance,
    ComparisonRequirementRecord,
    EvidenceRecord,
    ExecutionPlanRecord,
    NativeResults,
    ResultsDiagnostics,
    SmallEventControlsRecord,
    SmallEventSettings,
    TailGateControlsRecord,
    TailGateSettings,
    intern_reasons,
    reason_code,
)
from catmodel.contracts.results_schema import (
    DIAGNOSTICS_SCHEMA_FILE,
    DIALECT,
    REGENERATE_COMMAND,
    RESULTS_SCHEMA_FILE,
    results_schemas,
    schema_path,
    schema_text,
)
from catmodel.setup.store import setup_code_provenance
from catmodel.simulation.small_events import SmallEventControls
from catmodel.simulation.tail_gate import TailGateControls


def test_the_committed_schemas_are_generated_from_the_contracts():
    generated = results_schemas()
    directory = schema_path(RESULTS_SCHEMA_FILE).parent
    stale = [
        name
        for name, schema in generated.items()
        if not schema_path(name).is_file()
        or schema_path(name).read_text(encoding="utf-8") != schema_text(schema)
    ]
    assert not stale, (
        f"{', '.join(stale)} in contracts/results/ is not what the results "
        f"contracts generate; run `{REGENERATE_COMMAND}` and commit the result"
    )
    assert sorted(path.name for path in directory.iterdir()) == sorted(generated)


def test_every_section_of_a_report_is_in_exactly_one_file():
    """The headline sections, the exhibits and the view diagnostics partition
    a report, and each document holds its own sections and no other."""
    sections = [*HEADLINE_SECTIONS, *EXHIBIT_SECTIONS, VIEW_DIAGNOSTICS_SECTION]
    assert len(sections) == len(set(sections))
    assert set(sections) == set(typing.get_type_hints(NativeResults))
    documents = results_schemas()
    assert set(documents[RESULTS_SCHEMA_FILE]["properties"]) == {
        *HEADLINE_SECTIONS,
        "reasons",
    }
    # The diagnostics file holds the view diagnostics' own fields beside the
    # exhibits, and names the results it belongs to.
    assert set(documents[DIAGNOSTICS_SCHEMA_FILE]["properties"]) == {
        "schema_version",
        "analysis_sha256",
        *EXHIBIT_SECTIONS,
        *typing.get_type_hints(ResultsDiagnostics),
        "reasons",
    }
    for name, optional in (
        (RESULTS_SCHEMA_FILE, {"execution_context"}),
        (DIAGNOSTICS_SCHEMA_FILE, set()),
    ):
        document = documents[name]
        assert set(document["properties"]) - set(document["required"]) == optional


def _guide() -> str:
    path = Path(__file__).resolve().parents[1] / "docs" / "setup-run-results.md"
    return path.read_text(encoding="utf-8")


def test_the_results_guide_names_each_files_sections():
    """docs/setup-run-results.md lists the headline sections under
    ``results.json`` and the exhibits under the diagnostics file."""
    files = _guide().split("## Results files")[1].split("### Results file versions")[0]
    headline, diagnostics = files.split("\nThe diagnostics file holds ")
    headline = headline.split("`results.json` holds the headline:\n")[1]
    for section in HEADLINE_SECTIONS:
        assert f"`{section}`" in headline, section
    for section in EXHIBIT_SECTIONS:
        assert f"`{section}`" in diagnostics, section
        assert f"`{section}`" not in headline, section


def test_the_results_guide_states_the_current_schema_version():
    """docs/setup-run-results.md states the version as a number in prose."""
    versions = _guide().split("### Results file versions")[1].split("\n### ")[0]
    stated = re.findall(r"`schema_version` is an integer[^;]*; it is (\d+)\.", versions)
    assert stated == [str(RESULTS_SCHEMA_VERSION)]
    for name in (RESULTS_SCHEMA_FILE, DIAGNOSTICS_SCHEMA_FILE):
        committed = json.loads(schema_path(name).read_bytes())
        assert committed["properties"]["schema_version"]["const"] == int(stated[0])


def _records_and_fields(document: dict[str, Any]) -> tuple[int, int]:
    """A schema document's named definitions, and the fields of every record
    in it, the document's own included."""

    def fields(node: object) -> int:
        if isinstance(node, dict):
            return len(node.get("properties", {})) + sum(map(fields, node.values()))
        if isinstance(node, list):
            return sum(map(fields, node))
        return 0

    return len(document["$defs"]), fields(document)


def test_the_headline_document_keeps_its_size():
    """``results.json`` is the file others build on, so its layout stays
    small. The sizes are pinned so that a field added to the headline, or an
    exhibit moved into it, is a reviewed decision: a new report-only analysis
    belongs in ``ResultsExhibits``, which the diagnostics file holds and whose
    size is not pinned."""
    headline = json.loads(schema_path(RESULTS_SCHEMA_FILE).read_bytes())
    records, fields = _records_and_fields(headline)
    assert (records, fields) == (66, 315)
    # The results guide states the size it describes.
    assert f"{fields} fields in {records} record types" in _guide()
    diagnostics = json.loads(schema_path(DIAGNOSTICS_SCHEMA_FILE).read_bytes())
    # Most of a report's layout is in its exhibits.
    assert _records_and_fields(diagnostics)[1] > _records_and_fields(headline)[1]


def test_the_documents_are_2020_12_schemas_pinned_to_the_version():
    for schema in results_schemas().values():
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == DIALECT
        assert schema["additionalProperties"] is False
        properties = schema["properties"]
        assert properties["schema_version"]["const"] == RESULTS_SCHEMA_VERSION
        assert {"schema_version", "reasons"} <= set(schema["required"])
        # The reasons table maps each code to its text.
        table = Draft202012Validator(
            {**properties["reasons"], "$defs": schema["$defs"]}
        )
        assert properties["reasons"]["description"]
        assert schema["$defs"]["ReasonCode"]["description"]
        assert table.is_valid({reason_code("A reason."): "A reason."})
        assert not table.is_valid({"A reason.": "A reason."})
        assert not table.is_valid({reason_code("A reason."): 1})


class _Disclosure(TypedDict):
    status: Literal["available", "withheld"]
    reason: str | None
    interval_unavailable_reason: str | None
    state_reasons: dict[str, str]
    conditional_eal_unavailable_reasons_by_epistemic_group: dict[str, dict[str, str]]
    withheld_arms: dict[str, str]
    note: str
    count: NotRequired[int]


def _results_document(monkeypatch, contract) -> dict[str, Any]:
    """The ``results.json`` document generated with ``contract`` as the
    headline contract."""
    monkeypatch.setattr(results_schema, "HeadlineResults", contract)
    return results_schemas()[RESULTS_SCHEMA_FILE]


def test_a_contract_stores_its_reasons_as_intern_reasons_does(monkeypatch):
    """A string under a reason key, and every string under a key that holds
    reasons, is a code; other strings are stored as written."""
    validator = Draft202012Validator(_results_document(monkeypatch, _Disclosure))
    payload = {
        "status": "withheld",
        "reason": "Unconverged.",
        "interval_unavailable_reason": None,
        "state_reasons": {"state": "Not retained."},
        "conditional_eal_unavailable_reasons_by_epistemic_group": {
            "gross": {"respondent": "No finite mean."}
        },
        "withheld_arms": {"arm": "Unconverged."},
        "note": "Prose that is not a reason.",
    }
    stored = intern_reasons(payload)
    assert list(validator.iter_errors(stored)) == []
    assert stored["withheld_arms"] == payload["withheld_arms"]
    for changed in (
        {**payload, "reasons": stored["reasons"]},
        {**stored, "state_reasons": payload["state_reasons"]},
        {**stored, "notes": "A misspelt field."},
        {**stored, "count": 1.5},
        {**stored, "status": "done"},
        {key: value for key, value in stored.items() if key != "note"},
    ):
        assert not validator.is_valid(changed), changed


def test_a_contract_declares_every_value_type(monkeypatch):
    class Loose(TypedDict):
        value: Any

    with pytest.raises(TypeError, match="not Any"):
        _results_document(monkeypatch, Loose)


@pytest.mark.parametrize(
    "record, model, authored",
    [
        (ExecutionPlanRecord, ExecutionPlan, True),
        (AuthoredComparisonRequirement, ComparisonRequirement, True),
        (ComparisonRequirementRecord, ComparisonRequirement, False),
        (EvidenceRecord, Evidence, False),
    ],
)
def test_a_setup_record_has_its_models_fields(record, model, authored):
    """A dumped setup model holds every field; a plan file as authored may
    omit each field with a default."""
    fields = model.model_fields
    assert record.__required_keys__ | record.__optional_keys__ == set(fields)
    assert record.__required_keys__ == (
        {name for name, field in fields.items() if field.is_required()}
        if authored
        else set(fields)
    )


def test_the_control_and_provenance_records_have_their_producers_fields():
    gate = TailGateControls(views=("gross",), levels=(0.9,))
    assert TailGateControlsRecord.__required_keys__ == set(gate.record())
    assert TailGateSettings.__optional_keys__ == set(gate.record_controls())
    small = SmallEventControls().record()
    assert SmallEventControlsRecord.__required_keys__ == set(small)
    assert SmallEventSettings.__optional_keys__ == set(small)
    assert CodeProvenance.__required_keys__ == set(setup_code_provenance("fixture"))
