"""results.json and its diagnostics file: which sections each holds, interned
reasons, the schema version, the files' schemas and the analysis identity, and
the readable summary's study header."""

import copy
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.contracts.incident import ComparisonRequirement, ExecutionPlan
from catmodel.contracts.results import (
    EXHIBIT_SECTIONS,
    HEADLINE_SECTIONS,
    RESULTS_SCHEMA_VERSION,
    NativeResults,
    intern_reasons,
    reason_code,
    resolve_reasons,
)
from catmodel.contracts.results_schema import RESULTS_SCHEMA_FILE, schema_errors
from catmodel.hashing import canonical_json_sha256
from catmodel.results.execution import run_setup
from catmodel.results.results_export import export_results
from catmodel.results.results_file import (
    decode_results,
    diagnostics_path,
    load_results,
    results_payload,
    write_results,
)
from catmodel.results.results_summary import results_summary, write_results_summary
from catmodel.setup.store import build_setup
from test_support.run_setup import with_execution
from test_support.study import stipulated_study
from tests.test_reduction import draw, reduce

UNCONVERGED = "Unconverged at 64 years per state."


def test_every_reason_becomes_one_code_in_one_table():
    payload = {
        "statistics": {
            "var::0.9": {"status": "withheld", "reason": UNCONVERGED},
            "tvar::0.9": {"status": "withheld", "reason": UNCONVERGED},
            "annual_loss_median": {"status": "available", "reason": None},
        },
        "tail": {"tvar_interval_unavailable_reason": "No second moment."},
        "probability_withheld_reasons": [UNCONVERGED],
        "conditional_eal_unavailable_reasons_by_epistemic_group": {
            "gross": {"a": "No finite mean."}
        },
        "interpretation": "Prose that is not a reason stays as written.",
    }
    encoded = intern_reasons(payload)
    table = encoded["reasons"]
    assert table == {
        reason_code(text): text
        for text in (UNCONVERGED, "No second moment.", "No finite mean.")
    }
    code = reason_code(UNCONVERGED)
    assert len(code) == 12 and int(code, 16) >= 0
    assert encoded["statistics"]["var::0.9"]["reason"] == code
    assert encoded["statistics"]["tvar::0.9"]["reason"] == code
    assert encoded["statistics"]["annual_loss_median"]["reason"] is None
    assert encoded["probability_withheld_reasons"] == [code]
    assert encoded["interpretation"] == payload["interpretation"]
    assert UNCONVERGED not in json.dumps(
        {key: value for key, value in encoded.items() if key != "reasons"}
    )
    assert resolve_reasons(encoded) == payload


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda table, code: table.update({code: "A changed reason."}), "not the code"),
        (lambda table, code: table.pop(code), "has no entry"),
        (lambda table, code: table.update({reason_code("x"): "x"}), "no reason uses"),
    ],
)
def test_resolving_refuses_an_edited_or_incomplete_reasons_table(edit, message):
    encoded = intern_reasons({"reason": UNCONVERGED})
    edit(encoded["reasons"], reason_code(UNCONVERGED))
    with pytest.raises(ValueError, match=message):
        resolve_reasons(encoded)
    with pytest.raises(ValueError, match="must map reason codes"):
        resolve_reasons({"reason": "abc", "reasons": ["abc"]})


class Run(NamedTuple):
    report: NativeResults
    # The run's results file, with its diagnostics file beside it.
    results: Path


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Run:
    """A run of the stipulated study, with a reason inside its analysis
    identity: its report as the run returns it, and its results files."""
    root = tmp_path_factory.mktemp("results-file")
    plan = ExecutionPlan(
        plan_id="results-file",
        required_arms=("reference",),
        simulations=64,
        seed=7,
        requested_views=("gross",),
        comparison_requirements=(
            ComparisonRequirement(
                family_id="calendar",
                scenario_ids=("scenario",),
                status="unavailable",
                reason="Not identified.",
                held_fixed=("All quantities outside the perturbation's scope.",),
            ),
        ),
    )
    package = build_setup(
        destination=root / "setup",
        setup=with_execution(stipulated_study(), plan, setup_id="results-file"),
        code_commit="fixture",
    )
    report = run_setup(
        setup_path=package.root, output=root / "run", code_commit="fixture"
    )
    return Run(report, root / "run" / "results.json")


def _stored(run: Run) -> dict:
    return json.loads(run.results.read_bytes())


def _changed_results(run: Run, directory: Path, change) -> Path:
    """A copy of the run's results file in ``directory`` after ``change``
    edits its content."""
    payload = _stored(run)
    change(payload)
    directory.mkdir()
    path = directory / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# The sections a reader of headline numbers needs, and the report-only
# analyses beside them, written out here so that moving a section between the
# files is a change to this test as well as to the contract.
HEADLINE = {
    "schema_version",
    "analysis_sha256",
    "analysis_spec",
    "status",
    "study",
    "coverage",
    "limitations",
    "arms",
    "comparisons",
    "comparison_reference_arms",
    "reference_comparisons",
    "rankings_available",
    "rankings",
    "comparison_requirements",
    "comparison_families",
    "scenario_contributions",
    "producer_provenance",
    "execution_context",
}
EXHIBITS = {
    "sensitivity_summary",
    "capability_scenarios",
    "small_event_aggregation",
    "tail_gate",
    "scenario_groups",
    "workload",
}


def test_results_and_diagnostics_files_restore_the_report(run, tmp_path):
    report = run.report
    path = tmp_path / "results.json"
    write_results(path, report)
    assert diagnostics_path(path) == tmp_path / "results.diagnostics.json"
    stored = json.loads(path.read_bytes())
    assert stored["schema_version"] == RESULTS_SCHEMA_VERSION
    assert "Not identified." in stored["reasons"].values()
    # As JSON, which stores tuples as lists.
    original = json.loads(json.dumps(report))
    assert set(original) == HEADLINE | EXHIBITS | {"diagnostics"}
    assert (set(HEADLINE_SECTIONS), set(EXHIBIT_SECTIONS)) == (HEADLINE, EXHIBITS)
    # results.json holds the headline, and the diagnostics file the exhibits
    # and the view diagnostics, tied to the results by their analysis identity.
    assert set(stored) == HEADLINE | {"reasons"}
    diagnostics = json.loads(diagnostics_path(path).read_bytes())
    assert set(diagnostics) == EXHIBITS | {
        "schema_version",
        "analysis_sha256",
        "arms",
        "reasons",
    }
    assert diagnostics["analysis_sha256"] == report["analysis_sha256"]
    headline = cast(Mapping[str, Any], load_results(path))
    assert headline == {key: original[key] for key in HEADLINE}
    assert load_results(path, diagnostics=True) == original
    # A headline read alone says where an exhibit is, and the summary, which
    # would leave the exhibits out, refuses it.
    for section in (*sorted(EXHIBITS), "diagnostics"):
        with pytest.raises(ValueError, match=f"no `{section}`: it was read from"):
            headline[section]
        assert headline.get(section) is None
    with pytest.raises(KeyError):
        headline["no_such_section"]
    with pytest.raises(
        ValueError, match=r"no exhibits: it was read from results\.json"
    ):
        results_summary(headline)
    # The headline keeps each tail level's intervals; the per-state support
    # is in the diagnostics.
    tail = report["arms"]["reference"]["metrics"]["gross"]["tail_metrics"]["0.99"]
    assert set(tail) >= {"var_numerical_interval_95", "tvar_numerical_interval_95"}
    views = diagnostics["arms"]["reference"]["views"]
    assert (
        "tail_support_by_stratum" in views["gross"]["tail_diagnostics"]["tails"]["0.99"]
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda payload: payload.update({"schema_version": 0}), "schema version 0"),
        (
            lambda payload: payload.update(
                {"schema_version": float(RESULTS_SCHEMA_VERSION)}
            ),
            rf"schema version {RESULTS_SCHEMA_VERSION}\.0",
        ),
        (
            lambda payload: payload.update({"schema_version": True}),
            "schema version True",
        ),
        (lambda payload: payload.pop("schema_version"), "schema version is not stated"),
        (
            lambda payload: payload["analysis_spec"].update({"plan_sha256": "other"}),
            "not the digest of their analysis_spec",
        ),
        (
            lambda payload: payload.update({"analysis_sha256": "0" * 64}),
            "not the digest of their analysis_spec",
        ),
    ],
)
def test_reading_refuses_another_version_or_a_changed_identity(
    run, tmp_path, change, message
):
    changed = _changed_results(run, tmp_path / "changed", change)
    with pytest.raises(ValueError, match=message):
        load_results(changed)
    # The readable summary refuses it too, and writes nothing.
    with pytest.raises(ValueError, match=message):
        write_results_summary(results_path=changed, output=tmp_path / "summary")
    assert not (tmp_path / "summary").exists()


def test_a_diagnostics_file_must_belong_to_its_results(run, tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_results(first, run.report)
    changed = copy.deepcopy(run.report)
    changed["analysis_spec"]["plan_sha256"] = "other"
    changed["analysis_sha256"] = canonical_json_sha256(changed["analysis_spec"])
    write_results(second, changed)
    diagnostics_path(first).write_bytes(diagnostics_path(second).read_bytes())
    assert load_results(first)["analysis_sha256"] != changed["analysis_sha256"]
    with pytest.raises(ValueError, match="another analysis"):
        load_results(first, diagnostics=True)


def _gross(payload):
    return payload["arms"]["reference"]["metrics"]["gross"]


# Each edit keeps the analysis specification, so the analysis digest still
# matches; the path is the one the refusal names.
MALFORMED = {
    "a missing statistic": (
        lambda payload: _gross(payload).pop("tail_metrics"),
        "arms.reference.metrics.gross.tail_metrics is missing",
    ),
    "a retyped statistic": (
        lambda payload: _gross(payload).update(expected_annual_loss="fourteen"),
        'arms.reference.metrics.gross.expected_annual_loss is "fourteen", '
        "not a number or null",
    ),
    "a missing section": (lambda payload: payload.pop("study"), "study is missing"),
    "an unknown status": (
        lambda payload: payload.update(status="fine"),
        'status is "fine", not one of "complete", "partial"',
    ),
    "an unknown field": (
        lambda payload: payload.update(surprise=1),
        "surprise is not a field of the schema",
    ),
    "a missing section that holds reasons": (
        lambda payload: payload.pop("coverage"),
        "coverage is missing",
    ),
    "a level's retyped statistic": (
        lambda payload: _gross(payload)["tail_metrics"]["0.99"].update(var="110"),
        'arms.reference.metrics.gross.tail_metrics["0.99"].var is "110", '
        "not a number or null",
    ),
    "a reason stored as text": (
        lambda payload: payload["coverage"]["scenarios"][0].update(reason="Left out."),
        'coverage.scenarios[0].reason is "Left out.", which does not match',
    ),
}


def _refusal(run: Run, reader: str, results: Path, output: Path) -> str:
    """Why a reader of the results file ``results`` refuses it: the summary
    command, or an export given the file (``export_results``; the export
    command reduces the capture itself). Neither writes ``output``."""
    if reader == "summary":
        arguments = ["results", "summary", "--results", results, "--output", output]
        result = CliRunner().invoke(cli, [str(argument) for argument in arguments])
        # A refusal, not an error the command did not expect.
        assert isinstance(result.exception, SystemExit), result.exception
        assert result.exit_code == 1, result.output
        (error,) = result.output.splitlines()
        assert error.startswith("Error: ")
        reason = error.removeprefix("Error: ")
    else:
        root = run.results.parents[1]
        with pytest.raises(ValueError) as refusal:
            export_results(
                setup_path=root / "setup",
                capture_directory=root / "run" / "capture",
                results_path=results,
                output=output,
            )
        reason = str(refusal.value)
    assert not output.exists()
    return reason


@pytest.mark.parametrize("reader", ["summary", "export"])
@pytest.mark.parametrize("case", MALFORMED)
def test_the_readers_refuse_a_file_that_does_not_match_its_schema(
    run, tmp_path, case, reader
):
    """An edited results file that keeps its analysis specification is refused
    with the path of what is wrong, and its reader writes nothing."""
    change, problem = MALFORMED[case]
    changed = _changed_results(run, tmp_path / "changed", change)
    reason = _refusal(run, reader, changed, tmp_path / "output")
    assert reason.startswith(
        "the results file does not match its schema "
        "(contracts/results/results.schema.json): "
    )
    assert problem in reason


def test_the_unchanged_file_passes_both_readers(run, tmp_path):
    root = run.results.parents[1]
    summary = ["results", "summary", "--results", run.results]
    summary += ["--output", tmp_path / "summary"]
    result = CliRunner().invoke(cli, [str(argument) for argument in summary])
    assert result.exit_code == 0, result.output
    export_results(
        setup_path=root / "setup",
        capture_directory=root / "run" / "capture",
        results_path=run.results,
        output=tmp_path / "export",
    )


def test_the_export_command_exports_what_the_capture_reduces_to(run, tmp_path):
    """The command takes no results file: its results are those of the run,
    which reduced the same capture under the same plan."""
    root = run.results.parents[1]
    arguments = ["results", "export", "--setup", root / "setup"]
    arguments += ["--capture", root / "run" / "capture", "--output", tmp_path / "out"]
    result = CliRunner().invoke(cli, [str(argument) for argument in arguments])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(tmp_path / "out" / "export.json")
    exported = load_results(tmp_path / "out" / "native" / "results.json")
    assert exported["arms"] == load_results(run.results)["arms"]


def test_a_refusal_lists_the_first_departures_and_counts_the_rest(run, tmp_path):
    def change(payload):
        for key in (
            "study",
            "coverage",
            "producer_provenance",
            "limitations",
            "scenario_contributions",
        ):
            del payload[key]

    with pytest.raises(ValueError) as refusal:
        load_results(_changed_results(run, tmp_path / "changed", change))
    assert (
        "study is missing; coverage is missing; limitations is missing; "
        "and 2 more; derive current results"
    ) in str(refusal.value)


def _tail(payload):
    return _gross(payload)["tail_metrics"]["0.99"]


def _thresholds(payload):
    return payload["analysis_spec"]["reduction_spec"]["joint_thresholds"]


@pytest.mark.parametrize(
    "change, problem",
    [
        # An available view with another status is nearer an available view,
        # which it fails once, than an unavailable one, which has other fields.
        (
            lambda payload: _gross(payload).update(status="unavailable"),
            'arms.reference.metrics.gross.status is "unavailable", not "available"',
        ),
        # A record with two defects is nearer the record than null.
        (
            lambda payload: _tail(payload)["tvar_estimator"].update(
                method="guess", coefficient="high"
            ),
            'tail_metrics["0.99"].tvar_estimator.coefficient is "high", not a number',
        ),
        # A value of none of the union's types.
        (
            lambda payload: payload["arms"].update(reference="reduced"),
            'arms.reference is "reduced", not an object',
        ),
        (
            lambda payload: payload["coverage"]["scenarios"][0].update(reason=5),
            "coverage.scenarios[0].reason is 5, not a string or null",
        ),
        # A joint threshold is a view and a loss.
        (
            lambda payload: _thresholds(payload).append(["gross"]),
            "joint_thresholds[0] is an array of length 1, not 2",
        ),
        (
            lambda payload: _thresholds(payload).append(["gross", 1.0, 2.0]),
            "joint_thresholds[0] is an array of length 3, not 2",
        ),
        # A reasons table's keys are reason codes.
        (
            lambda payload: payload["reasons"].update({"Left out.": "Left out."}),
            'reasons has the key "Left out.", which does not match',
        ),
    ],
)
def test_a_refusal_names_the_value_and_what_the_schema_asks_of_it(run, change, problem):
    payload = _stored(run)
    assert schema_errors(payload, RESULTS_SCHEMA_FILE) == []
    change(payload)
    errors = schema_errors(payload, RESULTS_SCHEMA_FILE)
    assert any(problem in error for error in errors), errors


def test_a_diagnostics_file_must_match_its_schema(run, tmp_path):
    shutil.copy(run.results, tmp_path / "results.json")
    diagnostics = json.loads(diagnostics_path(run.results).read_bytes())
    view = diagnostics["arms"]["reference"]["views"]["gross"]
    view["mean_diagnostic"] = view.pop("mean_diagnostics")
    diagnostics_path(tmp_path / "results.json").write_text(json.dumps(diagnostics))
    assert load_results(tmp_path / "results.json") == load_results(run.results)
    with pytest.raises(ValueError) as refusal:
        load_results(tmp_path / "results.json", diagnostics=True)
    message = str(refusal.value)
    assert message.startswith(
        "the diagnostics file does not match its schema "
        "(contracts/results/results.diagnostics.schema.json): "
    )
    assert "arms.reference.views.gross.mean_diagnostics is missing" in message
    assert (
        "arms.reference.views.gross.mean_diagnostic is not a field of the schema"
        in message
    )


def test_an_exhibit_is_read_only_from_the_diagnostics_file(run, tmp_path):
    """The two files hold disjoint sections: an exhibit in a results payload
    is refused, a diagnostics file must hold every exhibit, and a report
    without its exhibits and view diagnostics has no diagnostics file to
    write."""
    stored = _stored(run)
    with pytest.raises(ValueError, match=r"results\.json does not hold tail_gate"):
        decode_results({**stored, "tail_gate": {}})
    shutil.copy(run.results, tmp_path / "results.json")
    diagnostics = json.loads(diagnostics_path(run.results).read_bytes())
    del diagnostics["tail_gate"]
    diagnostics_path(tmp_path / "results.json").write_text(json.dumps(diagnostics))
    with pytest.raises(ValueError, match="diagnostics file does not match its schema"):
        load_results(tmp_path / "results.json", diagnostics=True)
    with pytest.raises(ValueError, match=r"without sensitivity_summary, .*diagnostics"):
        write_results(tmp_path / "headline.json", load_results(run.results))
    assert not (tmp_path / "headline.json").exists()
    with pytest.raises(ValueError, match="neither results file holds surprise"):
        write_results(tmp_path / "unknown.json", {**run.report, "surprise": 1})


@pytest.mark.parametrize("reader", ["summary", "export"])
def test_the_readers_need_the_diagnostics_file_beside_the_results(
    run, tmp_path, reader
):
    """The summary and the export read the exhibits, so results copied without
    their diagnostics file are refused, with the file's name."""
    (tmp_path / "copied").mkdir()
    copied = tmp_path / "copied" / "results.json"
    shutil.copy(run.results, copied)
    assert load_results(copied) == load_results(run.results)
    assert _refusal(run, reader, copied, tmp_path / "output").startswith(
        "results.json has no diagnostics file results.diagnostics.json "
        "beside it, which holds its exhibits and view diagnostics; "
    )


def _report():
    """A reduction's report in memory, which has none of a run's sections."""
    report = reduce({"base": {"s": draw(values=(0, 80, 40, 40) * 16)}})
    report["schema_version"] = RESULTS_SCHEMA_VERSION
    return report


def test_decoding_in_memory_does_not_check_the_schema():
    """``decode_results`` restores what ``results_payload`` stores of a report
    in memory; only reading a file checks it against the results schema."""
    report = _report()
    decoded = decode_results(results_payload(report))
    assert "study" not in decoded
    assert decoded["arms"] == json.loads(json.dumps(report["arms"]))


def _metric(eal, status="available", reason=None):
    return {
        "status": "available",
        "basis": {"currency": "EUR"},
        "expected_annual_loss": eal,
        "statistics": {"expected_annual_loss": {"status": status, "reason": reason}},
        "tail_metrics": {},
    }


def _study():
    """A complete study record, as the summary's header reads it."""
    return {
        "setup_id": "lab-study",
        "scope_id": "frontier-lab",
        "title": "A stipulated frontier lab",
        "period_start": "2027-01-01T00:00:00Z",
        "period_end": "2028-01-01T00:00:00Z",
        "forecast_as_of": "2026-12-01T00:00:00Z",
        "data_cutoff": "2026-11-01T12:00:00Z",
        "availability_cutoff": "2026-11-01T00:00:00Z",
        "information_date": "2026-11-01T00:00:00Z",
        "scenario_ids": ["leak"],
        "monetary_bases": [{"currency": "EUR", "monetary_basis": "EUR, 2027."}],
        "entities": [
            {
                "entity_id": "lab",
                "kind": "lab",
                "display_name": "Frontier lab",
                "revision": None,
                "status": "hypothetical",
            },
            {
                "entity_id": "model",
                "kind": "model",
                "display_name": "Frontier model",
                "revision": "r2",
                "status": "hypothetical",
            },
        ],
        "deployments": [{"deployment_id": "api", "model_id": "model"}],
    }


def test_the_summary_names_its_study_and_headline_measures():
    report = copy.deepcopy(_report())
    report["study"] = _study()
    report["arms"]["base"]["metrics"] = {
        "welfare": _metric(2.5e9),
        "lab_loss::lab": _metric(4e8, "withheld", "The tail gate is unconverged."),
    }
    markdown, rows = results_summary(report)
    header = markdown.split("## Headline")[0]
    assert header.startswith("# Results summary: A stipulated frontier lab\n")
    for line in (
        "- **Setup:** `lab-study` (scope `frontier-lab`)",
        "- **Study period:** 2027-01-01 to 2028-01-01 (forecast as of 2026-12-01; data cutoff 2026-11-01T12:00:00Z)",
        "- **Lab:** Frontier lab (`lab`, hypothetical)",
        "- **Model:** Frontier model (`model`, revision `r2`, hypothetical)",
        "- **Deployment:** `api` (runs `model`)",
        "- **Currency and monetary basis:** EUR (EUR, 2027)",
        f"- **Analysis:** `{report['analysis_sha256'][:12]}`",
    ):
        assert line in header.splitlines()
    assert "| Societal harm | `welfare` | 2.5B EUR | available |" in markdown
    assert (
        "| Lab economic loss, Frontier lab | `lab_loss::lab` | (400M) EUR | "
        "withheld: The tail gate is unconverged. |"
    ) in markdown
    assert {
        row["unit"] for row in rows if row["statistic"] == "expected_annual_loss"
    } >= {"EUR"}


def _headline_rows(report) -> list[str]:
    """The headline table's rows below its header, as the summary prints them."""
    table = results_summary(report)[0].split("## Headline")[1].split("How to read")[0]
    return [line for line in table.splitlines() if line.startswith("|")][2:]


_UNAVAILABLE = {"status": "unavailable", "reason": "Component natures are missing."}
_PEER_ROWS_NOT_REQUESTED = [
    "| Societal harm | `welfare` | — | not requested: The view was not requested. |",
    "| Lab economic loss, Frontier lab | `lab_loss::lab` | — | not requested: The view was not requested. |",
]


@pytest.mark.parametrize(
    ("requested", "rows"),
    [
        # Neither peer measure: the pooled annual-loss views the plan requests,
        # gross first, and no occurrence maximum or per-scenario view.
        (
            {
                "financial": _metric(5.0),
                "oep_gross": _metric(12.5),
                "scenario::leak::gross": _metric(14.0),
                "gross": _metric(14.0),
            },
            [
                "| Gross component loss | `gross` | 14 EUR | available |",
                "| Insured payment | `financial` | 5 EUR | available |",
            ],
        ),
        (
            {
                "net_resource_cost": _metric(9.0, "withheld", "Unconverged."),
                "gross": _UNAVAILABLE,
            },
            [
                "| Gross component loss | `gross` | — | unavailable: Component natures are missing. |",
                "| All transfers cancelled | `net_resource_cost` | (9) EUR | withheld: Unconverged. |",
            ],
        ),
        # A requested peer measure keeps the peer rows, whatever its status
        # and whatever else the plan requests.
        (
            {"welfare": _UNAVAILABLE, "gross": _metric(14.0)},
            [
                "| Societal harm | `welfare` | — | unavailable: Component natures are missing. |",
                _PEER_ROWS_NOT_REQUESTED[1],
            ],
        ),
        (
            {"lab_loss::lab": _metric(4.0), "gross": _metric(14.0)},
            [
                _PEER_ROWS_NOT_REQUESTED[0],
                "| Lab economic loss, Frontier lab | `lab_loss::lab` | 4 EUR | available |",
            ],
        ),
        # No pooled annual-loss view either: the peer rows, not requested.
        (
            {"oep_gross": _metric(12.5), "scenario::leak::gross": _metric(14.0)},
            _PEER_ROWS_NOT_REQUESTED,
        ),
        ({}, _PEER_ROWS_NOT_REQUESTED),
    ],
)
def test_the_headline_rows_follow_the_views_the_plan_requests(requested, rows):
    """The headline gives societal harm and each named lab's loss wherever
    the reference arm reports either. Where it reports neither, the headline
    gives the requested ones of gross, net_resource_cost and financial."""
    report = copy.deepcopy(_report())
    report["study"] = _study()
    report["arms"]["base"]["metrics"] = requested
    assert _headline_rows(report) == rows


def test_a_study_without_a_named_lab_keeps_its_headline_rule():
    report = copy.deepcopy(_report())
    report["study"] = {
        **_study(),
        "entities": [row for row in _study()["entities"] if row["kind"] != "lab"],
    }
    lab = "| Lab economic loss | `lab_loss` | — | not requested: No named lab is authored in this study. |"
    report["arms"]["base"]["metrics"] = {"welfare": _metric(3.0)}
    assert _headline_rows(report) == [
        "| Societal harm | `welfare` | 3 EUR | available |",
        lab,
    ]
    report["arms"]["base"]["metrics"] = {"oep_gross": _metric(12.5)}
    assert _headline_rows(report) == [_PEER_ROWS_NOT_REQUESTED[0], lab]
    report["arms"]["base"]["metrics"] = {"gross": _metric(14.0)}
    assert _headline_rows(report) == [
        "| Gross component loss | `gross` | 14 EUR | available |"
    ]


def _evidence(origin: str) -> dict:
    """An evidence record as a result's current assessment holds it."""
    return {"evidence_id": f"{origin}-record", "origin": origin}


# Each limitation of a result's list, in the list's order: the setup's, the
# run's, then the model-wide items. An entry is the limitation's source, the
# number of the docs/limitations.md item it refers to (None without a
# reference) and what the item needs of the study to apply: elicited evidence,
# several scenarios, or nothing.
_LIMITATIONS = (
    ("setup", None, None),
    ("run", 4, None),
    ("model", 1, "elicited"),
    ("model", 3, None),
    ("model", 5, "several"),
    ("model", 6, None),
    ("model", 7, None),
    ("model", 8, "elicited"),
    ("model", 9, None),
    ("model", 10, None),
)


def _limitations_item(row) -> int | None:
    """The number of the docs/limitations.md item a limitation refers to."""
    reference = row.get("reference")
    if reference is None:
        return None
    document, _, anchor = reference.partition("#")
    assert document == "docs/limitations.md"
    return int(anchor.split("-")[0])


def _limitations_report(evidence, scenario_ids):
    """A report of a study of ``scenario_ids`` with the limitations list a
    run gives a setup without a sensitivity arm, whose current assessment
    records ``evidence``; None leaves the assessment out."""
    from types import SimpleNamespace

    from catmodel.contracts.run_setup import RunSetup
    from catmodel.results.result_disclosures import result_limitations

    report = copy.deepcopy(_report())
    report["study"] = {**_study(), "scenario_ids": list(scenario_ids)}
    package = SimpleNamespace(
        limitations=("A setup note.",),
        scope=SimpleNamespace(
            populations=[SimpleNamespace(scenario_id=key) for key in scenario_ids]
        ),
    )
    report["limitations"] = result_limitations(cast(RunSetup, package), report)
    assert [
        (row["source"], _limitations_item(row)) for row in report["limitations"]
    ] == [(source, item) for source, item, _ in _LIMITATIONS]
    if evidence is not None:
        report["analysis_spec"]["capture_provenance"] = {
            "current_assessment": {"evidence": evidence}
        }
    return report


@pytest.mark.parametrize(
    ("evidence", "scenario_ids", "left_out", "printed_count"),
    [
        # Authored, one scenario: no respondents and nothing to be independent of.
        ([], ("leak",), {"elicited", "several"}, 7),
        ([_evidence("stipulation")], ("leak",), {"elicited", "several"}, 7),
        # Authored, several scenarios: their independence still applies.
        ([_evidence("stipulation")], ("leak", "spill"), {"elicited"}, 8),
        (
            [_evidence("observation"), _evidence("fit")],
            ("leak", "spill", "theft"),
            {"elicited"},
            8,
        ),
        # Collected, several scenarios: the whole list.
        ([_evidence("fit"), _evidence("elicitation")], ("leak", "spill"), set(), 10),
        # Collected, one scenario: every item but the scenarios' independence.
        ([_evidence("elicitation")], ("leak",), {"several"}, 9),
        # A report that does not record its setup's evidence keeps the items
        # that describe a collected study.
        (None, ("leak", "spill"), set(), 10),
        (None, ("leak",), {"several"}, 9),
    ],
)
def test_the_summary_prints_the_limitations_that_apply_to_its_study(
    evidence, scenario_ids, left_out, printed_count
):
    """The summary leaves out the two model-wide limitations that describe a
    collected study where the setup's evidence holds no elicited judgment,
    and the one on the independence of scenarios where the study has one
    scenario. It keeps every other item in order and numbers what it prints
    consecutively. The results' own list is not changed."""
    report = _limitations_report(evidence, scenario_ids)
    listed = copy.deepcopy(report["limitations"])
    section = results_summary(report)[0].split("## Limitations")[1]
    printed = [line for line in section.splitlines() if line]
    applies = [needs not in left_out for _, _, needs in _LIMITATIONS]
    kept = [row for row, applied in zip(listed, applies, strict=True) if applied]
    assert len(printed) == len(kept) == printed_count
    for number, (line, row) in enumerate(zip(printed, kept, strict=True), start=1):
        assert line.startswith(f"{number}. {row['text']}")
    for row, applied in zip(listed, applies, strict=True):
        assert any(row["text"] in line for line in printed) == applied
    assert report["limitations"] == listed


def test_the_summary_shows_how_far_respondents_lie_apart():
    """The pooled EAL is a weighted mean of the respondents' own EALs. The
    summary reads each from the result and reports the spread, the weighted
    median and the share of the pooled EAL its largest respondent carries."""
    report = copy.deepcopy(_report())
    spec = report["analysis_spec"]
    # Four states: respondent a has two of weight 0.125 each, b and c one each.
    spec["state_weights"] = {"s1": 0.125, "s2": 0.125, "s3": 0.25, "s4": 0.5}
    spec["epistemic_groups"] = {"s1": "a", "s2": "a", "s3": "b", "s4": "c"}
    report["study"] = _study()
    report["arms"]["base"]["metrics"] = {"welfare": _metric(17.5)}
    report["arms"]["base"]["conditional_eal_by_epistemic_group"] = {
        "welfare": {"a": 40.0, "b": 10.0, "c": 10.0},
        # Not a pooled headline view, and a view lacking a respondent's EAL.
        "scenario::leak::welfare": {"a": 40.0, "b": 10.0, "c": 10.0},
        "gross": {"a": 40.0, "b": None, "c": 10.0},
    }
    markdown, rows = results_summary(report)
    section = markdown.split("## Respondent dispersion")[1].split("\n## ")[0]
    # By hand: weights a 0.25, b 0.25, c 0.5; mean 0.25*40 + 0.25*10 + 0.5*10
    # = 17.5; half the weight lies at or below 10; a carries 10 of 17.5.
    assert (
        "| `welfare` | 3 | 17.5 | 10 | 10 | 40 | 4× | 57.1% (`a`) |"
        in section.splitlines()
    )
    assert "scenario::leak" not in section and "`gross`" not in section
    found = {
        (row["statistic"], row["subject"]): row["value"]
        for row in rows
        if row["section"] == "respondent_dispersion"
    }
    assert found == {
        ("respondents", None): 3.0,
        ("lowest_respondent_eal", None): 10.0,
        ("weighted_median_respondent_eal", None): 10.0,
        ("highest_respondent_eal", None): 40.0,
        ("largest_respondent_share", "a"): pytest.approx(10 / 17.5),
    }
    # A result without respondents has no such section.
    del spec["epistemic_groups"]
    assert "Respondent dispersion" not in results_summary(report)[0]


def _contribution(scenario, eal, share):
    return {
        "scenario_id": scenario,
        "view": f"scenario::{scenario}::welfare",
        "status": "available",
        "reason": None,
        "expected_annual_loss": eal,
        "share_of_total": share,
    }


def _contributions_report():
    report = copy.deepcopy(_report())
    report["study"] = _study()
    report["scenario_contributions"] = {
        "status": "available",
        "measure": "welfare",
        "expected_annual_loss": 8e9,
        "total_reason": None,
        "scenarios": [
            _contribution("leak", 2e9, 0.25),
            {
                **_contribution("fraud", None, None),
                "status": "not_requested",
                "reason": "The view was not requested.",
            },
            _contribution("outage", 6e9, 0.75),
            _contribution("ban", 0.0, 0.0),
        ],
        "unallocated": {
            "expected_annual_loss": None,
            "reason": "Needs the pooled EAL and every scenario's EAL.",
            "annual_losses_reconcile": None,
        },
        "limitations": "Shares divide by the pooled EAL.",
    }
    return report


def _anchor_row(share, withheld=()):
    return {
        "status": "available",
        "gross_eal_share_at_or_above_large": share,
        "gross_eal_share_withheld_reasons": list(withheld),
    }


def _anchor_total(share):
    """A total row with the fields the summary's large-loss section reads."""
    return {
        **_anchor_row(share),
        "eal_alternatives": {},
        "two_anchor_tail": {"very_large_count": None},
        **dict.fromkeys(
            (
                "stated_large_count",
                "pooled_large_count",
                "modelled_large_count",
                "extrapolated_large_count",
                "stated_very_large_count",
                "pooled_very_large_count",
                "modelled_very_large_count",
            )
        ),
    }


def test_the_summary_gives_each_scenario_its_eal_and_its_large_incident_share():
    """The scenario table reads each scenario's EAL and share of the pooled
    EAL from ``scenario_contributions`` and, from the large-loss anchor
    exhibit, the share of its expected gross loss from incidents at or above
    the large-loss level, with the exhibit's own reason where it has none."""
    report = _contributions_report()
    report["sensitivity_summary"] = {
        "large_loss_anchor": {
            "levels": {"large_loss": 1e9, "very_large_loss": 1e11},
            "by_scenario": {
                "leak": _anchor_row(0.4),
                "outage": {
                    "status": "unavailable",
                    "reasons": ["No law at state a.", "No law at state b."],
                },
                "ban": _anchor_row(None, ("Components outside the shared total.",)),
            },
            "total": _anchor_total(0.125),
        }
    }
    markdown, rows = results_summary(report)
    section = markdown.split("## Expected annual loss by scenario")[1].split("\n## ")[0]
    lines = section.splitlines()
    assert lines[2] == (
        "Measure `welfare`. The large-incident share is the share of a scenario's "
        "expected gross loss from incidents of at least 1B EUR, under the run's laws; "
        "it is a share of gross loss whatever the measure."
    )
    # Scenarios in decreasing EAL, then the remainder and the pooled view.
    assert lines[4:12] == [
        "| Scenario | EAL | Share of EAL | Large-incident share | Note |",
        "| --- | --- | --- | --- | --- |",
        "| `outage` | 6B | 75.0% | — | large-incident share: No law at state a.; No law at state b. |",
        "| `leak` | 2B | 25.0% | 40.0% |  |",
        "| `ban` | 0 | 0.0% | — | large-incident share: Components outside the shared total. |",
        "| `fraud` | — | — | — | not_requested: The view was not requested. large-incident share: The scenario has no large-loss anchor. |",
        "| unallocated | — | — | — | Needs the pooled EAL and every scenario's EAL. |",
        "| total (pooled view) | 8B | 100.0% | 12.5% |  |",
    ]
    assert "Shares divide by the pooled EAL." in lines
    shares = {
        row["subject"]: (row["value"], row["unit"], row["status"], row["reason"])
        for row in rows
        if row["statistic"] == "gross_eal_share_at_or_above_large"
    }
    assert shares == {
        "leak": (0.4, "share", "available", None),
        "outage": (
            None,
            "share",
            "unavailable",
            "No law at state a.; No law at state b.",
        ),
        "ban": (None, "share", "unavailable", "Components outside the shared total."),
        "fraud": (
            None,
            "share",
            "unavailable",
            "The scenario has no large-loss anchor.",
        ),
        None: (0.125, "share", "available", None),
    }
    assert {
        (row["section"], row["view"])
        for row in rows
        if row["statistic"] == "gross_eal_share_at_or_above_large"
    } == {("scenario", "gross")}
    eal = {
        row["subject"]: (row["value"], row["status"])
        for row in rows
        if row["section"] == "scenario" and row["statistic"] == "expected_annual_loss"
    }
    assert eal == {
        "leak": (2e9, "available"),
        "fraud": (None, "not_requested"),
        "outage": (6e9, "available"),
        "ban": (0.0, "available"),
        "unallocated": (None, "unavailable"),
    }


def test_the_summary_says_why_large_incident_shares_are_missing():
    """A setup without a large-loss anchor has no such column; an exhibit
    withheld for a reweighted result gives its reason once."""
    report = _contributions_report()
    markdown, rows = results_summary(report)
    section = markdown.split("## Expected annual loss by scenario")[1].split("\n## ")[0]
    assert "| Scenario | EAL | Share of EAL | Note |" in section.splitlines()
    assert "large-incident" not in section.lower()
    assert not [
        row for row in rows if row["statistic"] == "gross_eal_share_at_or_above_large"
    ]
    report["sensitivity_summary"] = {
        "large_loss_anchor": {"status": "unavailable", "reason": "Reweighted."}
    }
    markdown, rows = results_summary(report)
    section = markdown.split("## Expected annual loss by scenario")[1].split("\n## ")[0]
    assert "| Scenario | EAL | Share of EAL | Note |" in section.splitlines()
    assert "Large-incident shares unavailable: Reweighted." in section.splitlines()
    assert [
        (row["subject"], row["value"], row["status"], row["reason"])
        for row in rows
        if row["statistic"] == "gross_eal_share_at_or_above_large"
    ] == [(None, None, "unavailable", "Reweighted.")]
    # A total with no anchored scenario says so.
    report["sensitivity_summary"] = {
        "large_loss_anchor": {
            "levels": None,
            "by_scenario": {},
            "total": {"status": "inapplicable"},
        }
    }
    markdown, rows = results_summary(report)
    section = markdown.split("## Expected annual loss by scenario")[1].split("\n## ")[0]
    assert (
        "| total (pooled view) | 8B | 100.0% | — | large-incident share: No scenario has a large-loss anchor. |"
        in section.splitlines()
    )
    assert "incidents of at least the large-loss level" in section


def test_a_results_file_without_scenario_contributions_says_so():
    report = copy.deepcopy(_report())
    report["study"] = _study()
    markdown, rows = results_summary(report)
    assert (
        "## Expected annual loss by scenario\n\nUnavailable: The results file has no scenario contributions."
        in markdown
    )
    report["scenario_contributions"] = {"status": "unavailable", "reason": "No view."}
    markdown, rows = results_summary(report)
    assert "## Expected annual loss by scenario\n\nUnavailable: No view." in markdown
    assert [
        (row["status"], row["reason"]) for row in rows if row["section"] == "scenario"
    ] == [("unavailable", "No view.")]
