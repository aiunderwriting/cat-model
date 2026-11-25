"""An export leaves captured prompts, responses and answers out unless asked,
records what it left out by digest, and still verifies."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.results.conditional import ConditionalCapture
from catmodel.results.execution import run_setup
from catmodel.results.results_export import export_results, load_results_export
from catmodel.setup.contextual_import import import_contextual_setup
from catmodel.setup.store import (
    MANIFEST,
    OMITTED_EVIDENCE,
    holds_captured_answers,
    load_setup,
    omitted_files,
)
from test_support.study import _package, _plan
from tests.test_contextual_study_import import _collection

# What the fixture's respondent writes as its reasoning, and how every prompt
# it is sent begins.
RESPONSE_TEXT = b"Explicitly synthetic software fixture"
PROMPT_TEXT = b"The lab, model and deployment described below are hypothetical"

SOURCE_CONVENTION = (
    "Native artifacts are authoritative and copied without redaction. This "
    "export does not confer permission to redistribute selected evidence."
)


def _files(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` by its path relative to it."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    ("name", "held"),
    [
        # A prompt as sent, a response, and the answers parsed from them.
        ("evidence/captured/calls/ab/request.json", True),
        ("evidence/captured/calls/ab/attempt-001.json", True),
        ("evidence/captured/contextual-risk.json", True),
        ("evidence/captured/contexts/3/calls/ab/attempt-002.json", True),
        ("evidence/captured/contexts/3/contextual-risk.json", True),
        ("evidence/captured/safeguard-weaker/contexts/0/contextual-risk.json", True),
        # A file of a kind no collection writes today, and a file that only
        # has a described file's name.
        ("evidence/captured/contexts/3/tool-log.json", True),
        ("evidence/captured/calls/ab/collection-plan.json", True),
        ("evidence/captured/calls/acquisition/ab.json", True),
        # What a study and a collection write about themselves.
        ("evidence/captured/public-study.json", False),
        ("evidence/captured/forecast.json", False),
        ("evidence/captured/public-frontier.json", False),
        ("evidence/captured/trend-posterior.json", False),
        ("evidence/captured/contexts/3/contextual-instrument.json", False),
        ("evidence/captured/contexts/3/collection-plan.json", False),
        ("evidence/captured/contexts/3/acquisition/ab.json", False),
        ("evidence/captured/parser-protocols/ab.json", False),
        # A source bundle, whatever its files are called.
        ("evidence/captured/source-bundle/evidence/source_cards.yaml", False),
        ("evidence/captured/contexts/3/source-bundle/evidence/calls/a.csv", False),
        # Everything outside the captured evidence.
        ("evidence/derived/excluded-samples.json", False),
        ("evidence/calls/ab/request.json", False),
        ("setup.json", False),
        ("alternatives/calls/contextual-risk.json", False),
    ],
)
def test_only_prompts_responses_and_answers_count_as_captured_answers(name, held):
    assert holds_captured_answers(name) is held


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    """A fixed-anchor collection imported, run and exported twice: by default
    (``default``) and with its answers (``whole``)."""
    root = tmp_path_factory.mktemp("export-answers")
    with pytest.MonkeyPatch.context() as patch:
        directory, source_bundle, _ = _collection(root, patch)
        package = import_contextual_setup(
            source=directory,
            source_bundle=source_bundle,
            destination=root / "setup",
            setup_id="offline-contextual",
            years_per_state=48,
            max_reference_state_years=4 * 4096,
            seed=19,
            code_commit="fixture",
        )
    run_setup(setup_path=package.root, output=root / "run", code_commit="fixture")
    indexes = {
        name: export_results(
            setup_path=package.root,
            capture_directory=root / "run/capture",
            output=root / name,
            code_commit="fixture",
            include_answers=include,
        )
        for name, include in (("default", False), ("whole", True))
    }
    return root, indexes


def _answers(setup: dict[str, bytes]) -> dict[str, str]:
    """The setup files that are a call's prompt or response or a collection's
    parsed answers, by what the fixture's collection writes, with the digest
    of the bytes the setup holds."""
    prefix = "evidence/captured/"
    return {
        name: _sha256(data)
        for name, data in setup.items()
        if name.startswith(prefix + "calls/") or name == prefix + "contextual-risk.json"
    }


def test_an_export_leaves_the_captured_answers_out_and_records_their_digests(
    imported,
):
    root, indexes = imported
    setup = _files(root / "setup")
    copy = _files(root / "default/native/setup")
    answers = _answers(setup)
    # Six calls, each a prompt and one response, and the parsed answers.
    assert len(answers) == 13
    # The copy holds every other file of the setup unchanged, and the record.
    assert set(copy) == set(setup) - set(answers) | {OMITTED_EVIDENCE}
    assert all(copy[name] == setup[name] for name in set(copy) - {OMITTED_EVIDENCE})
    assert json.loads(copy[OMITTED_EVIDENCE]) == {
        "schema_version": 1,
        "setup_manifest_sha256": _sha256(setup[MANIFEST]),
        "files": answers,
    }
    # The source bundle, the instrument and the derived exhibits stay.
    kept = {name for name in copy if name.startswith("evidence/")}
    assert "evidence/captured/source-bundle/evidence/source_cards.yaml" in kept
    assert "evidence/captured/contextual-instrument.json" in kept
    assert "evidence/derived/excluded-samples.json" in kept
    # No file of the export holds a response's reasoning or a prompt's text;
    # the whole export holds both.
    for text in (RESPONSE_TEXT, PROMPT_TEXT):
        assert not any(text in data for data in _files(root / "default").values())
        assert any(text in data for data in _files(root / "whole").values())
    index = indexes["default"]
    assert OMITTED_EVIDENCE in index.conventions["source"]
    assert index.source["setup"].sha256 == _sha256(setup[MANIFEST])


def test_an_export_without_answers_loads_and_verifies(imported):
    root, indexes = imported
    index = indexes["default"]
    assert load_results_export(root / "default") == index
    # The chain summary is the whole setup's, a row for the one scenario.
    assert index.chain_summary == indexes["whole"].chain_summary
    assert len(index.chain_summary) == 1
    # Against the setup the export was made from, the omitted files have
    # their recorded digests and the chain summary is rederived from them.
    assert load_results_export(root / "default", answers=root / "setup") == index
    # The copy loads as a setup, naming what it lacks, and serves the
    # conditional lookups the whole setup serves.
    package = load_setup(root / "default/native/setup")
    assert omitted_files(package) == _answers(_files(root / "setup"))
    whole = load_setup(root / "setup")
    assert omitted_files(whole) == {}
    context = whole.conditional_contexts[0].context_id
    kernels = [
        ConditionalCapture(setup, capture).kernel_at(context, 1, "reference")
        for setup, capture in (
            (package, root / "default/native/capture"),
            (whole, root / "run/capture"),
        )
    ]
    assert kernels[0].metrics("welfare") == kernels[1].metrics("welfare")


def test_including_the_answers_copies_the_setup_whole(imported):
    root, indexes = imported
    index = indexes["whole"]
    assert _files(root / "whole/native/setup") == _files(root / "setup")
    assert index.conventions["source"] == SOURCE_CONVENTION
    assert load_results_export(root / "whole") == index
    with pytest.raises(ValueError, match="leaves no file out"):
        load_results_export(root / "whole", answers=root / "setup")
    # The two exports differ in the setup's copy and in the index's two
    # conventions that describe it, and in nothing else.
    default, whole = _files(root / "default"), _files(root / "whole")
    outside = {
        name
        for name in default.keys() | whole.keys()
        if not name.startswith("native/setup/") and name != "export.json"
    }
    assert all(default[name] == whole[name] for name in outside)
    stated = [json.loads(files["export.json"]) for files in (default, whole)]
    assert {key for key in stated[0] if stated[0][key] != stated[1][key]} == {
        "conventions"
    }
    assert {
        key
        for key in stated[0]["conventions"]
        if stated[0]["conventions"][key] != stated[1]["conventions"][key]
    } == {"source", "chain_summary"}


def _copy(root: Path, tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(root / name, tmp_path / name))


def _edit_chain_summary(export: Path) -> None:
    path = export / "export.json"
    value = json.loads(path.read_text())
    value["chain_summary"][0]["direct_count_p50"] *= 2
    path.write_text(json.dumps(value))


def test_a_changed_chain_summary_is_refused_against_the_answers(imported, tmp_path):
    root, _ = imported
    whole = _copy(root, tmp_path, "whole")
    _edit_chain_summary(whole)
    with pytest.raises(ValueError, match="differs from its native"):
        load_results_export(whole)
    default = _copy(root, tmp_path, "default")
    _edit_chain_summary(default)
    with pytest.raises(ValueError, match="differs from its native"):
        load_results_export(default, answers=root / "setup")


def test_answers_that_differ_from_the_recorded_digests_are_refused(imported, tmp_path):
    root, _ = imported
    answers = Path(shutil.copytree(root / "setup", tmp_path / "answers"))
    artifact = answers / "evidence/captured/contextual-risk.json"
    artifact.write_bytes(artifact.read_bytes().replace(b"1.0", b"2.0", 1))
    with pytest.raises(ValueError, match="recorded SHA-256"):
        load_results_export(root / "default", answers=answers)
    artifact.unlink()
    with pytest.raises(ValueError, match="lacks an omitted file"):
        load_results_export(root / "default", answers=answers)


def _rehash_manifest(export: Path) -> None:
    """Record in the index the digest the setup copy's manifest now has."""
    path = export / "export.json"
    value = json.loads(path.read_text())
    manifest = export / value["source"]["setup"]["path"]
    value["source"]["setup"]["sha256"] = _sha256(manifest.read_bytes())
    path.write_text(json.dumps(value))


def test_tampering_with_an_export_without_answers_is_refused(imported, tmp_path):
    root, _ = imported
    export = _copy(root, tmp_path, "default")
    setup = export / "native/setup"
    # A file the copy holds: captured evidence that stays, and the laws.
    for name in ("evidence/captured/contextual-instrument.json", "setup.json"):
        original = (setup / name).read_bytes()
        (setup / name).write_bytes(original + b"\n")
        with pytest.raises(ValueError, match=f"checksum mismatch: {name}"):
            load_results_export(export)
        (setup / name).write_bytes(original)
    # The capture.
    capture = export / "native/capture/run-capture.json"
    original = capture.read_bytes()
    capture.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_results_export(export)
    capture.write_bytes(original)
    assert load_results_export(export)


def _record(setup: Path) -> dict:
    return json.loads((setup / OMITTED_EVIDENCE).read_text())


def test_a_setup_copy_holds_all_of_its_answers_or_records_all_as_left_out(
    imported, tmp_path
):
    root, _ = imported
    setup = _copy(root, tmp_path, "default") / "native/setup"
    record = _record(setup)
    source = root / "setup"

    def refused(match):
        with pytest.raises(ValueError, match=match):
            load_setup(setup)
        with pytest.raises(ValueError, match=match):
            load_results_export(setup.parent.parent)

    def write(payload):
        (setup / OMITTED_EVIDENCE).write_text(json.dumps(payload))

    listed = "must name its setup's manifest and list exactly"
    # Another digest for an omitted file, a file missing from the list, and
    # another manifest.
    name = next(iter(record["files"]))
    write({**record, "files": {**record["files"], name: "0" * 64}})
    refused(listed)
    write({**record, "files": {k: v for k, v in record["files"].items() if k != name}})
    refused(listed)
    write({**record, "setup_manifest_sha256": "0" * 64})
    refused(listed)
    # A file that holds no answers, removed and added to the list.
    kept = "evidence/captured/contextual-instrument.json"
    held = (setup / kept).read_bytes()
    (setup / kept).unlink()
    write({**record, "files": {**record["files"], kept: _sha256(held)}})
    refused(listed)
    (setup / kept).write_bytes(held)
    # Another schema version says why it is refused.
    write({**record, "schema_version": 2})
    refused("schema version 2 is not the current 1")
    # An omitted file put back beside the record.
    write(record)
    (setup / name).parent.mkdir(parents=True)
    (setup / name).write_bytes((source / name).read_bytes())
    refused(f"records as left out: {name}")
    (setup / name).unlink()
    assert load_setup(setup)
    # Without the record the files are missing, as from any setup.
    (setup / OMITTED_EVIDENCE).unlink()
    with pytest.raises((ValueError, OSError)):
        load_setup(setup)
    # With every omitted file back the copy is the whole setup, which is not
    # the setup copy the index describes.
    for name in record["files"]:
        (setup / name).parent.mkdir(parents=True, exist_ok=True)
        (setup / name).write_bytes((source / name).read_bytes())
    assert omitted_files(load_setup(setup)) == {}
    with pytest.raises(ValueError, match="differs from its native"):
        load_results_export(setup.parent.parent)


def test_a_whole_setup_takes_no_record_of_omissions(imported, tmp_path):
    root, _ = imported
    setup = Path(shutil.copytree(root / "setup", tmp_path / "setup"))
    shutil.copy(root / "default/native/setup" / OMITTED_EVIDENCE, setup)
    with pytest.raises(ValueError, match="records as left out"):
        load_setup(setup)
    # A setup with no captured answers has nothing to leave out.
    authored = _package(tmp_path, plan=_plan(16), name="authored")
    (authored.root / OMITTED_EVIDENCE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "setup_manifest_sha256": _sha256(
                    (authored.root / MANIFEST).read_bytes()
                ),
                "files": {},
            }
        )
    )
    with pytest.raises(ValueError, match="list exactly"):
        load_setup(authored.root)


def test_an_export_is_not_made_from_a_copy_without_its_answers(imported, tmp_path):
    root, _ = imported
    with pytest.raises(ValueError, match="leaves out its captured answers"):
        export_results(
            setup_path=root / "default/native/setup",
            capture_directory=root / "default/native/capture",
            output=tmp_path / "export",
        )
    assert not (tmp_path / "export").exists()


def test_an_authored_setups_export_is_the_same_with_or_without_the_option(tmp_path):
    package = _package(tmp_path, plan=_plan(32))
    run_setup(setup_path=package.root, output=tmp_path / "run", code_commit="fixture")
    indexes = {
        name: export_results(
            setup_path=package.root,
            capture_directory=tmp_path / "run/capture",
            output=tmp_path / name,
            code_commit="fixture",
            include_answers=include,
        )
        for name, include in (("default", False), ("whole", True))
    }
    default = _files(tmp_path / "default")
    assert default == _files(tmp_path / "whole")
    # The export holds what it held before the option: the index, the three
    # capability artifacts, the results and whole copies of setup and capture.
    setup = _files(package.root)
    capture = _files(tmp_path / "run/capture")
    assert set(default) == {
        "export.json",
        "capability_scenarios.json",
        "capability_scenarios.csv",
        "capability_nodes.csv",
        "native/results.json",
        "native/results.diagnostics.json",
        *(f"native/setup/{name}" for name in setup),
        *(f"native/capture/{name}" for name in capture),
    }
    assert all(default[f"native/setup/{name}"] == data for name, data in setup.items())
    stated = json.loads(default["export.json"])
    assert set(stated) == {
        "format",
        "source",
        "outputs",
        "scope",
        "series",
        "sample_sets",
        "conditioning",
        "chain_summary",
        "conventions",
    }
    assert stated["format"] == "cat.results/1"
    assert stated["conventions"]["source"] == SOURCE_CONVENTION
    assert load_results_export(tmp_path / "default") == indexes["default"]


def _cli(*arguments):
    return CliRunner().invoke(cli, ["--workers", "1", *map(str, arguments)])


def test_the_command_leaves_answers_out_unless_asked(imported, tmp_path):
    root, _ = imported
    result = _cli("results", "export", "--help")
    assert result.exit_code == 0, result.output
    assert "--include-answers" in result.output
    command = ("results", "export", "--setup", root / "setup")
    command += ("--capture", root / "run/capture")
    for name, options in (("default", ()), ("whole", ("--include-answers",))):
        result = _cli(*command, *options, "--output", tmp_path / name)
        assert result.exit_code == 0, result.output
        load_results_export(tmp_path / name)
    setup = _files(root / "setup")
    assert _files(tmp_path / "whole/native/setup") == setup
    copy = _files(tmp_path / "default/native/setup")
    assert set(setup) - set(copy) == set(_answers(setup))
    # The export's own setup and capture replay to the results it holds.
    result = _cli(
        *("results", "replay", "--setup", tmp_path / "default/native/setup"),
        *("--capture", tmp_path / "default/native/capture"),
        *("--output", tmp_path / "replay"),
    )
    assert result.exit_code == 0, result.output
    replayed = json.loads((tmp_path / "replay/results.json").read_text())
    exported = json.loads((tmp_path / "default/native/results.json").read_text())
    assert replayed["arms"] == exported["arms"]
