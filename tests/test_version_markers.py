"""A saved setup, run capture and export each state their version on disk."""

import json
import shutil

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.contracts.incident import ExecutionPlan
from catmodel.hashing import canonical_json_sha256
from catmodel.results.execution import load_run_capture, run_setup
from catmodel.results.results_export import export_results, load_results_export
from catmodel.setup.store import build_setup, load_setup
from test_support.conditional import conditional_study
from test_support.run_setup import with_execution
from test_support.version_markers import unstated_markers


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A built setup, its run and the run's export, read by every test and
    changed only in copies."""
    root = tmp_path_factory.mktemp("saved")
    package = build_setup(
        destination=root / "setup",
        setup=with_execution(
            conditional_study(),
            ExecutionPlan(
                plan_id="version-markers",
                required_arms=("reference",),
                simulations=48,
                seed=31,
                requested_views=("gross", "welfare", "lab_loss::lab"),
            ),
            setup_id="version-markers",
        ),
        code_commit="fixture",
    )
    run_setup(setup_path=package.root, output=root / "run", code_commit="fixture")
    export_results(
        setup_path=package.root,
        capture_directory=root / "run" / "capture",
        results_path=root / "run" / "results.json",
        output=root / "export",
    )
    return root


def _copies(saved, tmp_path, directory, name, key="schema_version", sealed=False):
    """Copies of the saved ``directory`` whose file ``name`` does not state
    its marker; ``sealed`` recomputes the file's own content digest."""
    written = json.loads((saved / directory / name).read_bytes())
    for label, payload in unstated_markers(written, key).items():
        if sealed:
            del payload["content_sha256"]
            payload["content_sha256"] = canonical_json_sha256(payload)
        copy = tmp_path / label.replace(" ", "-") / directory
        shutil.copytree(saved / directory, copy)
        (copy / name).write_text(json.dumps(payload))
        yield copy


def test_a_setup_manifest_must_state_its_schema(saved, tmp_path):
    package = load_setup(saved / "setup")
    for copy in _copies(saved, tmp_path, "setup", "setup-manifest.json"):
        with pytest.raises(ValueError, match="setup manifest schema version"):
            load_setup(copy)
        refused = CliRunner().invoke(cli, ["setup", "inspect", "--setup", str(copy)])
        assert refused.exit_code != 0
        assert "setup manifest schema version" in refused.output
    # The manifest as written is unchanged by being read.
    shown = CliRunner().invoke(
        cli, ["setup", "inspect", "--setup", str(saved / "setup")]
    )
    assert shown.exit_code == 0, shown.output
    assert load_setup(saved / "setup").manifest_sha256 == package.manifest_sha256


def test_a_run_capture_must_state_its_schema(saved, tmp_path):
    load_run_capture(saved / "run" / "capture")
    for copy in _copies(
        saved, tmp_path, "run/capture", "run-capture.json", sealed=True
    ):
        with pytest.raises(ValueError, match="run capture schema version"):
            load_run_capture(copy)


def test_an_export_index_must_state_its_format(saved, tmp_path):
    index = load_results_export(saved / "export")
    assert index.format == "cat.results/1"
    for copy in _copies(saved, tmp_path, "export", "export.json", key="format"):
        with pytest.raises(ValueError, match="format"):
            load_results_export(copy)
