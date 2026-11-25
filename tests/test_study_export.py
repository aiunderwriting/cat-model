"""Portable exports preserve native values, identities and conditional support."""

import hashlib
import json

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.contracts import _setup_codec
from catmodel.hashing import canonical_json_sha256
from catmodel.results.conditional import ConditionalCapture
from catmodel.results.execution import run_setup
from catmodel.results.results_export import export_results, load_results_export
from catmodel.results.results_file import diagnostics_path, load_results
from catmodel.setup.store import load_setup
from test_support.results import available_metric, reduced_arm
from test_support.study import _package, _plan
from tests.test_conditional_capture import _capture


def _export(tmp_path, *, lab_loss=True):
    bound = _capture(tmp_path, lab_loss=lab_loss)
    output = tmp_path / "export"
    index = export_results(
        setup_path=bound.setup.root,
        capture_directory=tmp_path / "run/capture",
        results_path=tmp_path / "run/results.json",
        output=output,
    )
    return bound, output, index


def test_export_is_portable_and_keeps_peer_joint_samples(tmp_path):
    original, output, index = _export(tmp_path)
    assert load_results_export(output) == index
    assert (output / index.source["results"].path).read_bytes() == (
        tmp_path / "run/results.json"
    ).read_bytes()
    # The diagnostics file beside it holds the run's exhibits and none of its
    # view diagnostics, which the export does not compare with the capture.
    assert set(index.source) == {"setup", "capture", "results", "diagnostics"}
    reported = json.loads(diagnostics_path(tmp_path / "run/results.json").read_bytes())
    exported = json.loads((output / index.source["diagnostics"].path).read_bytes())
    assert reported["arms"] and exported["arms"] == {}
    assert set(exported) == set(reported)
    for section in set(reported) - {"arms", "reasons"}:
        assert exported[section] == reported[section], section
    assert index.conditioning["status"] == "available"
    assert {row.measure for row in index.series} >= {
        "societal_harm",
        "lab_economic_loss",
    }
    package = load_setup((output / index.source["setup"].path).parent)
    bound = ConditionalCapture(package, (output / index.source["capture"].path).parent)
    assert bound.kernel_at("internal", 1, "reference").metrics(
        "welfare"
    ) == original.kernel_at("internal", 1, "reference").metrics("welfare")
    assert str(tmp_path) not in (output / "export.json").read_text()
    for key, ref in index.source.items():
        assert (
            hashlib.sha256((output / ref.path).read_bytes()).hexdigest() == ref.sha256
        ), key


def test_export_decodes_the_setup_once(tmp_path, monkeypatch):
    bound = _capture(tmp_path)
    original = _setup_codec._decode
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_setup_codec, "_decode", counted)
    export_results(
        setup_path=bound.setup.root,
        capture_directory=tmp_path / "run/capture",
        results_path=tmp_path / "run/results.json",
        output=tmp_path / "export",
    )
    assert len(calls) == 1


def test_missing_lab_peer_withholds_supported_context_not_zero(tmp_path):
    _, output, index = _export(tmp_path, lab_loss=False)
    assert load_results_export(output) == index
    lab = next(row for row in index.series if row.measure == "lab_economic_loss")
    assert lab.status == "unavailable" and not lab.curve
    context = index.conditioning["contexts"][0]
    assert context["status"] == "unavailable"
    assert len(context["failures"]) == 3


def test_native_results_tamper_is_rejected_even_with_unchanged_analysis_hash(tmp_path):
    bound = _capture(tmp_path)
    path = tmp_path / "run/results.json"
    result = json.loads(path.read_text())
    result["arms"]["reference"]["metrics"]["welfare"]["expected_annual_loss"] = 123456
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="do not reproduce"):
        export_results(
            setup_path=bound.setup.root,
            capture_directory=tmp_path / "run/capture",
            results_path=path,
            output=tmp_path / "export",
        )
    assert not (tmp_path / "export").exists()


def test_export_index_curve_tamper_is_rejected(tmp_path):
    _, output, _ = _export(tmp_path)
    path = output / "export.json"
    value = json.loads(path.read_text())
    value["series"][0]["curve"][0]["probability"] = 1.0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differs from its native"):
        load_results_export(output)


def test_rehashed_false_native_moment_claim_is_rejected(tmp_path):
    bound = _capture(tmp_path)
    path = tmp_path / "run/capture/run-capture.json"
    capture = json.loads(path.read_text())
    for record in capture["arms"]["reference"].values():
        record["draw"]["views"]["welfare"]["variance_exists"] = False
    capture["content_sha256"] = canonical_json_sha256(
        {k: v for k, v in capture.items() if k != "content_sha256"}
    )
    path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="moment availability"):
        export_results(
            setup_path=bound.setup.root,
            capture_directory=path.parent,
            output=tmp_path / "export",
            code_commit="fixture",
        )


def test_export_rejects_output_inside_immutable_capture(tmp_path):
    bound = _capture(tmp_path)
    with pytest.raises(ValueError, match="immutable evidence"):
        export_results(
            setup_path=bound.setup.root,
            capture_directory=tmp_path / "run/capture",
            output=tmp_path / "run/capture/export",
        )


def _rehash_source(output, name):
    path = output / "export.json"
    value = json.loads(path.read_text())
    data = (output / value["source"][name]["path"]).read_bytes()
    value["source"][name]["sha256"] = hashlib.sha256(data).hexdigest()
    path.write_text(json.dumps(value))


def test_an_export_holds_no_view_diagnostics_and_checks_its_exhibits(tmp_path):
    """An export's diagnostics file is the one beside its results, holds no
    view diagnostics, which nothing checks against the capture, and holds
    exhibits that reproduce from the capture, even when its checksum is
    recomputed."""
    _, output, index = _export(tmp_path)
    exported = output / index.source["diagnostics"].path
    assert exported == diagnostics_path(output / index.source["results"].path)
    original = exported.read_bytes()
    exported.write_bytes(diagnostics_path(tmp_path / "run/results.json").read_bytes())
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_results_export(output)
    _rehash_source(output, "diagnostics")
    with pytest.raises(ValueError, match="no view diagnostics"):
        load_results_export(output)
    changed = json.loads(original)
    changed["workload"]["arms"] += 1
    exported.write_text(json.dumps(changed))
    _rehash_source(output, "diagnostics")
    with pytest.raises(ValueError, match="do not reproduce"):
        load_results_export(output)
    exported.write_bytes(original)
    _rehash_source(output, "diagnostics")
    assert load_results_export(output) == index
    # A reference to another file is not the diagnostics file of the results.
    path = output / "export.json"
    value = json.loads(path.read_text())
    value["source"]["diagnostics"] = value["source"]["results"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="the file beside its results"):
        load_results_export(output)


def test_source_reference_traversal_is_rejected(tmp_path):
    _, output, _ = _export(tmp_path)
    path = output / "export.json"
    value = json.loads(path.read_text())
    value["source"]["capture"]["path"] = "../run/capture/run-capture.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="relative portable"):
        load_results_export(output)


def _cli(*arguments):
    return CliRunner().invoke(cli, ["--workers", "1", *map(str, arguments)])


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """A run whose two states have gross EALs of 200 and 400, weighted 0.25
    and 0.75, and a replay of it with all weight on the first and a tail
    level added."""
    root = tmp_path_factory.mktemp("export-replay")
    package = _package(root, plan=_plan(32))
    run_setup(setup_path=package.root, output=root / "run", code_commit="fixture")
    (root / "weights.json").write_text(json.dumps({"low": 1.0, "high": 0.0}))
    result = _cli(
        *("results", "replay", "--setup", package.root),
        *("--capture", root / "run/capture", "--weights", root / "weights.json"),
        *("--level", "0.95", "--output", root / "replay"),
    )
    assert result.exit_code == 0, result.output
    capture = root / "run/capture"
    return root, ("results", "export", "--setup", package.root, "--capture", capture)


def test_cli_exports_a_replayed_results_file_in_place_of_the_default(
    replayed, tmp_path
):
    root, export = replayed
    selected = root / "replay/results.json"
    exported = {}
    for name, options in (("default", ()), ("selected", ("--results", selected))):
        result = _cli(*export, *options, "--output", tmp_path / name)
        assert result.exit_code == 0, result.output
        index = load_results_export(tmp_path / name)
        exported[name] = tmp_path / name / index.source["results"].path
    reports = {name: load_results(path) for name, path in exported.items()}
    eal = {
        name: available_metric(
            reduced_arm(report["arms"]["reference"])["metrics"]["gross"]
        )["expected_annual_loss"]
        for name, report in reports.items()
    }
    # The default export reduces at the setup's weights; the selected results
    # are the replay's own file, at its weights and with its added level.
    assert eal == {
        "default": pytest.approx(0.25 * 200 + 0.75 * 400),
        "selected": pytest.approx(200),
    }
    assert exported["selected"].read_bytes() == selected.read_bytes()
    levels = {
        name: report["analysis_spec"]["reduction_spec"]["levels"]
        for name, report in reports.items()
    }
    assert 0.95 in levels["selected"] and 0.95 not in levels["default"]


def test_cli_refuses_results_that_do_not_reproduce_from_the_capture(replayed, tmp_path):
    root, export = replayed
    # A replay whose reference EAL was edited, and another run's results.
    forged = tmp_path / "forged/results.json"
    forged.parent.mkdir()
    report = json.loads((root / "replay/results.json").read_text())
    report["arms"]["reference"]["metrics"]["gross"]["expected_annual_loss"] = 350
    forged.write_text(json.dumps(report))
    diagnostics_path(forged).write_bytes(
        diagnostics_path(root / "replay/results.json").read_bytes()
    )
    other = _package(tmp_path, plan=_plan(16), name="other")
    run_setup(setup_path=other.root, output=tmp_path / "other-run", code_commit="x")
    for name, path in (
        ("forged", forged),
        ("other", tmp_path / "other-run/results.json"),
    ):
        output = tmp_path / f"export-{name}"
        result = _cli(*export, "--results", path, "--output", output)
        assert result.exit_code == 1, name
        assert "do not reproduce" in result.output, (name, result.output)
        assert not output.exists()
