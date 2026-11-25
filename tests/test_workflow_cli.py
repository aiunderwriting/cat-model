"""Public workflow commands preserve saved inputs and reuse immutable captures."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.resources import config_path
from catmodel.results.reduction import DEFAULT_TAIL_LEVELS
from catmodel.results.results_file import load_results
from catmodel.worktree import resolve_code_provenance
from test_support.results_schema import schema_violations


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _leaf_commands(group: click.Group, path: tuple[str, ...] = ()):
    for name, command in group.commands.items():
        if isinstance(command, click.Group):
            yield from _leaf_commands(command, (*path, name))
        else:
            yield (*path, name), command


def test_every_option_explains_itself():
    """Each option of the root group and every command has help text, and
    each command's help renders."""
    commands = list(_leaf_commands(cli))
    assert commands
    for path, command in [((), cli), *commands]:
        for option in command.params:
            if isinstance(option, click.Option):
                assert (option.help or "").strip(), (path, option.opts)
        result = CliRunner().invoke(cli, [*path, "--help"])
        assert result.exit_code == 0, (path, result.output)


def test_version_names_the_installed_distribution():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"catmodel, version {version('catmodel')}"


def test_a_global_option_after_the_command_says_where_it_goes():
    result = CliRunner().invoke(cli, ["results", "replay", "--workers", "2"])
    assert result.exit_code == 2
    assert "--workers is a global option; give it before the command" in (result.output)


def test_setup_run_results_reuses_capture_without_simulating(tmp_path, monkeypatch):
    from catmodel.results import execution
    from catmodel.setup.store import load_setup

    source = config_path("setups", "stipulated-campaign")
    plan = json.loads((source / "plan.json").read_bytes())
    plan["simulations"] = 24
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    setup_path = tmp_path / "setup"
    run_path = tmp_path / "run"
    runner = CliRunner()

    def invoke(*arguments):
        result = runner.invoke(cli, list(map(str, arguments)))
        assert result.exit_code == 0, result.output
        return result

    invoke(
        "setup",
        "build",
        "--spec",
        source / "setup.json",
        "--plan",
        plan_path,
        "--evidence",
        source / "evidence",
        "--setup-id",
        "cli-workflow",
        "--output",
        setup_path,
    )
    frozen = load_setup(setup_path)
    # A command records the identity the running code resolves for itself,
    # its checkout's commit and tree state or its installed version; none
    # asserts a commit.
    code = frozen.manifest.code_provenance
    resolved = resolve_code_provenance(None)
    assert (code["commit"], code["tree_state"]) == (
        resolved.commit,
        resolved.tree_state,
    )
    assert code["tree_state"] != "asserted"
    inspection = json.loads(invoke("setup", "inspect", "--setup", setup_path).output)
    assert inspection["setup_id"] == "cli-workflow"
    assert inspection["scope_id"] == frozen.scope.scope_id
    assert inspection["setup_manifest_sha256"] == frozen.manifest_sha256
    assert inspection["execution"]["simulations"] == 24
    # Inspection stays a compact summary: standalone assumptions and state IDs,
    # not the resolved laws themselves.
    assert inspection["assumptions"] == frozen.assumptions.model_dump(mode="json")
    assert set(inspection["assumptions"]) == {"independence"}
    assert inspection["joint_state_ids"] == [
        state.state_id for state in frozen.joint_states
    ]
    assert "joint_states" not in inspection
    before_setup = _tree_hashes(setup_path)

    invoke("run", "--setup", setup_path, "--output", run_path)
    assert {path.name for path in run_path.iterdir()} == {
        "capture",
        "results.json",
        "results.diagnostics.json",
        "tornado.svg",
    }
    initial = load_results(run_path / "results.json", diagnostics=True)
    assert initial["study"]["setup_id"] == "cli-workflow"
    # The stipulated example's results files conform to their JSON Schemas.
    assert schema_violations(run_path / "results.json") == []
    assert set(initial["diagnostics"]["arms"]) == set(initial["arms"])
    capture_path = run_path / "capture"
    before_capture = _tree_hashes(capture_path)
    assert "run-capture.json" in before_capture

    def forbid_simulation(*_args, **_kwargs):
        pytest.fail("results commands must reuse the persisted run capture")

    monkeypatch.setattr(execution, "simulate_incident_state", forbid_simulation)
    monkeypatch.setattr(execution, "run_setup", forbid_simulation)
    replay_path = tmp_path / "replay"
    printed = invoke(
        "results",
        "replay",
        "--setup",
        setup_path,
        "--capture",
        capture_path,
        "--output",
        replay_path,
    )
    # Like a run, a replay writes results.json, with its diagnostics file
    # beside it, into its new output directory.
    assert printed.output.strip() == str(replay_path / "results.json")
    assert sorted(path.name for path in replay_path.iterdir()) == [
        "results.diagnostics.json",
        "results.json",
    ]
    replay = load_results(replay_path / "results.json", diagnostics=True)
    assert schema_violations(replay_path / "results.json") == []
    assert replay["arms"] == initial["arms"]
    assert replay["diagnostics"] == initial["diagnostics"]
    assert replay["analysis_sha256"] == initial["analysis_sha256"]

    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps({"slow_fixed": 0.25, "fast_rising": 0.75}))
    reweighted_path = tmp_path / "reweighted"
    invoke(
        "results",
        "replay",
        "--setup",
        setup_path,
        "--capture",
        capture_path,
        "--weights",
        weights_path,
        "--level",
        "0.95",
        "--output",
        reweighted_path,
    )
    reweighted = load_results(reweighted_path / "results.json")
    assert schema_violations(reweighted_path / "results.json") == []
    assert reweighted["analysis_sha256"] != replay["analysis_sha256"]
    # A requested level is reported beside the default levels, not instead.
    levels = reweighted["analysis_spec"]["reduction_spec"]["levels"]
    assert levels == sorted({*DEFAULT_TAIL_LEVELS, 0.95})
    assert _tree_hashes(capture_path) == before_capture
    assert _tree_hashes(setup_path) == before_setup


def test_paths_that_are_not_what_an_option_reads_are_named(tmp_path):
    """An authored setup directory passed as --setup, a run directory passed
    as --capture and an existing --output each fail with a message naming the
    path and what to pass instead, before any work."""
    runner = CliRunner()
    authored = config_path("setups", "golden-poisson")
    unbuilt = runner.invoke(cli, ["setup", "inspect", "--setup", str(authored)])
    assert unbuilt.exit_code == 2
    assert "holds authored setup inputs, not a built setup" in unbuilt.output
    assert "catmodel setup build --spec" in unbuilt.output
    assert "Errno" not in unbuilt.output

    setup = tmp_path / "setup"
    setup.mkdir()
    (setup / "setup-manifest.json").write_text("{}")
    run = tmp_path / "run"
    (run / "capture").mkdir(parents=True)
    (run / "capture" / "run-capture.json").write_text("{}")
    result = runner.invoke(
        cli,
        [
            "results",
            "replay",
            "--setup",
            str(setup),
            "--capture",
            str(run),
            "--output",
            str(tmp_path / "replay"),
        ],
    )
    assert result.exit_code == 2
    assert f"--capture {run / 'capture'}" in result.output
    assert "Errno" not in result.output

    result = runner.invoke(
        cli,
        [
            "results",
            "replay",
            "--setup",
            str(setup),
            "--capture",
            str(run / "capture"),
            "--output",
            str(run),
        ],
    )
    assert result.exit_code == 2
    assert f"{run} already exists" in result.output


def test_setup_build_refuses_an_unsupported_numerical_control(tmp_path):
    """A plan naming an unsupported control fails at build, not when it runs."""
    source = config_path("setups", "stipulated-campaign")
    plan = json.loads((source / "plan.json").read_bytes())
    plan["numerical_controls"]["max_events"] = 200_000
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    result = CliRunner().invoke(
        cli,
        [
            "setup",
            "build",
            "--spec",
            str(source / "setup.json"),
            "--plan",
            str(plan_path),
            "--evidence",
            str(source / "evidence"),
            "--setup-id",
            "unsupported-control",
            "--output",
            str(tmp_path / "setup"),
        ],
    )
    assert result.exit_code != 0
    assert "unsupported numerical controls ['max_events']" in result.output
    assert not (tmp_path / "setup").exists()


@pytest.mark.parametrize(
    ("controls", "refusal"),
    [
        (
            {"tail_gate": {"views": "gross"}},
            "the tail gate views must be a nonempty list of view names",
        ),
        ({"tail_gate": {"views": ["gros"]}}, "tail gate views ['gros'] are not"),
        (
            {"event_batch_size": None},
            "numerical control event_batch_size must be a positive integer",
        ),
        (
            {"max_campaign_members_per_year": None},
            "numerical control max_campaign_members_per_year must be a positive integer",
        ),
    ],
    ids=["string-views", "misspelt-view", "null-batch-size", "null-allowance"],
)
@pytest.mark.parametrize("arm", [False, True], ids=["plan", "arm"])
def test_setup_build_refuses_a_control_that_would_disable_the_gate_or_stop_the_run(
    tmp_path, controls, refusal, arm
):
    """A tail gate whose views select no requested view, or a count control
    set to null, fails at build: the first would run and report its arm's
    drivers converged with no driver view, the second would stop the run."""
    source = config_path("setups", "golden-poisson")
    plan = json.loads((source / "plan.json").read_bytes())
    assert "gross" in plan["requested_views"]
    plan["numerical_controls_by_arm" if arm else "numerical_controls"] = (
        {"reference": controls} if arm else controls
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    result = CliRunner().invoke(
        cli,
        [
            "setup",
            "build",
            "--spec",
            str(source / "setup.json"),
            "--plan",
            str(plan_path),
            "--setup-id",
            "refused-control",
            "--output",
            str(tmp_path / "setup"),
        ],
    )
    assert result.exit_code != 0
    assert refusal in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "setup").exists()


@pytest.mark.parametrize(
    "arguments",
    [["no-such-command", "--help"], ["run", "--no-such-option", "missing"]],
)
def test_an_unknown_command_or_option_is_a_usage_error(arguments):
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code == 2
    assert "No such" in result.output
