"""The hypothetical-lab study script, offline: sources, plans and prompts."""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import yaml

from catmodel.config import EXAMPLE_PROFILES, PROFILES_ENV, load_model_profile
from catmodel.evidence import source_inputs
from catmodel.evidence.source_bundle import (
    ASSUMPTION_REGISTER_PATH,
    MANIFEST_PATH,
    SOURCE_CARDS_PATH,
    THREAT_MODELS_PATH,
    capture_source_bundle_snapshot,
)
from catmodel.evidence.source_inputs import build_source_bundle
from catmodel.forecast.public_frontier_epoch import EPOCH_ECI_SCORES_PATH
from catmodel.resources import config_path, data_path
from catmodel.setup.public_study import public_study_collections
from catmodel.worktree import CodeProvenance
from scripts import run_hypothetical_lab_study
from scripts.run_hypothetical_lab_study import main, stage_authored_source


@pytest.fixture
def clean_source_provenance(monkeypatch):
    monkeypatch.setattr(
        source_inputs,
        "resolve_code_provenance",
        lambda _: CodeProvenance("a" * 40, True, "clean", None),
    )


def _build_committed_bundle(stage_directory: Path, bundle_directory: Path):
    recipe = stage_authored_source(stage_directory)
    return build_source_bundle(recipe, bundle_directory)


def test_committed_recipe_builds_a_stable_bundle(tmp_path, clean_source_provenance):
    first, second = tmp_path / "first", tmp_path / "second"
    first_manifest = _build_committed_bundle(
        first / "authored-source", first / "bundle"
    )
    _build_committed_bundle(second / "authored-source", second / "bundle")

    first_snapshot = capture_source_bundle_snapshot(first / "bundle")
    assert (first / "bundle" / MANIFEST_PATH).read_bytes() == (
        second / "bundle" / MANIFEST_PATH
    ).read_bytes()
    assert first_manifest.bundle_id == "hypothetical-frontier-lab"
    assert first_manifest.bundle_version == "2026-11-25"
    assert set(first_manifest.files) == {
        EPOCH_ECI_SCORES_PATH,
        SOURCE_CARDS_PATH,
        THREAT_MODELS_PATH,
        ASSUMPTION_REGISTER_PATH,
    }
    committed_sources = {
        EPOCH_ECI_SCORES_PATH: data_path(
            "capability", "epoch_eci_scores.csv"
        ).read_bytes(),
        SOURCE_CARDS_PATH: data_path("elicitation_source_cards.yaml").read_bytes(),
        THREAT_MODELS_PATH: data_path("threat_models.yaml").read_bytes(),
        ASSUMPTION_REGISTER_PATH: config_path(
            "studies",
            "hypothetical-frontier-lab",
            "authored-source",
            "assumption_register.yaml",
        ).read_bytes(),
    }
    for path, content in committed_sources.items():
        assert first_snapshot.read_bytes(path) == content
        assert (second / "bundle" / path).read_bytes() == content

    assert len(first_manifest.external_sources) == 1
    assert first_manifest.external_sources[0].source_id == "epoch-eci-scores"
    assert first_manifest.external_sources[0].captured_paths == [EPOCH_ECI_SCORES_PATH]
    assert first_snapshot.manifest.bundle_id == "hypothetical-frontier-lab"


def _offline_environment(monkeypatch, directory: Path) -> None:
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "META_API_KEY",
        "OPENROUTER_API_KEY",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.chdir(directory)


def _one_model_profiles(path: Path) -> str:
    """Write a profiles file whose profile ``one`` is the example's test
    profile with its first model only, and return that model's ID."""
    payload = yaml.safe_load(data_path(EXAMPLE_PROFILES).read_text(encoding="utf-8"))
    test = payload["profiles"]["test"]
    payload["profiles"] = {"one": {**test, "models": test["models"][:1]}}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return test["models"][0]["model_id"]


def _cli_tail(command: list[str]) -> list[str]:
    """A recorded command's arguments after ``python -m catmodel.cli`` and its
    global ``--profile``."""
    assert command[1:3] == ["-m", "catmodel.cli"]
    tail = command[3:]
    return tail[2:] if tail[:1] == ["--profile"] else tail


VARIANT_DRAW_IDS = ["node-00", "node-1-4", "node-1-2", "node-3-4"]
"""A seven-node headline's no-progress node and embedded-rule nodes: a variant's
whole plan, all collected."""
EXPOSURE_STUDIES = ("exposure-half", "exposure-double")
SAFEGUARD_STUDIES = ("safeguard-weaker", "safeguard-stronger")


@pytest.mark.parametrize(
    ("exposure", "safeguard"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["default", "safeguard", "exposure", "both"],
)
def test_dry_run_and_resumed_run_cover_the_headline_and_variant_studies(
    tmp_path, monkeypatch, capfd, clean_source_provenance, exposure, safeguard
):
    """By default the script builds and collects the headline study only;
    each ``--variant`` adds the variant study it names."""
    bundle = tmp_path / "source-bundle"
    _build_committed_bundle(tmp_path / "authored-source", bundle)
    output = tmp_path / "run"
    # A profile of one model keeps the frozen prompts few.
    selected_model = _one_model_profiles(tmp_path / "profiles.yaml")
    _offline_environment(monkeypatch, tmp_path)
    arguments = [
        str(output),
        "--source-bundle",
        str(bundle),
        "--profile",
        "one",
        "--profiles",
        str(tmp_path / "profiles.yaml"),
        "--nodes",
        "7",
        # Given out of order: the studies are planned in the planner's order.
        *(("--variant", "stronger", "--variant", "weaker") if safeguard else ()),
        *(("--variant", "double", "--variant", "half") if exposure else ()),
    ]

    assert main([*arguments, "--dry-run"]) == 0

    samples = load_model_profile(
        "one", tmp_path / "profiles.yaml"
    ).settings.samples_per_question
    variant_names = (EXPOSURE_STUDIES if exposure else ()) + (
        SAFEGUARD_STUDIES if safeguard else ()
    )
    studies = {
        "public-study": (7, samples),
        **{name: (3, 3) for name in variant_names},
    }
    # With the bundle given, the output holds the studies, each with its
    # request record, and nothing else.
    assert sorted(path.name for path in output.iterdir()) == sorted(
        [*studies, *(f"{name}.request.json" for name in studies)]
    )
    # The headline's record holds the request and each input's digest.
    study = config_path("studies", "hypothetical-frontier-lab")
    forecast = data_path("capability", "public-frontier-forecast-template.json")
    assert json.loads((output / "public-study.request.json").read_bytes()) == {
        "nodes": 7,
        "variant": None,
        "profile": "one",
        "profiles_file_sha256": hashlib.sha256(
            (tmp_path / "profiles.yaml").read_bytes()
        ).hexdigest(),
        "forecast_sha256": hashlib.sha256(forecast.read_bytes()).hexdigest(),
        "context_sha256": hashlib.sha256(
            (study / "context.json").read_bytes()
        ).hexdigest(),
        "assumptions_sha256": hashlib.sha256(
            (study / "public-assembly-assumptions.json").read_bytes()
        ).hexdigest(),
        "source_bundle_manifest_sha256": capture_source_bundle_snapshot(
            bundle
        ).manifest_sha256,
    }
    template = json.loads(forecast.read_bytes())
    assert "releases" not in template
    for name, (nodes, study_samples) in studies.items():
        study_path = output / name / "public-study.json"
        plan = json.loads(study_path.read_bytes())
        # Each study keeps its forecast: the committed one, with its releases
        # filled from the bundle.
        forecast = json.loads((output / name / plan["forecast"]).read_bytes())
        assert forecast["releases"]
        assert forecast["forecast_id"] == template["forecast_id"]
        # Each source status at the interior nodes and the no-progress node.
        assert len(plan["contexts"]) == 2 * (nodes + 1)
        if name != "public-study":
            assert sorted({row["draw_id"] for row in plan["contexts"]}) == sorted(
                VARIANT_DRAW_IDS
            )
        collected = [capture for _, capture in public_study_collections(study_path)]
        frozen = [capture for capture in collected if capture.is_dir()]
        # Every study, a variant included, collects its whole plan.
        assert frozen == collected
        for capture in frozen:
            assert (capture / "collection-plan.json").is_file()
            assert not (capture / "contextual-risk.json").exists()
            instrument = json.loads(
                (capture / "contextual-instrument.json").read_bytes()
            )
            respondents = {row["respondent_id"] for row in instrument["respondents"]}
            assert len(respondents) == study_samples
            assert {row["model_id"] for row in instrument["respondents"]} == {
                selected_model
            }
            requests = [
                json.loads(path.read_bytes())["respondent"]["respondent_id"]
                for path in capture.glob("calls/*/request.json")
            ]
            # Every respondent gets each node's direct and chain question and
            # every taxonomy parent question.
            per_respondent = 2 * len(instrument["nodes"]) + len(instrument["parents"])
            assert Counter(requests) == dict.fromkeys(respondents, per_respondent)

    output_text = capfd.readouterr().out
    commands = [line for line in output_text.splitlines() if line.startswith("+ ")]
    # public-study plans seven nodes by default and a variant three, so the
    # script names each study's own count.
    plans = [
        command for command in commands if "capability-forecast public-study" in command
    ]
    assert sorted(command.split(" --nodes ")[1].split()[0] for command in plans) == [
        "3"
    ] * len(variant_names) + ["7"]
    assert sum(
        "capability-risk collect --study" in command and "--dry-run" in command
        for command in commands
    ) == 1 + len(variant_names)
    assert (
        any(
            "--variant half" in command or "--variant double" in command
            for command in commands
        )
        == exposure
    )
    assert (
        any(
            "--variant weaker" in command or "--variant stronger" in command
            for command in commands
        )
        == safeguard
    )
    assert not any(
        " validate " in f" {command} "
        or "setup import" in command
        or " run " in f" {command} "
        for command in commands
    )
    assert "no provider was called" in output_text

    # Resume the same directory as a full run with every provider-facing and
    # later step recorded instead of run: the frozen build steps validate and
    # are skipped, and each remaining step runs once, in order.
    recorded: list[list[str]] = []

    def record(command, check, env=None):
        assert check is False
        recorded.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_hypothetical_lab_study.subprocess, "run", record)
    assert main([*arguments, "--live", "--resume"]) == 0
    default_import = next(
        _cli_tail(command)
        for command in recorded
        if _cli_tail(command)[:2] == ["setup", "import"]
    )
    # The import names the study, its variant studies and every comparison
    # family; its numerical controls and seed are the fixed defaults.
    assert default_import == [
        "setup",
        "import",
        "--study",
        str(output / "public-study" / "public-study.json"),
        "--setup-id",
        "hypothetical-frontier-lab",
        "--comparison-family",
        "all",
        *(
            argument
            for name in variant_names
            for argument in (
                "--variant-study",
                str(output / name / "public-study.json"),
            )
        ),
        "--output",
        str(output / "setup"),
    ]
    output_text = capfd.readouterr().out
    # Each study's plan.
    assert output_text.count("= complete, skipped: ") == len(studies)
    tails = [_cli_tail(command) for command in recorded]
    collections = [tail for tail in tails if tail[:2] == ["capability-risk", "collect"]]
    assert [Path(tail[3]).parent.name for tail in collections] == list(studies)
    for tail in collections:
        assert tail[2] == "--study"
        assert "--dry-run" not in tail
    validated = [
        Path(tail[3]) for tail in tails if tail[:2] == ["capability-risk", "validate"]
    ]
    assert validated == [
        capture
        for name in studies
        for _, capture in public_study_collections(output / name / "public-study.json")
    ]
    assert [tail[:2] for tail in tails[len(collections) + len(validated) :]] == [
        ["setup", "import"],
        ["run", "--setup"],
        ["results", "summary"],
        ["results", "export"],
    ]
    # The export reduces the run's capture itself.
    assert tails[-1] == [
        "results",
        "export",
        "--setup",
        str(output / "setup"),
        "--capture",
        str(output / "model-run" / "capture"),
        "--output",
        str(output / "export"),
    ]

    def options(tail: list[str], name: str) -> list[str]:
        return [tail[index + 1] for index, value in enumerate(tail) if value == name]

    assert options(default_import, "--comparison-family") == ["all"]
    assert options(default_import, "--study") == [
        str(output / "public-study" / "public-study.json")
    ]
    # A family without its variant studies is recorded unavailable at import.
    assert options(default_import, "--variant-study") == [
        str(output / name / "public-study.json") for name in variant_names
    ]

    # The plans were built for seven nodes: a resumed request for another
    # count is refused before any step runs.
    recorded.clear()
    other_nodes = [("11" if argument == "7" else argument) for argument in arguments]
    assert main([*other_nodes, "--dry-run", "--resume"]) == 2
    captured = capfd.readouterr()
    assert "nodes was 7 and is now 11" in captured.err
    assert "new output directory" in captured.err
    assert not recorded and "= complete, skipped: " not in captured.out


def test_a_profiles_file_reaches_every_step_that_reads_the_profile(
    tmp_path, monkeypatch, capsys
):
    profiles = tmp_path / "my-profiles.yaml"
    profiles.write_text(
        data_path(EXAMPLE_PROFILES)
        .read_text(encoding="utf-8")
        .replace("\n  test:\n", "\n  mine:\n"),
        encoding="utf-8",
    )
    recorded: list[list[str]] = []
    environments: list[dict[str, str] | None] = []

    def record(command, check, env=None):
        recorded.append(command)
        environments.append(env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_hypothetical_lab_study.subprocess, "run", record)
    monkeypatch.setattr(
        run_hypothetical_lab_study, "public_study_collections", lambda _path: []
    )
    arguments = ["--source-bundle", str(tmp_path), "--profile", "mine", "--live"]

    # The packaged example has no profile "mine": the script stops before it
    # creates its output or runs a step.
    assert main([str(tmp_path / "refused"), *arguments]) == 2
    assert "Unknown model profile 'mine'" in capsys.readouterr().err
    assert not (tmp_path / "refused").exists() and not recorded

    full = [*arguments, "--profiles", str(profiles)]
    assert (
        main(
            [
                str(tmp_path / "run"),
                *full,
                "--variant",
                "weaker",
                "--variant",
                "stronger",
            ]
        )
        == 0
    )
    # Each command that reads the profile takes the global --profile, and
    # every step gets the file as CATMODEL_PROFILES.
    reading = [command for command in recorded if "--profile" in command]
    for command in reading:
        at = command.index("--profile")
        assert command[at : at + 2] == ["--profile", "mine"]
        assert "--profiles" not in command
    assert all(env and env[PROFILES_ENV] == str(profiles) for env in environments)
    assert Counter(" ".join(tail[:2]) for tail in map(_cli_tail, reading)) == {
        "capability-forecast public-study": 3,
        "capability-risk collect": 3,
    }
    # The headline's 15 nodes and each variant's embedded seven.
    assert [
        tail[tail.index("--nodes") + 1]
        for tail in map(_cli_tail, recorded)
        if tail[:2] == ["capability-forecast", "public-study"]
    ] == ["15", "7", "7"]

    # CATMODEL_PROFILES names the file for the script and, through the
    # environment it passes on, for each step it runs.
    recorded.clear()
    environments.clear()
    monkeypatch.setenv(PROFILES_ENV, str(profiles))
    assert main([str(tmp_path / "environment"), *arguments]) == 0
    assert not any("--profiles" in command for command in recorded)
    # The headline's plan and its collection.
    assert sum("--profile" in command for command in recorded) == 2
    assert environments == [None] * len(recorded)


def test_a_named_variant_is_planned_collected_and_imported_alone(
    tmp_path, monkeypatch, capsys
):
    """Naming one variant plans, collects and imports that variant study
    alone, and a variant the planner does not have is refused before any step
    runs."""
    recorded: list[list[str]] = []

    def record(command, check, env=None):
        recorded.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_hypothetical_lab_study.subprocess, "run", record)
    monkeypatch.setattr(
        run_hypothetical_lab_study, "public_study_collections", lambda _path: []
    )
    output = tmp_path / "run"
    common = ["--source-bundle", str(tmp_path), "--live", "--variant"]

    assert main([str(output), *common, "stronger"]) == 0
    tails = [_cli_tail(command) for command in recorded]
    plans = [
        tail for tail in tails if tail[:2] == ["capability-forecast", "public-study"]
    ]
    # The headline and the one named variant.
    assert [
        tail[tail.index("--variant") + 1] if "--variant" in tail else None
        for tail in plans
    ] == [None, "stronger"]
    assert sum(tail[:2] == ["capability-risk", "collect"] for tail in tails) == 2
    (imported,) = [tail for tail in tails if tail[:2] == ["setup", "import"]]
    assert [
        imported[index + 1]
        for index, value in enumerate(imported)
        if value == "--variant-study"
    ] == [str(output / "safeguard-stronger" / "public-study.json")]

    recorded.clear()
    with pytest.raises(SystemExit) as refused:
        main([str(tmp_path / "refused"), *common, "strongest"])
    assert refused.value.code == 2
    assert "invalid choice: 'strongest'" in capsys.readouterr().err
    assert not recorded and not (tmp_path / "refused").exists()


def _stand_in_run(recorded: list[list[str]]):
    """A ``subprocess.run`` that records each command instead of running it
    and writes, as the command's ``--output``, a directory holding the files
    later steps read: each named for the step and holding its command."""

    def run(command, check, env=None):
        assert check is False
        recorded.append(command)
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            for name in (
                "public-study.json",
                "setup-manifest.json",
                "results.json",
                "capture/run-capture.json",
            ):
                (output / name).parent.mkdir(parents=True, exist_ok=True)
                (output / name).write_text(" ".join(command), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    return run


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    """A completed live run of a seven-node headline with the stronger
    safeguard variant, whose steps were stood in for (``_stand_in_run``) and
    whose outputs every check accepts; returns the commands recorded, the
    output directory and the run's arguments."""
    recorded: list[list[str]] = []
    monkeypatch.setattr(
        run_hypothetical_lab_study.subprocess, "run", _stand_in_run(recorded)
    )
    for check in (
        "public_study_collections",
        "load_setup",
        "_check_model_run",
        "_check_summary",
        "load_results_export",
    ):
        monkeypatch.setattr(run_hypothetical_lab_study, check, lambda _path: [])
    profiles = tmp_path / "profiles.yaml"
    profiles.write_bytes(data_path(EXAMPLE_PROFILES).read_bytes())
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / MANIFEST_PATH).write_text("bundle_id: first\n", encoding="utf-8")
    output = tmp_path / "run"
    arguments = [
        str(output),
        "--source-bundle",
        str(bundle),
        "--profiles",
        str(profiles),
        "--nodes",
        "7",
        "--variant",
        "stronger",
    ]
    assert main([*arguments, "--live"]) == 0
    recorded.clear()
    return recorded, output, arguments


RESUMED_OUTPUTS = (
    "public-study",
    "safeguard-stronger",
    "setup",
    "model-run",
    "summary",
    "export",
)
"""The outputs of ``completed_run`` that ``--resume`` can skip."""


def test_resume_skips_every_output_built_from_the_same_request(completed_run, capsys):
    """The same request resumes whether it is given as the full run or as a
    dry run, and each skipped output has a request record beside it."""
    recorded, output, arguments = completed_run
    capsys.readouterr()

    for mode, outputs in (
        ("--live", RESUMED_OUTPUTS),
        ("--dry-run", RESUMED_OUTPUTS[:2]),
    ):
        recorded.clear()
        assert main([*arguments, mode, "--resume"]) == 0
        captured = capsys.readouterr()
        assert captured.out.count("= complete, skipped: ") == len(outputs)
        assert not any("--output" in command for command in recorded)
    for name in RESUMED_OUTPUTS:
        assert json.loads((output / f"{name}.request.json").read_bytes())


def _replace(arguments: list[str], option: str, value: str) -> list[str]:
    at = arguments.index(option) + 1
    return [*arguments[:at], value, *arguments[at + 1 :]]


@pytest.mark.parametrize(
    ("change", "refused", "difference"),
    [
        (
            lambda arguments: _replace(arguments, "--nodes", "15"),
            "public-study",
            "nodes was 7 and is now 15",
        ),
        (
            lambda arguments: [*arguments, "--profile", "test"],
            "public-study",
            'profile was "example" and is now "test"',
        ),
        (
            lambda arguments: [*arguments, "--variant", "weaker"],
            "setup",
            'variants was ["stronger"] and is now ["weaker", "stronger"]',
        ),
        (
            lambda arguments: arguments[:-2],
            "setup",
            'variants was ["stronger"] and is now []',
        ),
    ],
    ids=["other-nodes", "other-profile", "variant-added", "variant-dropped"],
)
def test_resume_refuses_a_changed_request_before_any_step_runs(
    completed_run, capsys, change, refused, difference
):
    """A completed run does not answer another request: resuming it with
    other nodes, another profile or other variants names the first output
    built from the earlier request and what differs, and runs nothing, so no
    added variant is collected for a setup that cannot take it."""
    recorded, output, arguments = completed_run
    capsys.readouterr()

    for mode in ("--dry-run", "--live"):
        assert main([*change(arguments), mode, "--resume"]) == 2
        captured = capsys.readouterr()
        assert f"{output / refused} was built from another request" in captured.err
        assert difference in captured.err
        assert "new output directory" in captured.err
        assert not recorded and "= complete, skipped: " not in captured.out


def test_resume_refuses_an_output_whose_inputs_have_changed(
    completed_run, tmp_path, capsys
):
    """Each skipped output is bound to the files it was built from: another
    profiles file or source bundle refuses the plans, and a plan rebuilt
    after the import refuses the setup, each naming the input that differs."""
    recorded, output, arguments = completed_run
    capsys.readouterr()

    def refusal(resumed: list[str]) -> str:
        assert main([*resumed, "--live", "--resume"]) == 2
        captured = capsys.readouterr()
        assert "new output directory" in captured.err
        assert not any("--output" in command for command in recorded)
        return captured.err

    profiles = tmp_path / "profiles.yaml"
    original = profiles.read_bytes()
    profiles.write_bytes(original + b"\n")
    error = refusal(arguments)
    assert f"{output / 'public-study'} was built from another request" in error
    assert "profiles_file_sha256 was " in error
    profiles.write_bytes(original)

    other_bundle = tmp_path / "other-bundle"
    other_bundle.mkdir()
    (other_bundle / MANIFEST_PATH).write_text("bundle_id: second\n", encoding="utf-8")
    error = refusal(_replace(arguments, "--source-bundle", str(other_bundle)))
    assert f"{output / 'public-study'} was built from another request" in error
    assert "source_bundle_manifest_sha256 was " in error

    plan = output / "safeguard-stronger" / "public-study.json"
    plan.write_bytes(plan.read_bytes() + b" ")
    error = refusal(arguments)
    assert f"{output / 'setup'} was built from another request" in error
    assert "safeguard-stronger/public-study.json was " in error


def test_resume_refuses_an_output_without_its_request_record(completed_run, capsys):
    recorded, output, arguments = completed_run
    capsys.readouterr()
    (output / "setup.request.json").unlink()

    assert main([*arguments, "--live", "--resume"]) == 2
    captured = capsys.readouterr()
    assert (
        f"{output / 'setup'} exists without a readable request record" in captured.err
    )
    assert "new output directory" in captured.err
    assert not any("--output" in command for command in recorded)


def test_resume_refuses_an_output_that_does_not_validate(tmp_path, capsys):
    output = tmp_path / "run"
    (output / "public-study").mkdir(parents=True)
    (output / "public-study" / "public-study.json").write_text(
        '{"schema_version"', "utf-8"
    )

    assert (
        main([str(output), "--source-bundle", str(tmp_path), "--dry-run", "--resume"])
        == 2
    )
    captured = capsys.readouterr()
    assert "public-study exists but does not validate" in captured.err
    assert "+ " not in captured.out


def test_a_small_study_names_its_scenarios_and_may_have_three_nodes(
    tmp_path, monkeypatch, capsys
):
    """A study without variants takes any odd node count of at least 3, and
    ``--scenario`` narrows it to the scenarios named; a variant still needs a
    headline whose embedded rule has an odd number of at least 3 nodes."""
    recorded: list[list[str]] = []
    monkeypatch.setattr(
        run_hypothetical_lab_study.subprocess, "run", _stand_in_run(recorded)
    )
    monkeypatch.setattr(
        run_hypothetical_lab_study, "public_study_collections", lambda _path: []
    )
    output = tmp_path / "run"
    common = [str(output), "--source-bundle", str(tmp_path), "--dry-run"]

    for refused, reason in (
        (["--nodes", "3", "--variant", "weaker"], "embedded rule"),
        (["--nodes", "9", "--variant", "weaker"], "embedded rule"),
        (["--nodes", "1"], "odd number of at least 3"),
        (["--nodes", "4"], "odd number of at least 3"),
    ):
        with pytest.raises(SystemExit) as stopped:
            main([*common, *refused])
        assert stopped.value.code == 2
        assert reason in capsys.readouterr().err
    assert not recorded and not output.exists()

    scenarios = ["output_reliance_legal", "fraud_organizations"]
    small = [*common, "--nodes", "3"]
    for scenario in scenarios:
        small += ["--scenario", scenario]
    assert main(small) == 0
    plan, collection = map(_cli_tail, recorded)
    assert plan[:2] == ["capability-forecast", "public-study"]
    assert plan[plan.index("--nodes") + 1] == "3"
    # In place of the complete catalog, in the planner's order.
    assert [
        plan[index + 1] for index, value in enumerate(plan) if value == "--scenario"
    ] == sorted(scenarios)
    assert collection[:2] == ["capability-risk", "collect"]
    assert "--dry-run" in collection

    # The plan answers that selection: a resumed request for another is
    # refused before any step runs.
    recorded.clear()
    capsys.readouterr()
    assert main([*small[:-2], "--resume"]) == 2
    assert (
        'scenarios was ["fraud_organizations", "output_reliance_legal"] and is '
        'now ["output_reliance_legal"]'
    ) in capsys.readouterr().err
    assert not recorded


@pytest.mark.parametrize("modes", [(), ("--dry-run", "--live")])
def test_a_run_names_exactly_one_of_dry_run_and_live(tmp_path, capsys, modes):
    """A collection that calls providers is never started by omission: the
    script stops before it writes anything unless one mode is named."""
    output = tmp_path / "run"

    assert main([str(output), *modes]) == 2
    captured = capsys.readouterr()
    assert "--dry-run" in captured.err and "--live" in captured.err
    assert not output.exists()
    assert "+ " not in captured.out


def test_existing_output_is_refused(tmp_path, capsys):
    output = tmp_path / "already-exists"
    output.mkdir()

    assert main([str(output), "--dry-run"]) == 2
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert "--resume" in captured.err
    assert "+ " not in captured.out


def test_output_inside_the_checkout_is_refused(tmp_path, monkeypatch, capsys):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    output = checkout / "not-created" / "run"
    monkeypatch.setattr(
        run_hypothetical_lab_study, "source_checkout_root", lambda: checkout
    )

    assert main([str(output), "--dry-run"]) == 2

    captured = capsys.readouterr()
    assert "inside the source checkout" in captured.err
    assert "write outside the checkout" in captured.err
    assert "+ " not in captured.out
    assert not output.exists()
    assert not output.parent.exists()
