"""Contextual and panel collection share commands without sharing meanings."""

import json

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.config import PROFILES_ENV, load_model_profile
from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from test_support.contextual_risk import contextual_artifact, contextual_instrument
from test_support.version_markers import unstated_markers

MY_PROFILES = """\
profiles:
  mine:
    models:
      - provider: openai
        model_id: my-model
        reasoning_effort: medium
        max_output_tokens: 1000
    settings:
      samples_per_question: 2
"""


def test_build_requires_a_context(tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "capability-risk",
            "build-instrument",
            "--source-bundle",
            str(tmp_path),
            "--output",
            str(tmp_path / "output.json"),
        ],
    )
    assert result.exit_code == 2
    assert "Missing option '--context'" in result.output
    assert not (tmp_path / "output.json").exists()


def test_context_build_accepts_explicit_nodes_without_scenario(tmp_path, monkeypatch):
    import catmodel.setup.contextual_inputs as inputs

    instrument = contextual_instrument()
    context = tmp_path / "context.json"
    context.write_text(instrument.model_dump_json())
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return instrument

    monkeypatch.setattr(inputs, "build_contextual_instrument", build)
    output = tmp_path / "instrument.json"
    result = CliRunner().invoke(
        cli,
        [
            "--profile",
            "test",
            "capability-risk",
            "build-instrument",
            "--source-bundle",
            str(tmp_path),
            "--context",
            str(context),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["scenario_ids"] == ()
    assert (
        ContextualRiskInstrument.model_validate_json(output.read_bytes()) == instrument
    )


@pytest.mark.parametrize("variant", [None, "double", "weaker"])
def test_the_study_command_hands_the_planner_its_inputs_and_the_profiles_respondents(
    tmp_path, monkeypatch, variant
):
    """public-study plans a headline or, with --variant, a variant study, which
    asks three samples of each model of the profile instead of its
    samples_per_question."""
    import catmodel.setup.public_study as planner

    paths = {}
    for name in ("forecast", "context", "assumptions"):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text("{}")
    calls = []

    def plan(**kwargs):
        calls.append(kwargs)
        return {"trajectory_nodes": 8}

    monkeypatch.setattr(planner, "plan_public_frontier_study", plan)
    result = CliRunner().invoke(
        cli,
        [
            "capability-forecast",
            "public-study",
            "--source-bundle",
            str(tmp_path),
            "--forecast",
            str(paths["forecast"]),
            "--context",
            str(paths["context"]),
            "--scenario",
            "all",
            *(("--variant", variant) if variant else ()),
            "--assumptions",
            str(paths["assumptions"]),
            "--output",
            str(tmp_path / "study"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"trajectory_nodes": 8}
    (call,) = calls
    # The default profile, dev, asks five samples per question.
    profile = load_model_profile("dev")
    assert profile.settings.samples_per_question != 3
    samples = 3 if variant else profile.settings.samples_per_question
    respondents = call.pop("respondents")
    assert [(row.model_id, row.sample_index) for row in respondents] == [
        (spec.model_id, sample) for spec in profile.models for sample in range(samples)
    ]
    assert call == {
        "source_bundle": tmp_path,
        "forecast_path": paths["forecast"],
        "context_path": paths["context"],
        "scenario_ids": ("all",),
        "variant": variant,
        "assumptions_path": paths["assumptions"],
        "nodes": None,
        "output": tmp_path / "study",
    }
    unknown = CliRunner().invoke(
        cli, ["capability-forecast", "public-study", "--variant", "triple"]
    )
    assert unknown.exit_code == 2
    assert "'triple' is not one of 'half', 'double', 'weaker', 'stronger'" in (
        unknown.output
    )


@pytest.mark.parametrize("variant", [None, "half"])
def test_the_study_command_refuses_a_node_count_no_rule_takes_before_planning(
    tmp_path, monkeypatch, variant
):
    """--nodes is an odd count of at least three, checked with the planner's
    own rule when the command line is read, so a refused count is a usage
    error and never reaches the planner. Every such count is planned,
    including a headline's whose embedded rule no variant study can be
    planned on."""
    import catmodel.setup.public_study as planner
    from catmodel.forecast.public_frontier_quadrature import require_trajectory_nodes

    calls = []

    def plan(**kwargs):
        calls.append(kwargs["nodes"])
        return {}

    monkeypatch.setattr(planner, "plan_public_frontier_study", plan)
    for name in ("forecast", "context", "assumptions"):
        (tmp_path / f"{name}.json").write_text("{}")
    arguments = [
        "capability-forecast",
        "public-study",
        "--source-bundle",
        str(tmp_path),
        "--forecast",
        str(tmp_path / "forecast.json"),
        "--context",
        str(tmp_path / "context.json"),
        "--assumptions",
        str(tmp_path / "assumptions.json"),
        *(("--variant", variant) if variant else ()),
        "--output",
        str(tmp_path / "study"),
    ]
    for nodes in ("-3", "0", "1", "2", "4", "6", "16"):
        refused = CliRunner().invoke(cli, [*arguments, "--nodes", nodes])
        assert refused.exit_code == 2, (nodes, refused.output)
        assert "Invalid value for '--nodes'" in refused.output
        assert "odd number of at least 3 nodes" in refused.output
        with pytest.raises(ValueError, match="odd number of at least 3 nodes"):
            require_trajectory_nodes(int(nodes))
    assert calls == []
    accepted = (3, 5, 7, 9, 11, 13, 15, 31)
    for nodes in accepted:
        assert require_trajectory_nodes(nodes) == nodes
        result = CliRunner().invoke(cli, [*arguments, "--nodes", str(nodes)])
        assert result.exit_code == 0, (nodes, result.output)
    assert calls == list(accepted)


def test_build_reads_the_roster_from_the_profiles_file_the_environment_names(
    tmp_path, monkeypatch
):
    import catmodel.setup.contextual_inputs as inputs

    instrument = contextual_instrument()
    context = tmp_path / "context.json"
    context.write_text(instrument.model_dump_json())
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return instrument

    monkeypatch.setattr(inputs, "build_contextual_instrument", build)
    profiles = tmp_path / "my-profiles.yaml"
    profiles.write_text(MY_PROFILES, encoding="utf-8")
    runner = CliRunner()

    def invoke(name, profiles_file=None):
        return runner.invoke(
            cli,
            [
                "--profile",
                "mine",
                "capability-risk",
                "build-instrument",
                "--source-bundle",
                str(tmp_path),
                "--context",
                str(context),
                "--output",
                str(tmp_path / f"{name}.json"),
            ],
            env={PROFILES_ENV: None if profiles_file is None else str(profiles_file)},
        )

    by_environment = invoke("environment", profiles)
    assert by_environment.exit_code == 0, by_environment.output
    assert [(row.model_id, row.sample_index) for row in calls[0]["respondents"]] == [
        ("my-model", 0),
        ("my-model", 1),
    ]

    # Without it, the profile is read from the packaged example.
    example = invoke("example")
    assert example.exit_code == 1
    assert "Unknown model profile 'mine'" in example.output
    assert "Available profiles: example, test, dev" in example.output
    absent = invoke("absent", tmp_path / "absent.yaml")
    assert absent.exit_code == 1
    assert f"model profiles file not found: {tmp_path / 'absent.yaml'}" in (
        absent.output
    )
    profiles.write_text(MY_PROFILES.replace("settings:", "setings:"), encoding="utf-8")
    malformed = invoke("malformed", profiles)
    assert malformed.exit_code == 1
    assert "profile 'mine'" in malformed.output
    assert "Extra inputs are not permitted" in malformed.output
    assert len(calls) == 1


def test_context_collect_dispatches_without_panel_source_loading(tmp_path, monkeypatch):
    import catmodel.collection.contextual_collection as collection

    instrument = contextual_instrument()
    path = tmp_path / "instrument.json"
    path.write_text(instrument.model_dump_json())
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(MY_PROFILES, encoding="utf-8")
    calls = []

    def collect(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(collection, "run_contextual_collection", collect)

    def invoke():
        return CliRunner().invoke(
            cli,
            [
                "--profile",
                "mine",
                "capability-risk",
                "collect",
                "--source-bundle",
                str(tmp_path),
                "--instrument",
                str(path),
                "--run-directory",
                str(tmp_path / "run"),
                "--dry-run",
            ],
            env={PROFILES_ENV: str(profiles)},
        )

    result = invoke()
    assert result.exit_code == 0, result.output
    assert "Validated frozen collection plan" in result.output
    assert calls == [
        {
            "instrument": instrument,
            "source_bundle": tmp_path,
            "run_directory": tmp_path / "run",
            "profile": "mine",
            "dry_run": True,
            "answers": None,
        }
    ]
    # An instrument file that does not state its schema reaches no collection.
    for payload in unstated_markers(json.loads(instrument.model_dump_json())).values():
        path.write_text(json.dumps(payload))
        refused = invoke()
        assert refused.exit_code != 0
        assert "contextual instrument schema" in refused.output
    assert len(calls) == 1


def test_study_collect_dispatches_the_plan_and_trajectory_nodes(tmp_path, monkeypatch):
    import catmodel.collection.contextual_collection as collection

    study = tmp_path / "public-study.json"
    study.write_text("{}")
    calls = []

    def collect(**kwargs):
        calls.append(kwargs)
        return None if kwargs["dry_run"] else (object(), object())

    monkeypatch.setattr(collection, "run_study_collection", collect)

    def invoke(*options):
        result = CliRunner().invoke(
            cli,
            [
                "--profile",
                "test",
                "--workers",
                "3",
                "capability-risk",
                "collect",
                "--study",
                str(study),
                "--source-bundle",
                str(tmp_path),
                *options,
            ],
        )
        assert result.exit_code == 0, result.output
        return result.stdout

    assert invoke("--dry-run") == "Validated frozen collection plans\n"
    assert invoke("--answers", str(tmp_path / "answers")) == "Collected 2 contexts\n"
    # The global --workers is the study's collector processes.
    dispatched = {
        "study": study,
        "source_bundle": tmp_path,
        "profile": "test",
        "workers": 3,
    }
    assert calls == [
        {**dispatched, "dry_run": True, "answers": None},
        {**dispatched, "dry_run": False, "answers": tmp_path / "answers"},
    ]


@pytest.mark.parametrize(
    "selection",
    [
        [],
        ["--study", "public-study.json", "--instrument", "instrument.json"],
        ["--instrument", "instrument.json"],
        ["--study", "public-study.json", "--run-directory", "run"],
    ],
    ids=["neither", "both", "instrument-without-directory", "study-with-directory"],
)
def test_collect_takes_a_study_or_one_instrument_and_its_directory(
    tmp_path, monkeypatch, selection
):
    for name in ("public-study.json", "instrument.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli,
        ["capability-risk", "collect", "--source-bundle", ".", *selection],
    )
    assert result.exit_code == 2
    assert "--study" in result.output or "--run-directory" in result.output


def test_context_validate_uses_frozen_contextual_loader(tmp_path, monkeypatch):
    import catmodel.setup.contextual_inputs as inputs

    instrument = contextual_instrument()
    artifact = contextual_artifact(instrument)
    (tmp_path / "contextual-instrument.json").write_text(instrument.model_dump_json())
    monkeypatch.setattr(
        inputs,
        "load_contextual_collection",
        lambda directory: (instrument, artifact, {}),
    )
    result = CliRunner().invoke(
        cli, ["capability-risk", "validate", "--run-directory", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "evidence_status": "fixture",
        "complete_cases": 3,
        "unanswered_cases": [],
        "unanswered_scenarios": [],
        "exhausted_calls": [],
        "p999_implied_gpd_shapes": [],
        "call_limits": {},
        # Three cases, each with a direct and a chain answer that cite nothing.
        "reasoning_use": [
            {
                "node_id": "synthetic-node",
                "model_id": "fixture-model",
                "answers": 6,
                "construction": 0,
                "citing_excerpts": 0,
                "citing_threat_model": None,
            }
        ],
    }


def test_contextual_all_selects_current_catalog_and_is_exclusive(tmp_path):
    from catmodel.resources import scenarios_path
    from catmodel.scenarios.loader import load_scenarios
    from catmodel.setup.contextual_inputs import build_contextual_instrument
    from test_support.capability_sources import write_source_bundle

    source = write_source_bundle(tmp_path / "source")
    original = contextual_instrument()
    payload = original.model_dump(
        mode="json", exclude={"nodes", "respondents", "source_bundle_manifest_sha256"}
    )
    instrument = build_contextual_instrument(
        payload,
        source_bundle=source,
        respondents=original.respondents,
        scenario_ids=("all",),
    )
    assert {node.node_id for node in instrument.nodes} == {
        scenario.id for scenario in load_scenarios(scenarios_path())
    }
    assert all(node.unavailable_reason is None for node in instrument.nodes)
    with pytest.raises(ValueError, match="all must be used alone"):
        build_contextual_instrument(
            payload,
            source_bundle=source,
            respondents=original.respondents,
            scenario_ids=("all", "influence_manipulation"),
        )


def test_equivalent_scenario_selections_build_the_same_instrument(tmp_path):
    """Naming every scenario is the selection ``all``: the same nodes in the
    same order and the same taxonomy parents. A selection is a set, so its
    order does not reach the instrument, and one short of the catalog asks
    no parent."""
    from catmodel.elicitation.contextual_hierarchy import hierarchy_parents
    from catmodel.resources import scenarios_path
    from catmodel.scenarios.loader import load_scenarios
    from catmodel.setup.contextual_inputs import build_contextual_instrument
    from test_support.capability_sources import write_source_bundle

    source = write_source_bundle(tmp_path / "source")
    original = contextual_instrument()
    payload = original.model_dump(
        mode="json", exclude={"nodes", "respondents", "source_bundle_manifest_sha256"}
    )

    def build(*scenario_ids):
        return build_contextual_instrument(
            payload,
            source_bundle=source,
            respondents=original.respondents,
            scenario_ids=scenario_ids,
        )

    scenarios = {row.id: row for row in load_scenarios(scenarios_path())}
    catalog = sorted(scenarios)
    assert len(catalog) > 2
    everything = build("all")
    assert [node.node_id for node in everything.nodes] == catalog
    assert everything.parents == hierarchy_parents(everything.nodes, scenarios)
    assert everything.parents
    for named in (catalog, catalog[::-1], [*catalog[1:], catalog[0]]):
        assert build(*named).model_dump_json() == everything.model_dump_json()
    for partial in (catalog[:-1], catalog[1:], catalog[1::2]):
        instrument = build(*partial[::-1])
        assert [node.node_id for node in instrument.nodes] == partial
        assert instrument.parents == ()
        assert instrument == build(*partial)
    with pytest.raises(ValueError, match="unique known identifiers"):
        build(*catalog, catalog[0])
    with pytest.raises(ValueError, match="all must be used alone"):
        build(*catalog, "all")
