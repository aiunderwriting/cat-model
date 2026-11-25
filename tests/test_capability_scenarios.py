"""Replay-time capability paths over one offline public-study run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.stats import truncnorm

import catmodel.results.capability_scenarios as capability_module
from catmodel.artifacts import frozen_json
from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from catmodel.forecast.public_frontier_quadrature import (
    base_progress_cdf,
    base_progress_quantile,
)
from catmodel.results.execution import (
    arm_weights,
    load_run_capture,
    replay_capture,
    run_setup,
)
from catmodel.results.respondent_sampling import SampleGroups, sampling_view
from catmodel.results.results_export import export_results, load_results_export
from catmodel.results.results_file import diagnostics_path
from catmodel.setup.public_study import import_public_frontier_study
from catmodel.simulation.capability_mixture import weighted_var_tvar
from catmodel.simulation.incident_runtime import state_key
from catmodel.simulation.tail_gate import MIN_TAIL_EFFECTIVE_UNITS
from test_support.contextual_risk import (
    governed_contextual_answer,
    governed_contextual_instrument,
)
from test_support.public_frontier import synthetic_forecast
from test_support.results_schema import schema_violations
from tests.test_public_frontier_study import _collect, _Provider, _study

SCENARIO_COLUMNS = (
    "scenario",
    "label",
    "scenario_status",
    "forecast_probability",
    "percentile",
    "progress",
    "progress_cdf",
    "eci_period_start",
    "eci_period_end",
    "lower_node",
    "upper_node",
    "interpolation_weight",
    "view",
    "measure",
    "source_status",
    "quantity",
    "non_exceedance_probability",
    "return_period_years",
    "loss_threshold",
    "value",
    "status",
    "reason",
    "sampling_standard_error",
    "sampling_reason",
    "half_width",
    "relative_half_width",
    "minimum_batch_exceedances",
    "years_per_state",
)
NODE_COLUMNS = (
    "node_id",
    "progress",
    "progress_cdf",
    "eci_period_start",
    "eci_period_end",
    "base_weight",
    "arm",
    "samples",
    "view",
    "measure",
    "source_status",
    "expected_annual_loss",
    "status",
    "reason",
    "sampling_standard_error",
    "sampling_reason",
)


@pytest.fixture(scope="module")
def capability_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("capability-scenarios")
    models = ("fixture-a", "fixture-b", "fixture-c")
    template_payload = governed_contextual_instrument(
        ("compromise_critical_infrastructure",), models
    ).model_dump(mode="json")
    template = ContextualRiskInstrument.model_validate(template_payload)
    providers = [
        _Provider(
            model_id=model,
            answer=lambda call, node=template.nodes[0], scale=index + 1: (
                governed_contextual_answer(node, scale=scale * (1 + (call - 1) // 2))
            ),
        )
        for index, model in enumerate(models)
    ]
    patcher = pytest.MonkeyPatch()
    try:
        study, bundle, _ = _study(
            root,
            patcher,
            nodes=3,
            template=template,
            providers=providers,
            forecast_input=synthetic_forecast().model_copy(update={"pace_shape": 5.0}),
        )

        # Keep fake collection deterministic and avoid executor teardown for
        # these small, synchronous chain-law calculations.
        async def inline_to_thread(
            function: Callable[..., Any], *args: Any, **kwargs: Any
        ) -> Any:
            return function(*args, **kwargs)

        patcher.setattr("asyncio.to_thread", inline_to_thread)
        _collect(study, bundle)
    finally:
        patcher.undo()

    package = import_public_frontier_study(
        source=study / "public-study.json",
        destination=root / "package",
        setup_id="capability-scenarios-fixture",
        years_per_state=16,
        max_reference_state_years=12 * 4096,
        max_comparison_state_years=12 * 1024,
        comparison_families=("public_forecast",),
        seed=3,
        code_commit="fixture",
    )
    run_directory = root / "run"
    report = run_setup(
        setup_path=package.root,
        output=run_directory,
        code_commit="fixture",
    )
    draws, capture = load_run_capture(run_directory / "capture")
    targets = arm_weights(package, None)

    started = time.perf_counter()
    direct_record = capability_module.capability_scenarios(
        package, draws, capture, report, targets, reweighted=False
    )
    reduction_seconds = time.perf_counter() - started
    tracemalloc.start()
    try:
        capability_module.capability_scenarios(
            package, draws, capture, report, targets, reweighted=False
        )
        _, reduction_peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert direct_record == report["capability_scenarios"]
    # Without safeguard-level variants the exhibit has no grid section.
    assert "safeguard_grid" not in direct_record
    print(
        "capability_scenarios added reduction: "
        f"{reduction_seconds:.3f}s; traced peak {reduction_peak_bytes / 1024**2:.2f} MiB"
    )

    export_directory = root / "export"
    export_index = export_results(
        setup_path=package.root,
        capture_directory=run_directory / "capture",
        output=export_directory,
        code_commit="fixture",
    )
    trajectory = next(
        row.acquisition["trajectory_quadrature"]
        for row in package.evidence.records
        if row.acquisition and "trajectory_quadrature" in row.acquisition
    )
    return {
        "root": root,
        "package": package,
        "run_directory": run_directory,
        "report": report,
        "draws": draws,
        "capture": capture,
        "targets": targets,
        "trajectory": trajectory,
        "record": report["capability_scenarios"],
        "reduction_seconds": reduction_seconds,
        "reduction_peak_bytes": reduction_peak_bytes,
        "export_directory": export_directory,
        "export_index": export_index,
    }


def _node_samples(fixture, node_id: str, view: str):
    package = fixture["package"]
    trajectory = fixture["trajectory"]
    arm = "public_forecast::no_progress" if node_id == "node-00" else "reference"
    target = fixture["targets"][arm]
    members = []
    total = math.fsum(
        target[state_key(state.state_id)]
        for state in package.joint_states
        if trajectory["state_draws"].get(state.state_id) == node_id
        and target[state_key(state.state_id)] > 0
    )
    for state in package.joint_states:
        if trajectory["state_draws"].get(state.state_id) != node_id:
            continue
        key = state_key(state.state_id)
        state_weight = target[key]
        if state_weight <= 0:
            continue
        draw = fixture["draws"].get(arm, {}).get(key)
        assert draw is not None
        state_weight /= total
        losses = draw.annual_losses[view]
        masses = state_weight * draw.normalized_weights()
        members.append((state, draw, state_weight, losses, masses))
    return (
        np.concatenate([row[3] for row in members]),
        np.concatenate([row[4] for row in members]),
        members,
    )


def _rehash_export_output(directory, name: str, data: bytes) -> None:
    index_path = directory / "export.json"
    index = json.loads(index_path.read_bytes())
    index["outputs"][name]["sha256"] = hashlib.sha256(data).hexdigest()
    index_path.write_text(
        json.dumps(index, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_the_public_study_results_files_match_their_schemas(capability_fixture):
    """A synthetic public study's results and diagnostics files conform to
    their JSON Schema documents, and so do the export's own, reduced from the
    capture: they hold no execution context, and the diagnostics file holds
    the exhibits and no view diagnostics."""
    results = capability_fixture["run_directory"] / "results.json"
    assert schema_violations(results) == []
    exported = capability_fixture["export_directory"] / "native" / "results.json"
    assert "execution_context" not in json.loads(exported.read_bytes())
    assert schema_violations(exported) == []
    diagnostics = json.loads(diagnostics_path(exported).read_bytes())
    assert diagnostics["arms"] == {}
    assert diagnostics["capability_scenarios"]["status"] == "available"


def test_a_study_reports_count_laws_at_face_value(capability_fixture):
    record = capability_fixture["record"]
    assert record["respondent_spread"] == {
        "k": 1.0,
        "basis": "face_value",
        "level_bias": 1.0,
    }
    assert (
        "EALs are conditional on respondent spread k = 1 (face value)."
        in record["disclosures"]
    )


def test_current_frontier_reconciles_every_requested_view(capability_fixture):
    fixture = capability_fixture
    current = fixture["record"]["scenarios"][0]
    assert current["scenario"] == "current_frontier"
    assert current["forecast_probability"] == 0
    assert "counterfactual" in current["forecast_probability_note"]
    no_progress = fixture["report"]["arms"]["public_forecast::no_progress"]
    assert set(current["by_view"]) == {
        view for view in no_progress["metrics"] if sampling_view(view)
    }
    for view, data in current["by_view"].items():
        expected = no_progress["metrics"][view].get("expected_annual_loss")
        reconciliation = fixture["record"]["reconciliation"][view]
        if expected is None:
            assert data["expected_annual_loss"] is None
            assert reconciliation["status"] == "unavailable"
        else:
            assert data["expected_annual_loss"] == pytest.approx(
                expected, rel=1e-9, abs=0
            )
            assert reconciliation["status"] == "available"
            assert reconciliation["scenario"] == "current_frontier"
            assert reconciliation["arm"] == "public_forecast::no_progress"
            assert (
                reconciliation["current_frontier_expected_annual_loss"]
                == (data["expected_annual_loss"])
            )
            assert reconciliation["no_progress_arm_expected_annual_loss"] == expected
        assert "difference" not in reconciliation
        assert "matches" not in reconciliation


def test_base_progress_percentiles_match_cdf_and_independent_sampling(
    capability_fixture,
):
    """Each percentile path's progress inverts the base law's CDF, and the CDF
    matches direct sampling of independent lab paths: the pooled rate from its
    positive-truncated kernel and each lab's own log-logistic pace, redrawn
    while it lies beyond the pace cap."""
    base = capability_fixture["trajectory"]["base_law"]
    pace_shape, pace_cap = base["pace_shape"], base["pace_cap"]
    assert (pace_shape, pace_cap) == (5, 4)
    posterior = base["posterior"]
    elapsed = base["progress_elapsed_years"]
    location = posterior["rate_location"]
    scale = math.sqrt(posterior["rate_variance"])
    anchors = np.asarray([row["as_of_value"] for row in posterior["labs"].values()])
    rng = np.random.default_rng(17)
    size = 400_000
    rates = truncnorm.rvs(
        -location / scale,
        np.inf,
        loc=location,
        scale=scale,
        size=size,
        random_state=rng,
    )
    pace = np.full((size, anchors.size), np.inf)
    while (beyond := pace > pace_cap).any():
        uniforms = rng.random(int(beyond.sum()))
        pace[beyond] = np.exp((np.log(uniforms) - np.log1p(-uniforms)) / pace_shape)
    progress = (anchors + rates[:, None] * pace * elapsed).max(axis=1) - anchors.max()
    for probability in (0.1, 0.5, 0.9):
        point = float(
            base_progress_quantile(
                pace_shape, posterior, elapsed, probability, pace_cap=pace_cap
            )
        )
        assert base_progress_cdf(
            pace_shape, posterior, elapsed, point, pace_cap=pace_cap
        ) == pytest.approx(probability, abs=2e-10)
        sampled = float(np.mean(progress <= point))
        tolerance = 5 * math.sqrt(probability * (1 - probability) / size)
        assert sampled == pytest.approx(probability, abs=tolerance)
        scenario = next(
            row
            for row in capability_fixture["record"]["scenarios"]
            if row["percentile"] == probability
        )
        assert scenario["progress"] == pytest.approx(point, rel=1e-12)
        assert scenario["progress_cdf"] == pytest.approx(probability, abs=2e-10)


def test_percentile_exactly_on_a_node_reuses_its_eal_and_curves(
    capability_fixture, monkeypatch
):
    fixture = capability_fixture
    node = next(row for row in fixture["record"]["nodes"] if row["progress"] > 0)
    percentile = float(
        base_progress_cdf(
            fixture["trajectory"]["base_law"]["pace_shape"],
            fixture["trajectory"]["base_law"]["posterior"],
            fixture["trajectory"]["base_law"]["progress_elapsed_years"],
            node["progress"],
            pace_cap=fixture["trajectory"]["base_law"]["pace_cap"],
        )
    )
    monkeypatch.setattr(
        capability_module,
        "REPRESENTATIVE_PERCENTILES",
        (percentile, *capability_module.REPRESENTATIVE_PERCENTILES[1:]),
    )
    exact = capability_module.capability_scenarios(
        fixture["package"],
        fixture["draws"],
        fixture["capture"],
        fixture["report"],
        fixture["targets"],
        reweighted=False,
    )
    scenarios = cast(list[dict[str, Any]], exact["scenarios"])
    scenario = scenarios[1]
    interpolation = scenario["interpolation"]
    assert interpolation["exact_node"] is True
    assert interpolation["weight"] == 0
    assert interpolation["lower_node"]["node_id"] == node["node_id"]
    assert interpolation["upper_node"]["node_id"] == node["node_id"]
    for view, data in scenario["by_view"].items():
        node_view = node["by_view"][view]
        assert data["expected_annual_loss"] == node_view["expected_annual_loss"]
        losses, masses, _ = _node_samples(fixture, node["node_id"], view)
        for point in data["quantiles"]:
            expected = capability_module.vincentized_quantile(
                losses,
                masses,
                losses,
                masses,
                0,
                point["non_exceedance_probability"],
            )
            assert point["annual_loss"] == expected
        for point in data["exceedance"]:
            expected = capability_module.vincentized_exceedance_probability(
                losses,
                masses,
                losses,
                masses,
                0,
                point["loss"],
            )
            assert point["probability"] == expected


def test_scenario_eci_interpolates_linearly_in_progress(capability_fixture):
    fixture = capability_fixture
    trajectory = fixture["trajectory"]
    node_eci = trajectory["node_eci"]
    nodes = {row["node_id"]: row for row in fixture["record"]["nodes"]}
    checked = 0
    for scenario in fixture["record"]["scenarios"][1:]:
        interpolation = scenario["interpolation"]
        lower = nodes[interpolation["lower_node"]["node_id"]]
        upper = nodes[interpolation["upper_node"]["node_id"]]
        if lower["node_id"] == upper["node_id"]:
            continue
        weight = (scenario["progress"] - lower["progress"]) / (
            upper["progress"] - lower["progress"]
        )
        lower_eci = node_eci[lower["node_id"]]
        upper_eci = node_eci[upper["node_id"]]
        for lab, values in scenario["eci_by_lab"].items():
            for period in ("period_start", "period_end"):
                expected = lower_eci[period][lab] + weight * (
                    upper_eci[period][lab] - lower_eci[period][lab]
                )
                assert values[period] == pytest.approx(expected, rel=1e-14)
        checked += 1
    assert checked > 0


def test_percentile_scenarios_follow_representative_percentiles(
    capability_fixture, monkeypatch
):
    fixture = capability_fixture
    percentiles = (0.2, 0.5, 0.8)
    monkeypatch.setattr(capability_module, "REPRESENTATIVE_PERCENTILES", percentiles)
    record = cast(
        dict[str, Any],
        capability_module.capability_scenarios(
            fixture["package"],
            fixture["draws"],
            fixture["capture"],
            fixture["report"],
            fixture["targets"],
            reweighted=False,
        ),
    )
    base = fixture["trajectory"]["base_law"]
    scenarios = cast(list[dict[str, Any]], record["scenarios"])
    assert record["percentiles"]["values"] == list(percentiles)
    assert [row["scenario"] for row in scenarios] == [
        "current_frontier",
        "p20",
        "p50",
        "p80",
    ]
    assert scenarios[1]["label"] == "20th percentile path"
    assert scenarios[3]["label"] == "80th percentile path"
    assert scenarios[1]["progress"] == pytest.approx(
        base_progress_quantile(
            base["pace_shape"],
            base["posterior"],
            base["progress_elapsed_years"],
            0.2,
            pace_cap=base["pace_cap"],
        ),
        rel=1e-12,
    )


def test_vincentized_step_laws_handle_atoms_endpoints_and_equal_laws():
    lower_losses, lower_masses = [0, 10], [0.4, 0.6]
    upper_losses, upper_masses = [0, 40], [0.2, 0.8]
    assert (
        capability_module.vincentized_quantile(
            lower_losses, lower_masses, upper_losses, upper_masses, 0.5, 0.1
        )
        == 0
    )
    expected = math.expm1(0.5 * math.log1p(40))
    assert capability_module.vincentized_quantile(
        lower_losses, lower_masses, upper_losses, upper_masses, 0.5, 0.3
    ) == pytest.approx(expected)
    assert (
        capability_module.vincentized_quantile(
            lower_losses, lower_masses, upper_losses, upper_masses, 0, 0.3
        )
        == 0
    )
    assert (
        capability_module.vincentized_quantile(
            lower_losses, lower_masses, upper_losses, upper_masses, 1, 0.3
        )
        == 40
    )
    assert capability_module.vincentized_exceedance_probability(
        lower_losses, lower_masses, upper_losses, upper_masses, 0, 0
    ) == pytest.approx(0.6)
    midpoint = math.expm1(0.5 * math.log1p(40))
    assert capability_module.vincentized_exceedance_probability(
        lower_losses, lower_masses, upper_losses, upper_masses, 0.5, 5
    ) == pytest.approx(0.8)
    assert capability_module.vincentized_exceedance_probability(
        lower_losses, lower_masses, upper_losses, upper_masses, 0.5, midpoint
    ) == pytest.approx(0.6)
    assert (
        capability_module.vincentized_exceedance_probability(
            lower_losses, lower_masses, upper_losses, upper_masses, 1, 40
        )
        == 0
    )
    assert (
        capability_module.vincentized_exceedance_probability(
            lower_losses, lower_masses, upper_losses, upper_masses, 0.5, -1
        )
        == 1
    )
    assert (
        capability_module.vincentized_quantile(
            lower_losses, lower_masses, lower_losses, lower_masses, 0.37, 0.7
        )
        == 10
    )
    assert (
        capability_module.vincentized_exceedance_probability(
            lower_losses, lower_masses, lower_losses, lower_masses, 0.37, 10
        )
        == 0
    )

    shared_atom_probability = capability_module.vincentized_exceedance_probability(
        [0, 5e10, 2e11],
        [0.5, 0.25, 0.25],
        [0, 5e10, 2e11],
        [0.2, 0.55, 0.25],
        0.5,
        5e10,
    )
    assert shared_atom_probability == 0.25

    lower_losses, upper_losses = [0, 1e5], [0, 2e5]
    quantile = capability_module.vincentized_quantile(
        lower_losses, [0.5, 0.5], upper_losses, [0.5, 0.5], 0.1, 0.9
    )
    assert (
        capability_module.vincentized_exceedance_probability(
            lower_losses, [0.5, 0.5], upper_losses, [0.5, 0.5], 0.1, quantile
        )
        == 0
    )


def test_vincentized_lower_quantile_matches_weighted_var_tvar():
    losses = np.arange(1, 11, dtype=float)
    masses = np.ones(10)
    expected, _ = weighted_var_tvar(losses, masses, 0.8)
    actual = capability_module.vincentized_quantile(
        losses, masses, losses, masses, 0.37, 0.8
    )
    assert expected == 8
    assert actual == expected


def test_step_law_cumulative_mass_stays_monotone_and_bounded():
    losses = np.arange(51, dtype=float) * 1e5
    masses = np.array([1.0] * 50 + [1e-15])
    lower = capability_module._StepLaw.from_samples(losses, masses)
    upper = capability_module._StepLaw.from_samples(losses + 1, masses)
    assert np.all(np.diff(lower.cumulative) >= 0)
    assert np.all((lower.cumulative >= 0) & (lower.cumulative <= 1))
    probability = capability_module.vincentized_exceedance_probability(
        losses, masses, losses + 1, masses, 0.5, 2.5e6
    )
    assert 0 <= probability <= 1
    assert np.all(np.diff(upper.cumulative) >= 0)


def test_curves_are_monotone_supported_and_match_their_quantiles(capability_fixture):
    fixture = capability_fixture
    record = fixture["record"]
    nodes = {row["node_id"]: row for row in record["nodes"]}
    for scenario in record["scenarios"]:
        for _view, data in scenario["by_view"].items():
            quantiles = data["quantiles"]
            values = [row["annual_loss"] for row in quantiles]
            present = [value for value in values if value is not None]
            assert present == sorted(present)
            exceedance = data["exceedance"]
            probabilities = [row["probability"] for row in exceedance]
            present_probabilities = [
                value for value in probabilities if value is not None
            ]
            assert present_probabilities == sorted(present_probabilities, reverse=True)
            for point in (*quantiles, *exceedance):
                assert point["status"] in {"available", "withheld", "unavailable"}
                if point["status"] != "available":
                    assert point["reason"]

        if scenario["status"] != "available" or "welfare" not in scenario["by_view"]:
            continue
        interpolation = scenario["interpolation"]
        lower = nodes[interpolation["lower_node"]["node_id"]]
        upper = nodes[interpolation["upper_node"]["node_id"]]
        low_losses, low_masses, _ = _node_samples(fixture, lower["node_id"], "welfare")
        high_losses, high_masses, _ = _node_samples(
            fixture, upper["node_id"], "welfare"
        )
        weight = interpolation["weight"]
        for level in record["levels"]:
            loss = capability_module.vincentized_quantile(
                low_losses,
                low_masses,
                high_losses,
                high_masses,
                weight,
                level,
            )
            at_loss = capability_module.vincentized_exceedance_probability(
                low_losses,
                low_masses,
                high_losses,
                high_masses,
                weight,
                loss,
            )
            epsilon = max(1.0, abs(loss) * 1e-9)
            below_loss = loss - epsilon
            below = capability_module.vincentized_exceedance_probability(
                low_losses,
                low_masses,
                high_losses,
                high_masses,
                weight,
                below_loss,
            )
            assert at_loss <= 1 - level
            assert below >= 1 - level


def test_node_sampling_error_matches_hand_calculation():
    def draw(native_eal: float, offset: float):
        losses = np.array([1, 2, 3, 4], dtype=float) + offset
        return SimpleNamespace(
            annual_losses={"welfare": losses},
            n_simulations=len(losses),
            normalized_weights=lambda: np.full(len(losses), 1 / len(losses)),
            views={
                "welfare": SimpleNamespace(
                    native_mean=SimpleNamespace(value=native_eal, precision={}),
                    mean_exists=True,
                    array=None,
                )
            },
        )

    states = [
        capability_module._NodeState("s0", "s0", 0.5, draw(100, 0)),
        capability_module._NodeState("s1", "s1", 0.5, draw(200, 1)),
    ]
    node = capability_module._Node(
        node_id="node-1",
        progress=1,
        progress_cdf=0.5,
        arm="reference",
        base_weight=1,
        eci_by_lab={"lab": {"period_start": 0.25, "period_end": 0.5}},
        states=states,
        by_view={},
        controls=capability_module.TailGateControls(batches=2),
    )
    capture = SimpleNamespace(arms={}, supports={})
    groups = SampleGroups({"s0": "node-1::model-a", "s1": "node-1::model-a"})

    view = capability_module._node_view(node, "welfare", capture, groups, batches=2)

    mean = math.fsum((100, 200)) / 2
    sample_variance = math.fsum((value - mean) ** 2 for value in (100, 200))
    expected = math.sqrt(sample_variance / 2)
    assert view.expected_annual_loss == pytest.approx(150)
    assert view.sampling_standard_error == pytest.approx(expected, rel=1e-12)


def test_interpolated_eal_sampling_error_matches_hand_formula(monkeypatch):
    def node(node_id: str, progress: float, eal: float, error: float):
        data = capability_module._NodeView(
            measure="welfare",
            source_status="total",
            status="available",
            reason=None,
            expected_annual_loss=eal,
            sampling_standard_error=error,
            sampling_reason=None,
            years_per_state=16,
            law=None,
            batch_laws=None,
            batch_samples=None,
            batch_reason=None,
            var_reason=None,
            structural_zero=False,
        )
        return capability_module._Node(
            node_id=node_id,
            progress=progress,
            progress_cdf=0.5,
            arm="reference",
            base_weight=0.5,
            eci_by_lab={"lab": {"period_start": progress, "period_end": 2 * progress}},
            states=[],
            by_view={"welfare": data},
            controls=capability_module.TailGateControls(batches=2),
        )

    low, high = node("low", 1, 100, 3), node("high", 4, 200, 4)
    monkeypatch.setattr(capability_module, "_scenario_curve", lambda *_args: ([], []))
    scenario = capability_module._scenario_for_progress(
        "p50",
        "Expected progress (median)",
        "description",
        0.5,
        2,
        0.5,
        (low, high),
        ("welfare",),
    )

    weight = scenario["interpolation"]["weight"]
    expected = math.sqrt(((1 - weight) * 3) ** 2 + (weight * 4) ** 2)
    data = scenario["by_view"]["welfare"]
    assert data["expected_annual_loss"] == pytest.approx(150)
    assert data["sampling_standard_error"] == pytest.approx(expected, rel=1e-12)


def test_replay_reproduces_exhibit_and_reweight_marks_it_unavailable(
    capability_fixture,
):
    fixture = capability_fixture
    replayed = replay_capture(
        setup_path=fixture["package"].root,
        capture_directory=fixture["run_directory"] / "capture",
        output=fixture["root"] / "replay.json",
        code_commit="fixture",
    )
    assert replayed["capability_scenarios"] == fixture["record"]

    key_to_state = {
        state_key(state.state_id): state.state_id
        for state in fixture["package"].joint_states
    }
    raw_weights = {
        arm: {key_to_state[key]: weight for key, weight in values.items()}
        for arm, values in fixture["targets"].items()
    }
    positive_by_node = defaultdict(list)
    for state in fixture["package"].joint_states:
        key = state_key(state.state_id)
        weight = fixture["targets"]["reference"][key]
        node_id = fixture["trajectory"]["state_draws"].get(state.state_id)
        if weight > 0 and node_id is not None:
            positive_by_node[node_id].append((state.state_id, weight))
    first, second = next(
        rows[:2] for rows in positive_by_node.values() if len(rows) >= 2
    )
    delta = min(first[1], second[1]) / 10
    raw_weights["reference"][first[0]] -= delta
    raw_weights["reference"][second[0]] += delta
    changed = replay_capture(
        setup_path=fixture["package"].root,
        capture_directory=fixture["run_directory"] / "capture",
        output=fixture["root"] / "reweighted.json",
        state_weights_by_arm=raw_weights,
        code_commit="fixture",
    )
    assert changed["capability_scenarios"]["status"] == "unavailable"
    assert (
        changed["capability_scenarios"]["reason"]
        == capability_module.REWEIGHTED_EXHIBIT_REASON
    )
    assert "fixed authored text" in changed["capability_scenarios"]["disclosures"][0]


def test_results_export_without_results_path_roundtrips_and_rejects_rehashed_tampering(
    capability_fixture,
):
    fixture = capability_fixture
    directory = fixture["export_directory"]
    index = fixture["export_index"]
    assert index.format == "cat.results/1"
    assert set(index.outputs) == {
        "capability_scenarios.json",
        "capability_scenarios.csv",
        "capability_nodes.csv",
    }
    for _name, reference in index.outputs.items():
        path = directory / reference.path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference.sha256
    with (directory / "capability_scenarios.csv").open() as handle:
        assert next(csv.reader(handle)) == list(SCENARIO_COLUMNS)
    with (directory / "capability_nodes.csv").open() as handle:
        assert next(csv.reader(handle)) == list(NODE_COLUMNS)
    assert load_results_export(directory) == index

    index_bytes = (directory / "export.json").read_bytes()
    scenario_json = (directory / "capability_scenarios.json").read_bytes()
    # The export writes the record as the artifact writer freezes it.
    assert scenario_json == frozen_json(json.loads(scenario_json))
    scenario_csv = (directory / "capability_scenarios.csv").read_bytes()
    try:
        changed_csv = scenario_csv + b"#\n"
        (directory / "capability_scenarios.csv").write_bytes(changed_csv)
        _rehash_export_output(directory, "capability_scenarios.csv", changed_csv)
        with pytest.raises(
            ValueError,
            match=r"capability_scenarios\.csv differs from its native results",
        ):
            load_results_export(directory)

        (directory / "capability_scenarios.csv").write_bytes(scenario_csv)
        (directory / "export.json").write_bytes(index_bytes)
        changed_record = json.loads(scenario_json)
        changed_record["interpretation"] += " Tampered."
        changed_json = (
            json.dumps(changed_record, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        (directory / "capability_scenarios.json").write_bytes(changed_json)
        _rehash_export_output(directory, "capability_scenarios.json", changed_json)
        with pytest.raises(
            ValueError,
            match=r"capability_scenarios\.json differs from its native results",
        ):
            load_results_export(directory)
    finally:
        (directory / "capability_scenarios.csv").write_bytes(scenario_csv)
        (directory / "capability_scenarios.json").write_bytes(scenario_json)
        (directory / "export.json").write_bytes(index_bytes)


def test_the_scenario_table_states_why_a_standard_error_is_missing(capability_fixture):
    """Each row of capability_scenarios.csv carries its view's standard error
    or the reason it has none, as the results and the node table do. In this
    study no state reads a donor's answers, and no reason says one does."""
    fixture = capability_fixture
    stated = {
        (scenario["scenario"], view): data
        for scenario in fixture["record"]["scenarios"]
        for view, data in scenario["by_view"].items()
    }
    with (fixture["export_directory"] / "capability_scenarios.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["scenario"], row["view"]) for row in rows} == set(stated)
    for row in rows:
        data = stated[row["scenario"], row["view"]]
        assert row["sampling_reason"] == (data["sampling_reason"] or "")
        assert (row["sampling_standard_error"] == "") == (
            data["sampling_standard_error"] is None
        )
        assert (row["sampling_standard_error"] == "") != (row["sampling_reason"] == "")
        assert "POOL-UNANSWERED-SCENARIOS" not in row["sampling_reason"]


def test_interpolated_eal_and_error_follow_the_log_progress_weight(capability_fixture):
    fixture = capability_fixture
    nodes = {row["node_id"]: row for row in fixture["record"]["nodes"]}
    checked = 0
    for scenario in fixture["record"]["scenarios"][1:]:
        interpolation = scenario["interpolation"]
        lower = nodes[interpolation["lower_node"]["node_id"]]
        upper = nodes[interpolation["upper_node"]["node_id"]]
        if lower["node_id"] == upper["node_id"]:
            continue
        weight = math.log(scenario["progress"] / lower["progress"]) / math.log(
            upper["progress"] / lower["progress"]
        )
        assert interpolation["weight"] == pytest.approx(weight, rel=1e-12)
        assert not math.isclose(weight, 0.5)
        for view, row in scenario["by_view"].items():
            low, high = lower["by_view"][view], upper["by_view"][view]
            if row["expected_annual_loss"] is None:
                continue
            assert low["expected_annual_loss"] != high["expected_annual_loss"]
            assert row["expected_annual_loss"] == pytest.approx(
                (1 - weight) * low["expected_annual_loss"]
                + weight * high["expected_annual_loss"],
                rel=1e-12,
            )
            checked += 1
    assert checked > 0


def test_a_narrow_curve_point_with_sparse_batches_is_withheld():
    controls = capability_module.TailGateControls()
    fields = {"non_exceedance_probability": 0.99}
    batches = [1.0] * controls.batches
    sparse = capability_module._curve_point(
        fields,
        "annual_loss",
        1.0,
        functional="var",
        controls=controls,
        batch_values=batches,
        minimum_batch_exceedances=MIN_TAIL_EFFECTIVE_UNITS - 1,
        years_per_state=64,
    )
    assert sparse["status"] == "withheld"
    assert sparse["relative_half_width"] == 0
    assert "sparsest" in str(sparse["reason"])
    supported = capability_module._curve_point(
        fields,
        "annual_loss",
        1.0,
        functional="var",
        controls=controls,
        batch_values=batches,
        minimum_batch_exceedances=MIN_TAIL_EFFECTIVE_UNITS,
        years_per_state=64,
    )
    assert supported["status"] == "available"
    assert supported["reason"] is None
