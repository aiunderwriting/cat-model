"""The capability-by-safeguard grid over a public study with both safeguard variants."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest

import catmodel.results.capability_scenarios as capability_module
from catmodel.results.execution import (
    arm_weights,
    load_run_capture,
    replay_capture,
    run_setup,
)
from catmodel.results.results_export import export_results, load_results_export
from catmodel.results.sensitivity_results import REWEIGHTED_EXHIBIT_REASON
from catmodel.setup.public_frontier_comparisons import SAFEGUARD_CONTROL
from catmodel.simulation.incident_runtime import state_key
from catmodel.simulation.tail_gate import MIN_TAIL_EFFECTIVE_UNITS
from tests.test_safeguard_level_variants import (
    POSTURES,
    _headline,
    _importer,
    _variant_study,
)


@pytest.fixture(scope="module")
def grid_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("safeguard-grid")
    patcher = pytest.MonkeyPatch()
    try:
        headline, bundle = _headline(root, patcher)
        weaker = _variant_study(root, bundle, "weaker")
        stronger = _variant_study(root, bundle, "stronger")
    finally:
        patcher.undo()
    package = _importer(root, headline)("package", (weaker, stronger))
    run = root / "run"
    report = run_setup(setup_path=package.root, output=run, code_commit="fixture")
    draws, capture = load_run_capture(run / "capture")
    return {
        "root": root,
        "package": package,
        "run": run,
        "report": report,
        "draws": draws,
        "capture": capture,
        "targets": arm_weights(package, None),
        "record": report["capability_scenarios"],
    }


def _reduce(fixture, *, reweighted=False) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        capability_module.capability_scenarios(
            fixture["package"],
            fixture["draws"],
            fixture["capture"],
            fixture["report"],
            fixture["targets"],
            reweighted=reweighted,
        ),
    )


def _cells(record) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (cell["safeguard_level"], cell["scenario"]): cell
        for cell in record["safeguard_grid"]["cells"]
    }


def _nodes(record, level: str) -> dict[str, dict[str, Any]]:
    (row,) = (
        row
        for row in record["safeguard_grid"]["levels"]
        if row["safeguard_level"] == level
    )
    return {node["node_id"]: node for node in row["nodes"]}


def test_grid_covers_every_level_and_scenario_from_its_own_nodes(grid_fixture):
    record = grid_fixture["record"]
    grid = record["safeguard_grid"]
    assert grid["status"] == "available"
    assert [row["safeguard_arm"] for row in grid["levels"]] == [
        "safeguard_level::weaker",
        SAFEGUARD_CONTROL,
        "safeguard_level::stronger",
    ]
    scenarios = [row["scenario"] for row in record["scenarios"]]
    assert list(_cells(record)) == [
        (level, scenario)
        for level in ("weaker", "current", "stronger")
        for scenario in scenarios
    ]
    # Variants are collected at the embedded rule's interior nodes and node-00;
    # every level reduces exactly those interior nodes from its own arm's
    # states, and each variant also lists its own node-00.
    trajectory = next(
        row.acquisition["trajectory_quadrature"]
        for row in grid_fixture["package"].evidence.records
        if row.acquisition and "trajectory_quadrature" in row.acquisition
    )
    embedded = {
        node_id
        for node_id, weight in trajectory["draw_weights"]["reference"][
            "embedded"
        ].items()
        if weight > 0
    }
    assert 0 < len(embedded) < len(record["nodes"]) - 1
    for level in ("weaker", "current", "stronger"):
        nodes = _nodes(record, level)
        assert set(nodes) == (
            embedded if level == "current" else embedded | {"node-00"}
        )
        arm = "safeguard_level::" + (
            "embedded_control" if level == "current" else level
        )
        assert {node["arm"] for node in nodes.values()} == {arm}
    # The synthetic posture scales every variant node's EAL by its factor, node-00
    # against the reference's no-progress node.
    current = {
        **_nodes(record, "current"),
        "node-00": next(
            node for node in record["nodes"] if node["node_id"] == "node-00"
        ),
    }
    for level, (_, factor) in POSTURES.items():
        for node_id, node in _nodes(record, level).items():
            assert node["by_view"]["welfare"]["expected_annual_loss"] == pytest.approx(
                factor * current[node_id]["by_view"]["welfare"]["expected_annual_loss"],
                rel=1e-9,
            )


def test_interior_percentile_interpolates_between_variant_nodes(grid_fixture):
    record = grid_fixture["record"]
    for level in ("weaker", "current", "stronger"):
        cell = _cells(record)[(level, "p10")]
        nodes = _nodes(record, level)
        interpolation = cell["interpolation"]
        lower = nodes[interpolation["lower_node"]["node_id"]]
        upper = nodes[interpolation["upper_node"]["node_id"]]
        assert lower["progress"] < cell["progress"] < upper["progress"]
        assert interpolation["exact_node"] is False
        assert cell["capability_clamped"] is False
        assert cell["clamped_node"] is None and cell["clamp_log_progress_gap"] is None
        weight = (math.log(cell["progress"]) - math.log(lower["progress"])) / (
            math.log(upper["progress"]) - math.log(lower["progress"])
        )
        assert interpolation["weight"] == pytest.approx(weight, rel=1e-12)
        for view, data in cell["by_view"].items():
            low = lower["by_view"][view]["expected_annual_loss"]
            high = upper["by_view"][view]["expected_annual_loss"]
            assert data["expected_annual_loss"] == pytest.approx(
                (1 - weight) * low + weight * high, rel=1e-12
            )
        assert cell["eci_period_end"] is not None
    # The reference's p10 brackets on the headline's denser nodes instead.
    reference = next(row for row in record["scenarios"] if row["scenario"] == "p10")
    assert (
        reference["interpolation"]["upper_node"]["node_id"]
        != _cells(record)[("current", "p10")]["interpolation"]["upper_node"]["node_id"]
    )


def test_percentile_on_a_node_uses_the_node_directly(grid_fixture):
    record = grid_fixture["record"]
    for level in ("weaker", "current", "stronger"):
        cell = _cells(record)[(level, "p50")]
        interpolation = cell["interpolation"]
        assert interpolation["exact_node"] is True
        assert interpolation["weight"] == 0
        node = _nodes(record, level)[interpolation["lower_node"]["node_id"]]
        assert interpolation["upper_node"]["node_id"] == node["node_id"]
        assert cell["capability_clamped"] is False
        for view, data in cell["by_view"].items():
            assert (
                data["expected_annual_loss"]
                == (node["by_view"][view]["expected_annual_loss"])
            )


def test_out_of_range_percentile_clamps_to_the_nearest_node_and_is_flagged(
    grid_fixture, monkeypatch
):
    monkeypatch.setattr(
        capability_module, "REPRESENTATIVE_PERCENTILES", (0.01, 0.5, 0.99)
    )
    record = _reduce(grid_fixture)
    cells = _cells(record)
    for level in ("weaker", "current", "stronger"):
        nodes = sorted(
            (row for row in _nodes(record, level).values() if row["progress"] > 0),
            key=lambda row: row["progress"],
        )
        for scenario, nearest in (("p1", nodes[0]), ("p99", nodes[-1])):
            cell = cells[(level, scenario)]
            assert cell["status"] == "available"
            assert cell["capability_clamped"] is True
            assert cell["clamped_node"]["node_id"] == nearest["node_id"]
            gap = math.log(cell["progress"]) - math.log(nearest["progress"])
            assert cell["clamp_log_progress_gap"] == pytest.approx(gap, rel=1e-12)
            assert (gap < 0) == (scenario == "p1")
            assert cell["interpolation"]["exact_node"] is False
            assert cell["interpolation"]["lower_node"]["node_id"] == nearest["node_id"]
            assert cell["interpolation"]["upper_node"]["node_id"] == nearest["node_id"]
            # No extrapolation: the cell is the node's own law and EAL.
            for view, data in cell["by_view"].items():
                assert (
                    data["expected_annual_loss"]
                    == (nearest["by_view"][view]["expected_annual_loss"])
                )
            assert cell["eci_period_end"] == nearest["eci_period_end"]
    # The reference exhibit keeps its own rule: no clamping, unavailable instead.
    reference = {row["scenario"]: row for row in record["scenarios"]}
    assert "capability_clamped" not in reference["p99"]


def test_current_frontier_is_each_variants_own_no_progress_node_and_the_reference_for_current(
    grid_fixture,
):
    record = grid_fixture["record"]
    cells = _cells(record)
    reference_current = record["scenarios"][0]
    current = cells[("current", "current_frontier")]
    assert current["status"] == "available"
    assert {
        key: value
        for key, value in current.items()
        if key
        not in {
            "safeguard_level",
            "safeguard_arm",
            "capability_clamped",
            "clamped_node",
            "clamp_log_progress_gap",
        }
    } == reference_current
    assert current["capability_clamped"] is False
    # Each variant collected node-00, simulated at weight zero by its own arm, so
    # its current-frontier cell is that node at the variant's posture.
    for level, (_, factor) in POSTURES.items():
        cell = cells[(level, "current_frontier")]
        assert cell["status"] == "available"
        assert cell["arm"] == f"safeguard_level::{level}"
        assert cell["interpolation"]["exact_node"] is True
        assert cell["interpolation"]["lower_node"]["node_id"] == "node-00"
        assert cell["capability_clamped"] is False
        assert cell["by_view"]["welfare"]["expected_annual_loss"] == pytest.approx(
            factor * reference_current["by_view"]["welfare"]["expected_annual_loss"],
            rel=1e-9,
        )


def test_a_variant_without_no_progress_states_has_an_unavailable_current_frontier(
    grid_fixture,
):
    """A variant study that did not collect node-00 leaves its current-frontier
    cell unavailable rather than estimating it from other nodes."""
    package = grid_fixture["package"]
    record = cast(
        dict[str, Any],
        capability_module.capability_scenarios(
            SimpleNamespace(
                joint_states=tuple(
                    state
                    for state in package.joint_states
                    if all(
                        capability_module._variant_node(state.state_id, level)
                        != "node-00"
                        for level in ("weaker", "stronger")
                    )
                ),
                execution=package.execution,
                evidence=package.evidence,
            ),
            grid_fixture["draws"],
            grid_fixture["capture"],
            grid_fixture["report"],
            grid_fixture["targets"],
            reweighted=False,
        ),
    )
    cells = _cells(record)
    for level in ("weaker", "stronger"):
        cell = cells[(level, "current_frontier")]
        assert cell["status"] == "unavailable"
        assert cell["reason"] == capability_module.VARIANT_NO_PROGRESS_REASON
        assert cell["arm"] == f"safeguard_level::{level}"
        for data in cell["by_view"].values():
            assert data["expected_annual_loss"] is None
            assert data["status"] == "unavailable"
            assert {point["status"] for point in data["quantiles"]} == {"unavailable"}
            assert {point["status"] for point in data["exceedance"]} == {"unavailable"}
        assert "node-00" not in _nodes(record, level)
    skip = {("weaker", "current_frontier"), ("stronger", "current_frontier")}
    assert {key: value for key, value in cells.items() if key not in skip} == {
        key: value
        for key, value in _cells(grid_fixture["record"]).items()
        if key not in skip
    }


def test_each_cell_has_its_own_sampling_error_and_tail_gate(grid_fixture):
    record = grid_fixture["record"]
    controls = grid_fixture["capture"].execution_plan["numerical_controls"]["tail_gate"]
    for (level, scenario), cell in _cells(record).items():
        if cell["status"] != "available":
            continue
        nodes = _nodes(record, level)
        interpolation = cell["interpolation"]
        if scenario == "current_frontier":
            lower = upper = (
                nodes["node-00"]
                if level != "current"
                else next(
                    node for node in record["nodes"] if node["node_id"] == "node-00"
                )
            )
        else:
            lower = nodes[interpolation["lower_node"]["node_id"]]
            upper = nodes[interpolation["upper_node"]["node_id"]]
        weight = interpolation["weight"]
        for view, data in cell["by_view"].items():
            low = lower["by_view"][view]["sampling_standard_error"]
            high = upper["by_view"][view]["sampling_standard_error"]
            if low is None or high is None:
                assert data["sampling_standard_error"] is None
                assert data["sampling_reason"]
            else:
                assert data["sampling_standard_error"] == pytest.approx(
                    math.hypot((1 - weight) * low, weight * high), rel=1e-12
                )
            for point in (*data["quantiles"], *data["exceedance"]):
                if point["status"] == "available":
                    assert point["reason"] is None
                    assert point["relative_half_width"] <= controls["tolerance"]
                elif point["status"] == "withheld":
                    assert point["reason"]
                    assert (
                        point["relative_half_width"] is None
                        or point["relative_half_width"] > controls["tolerance"]
                        or point["minimum_batch_exceedances"] < MIN_TAIL_EFFECTIVE_UNITS
                        or point.get("probability") == 0
                    )
    # Each variant has one respondent, so its own sampling groups hold one sample
    # and give no standard error; the control's two respondents give one.
    weaker = _cells(record)[("weaker", "p50")]["by_view"]["welfare"]
    assert weaker["sampling_standard_error"] is None
    assert "safeguard-weaker-" in weaker["sampling_reason"]
    current = _cells(record)[("current", "p50")]["by_view"]["welfare"]
    assert current["sampling_standard_error"] is not None


def test_replay_and_export_reproduce_the_grid(grid_fixture):
    fixture = grid_fixture
    replayed = replay_capture(
        setup_path=fixture["package"].root,
        capture_directory=fixture["run"] / "capture",
        output=fixture["root"] / "replay.json",
        code_commit="fixture",
    )
    assert replayed["capability_scenarios"] == fixture["record"]
    export = fixture["root"] / "export"
    export_results(
        setup_path=fixture["package"].root,
        capture_directory=fixture["run"] / "capture",
        output=export,
        code_commit="fixture",
    )
    load_results_export(export)
    exported = json.loads((export / "capability_scenarios.json").read_text())
    assert exported["safeguard_grid"] == json.loads(
        json.dumps(fixture["record"]["safeguard_grid"])
    )

    unavailable = _reduce(fixture, reweighted=True)
    assert unavailable["safeguard_grid"]["status"] == "unavailable"
    assert unavailable["safeguard_grid"]["reason"] == REWEIGHTED_EXHIBIT_REASON
    assert unavailable["safeguard_grid"]["cells"] == []


VARIANT_ARM = "safeguard_level::weaker"
NO_PROGRESS = "public_forecast::no_progress"


def _variant_zero_states(package, level: str = "weaker"):
    return [
        state
        for state in package.joint_states
        if capability_module._variant_node(state.state_id, level) == "node-00"
    ]


def _with_variant_zero_states(
    fixture, *, simulated: int | None = None, weighted=False, copies=0
):
    """The grid with the weaker variant's own node-00 states reweighted or their
    draws withheld: zero weight in every arm unless ``weighted`` (then the variant
    arm weights them by the pooled rule), and the variant's capture holding the
    draws of only the first ``simulated`` of them (all when None). ``copies``
    adds that many further variant respondents at node-00, copies of the first."""
    package = fixture["package"]
    variant_states = _variant_zero_states(package)
    assert variant_states
    targets = {arm: dict(weights) for arm, weights in fixture["targets"].items()}
    draws = {arm: dict(rows) for arm, rows in fixture["draws"].items()}
    first = variant_states[0]
    added = [
        first.model_copy(
            update={
                "state_id": f"{first.state_id}-copy{index}",
                "sampling_group_id": f"{first.sampling_group_id}-copy{index}",
            }
        )
        for index in range(copies)
    ]
    for state in added:
        key = state_key(state.state_id)
        for weights in targets.values():
            weights[key] = 0.0
        draws[VARIANT_ARM][key] = draws[VARIANT_ARM][state_key(first.state_id)]
    variant_states = [*variant_states, *added]
    pooled = capability_module.pooled_node_weights(variant_states)
    for index, state in enumerate(variant_states):
        key = state_key(state.state_id)
        assert all(weights.get(key, 0.0) == 0 for weights in targets.values())
        assert key in draws[VARIANT_ARM]
        if weighted:
            targets[VARIANT_ARM][key] = pooled[key]
        if simulated is not None and index >= simulated:
            for rows in draws.values():
                rows.pop(key, None)
    record = cast(
        dict[str, Any],
        capability_module.capability_scenarios(
            SimpleNamespace(
                joint_states=(*package.joint_states, *added),
                execution=package.execution,
                evidence=package.evidence,
            ),
            draws,
            fixture["capture"],
            fixture["report"],
            targets,
            reweighted=False,
        ),
    )
    return record, targets


def _others_unchanged(record, fixture) -> bool:
    """Every cell but the weaker current frontier matches the fixture's grid."""
    skip = ("weaker", "current_frontier")
    return {key: value for key, value in _cells(record).items() if key != skip} == {
        key: value for key, value in _cells(fixture["record"]).items() if key != skip
    }


@pytest.mark.parametrize("weighted", [False, True])
def test_a_variant_with_no_progress_states_reduces_its_own_current_frontier(
    grid_fixture, weighted
):
    """A variant's current-frontier cell comes from its own node-00 states: by the
    pooled rule when no arm weights them, else by the arm that does."""
    package = grid_fixture["package"]
    # The within-node rule reproduces the no-progress arm's node-00 weights.
    headline_zero = [
        state for state in package.joint_states if state.state_id.startswith("node-00:")
    ]
    pooled = capability_module.pooled_node_weights(headline_zero)
    targets = grid_fixture["targets"]
    total = sum(targets[NO_PROGRESS][key] for key in pooled)
    assert pooled == pytest.approx(
        {key: targets[NO_PROGRESS][key] / total for key in pooled}, rel=1e-12
    )
    record, _ = _with_variant_zero_states(grid_fixture, weighted=weighted)
    cell = _cells(record)[("weaker", "current_frontier")]
    assert cell["status"] == "available"
    assert cell["arm"] == VARIANT_ARM
    assert cell["interpolation"]["lower_node"]["node_id"] == "node-00"
    assert cell == _cells(grid_fixture["record"])[("weaker", "current_frontier")]
    assert "node-00" in _nodes(record, "weaker")
    assert _others_unchanged(record, grid_fixture)


@pytest.mark.parametrize(
    ("simulated", "copies", "reason"),
    [
        (0, 0, capability_module.VARIANT_NOT_SIMULATED_REASON),
        (1, 1, capability_module.VARIANT_PARTLY_SIMULATED_REASON),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_a_variant_whose_no_progress_states_lack_draws_is_unavailable(
    grid_fixture, simulated, copies, reason, weighted
):
    """Missing node-00 draws leave the cell unavailable rather than reweighted
    over the simulated states, which would change its respondent pool."""
    record, _ = _with_variant_zero_states(
        grid_fixture, simulated=simulated, weighted=weighted, copies=copies
    )
    cell = _cells(record)[("weaker", "current_frontier")]
    assert cell["status"] == "unavailable"
    assert cell["reason"] == reason
    assert cell["arm"] == VARIANT_ARM
    for data in cell["by_view"].values():
        assert data["expected_annual_loss"] is None
        assert {point["status"] for point in data["quantiles"]} == {"unavailable"}
    assert "node-00" not in _nodes(record, "weaker")
    assert _others_unchanged(record, grid_fixture)


def test_pooled_node_weights_share_equally_by_model_after_exclusions():
    """Two models at a node, one with a sample excluded: each model keeps half."""
    states = [
        SimpleNamespace(
            state_id=f"node-00:{sample}", sampling_group_id=f"node-00:{model}"
        )
        for sample, model in (("a0", "a"), ("a1", "a"), ("a2", "a"), ("b0", "b"))
    ]
    weights = capability_module.pooled_node_weights(states)
    assert weights == pytest.approx(
        {
            state_key("node-00:a0"): 1 / 6,
            state_key("node-00:a1"): 1 / 6,
            state_key("node-00:a2"): 1 / 6,
            state_key("node-00:b0"): 1 / 2,
        }
    )
