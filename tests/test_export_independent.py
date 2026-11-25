"""Independent export oracles for numerical order and conditional completeness."""

import hashlib

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.incident import ExecutionPlan
from catmodel.results.execution import replay_capture, run_setup
from catmodel.results.results_export import _curve, export_results, load_results_export
from catmodel.setup.store import build_setup
from catmodel.simulation.capability_mixture import (
    ConditionalCapabilityDraws,
    ConditionalView,
)
from test_support.conditional import conditional_study
from test_support.run_setup import setup_from_payload, with_execution


def _run(tmp_path, study):
    package = build_setup(
        destination=tmp_path / "package",
        setup=with_execution(
            study,
            ExecutionPlan(
                plan_id="independent-export",
                required_arms=("reference",),
                simulations=48,
                seed=173,
                requested_views=("gross", "welfare", "lab_loss::lab"),
            ),
            setup_id="independent-export",
        ),
        code_commit="fixture",
    )
    run_setup(
        setup_path=package.root,
        output=tmp_path / "run",
        code_commit="fixture",
    )
    return package


def test_reversed_weight_object_order_preserves_full_report_and_export(tmp_path):
    package = _run(tmp_path, conditional_study(two_templates=True))
    weights = {state.state_id: 1 / 6 for state in package.joint_states}
    reversed_weights = dict(reversed(list(weights.items())))
    reports = []
    for name, vector in (("forward", weights), ("reverse", reversed_weights)):
        reports.append(
            replay_capture(
                setup_path=package.root,
                capture_directory=tmp_path / "run/capture",
                state_weights=vector,
                output=tmp_path / f"{name}.json",
                code_commit="fixture",
            )
        )
    assert reports[0] == reports[1]
    source = tmp_path / "reverse.json"
    original = source.read_bytes()
    index = export_results(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        results_path=source,
        output=tmp_path / "export",
    )
    assert load_results_export(tmp_path / "export") == index
    copied = tmp_path / "export" / index.source["results"].path
    assert copied.read_bytes() == original
    assert index.source["results"].sha256 == hashlib.sha256(original).hexdigest()


def test_missing_zero_prior_interior_peer_withholds_whole_supported_context(tmp_path):
    payload = encode_scientific(conditional_study())
    interior = payload["joint_states"][1]
    assert interior["weight"] == 0
    interior["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]["components"][
        0
    ].pop("lab_loss_weights")
    package = _run(tmp_path, setup_from_payload(payload))
    index = export_results(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        results_path=tmp_path / "run/results.json",
        output=tmp_path / "export",
    )
    assert load_results_export(tmp_path / "export") == index
    assert (
        next(row for row in index.series if row.measure == "lab_economic_loss").status
        == "available"
    )
    assert index.conditioning["status"] == "unavailable"
    context = index.conditioning["contexts"][0]
    assert context["status"] == "unavailable"
    assert context["failures"] == [
        {
            "eci": 1,
            "arm": "reference",
            "reason": "conditional distribution unavailable: societal or lab loss is missing",
        }
    ]


def test_tiny_positive_tail_retains_support_and_strict_exceedance():
    def draw(key, values):
        return ConditionalCapabilityDraws(
            stratum_sha256=key,
            response_sha256="response",
            seed=1,
            views={"welfare": ConditionalView(np.array(values, dtype=float))},
        )

    rows = {"ordinary": draw("ordinary", [0]), "rare": draw("rare", [10])}
    curve = _curve(rows, {"ordinary": 1.0, "rare": 1e-200}, "welfare", (5, 10))
    assert curve[0].probability == pytest.approx(1e-200, abs=0)
    assert curve[0].useful_exceedances == 1
    assert curve[0].tail_weight_ess == 1
    assert curve[1].probability == 0
    assert curve[1].useful_exceedances == 0
    assert curve[1].tail_weight_ess == 0


def test_export_without_selected_report_roundtrips_after_json_key_sorting(tmp_path):
    package = _run(tmp_path, conditional_study(two_templates=True))
    index = export_results(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "export",
        code_commit="fixture",
    )
    assert load_results_export(tmp_path / "export") == index
