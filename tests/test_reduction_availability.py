"""Public reduction keeps unavailable functionals and comparisons in their scope."""

import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.incident import ComparisonRequirement
from catmodel.results.execution import replay_capture, run_setup
from test_support.results import (
    available_metric,
    present,
    reduced_arm,
    statistic_ranking,
    view_diagnostics,
)
from test_support.run_setup import setup_from_payload
from test_support.study import _package, _plan, _study


def test_numeric_underflow_stays_unavailable_within_its_epistemic_group(tmp_path):
    payload = encode_scientific(_study(rate=1e-316))
    low, high = payload["joint_states"]
    low["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
        "amount"
    ] = 1e-9
    high["laws"][0]["reference_annual_rate"] = 1
    payload["alternatives"] = []
    package = _package(
        tmp_path,
        study=setup_from_payload(payload),
        plan=_plan(4).model_copy(
            update={"required_arms": ("reference",), "requested_views": ("gross",)}
        ),
    )
    result = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="producer"
    )
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "replay.json",
        code_commit="reducer",
    )
    assert result["arms"] == replay["arms"]
    arm = reduced_arm(result["arms"]["reference"])
    metric = available_metric(arm["metrics"]["gross"])
    statistics = metric["statistics"]
    # A finite mean below the output range is withheld, not unavailable.
    assert statistics["expected_annual_loss"]["status"] == "withheld"
    reason = present(statistics["expected_annual_loss"]["reason"])
    assert "numeric output range" in reason
    assert metric["expected_annual_loss"] is None
    assert arm["conditional_eal_by_epistemic_group"]["gross"] == {
        "judgment-a": None,
        "judgment-b": 200,
    }
    assert statistics["annual_loss_variance"]["status"] == "withheld"
    variance_reason = present(statistics["annual_loss_variance"]["reason"])
    assert reason in variance_reason
    for key in ("annual_loss_variance", "normalized_empirical_variance"):
        variance_reason = statistics[key]["reason"]
        assert not variance_reason or "second moment" not in variance_reason
    assert "second moment" not in present(
        metric["eal_precision"]["precision_suppression_reason"]
    )
    diagnostics = view_diagnostics(result, "reference", "gross")
    for tail in diagnostics["tail_diagnostics"]["tails"].values():
        for key in (
            "var_interval_unavailable_reason",
            "tvar_interval_unavailable_reason",
        ):
            reason = tail[key]
            assert not reason or "second moment" not in reason
    changed = reduced_arm(
        replay_capture(
            setup_path=package.root,
            capture_directory=tmp_path / "run/capture",
            output=tmp_path / "reweighted.json",
            state_weights={"low": 0, "high": 1},
            code_commit="reducer",
        )["arms"]["reference"]
    )
    assert available_metric(changed["metrics"]["gross"])["expected_annual_loss"] == 200
    assert changed["conditional_eal_by_epistemic_group"]["gross"] == {"judgment-b": 200}


@pytest.mark.parametrize("unavailable_requirement", [False, True])
def test_optional_finance_does_not_control_gross_ranking(
    tmp_path, unavailable_requirement
):
    requirements = (
        (
            ComparisonRequirement(
                family_id="unidentified_comparison",
                scenario_ids=("loss",),
                status="unavailable",
                held_fixed=("Original capture and all other assumptions.",),
                reason="The source does not identify the required alternative.",
            ),
        )
        if unavailable_requirement
        else ()
    )
    package = _package(
        tmp_path,
        plan=_plan(4).model_copy(
            update={
                "requested_views": ("gross", "financial"),
                "comparison_requirements": requirements,
            }
        ),
    )
    result = run_setup(
        setup_path=package.root, output=tmp_path / "run", code_commit="producer"
    )
    replay = replay_capture(
        setup_path=package.root,
        capture_directory=tmp_path / "run/capture",
        output=tmp_path / "replay.json",
        code_commit="reducer",
    )
    assert result["rankings"] == replay["rankings"]
    assert result["status"] == "partial"
    assert not result["rankings_available"]
    assert reduced_arm(result["arms"]["reference"])["metrics"]["financial"][
        "status"
    ] == ("unavailable")
    assert result["rankings"]["financial"]["status"] == "unavailable"

    ranking = statistic_ranking(result["rankings"], "gross", "expected_annual_loss")
    if unavailable_requirement:
        assert ranking["status"] == "unavailable"
        assert "comparison coverage" in ranking["reason"]
    else:
        assert ranking["status"] == "available"
        assert [row["arm"] for row in ranking["arms"]] == ["contained"]


def test_group_unavailability_keeps_distinct_positive_state_reasons():
    from tests.test_reduction import basis, draw, reduce, support_metadata

    underflow = "The positive target mean is below numeric output range."
    infinite = "The target law has no finite mean."
    draws = {
        "tiny": draw("tiny", values=(0, 0), variance_exists=False),
        "heavy": draw(
            "heavy",
            values=(1, 2),
            mean_exists=False,
            variance_exists=False,
        ),
        "finite": draw("finite", values=(20, 20)),
    }
    records = {
        state: {
            "view_bases": {"pooled": basis()},
            "unavailable_functionals": {"gross": {"expected_annual_loss": reason}},
        }
        for state, reason in (("tiny", underflow), ("heavy", infinite))
    }
    records["finite"] = {"view_bases": {"pooled": basis()}}
    result = reduce(
        {"base": draws},
        state_weights={"tiny": 0.25, "heavy": 0.25, "finite": 0.5},
        epistemic_groups={"tiny": "tiny", "heavy": "heavy", "finite": "finite"},
        capture_provenance=support_metadata({"base": records}),
    )["arms"]["base"]
    assert result["conditional_eal_by_epistemic_group"]["gross"] == {
        "tiny": None,
        "heavy": None,
        "finite": 20,
    }
    assert result["conditional_eal_unavailable_reasons_by_epistemic_group"] == {
        "gross": {"tiny": underflow, "heavy": infinite}
    }
    statistics = result["metrics"]["gross"]["statistics"]
    # One state's mean is infinite, so the pooled mean is unavailable.
    assert statistics["expected_annual_loss"]["status"] == "unavailable"
    reason = statistics["expected_annual_loss"]["reason"]
    assert underflow in reason and infinite in reason
    assert statistics["annual_loss_variance"]["status"] == "unavailable"
    precision_reason = statistics["annual_loss_variance"]["reason"]
    assert underflow in precision_reason and "second moment" in precision_reason
