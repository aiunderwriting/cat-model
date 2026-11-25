"""Extending a simulation by later years gives the draws of one simulation.

A year's draws depend only on the plan seed, the state and the year, so the
tail gate's doubling rounds extend each state's years instead of redrawing
them.
"""

import json
from copy import deepcopy

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.incident import ExecutionPlan
from catmodel.elicitation.epistemic import epistemic_mean
from catmodel.results.execution import load_run_capture, replay_capture, run_setup
from catmodel.setup.store import build_setup
from catmodel.simulation import incident_runtime as runtime
from catmodel.simulation.incident import compile_incident_state
from test_support.incident import common_case
from test_support.run_setup import setup_from_payload, with_execution
from test_support.study import (
    _count_control_study,
    _study,
    partial_financial_payload,
    study_payload,
)
from tests.test_incident_linear_policy import LINEAR_VIEWS, POLICY_VIEWS, _rich_payload
from tests.test_small_events import (
    VIEWS,
    _kernel,
    _lognormal_count,
    _two_origin_state,
)


def _assert_same(actual, expected):
    """Equal draws, withheld views and allocations, array bits included."""
    draw, reasons, allocation = actual
    reference, expected_reasons, expected_allocation = expected
    assert reasons == expected_reasons
    assert allocation == expected_allocation
    assert draw.views.keys() == reference.views.keys()
    for view, row in draw.views.items():
        other = reference.views[view]
        assert row.array.tobytes() == other.array.tobytes(), view
        assert (row.mean_exists, row.variance_exists, row.native_mean) == (
            other.mean_exists,
            other.variance_exists,
            other.native_mean,
        ), view
    assert (
        draw.stratum_sha256,
        draw.response_sha256,
        draw.seed,
        draw.joint_law_semantics,
        draw.evaluator_provenance_by_risk_unit,
    ) == (
        reference.stratum_sha256,
        reference.response_sha256,
        reference.seed,
        reference.joint_law_semantics,
        reference.evaluator_provenance_by_risk_unit,
    )


def _extended(plan, compiled, years, **options):
    """``years[-1]`` years, simulated as successive extensions."""
    result, cache = None, {}
    for count in years:
        result = runtime.simulate_incident_state(
            plan,
            compiled,
            response_sha256="a" * 64,
            years=count,
            previous=result,
            split_cache=cache,
            **options,
        )
    assert result is not None
    return result


def _fixed_counts(counts):
    """An ``_origin_counts`` stand-in: the given counts in every year from
    ``first``, zero in the years an extended simulation already holds."""

    def origin_counts(_compiled, _plan, _scores, simulations: int, first: int = 0):
        return {
            origin: np.where(np.arange(simulations) >= first, values[:simulations], 0)
            for origin, values in counts.items()
        }

    return origin_counts


def _ranked_small_events():
    plan = ExecutionPlan(
        plan_id="ranked",
        required_arms=("reference",),
        simulations=48,
        seed=7,
        requested_views=VIEWS,
        numerical_controls={"event_batch_size": 997},
    )
    return plan, _two_origin_state()


def _shared_contracts():
    """A single-member origin sharing contracts with a campaign origin, so
    one reduction applies their annual terms, with severities, payments and
    defense."""
    payload = _rich_payload()
    for state in payload["joint_states"]:
        pairs = deepcopy(state["laws"][0])
        pairs["origin_id"] = "pairs"
        pairs["reference_annual_rate"] = 0.4
        pairs["segments"][0]["kernel"]["cluster"]["support"] = [
            {"count": 2, "probability": 1}
        ]
        state["laws"].append(pairs)
    compiled = compile_incident_state(setup_from_payload(payload), "low")
    _, plan, _, _ = common_case()
    plan = plan.model_copy(
        update={
            "simulations": 40,
            "requested_views": (*LINEAR_VIEWS, *POLICY_VIEWS),
            "numerical_controls": {"event_batch_size": 7},
        }
    )
    return plan, compiled


def _count_experiment(kind):
    study = _count_control_study()
    if kind == "shared_gamma_intensity":
        payload = encode_scientific(study)
        for alternative in payload["alternatives"]:
            shared = alternative["alternative_id"] == "stress"
            alternative["count_experiment"] = {
                "kind": kind,
                "origin_groups": {
                    "campaign": "all" if shared else "first",
                    "other-campaign": "all" if shared else "second",
                },
                "coefficient_of_variation": 0.8,
            }
        study = setup_from_payload(payload)
    _, plan, _, _ = common_case()
    plan = plan.model_copy(update={"simulations": 40})
    return plan, compile_incident_state(study, "low", "stress")


def _negative_binomial():
    payload = encode_scientific(_study(rate=3.0))
    for state in payload["joint_states"]:
        state["laws"][0]["count_family"] = "negative_binomial"
        state["laws"][0]["count_variance_to_mean"] = 3
    _, plan, _, _ = common_case()
    plan = plan.model_copy(update={"simulations": 40})
    return plan, compile_incident_state(setup_from_payload(payload), "low")


CASES = {
    "ranked-small-events": _ranked_small_events,
    "shared-contracts": _shared_contracts,
    "gaussian-count-experiment": lambda: _count_experiment("fixed_marginal_gaussian"),
    "gamma-count-experiment": lambda: _count_experiment("shared_gamma_intensity"),
    "negative-binomial": _negative_binomial,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_extending_a_simulation_gives_the_draws_of_one_simulation(case):
    plan, compiled = CASES[case]()
    years = (plan.simulations, 2 * plan.simulations, 4 * plan.simulations)
    direct = runtime.simulate_incident_state(
        plan, compiled, response_sha256="a" * 64, years=years[-1]
    )
    assert direct.audit.total_initiators > 0
    _assert_same(_extended(plan, compiled, years), direct)
    # The first years are those of a shorter simulation.
    short = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)
    for view, row in short[0].views.items():
        np.testing.assert_array_equal(
            row.array, direct[0].views[view].array[: plan.simulations]
        )


def test_annual_terms_shared_with_a_later_campaign_origin_extend_unchanged(
    monkeypatch,
):
    """A campaign origin sharing a contract with a single-member one first
    fires after the first round; the shared annual terms take both origins'
    events in each year, so the extension still gives the draws of one
    simulation."""
    plan, compiled = _shared_contracts()
    counts = {
        "campaign": np.tile(np.array([3, 0, 5, 1], dtype=np.int64), 40),
        "pairs": np.where(np.arange(160) < 100, 0, 1).astype(np.int64),
    }
    monkeypatch.setattr(
        runtime,
        "_origin_counts",
        _fixed_counts(counts),
    )
    short = runtime.simulate_incident_state(
        plan, compiled, response_sha256="a" * 64, years=80
    )
    assert short.audit.campaign_members == 0
    provenance = short[0].evaluator_provenance_by_risk_unit
    assert provenance is not None
    assert provenance["campaign"]["method"] == "vectorized_native_incident"
    direct = runtime.simulate_incident_state(
        plan, compiled, response_sha256="a" * 64, years=160
    )
    # Each of the 60 later episodes passes its shared gate with
    # probability 0.7 and then has two members.
    assert 0 < direct.audit.campaign_members <= 120
    _assert_same(_extended(plan, compiled, (80, 160)), direct)


@pytest.mark.parametrize("financial", [True, False], ids=["financial", "linear"])
def test_a_campaign_origin_firing_only_in_earlier_years_keeps_its_audit(
    monkeypatch, financial
):
    """A campaign origin whose events all lie in the kept years still counts
    toward the member audit and budget of the extended simulation."""
    plan, compiled = _shared_contracts()
    if not financial:
        plan = plan.model_copy(update={"requested_views": LINEAR_VIEWS})
    counts = {
        "campaign": np.tile(np.array([3, 0, 5, 1], dtype=np.int64), 40),
        "pairs": np.where(np.arange(160) < 40, 1, 0).astype(np.int64),
    }
    monkeypatch.setattr(
        runtime,
        "_origin_counts",
        _fixed_counts(counts),
    )
    direct = runtime.simulate_incident_state(
        plan, compiled, response_sha256="a" * 64, years=160
    )
    assert 0 < direct.audit.campaign_members <= 80
    _assert_same(_extended(plan, compiled, (80, 160)), direct)


# The work allowance each case varies, the work count it bounds, and the work
# each event costs: one individually drawn event, or the two members of a
# campaign episode.
WORK_CASES = {
    "vectorized": ("max_vectorized_events_per_year", "vectorized_initiators", 1),
    "members": ("max_campaign_members_per_year", "campaign_members", 2),
}


# Events in each of a first round's four years and its extension's four more,
# with the allowance per unit of work per event that admits the first round
# but not all eight years: rising counts, whose later years alone already
# exceed it, and level ones, whose later years alone are within it.
WORK_PATTERNS = {
    "rising": ((1, 1, 1, 1, 5, 5, 5, 5), 2),
    "level": ((2, 2, 2, 2, 3, 3, 3, 3), 2),
}


@pytest.mark.parametrize("pattern", sorted(WORK_PATTERNS))
@pytest.mark.parametrize("case", sorted(WORK_CASES))
def test_every_round_limits_the_work_of_all_its_years(monkeypatch, case, pattern):
    """An allowance that admits the first four years refuses the extension to
    eight once their work over all eight years exceeds it, whatever the new
    years alone draw, and an event allowance does so before the extension
    draws any event; one more a year admits both rounds (exactly, for the
    rising counts)."""
    allowance, count, work = WORK_CASES[case]
    counts, refused = WORK_PATTERNS[pattern]
    if case == "vectorized":
        _, plan, compiled, _ = common_case()
        # The member allowance does not bound single-member events.
        controls = {"max_campaign_members_per_year": 1}
    else:
        _, plan, _, _ = common_case()
        compiled = compile_incident_state(_study(), "low")
        controls = {}
    monkeypatch.setattr(
        runtime,
        "_origin_counts",
        _fixed_counts({"campaign": np.array(counts, dtype=np.int64)}),
    )
    # Every round that passes its event checks reduces its origins' events.
    drawing = []
    vectorized = runtime._vectorized_incidents

    def tracked(*args, **kwargs):
        drawing.append(True)
        return vectorized(*args, **kwargs)

    monkeypatch.setattr(runtime, "_vectorized_incidents", tracked)

    def simulate(per_year, years, previous=None):
        return runtime.simulate_incident_state(
            plan.model_copy(
                update={
                    "simulations": 4,
                    "requested_views": ("gross",),
                    "numerical_controls": {**controls, allowance: per_year},
                }
            ),
            compiled,
            response_sha256="a" * 64,
            years=years,
            previous=previous,
        )

    admitted = (refused + 1) * work
    first = simulate(admitted, 4)
    assert (getattr(first.audit, count), first.audit.work_limits[count]) == (
        sum(counts[:4]) * work,
        4 * admitted,
    )
    extended = simulate(admitted, 8, first)
    assert getattr(extended.audit, count) == sum(counts) * work
    assert extended.audit.work_limits[count] == 8 * admitted
    first = simulate(refused * work, 4)
    assert (
        getattr(first.audit, count)
        <= first.audit.work_limits[count]
        == 4 * refused * work
    )
    assert sum(counts) * work > 8 * refused * work
    drawing.clear()
    with pytest.raises(
        ValueError, match=f"above the limit of {8 * refused * work}, {allowance} "
    ):
        simulate(refused * work, 8, first)
    # Campaign members are counted with each batch of episodes, so they are
    # refused as a batch reserves them, before any member is drawn.
    assert drawing == ([True] if case == "members" else [])


def test_an_extension_must_continue_the_same_state_plan_and_streams():
    plan, compiled = _ranked_small_events()
    previous = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)
    changed = {
        "seed": plan.model_copy(update={"seed": 8}),
        "controls": plan.model_copy(
            update={
                "numerical_controls": {
                    **plan.numerical_controls,
                    "small_events": {"target_large_events_per_year": 64},
                }
            }
        ),
    }
    for other in changed.values():
        with pytest.raises(ValueError, match="continue the same"):
            runtime.simulate_incident_state(
                other,
                compiled,
                response_sha256="a" * 64,
                years=2 * plan.simulations,
                previous=previous,
            )
    with pytest.raises(ValueError, match="continue the same"):
        runtime.simulate_incident_state(
            plan,
            compiled,
            response_sha256="b" * 64,
            years=2 * plan.simulations,
            previous=previous,
        )
    with pytest.raises(ValueError, match="above the extended"):
        runtime.simulate_incident_state(
            plan,
            compiled,
            response_sha256="a" * 64,
            years=plan.simulations,
            previous=previous,
        )


def _gate_payload():
    """A ranked origin with summed small events and a lab, and a campaign
    with policy terms, in one state."""
    payload = partial_financial_payload(complete=True, payment=1)
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Synthetic lab",
            "display_name": "lab",
            "status": "hypothetical",
        }
    )
    payload["contracts"][0].update(
        occurrence_retention=5, annual_retention=15, annual_limit=200
    )
    (campaign,) = payload["joint_states"][0]["laws"]
    campaign.update(origin_id="campaign", reference_annual_rate=0.5)
    campaign["segments"][0]["kernel"]["cluster"]["support"] = [
        {"count": 2, "probability": 1}
    ]
    count = _lognormal_count(800.0, 30.0)
    ranked = json.loads(json.dumps(study_payload()["joint_states"][0]["laws"][0]))
    ranked.update(
        reference_annual_rate=epistemic_mean(count),
        parameter_rank={
            "group_id": "group",
            "count": count.model_dump(),
            "tail_form": "calibrated_p95_peak",
        },
    )
    ranked["segments"][0]["kernel"] = _kernel(member=0.9).model_dump(mode="json")
    payload["joint_states"][0]["laws"].append(ranked)
    payload["alternatives"] = [
        {
            "alternative_id": "audit",
            "purpose": "numerical_refinement",
            "description": "More individually drawn large events.",
        }
    ]
    return payload


def _gate_run(tmp_path, name, *, simulations):
    plan = ExecutionPlan(
        plan_id=name,
        required_arms=("reference", "audit"),
        simulations=simulations,
        seed=29,
        requested_views=(
            "gross",
            "oep_gross",
            "welfare",
            "lab_loss::lab",
            "scenario::scenario::gross",
            "financial",
            "oep_financial",
        ),
        numerical_controls={
            # Too few years for any batch to hold ten 99.9% exceedances, so
            # the gate doubles to its cap: 4,096 years of the one state.
            "tail_gate": {
                "batches": 16,
                "max_reference_state_years": 4096,
                "max_comparison_state_years": 4096,
                "levels": [0.999],
            },
            "max_campaign_members_per_year": 25,
        },
        numerical_controls_by_arm={
            "audit": {"small_events": {"target_large_events_per_year": 64}}
        },
    )
    package = build_setup(
        destination=tmp_path / f"{name}-setup",
        setup=with_execution(setup_from_payload(_gate_payload()), plan, setup_id=name),
        code_commit="fixture",
    )
    run_setup(setup_path=package.root, output=tmp_path / name, code_commit="fixture")
    return package, tmp_path / name / "capture"


def test_gate_rounds_extend_the_years_of_one_simulation(tmp_path):
    """Rounds of 1,024, 2,048 and 4,096 years capture the arrays and records
    of a single 4,096-year run, in every arm, state and view."""
    package, gated = _gate_run(tmp_path, "gated", simulations=1024)
    _, direct = _gate_run(tmp_path, "direct", simulations=4096)
    draws, capture = load_run_capture(gated)
    reference_draws, reference = load_run_capture(direct)
    assert [row["doubling"] for row in capture.tail_gate_rounds] == [0, 1, 2]
    assert len(reference.tail_gate_rounds) == 1
    assert capture.supports == reference.supports
    assert capture.arms == reference.arms
    for arm, arm_draws in draws.items():
        assert arm_draws.keys() == reference_draws[arm].keys()
        for state, draw in arm_draws.items():
            other = reference_draws[arm][state]
            assert draw.n_simulations == 4096
            for view, row in draw.views.items():
                assert row.array.tobytes() == other.views[view].array.tobytes()
    # The arms' numerical controls differ, so they do not share a simulation.
    (state,) = draws["reference"]
    assert not np.array_equal(
        draws["reference"][state].annual_losses["gross"],
        draws["audit"][state].annual_losses["gross"],
    )
    execution = capture.arms["reference"][state].execution
    assert execution["summed_small_initiators"] > 0
    assert execution["campaign_members"] > 0
    # Replay re-derives every round from the captured years' prefixes.
    replay_capture(
        setup_path=package.root,
        capture_directory=gated,
        output=tmp_path / "replay.json",
        code_commit="fixture",
    )
