"""Policy terms on event-addressed draws match an exact oracle of the same events."""

from copy import deepcopy

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.probability.distributions import Distribution
from catmodel.simulation import incident_runtime as runtime
from catmodel.simulation.incident import compile_incident_state
from test_support.incident import common_case
from test_support.policy_oracle import event_rows, exact_policy_views
from test_support.run_setup import setup_from_payload

RISK_UNIT = "risk_unit::campaign::occurrence_net_financial"
POLICY_VIEWS = ("financial", "oep_financial", RISK_UNIT)
LINEAR_VIEWS = ("gross", "welfare", "net_resource_cost", "scenario::loss::gross")


def _liability():
    return {
        "mean": 0.4,
        "concentration": 10,
        "recovery_rate": 0.8,
        "severity_slope": 0.05,
        "median_severity": 10,
        "regulatory_recovery_threshold": 100,
        "regulatory_recovery_uplift": 0,
    }


def _rich_payload(*, shared_probability=0.7):
    """A gated native law with a payment proxy, defense loadings and two contracts."""
    study, *_ = common_case(
        sampler=Distribution(1, 3, 10, 40, 100, gpd_shape=0.3), financial=True
    )
    payload = encode_scientific(study)
    payload["contracts"] = [
        {
            "contract_id": "policy",
            "contract_year": "2027",
            "entity_id": "model",
            "occurrence_retention": 2,
            "occurrence_limit": 30,
            "annual_retention": 15,
            "annual_limit": 120,
            "defense_loading": 0.25,
            "defense_outside_limits": True,
        },
        {
            "contract_id": "excess",
            "contract_year": "2027",
            "entity_id": "model",
            "occurrence_retention": 1,
            "annual_retention": 3,
            "annual_limit": 50,
            "outside_limits_subject_to_aggregate": False,
        },
    ]
    for state in payload["joint_states"]:
        kernel = state["laws"][0]["segments"][0]["kernel"]
        kernel["shared_probability"] = shared_probability
        outcome = kernel["member_outcomes"][0]
        harm = outcome["components"][0]
        fees = {
            **harm,
            "component_id": "fees",
            "nature": "transfer",
            "loss_type": "indemnity",
            "severity": None,
            "rank_group_id": None,
            "amount": 4,
        }
        outcome["components"] = [harm, fees]
        outcome["obligations"] = [
            {"component_id": "fees", "entity_id": "model", "fraction": 1}
        ]
        outcome["payments"] = [
            {"component_id": "fees", "entity_id": "model", "fraction": 0.5}
        ]
        outcome["payment_kernel"] = {
            "source_severity": harm["severity"],
            "rank_group_id": "severity",
            "entity_id": "model",
            "liability_component_ids": ["harm"],
            "liability": _liability(),
        }
        outcome["memberships"] = [
            {
                "component_id": "harm",
                "contract_id": "policy",
                "obligated_entity_id": "model",
                "eligible": True,
                "covered_fraction": 0.9,
                "occurrence_id": "incident",
                "occurrence_scope": "member",
            },
            {
                "component_id": "fees",
                "contract_id": "policy",
                "obligated_entity_id": "model",
                "eligible": True,
                "occurrence_id": "incident",
                "occurrence_scope": "member",
            },
            {
                "component_id": "harm",
                "contract_id": "excess",
                "obligated_entity_id": "model",
                "eligible": True,
                "outside_limits": True,
                "defense_loading": 0.1,
                "occurrence_id": "incident",
                "occurrence_scope": "member",
            },
            {
                "component_id": "fees",
                "contract_id": "excess",
                "obligated_entity_id": "model",
                "eligible": True,
            },
        ]
    return payload


def _run(payload, views, monkeypatch, *, counts=None, controls=None):
    compiled = compile_incident_state(setup_from_payload(payload), "low")
    _study, plan, _compiled, _fixed = common_case()
    simulations = 400 if counts is None else len(counts)
    plan = plan.model_copy(
        update={
            "simulations": simulations,
            "requested_views": views,
            "numerical_controls": controls or {},
        }
    )
    if counts is not None:
        monkeypatch.setattr(runtime, "_origin_counts", lambda *_: {"campaign": counts})
    return runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)


def test_vectorized_policy_terms_match_the_exact_oracle_of_the_same_draws(
    monkeypatch,
):
    """Every year's policy views equal an exact evaluation of the same events'
    covered parts (``exact_policy_views``), up to the order of the sums (the
    oracle's are exact, the arrays' left to right)."""
    payload = _rich_payload()
    views = (*LINEAR_VIEWS, *POLICY_VIEWS)
    counts = runtime._origin_counts
    drawn = {}

    def recorded(*args):
        drawn.update(counts(*args))
        return drawn

    monkeypatch.setattr(runtime, "_origin_counts", recorded)
    vector, reasons, audit = _run(payload, views, monkeypatch)
    assert reasons == {}
    assert audit.campaign_members == 0
    compiled = compile_incident_state(setup_from_payload(payload), "low")
    rows = event_rows(compiled, "campaign", drawn["campaign"], seed=vector.seed)
    exact = [
        exact_policy_views(rows.get(year, ()), compiled.contracts)
        for year in range(vector.n_simulations)
    ]
    oracle = {
        "financial": [values["financial"] for values in exact],
        "oep_financial": [values["oep_financial"] for values in exact],
        RISK_UNIT: [values["occurrences"] for values in exact],
    }
    assert sum(len(values) for values in rows.values()) > 100
    for view in POLICY_VIEWS:
        np.testing.assert_allclose(
            vector.views[view].array, oracle[view], rtol=1e-12, atol=0
        )
    financial = vector.views["financial"].array
    # Annual retentions bind in some years and limits in others; the oracle
    # is not exercised by a law whose payments are all zero or all capped.
    assert 0 < np.count_nonzero(financial) < len(financial)
    assert np.any(vector.views["oep_financial"].array > 0)
    linear_only, _, _ = _run(payload, LINEAR_VIEWS, monkeypatch)
    for view in LINEAR_VIEWS:
        np.testing.assert_array_equal(
            linear_only.views[view].array, vector.views[view].array
        )


def _deterministic_case(rate, *, scope="member"):
    study, _plan, _compiled, counts = common_case(financial=True, rate=rate)
    payload = encode_scientific(study)
    for state in payload["joint_states"]:
        outcome = state["laws"][0]["segments"][0]["kernel"]["member_outcomes"][0]
        outcome["memberships"][0]["occurrence_scope"] = scope
    return payload, counts["campaign"]


# Annual terms for the deterministic case, and the annual payments they give
# for 0, 1, 3, 12, 13 and 40 occurrences that each pay 100 - 60 = 40.
ANNUAL_TERMS = {
    # 40n - 500: a year with up to 12 occurrences pays nothing in total.
    "annual_retention": ({"annual_retention": 500}, [0, 0, 0, 0, 20, 1100]),
    # min(40n, 30): every loss year pays less than its largest occurrence.
    "annual_limit": (
        {"annual_retention": 0, "annual_limit": 30},
        [0, 30, 30, 30, 30, 30],
    ),
}


@pytest.mark.parametrize("terms", sorted(ANNUAL_TERMS))
@pytest.mark.parametrize("evaluator", ["vectorized", "campaign_sampler"])
def test_deterministic_policy_terms_match_the_annual_oracle(
    monkeypatch, evaluator, terms
):
    contract_terms, annual_payments = ANNUAL_TERMS[terms]
    payload, _ = _deterministic_case(10)
    payload["contracts"][0].update(contract_terms)
    counts = np.array([0, 1, 3, 12, 13, 40], dtype=np.int64)
    if evaluator == "campaign_sampler":
        monkeypatch.setattr(runtime, "single_member_law", lambda _kernel: None)
    draw, _, audit = _run(payload, ("gross", *POLICY_VIEWS), monkeypatch, counts=counts)
    # The campaign sampler draws each event's one member as an episode member.
    assert audit.campaign_members == (
        0 if evaluator == "vectorized" else int(counts.sum())
    )
    np.testing.assert_array_equal(draw.views["gross"].array, 100 * counts)
    financial = draw.views["financial"].array
    occurrence_maximum = draw.views["oep_financial"].array
    np.testing.assert_allclose(financial, annual_payments, rtol=1e-15)
    # oep_financial is the largest occurrence payment after occurrence terms
    # and before annual terms, as is the risk-unit occurrence-net sum.
    np.testing.assert_array_equal(occurrence_maximum, np.where(counts > 0, 40, 0))
    np.testing.assert_allclose(draw.views[RISK_UNIT].array, 40 * counts, rtol=1e-15)
    loss_years = counts > 0
    if terms == "annual_retention":
        assert np.any((financial == 0) & (occurrence_maximum > 0))
    else:
        assert np.all(occurrence_maximum[loss_years] > financial[loss_years])


def test_campaign_members_take_their_own_allowance(monkeypatch):
    """Single-member events count only as events; the same events drawn as
    campaign episodes count their members too, against their own allowance,
    and pay the same policy terms."""
    payload, counts = _deterministic_case(10_000)
    views = ("gross", "financial", "oep_financial")
    draw, _, audit = _run(payload, views, monkeypatch, counts=counts)
    assert audit.campaign_members == 0
    expected = np.full(len(counts), 40 * 10_000 - 500)
    np.testing.assert_allclose(draw.views["financial"].array, expected)
    with monkeypatch.context() as campaign:
        campaign.setattr(runtime, "single_member_law", lambda _kernel: None)
        draw, _, audit = _run(payload, views, monkeypatch, counts=counts)
        assert audit.campaign_members == int(counts.sum())
        np.testing.assert_allclose(draw.views["financial"].array, expected)
        with pytest.raises(ValueError, match="max_campaign_members_per_year"):
            _run(
                payload,
                ("financial",),
                monkeypatch,
                counts=counts,
                controls={"max_campaign_members_per_year": 9_999},
            )


def test_contract_year_occurrences_sum_across_events(monkeypatch):
    payload, _ = _deterministic_case(10, scope="contract_year")
    counts = np.array([0, 2, 7], dtype=np.int64)
    draw, _, audit = _run(payload, ("gross", "financial"), monkeypatch, counts=counts)
    provenance = draw.evaluator_provenance_by_risk_unit
    assert provenance is not None and provenance["campaign"] is not None
    assert provenance["campaign"]["method"] == "vectorized_native_incident"
    assert audit.campaign_members == 0
    # One contract-year occurrence: 100n - 60 before the 500 annual retention.
    np.testing.assert_array_equal(
        draw.views["financial"].array, np.maximum(100 * counts - 560, 0)
    )
    np.testing.assert_array_equal(draw.views["gross"].array, 100 * counts)


def test_contract_shared_with_a_campaign_origin_takes_annual_terms_once(monkeypatch):
    payload, _ = _deterministic_case(10)
    for state in payload["joint_states"]:
        other = deepcopy(state["laws"][0])
        other["origin_id"] = "pairs"
        other["segments"][0]["kernel"]["cluster"]["support"] = [
            {"count": 2, "probability": 1}
        ]
        state["laws"].append(other)
    compiled = compile_incident_state(setup_from_payload(payload), "low")
    first = np.array([3, 20, 0], dtype=np.int64)
    pairs = np.array([4, 0, 9], dtype=np.int64)
    monkeypatch.setattr(
        runtime, "_origin_counts", lambda *_: {"campaign": first, "pairs": pairs}
    )
    _study, plan, _compiled, _counts = common_case()
    plan = plan.model_copy(
        update={"simulations": 3, "requested_views": ("gross", "financial")}
    )
    draw, _, audit = runtime.simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    provenance = draw.evaluator_provenance_by_risk_unit
    assert provenance is not None and provenance["campaign"] is not None
    assert provenance["campaign"]["method"] == "vectorized_native_incident"
    assert provenance["pairs"]["method"] == "vectorized_native_incident"
    assert audit.campaign_members == 2 * int(pairs.sum())
    members = first + 2 * pairs
    np.testing.assert_array_equal(
        draw.views["financial"].array, np.maximum(40 * members - 500, 0)
    )
    unrelated, _, _ = runtime.simulate_incident_state(
        plan.model_copy(update={"requested_views": ("gross", "oep_financial")}),
        compiled,
        response_sha256="a" * 64,
    )
    # Occurrence-level views have no annual coupling across origins.
    np.testing.assert_array_equal(unrelated.views["oep_financial"].array, [40, 40, 40])


def test_vectorized_event_budget_fails_before_drawing(monkeypatch):
    payload, _ = _deterministic_case(10)
    calls = []
    original = runtime.draw_single_member_events

    def tracked(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "draw_single_member_events", tracked)
    with pytest.raises(ValueError, match="max_vectorized_events_per_year"):
        _run(
            payload,
            ("gross",),
            monkeypatch,
            counts=np.array([3, 4], dtype=np.int64),
            controls={"max_vectorized_events_per_year": 3},
        )
    assert calls == []
    draw, _, audit = _run(
        payload,
        ("gross",),
        monkeypatch,
        counts=np.array([3, 3], dtype=np.int64),
        controls={"max_vectorized_events_per_year": 3},
    )
    assert audit.resolved_numerical_controls["max_vectorized_events_per_year"] == 3
    assert audit.work_limits["vectorized_initiators"] == 6
    np.testing.assert_array_equal(draw.views["gross"].array, [300, 300])
