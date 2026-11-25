"""Exact annual outputs when campaign episodes are drawn in batches.

Every episode's and member's uniforms are event-addressed, and a batch's
members are drawn in runs bounded by ``event_batch_size``, so any batch size
gives the same draws, sums and audit.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.incident import ExecutionPlan
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.probability.distributions import Distribution, MixtureSeverityDistribution
from catmodel.simulation import incident_runtime as runtime
from catmodel.simulation.incident import compile_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import _native_payment_study


def _compiled(*, explicit):
    payload = encode_scientific(_native_payment_study(mean=0.4, recovery_rate=0.8))
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Synthetic loss-bearing lab",
            "display_name": "Synthetic lab",
            "status": "hypothetical",
        }
    )
    severity = native_severity_from_sampler(
        MixtureSeverityDistribution(
            components=[
                Distribution(1, 2, 3, 4, 5, gpd_shape=0.3, gpd_scale=1),
                Distribution(2, 4, 6, 8, 10, gpd_shape=0.7, gpd_scale=2),
            ],
            weights=np.array([0.4, 0.6]),
            severity_cap=100,
        )
    ).model_dump(mode="json")
    for state in payload["joint_states"]:
        law = state["laws"][0]
        segment = law["segments"][0]
        kernel = segment["kernel"]
        kernel["shared_probability"] = 0.9
        kernel["unit_probability"] = 0.75
        member = kernel["member_outcomes"][0]
        component = member["components"][0]
        # Every component allocates half its amount to the lab, so the lab view
        # adds these allocations to the episode's aggregate lab draw.
        component.update(
            severity=severity, rank_scope="campaign", lab_loss_weights={"lab": 0.5}
        )
        member["payment_kernel"].update(source_severity=severity, rank_scope="campaign")
        member["components"].append(
            {
                **deepcopy(component),
                "component_id": "member-harm",
                "rank_group_id": "member-rank",
                "rank_scope": "member",
                "multiplier": 0.5,
            }
        )
        kernel["campaign_components"] = [
            {
                **deepcopy(component),
                "component_id": "response",
                "post_multiplier": 0.25,
            }
        ]
        kernel["campaign_financial_mapping_complete"] = True
        kernel["campaign_lab_losses"] = [{"lab_id": "lab", "severity": severity}]
        if explicit:
            other_member = deepcopy(member)
            other_member["components"][1]["multiplier"] = 0.75
            kernel.pop("cluster")
            kernel.pop("member_outcomes")
            kernel["explicit_joint_members"] = [
                {"probability": 0.4, "members": [member]},
                {"probability": 0.6, "members": [member, other_member]},
            ]
        segment.update(
            end=0.5,
            reference_exposure_share=0.5,
            source_model_id="model",
            source_model_revision="one",
            source_public_status="public",
        )
        second = deepcopy(segment)
        second.update(start=0.5, end=1, source_public_status="not_yet_public")
        law["segments"] = [segment, second]
        other_origin = deepcopy(law)
        other_origin["origin_id"] = "other-campaign"
        state["laws"].append(other_origin)
    return compile_incident_state(setup_from_payload(payload), "low")


def _plan(*, members_per_year=125, batch_size=None):
    return ExecutionPlan(
        plan_id="batched-event-oracle",
        required_arms=("reference",),
        simulations=8,
        seed=42,
        requested_views=(
            "gross",
            "welfare",
            "financial",
            "lab_loss::lab",
            "source_status::public::welfare",
            "source_status::not_yet_public::lab_loss::lab",
        ),
        numerical_controls={
            "max_campaign_members_per_year": members_per_year,
            **({} if batch_size is None else {"event_batch_size": batch_size}),
        },
    )


def _assert_same(actual, expected):
    draw, reasons, audit = actual
    baseline, baseline_reasons, baseline_audit = expected
    assert reasons == baseline_reasons
    # Only the recorded batch size may differ.
    assert {
        **audit.resolved_numerical_controls,
        "event_batch_size": None,
    } == {**baseline_audit.resolved_numerical_controls, "event_batch_size": None}
    assert (
        replace(
            audit,
            resolved_numerical_controls=baseline_audit.resolved_numerical_controls,
        )
        == baseline_audit
    )
    for name in (
        "seed",
        "stratum_sha256",
        "response_sha256",
        "joint_law_semantics",
        "evaluator_provenance_by_risk_unit",
    ):
        assert getattr(draw, name) == getattr(baseline, name)
    assert (
        draw.normalized_weights().tobytes() == baseline.normalized_weights().tobytes()
    )
    assert draw.views.keys() == baseline.views.keys()
    for name, view in draw.views.items():
        reference = baseline.views[name]
        assert view.array.tobytes() == reference.array.tobytes()
        assert view.mean_exists == reference.mean_exists
        assert view.variance_exists == reference.variance_exists
        assert view.native_mean == reference.native_mean


@pytest.mark.parametrize("explicit", [False, True])
def test_batched_episodes_preserve_complete_annual_capture(monkeypatch, explicit):
    compiled = _compiled(explicit=explicit)
    counts = np.array([0, 2, 0, 1, 3, 0, 1, 0])
    other_counts = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    origin_counts = {"campaign": counts, "other-campaign": other_counts}
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: origin_counts)
    baseline = runtime.simulate_incident_state(
        _plan(), compiled, response_sha256="a" * 64
    )
    assert baseline.audit.total_initiators == int(counts.sum() + other_counts.sum())
    assert baseline.audit.campaign_members > 0
    assert not baseline[1]
    assert all(
        np.all(view.array[(counts + other_counts) == 0] == 0)
        for view in baseline[0].views.values()
    )
    # A batch of one event or one member's worth splits every episode's
    # members from the next; a large batch holds every event.
    for size in (1, 2, 3, 1024):
        actual = runtime.simulate_incident_state(
            _plan(batch_size=size), compiled, response_sha256="a" * 64
        )
        _assert_same(actual, baseline)


@pytest.mark.parametrize("batch_size", [1, 3, 1024])
def test_batched_episodes_preserve_cumulative_member_budget(monkeypatch, batch_size):
    compiled = _compiled(explicit=False)
    monkeypatch.setattr(
        runtime,
        "_origin_counts",
        lambda *_: {
            "campaign": np.full(8, 2),
            "other-campaign": np.zeros(8, dtype=int),
        },
    )
    with pytest.raises(ValueError, match="max_campaign_members_per_year"):
        runtime.simulate_incident_state(
            _plan(members_per_year=1, batch_size=batch_size),
            compiled,
            response_sha256="a" * 64,
        )


def test_member_runs_keep_every_event_in_order_and_split_large_episodes():
    """Runs of a batch's events hold at most the batch size of members, except
    an episode holding more on its own; no event is lost or reordered."""
    members = np.array([0, 3, 0, 9, 1, 1, 0, 2, 0])
    runs = list(runtime._member_chunks(members, 4))
    assert runs == [(0, 3), (3, 4), (4, 9)]
    assert [members[start:stop].sum() for start, stop in runs] == [3, 9, 4]
    assert list(runtime._member_chunks(np.zeros(5, dtype=np.int64), 4)) == [(0, 5)]
    assert list(runtime._member_chunks(np.zeros(0, dtype=np.int64), 4)) == []
