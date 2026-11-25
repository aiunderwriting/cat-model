"""Exact native amounts and named dependence in vectorized incident marks."""

import math

import numpy as np
import pytest

from catmodel.contracts.components import AggregateLabLossLaw, ComponentAmountLaw
from catmodel.contracts.incident import (
    ClusterCount,
    JointMarkOutcome,
    MarkKernel,
    UnitCountMass,
)
from catmodel.contracts.native_severity import (
    native_severity_from_sampler,
    native_severity_sampler,
)
from catmodel.probability.distributions import (
    Distribution,
    MixtureSeverityDistribution,
    sample_affine_capped_severity,
)
from catmodel.probability.keyed_streams import event_uniforms
from catmodel.probability.numerics import scaled_product
from catmodel.simulation.components import (
    component_view_weight,
)
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_linear import (
    draw_single_member_events,
    linear_projection,
    linear_values,
    single_member_law,
)
from test_support.run_setup import setup_from_payload


def _kernel(*, labs=True):
    severity = native_severity_from_sampler(
        Distribution(1, 3, 10, 40, 100, gpd_shape=0.5)
    )
    lab_severity = native_severity_from_sampler(
        MixtureSeverityDistribution(
            [
                Distribution(0, 0, 0, 0, 0),
                Distribution(2, 4, 20, 60, 200, gpd_shape=0.4),
            ],
            np.array([0.3, 0.7]),
        )
    )
    return MarkKernel(
        cluster=ClusterCount(
            family="finite", support=(UnitCountMass(count=1, probability=1),)
        ),
        shared_probability=1,
        unit_probability=1,
        gate_meaning="One qualified event",
        selection_assumption="One member",
        displacement_assumption="No displacement",
        campaign_lab_losses=(
            AggregateLabLossLaw(lab_id="lab", severity=lab_severity, episode_cap=150),
            AggregateLabLossLaw(lab_id="other", amount=7),
        )
        if labs
        else (),
        member_outcomes=(
            JointMarkOutcome(
                probability=1,
                components=(
                    ComponentAmountLaw(
                        component_id="campaign",
                        nature="resource_cost",
                        lab_loss_weights={},
                        severity=severity,
                        rank_scope="campaign",
                        rank_group_id="shared",
                        multiplier=0.25,
                        post_multiplier=3,
                    ),
                    ComponentAmountLaw(
                        component_id="member",
                        nature="resource_cost",
                        lab_loss_weights={"lab": 0.5} if labs else {},
                        severity=severity,
                        rank_group_id="shared",
                        post_multiplier=2,
                    ),
                    ComponentAmountLaw(
                        component_id="transfer",
                        nature="transfer",
                        lab_loss_weights={},
                        severity=severity,
                        rank_group_id="shared",
                        post_multiplier=0.1,
                    ),
                    ComponentAmountLaw(
                        component_id="disabled",
                        nature="resource_cost",
                        lab_loss_weights={},
                        severity=severity,
                        rank_group_id="disabled",
                        multiplier=0,
                    ),
                    ComponentAmountLaw(
                        component_id="fixed",
                        nature="injury_valuation",
                        lab_loss_weights={},
                        amount=9,
                    ),
                ),
            ),
        ),
    )


def _draw(kernel, years, ordinals):
    law = single_member_law(kernel)
    assert law is not None
    return draw_single_member_events(
        law, years, ordinals, root_seed=42, state_id="state", origin_id="origin"
    )


def _values(kernel, views, years, ordinals, **kwargs):
    law = single_member_law(kernel)
    assert law is not None
    projection = linear_projection(law, views, **kwargs)
    assert projection is not None
    return linear_values(projection, _draw(kernel, years, ordinals))


def _rank(years, ordinals, purpose):
    return event_uniforms(
        years,
        ordinals,
        root_seed=42,
        state_id="state",
        origin_id="origin",
        purpose=purpose,
    )


def test_vectorized_marks_match_scalar_native_inverse_and_declared_rank_sharing():
    kernel = _kernel()
    views = ("gross", "oep_gross", "welfare", "lab_loss::lab", "lab_loss::other")
    years = np.repeat(np.arange(6), 80)
    ordinals = np.tile(np.arange(80), 6)
    draws = _draw(kernel, years, ordinals)
    actual = _values(kernel, views, years, ordinals)
    terms = []
    for law in kernel.member_outcomes[0].components:
        if law.amount is not None:
            amounts = np.full(
                years.shape,
                scaled_product(law.amount, law.multiplier, law.post_multiplier),
            )
        elif law.multiplier == 0:
            amounts = np.zeros(years.shape)
        else:
            sampler = native_severity_sampler(law.severity)
            purpose = (
                'physical-rank::["' + law.rank_scope + '","' + law.rank_group_id + '"]'
            )
            ranks = _rank(years, ordinals, purpose)
            # Independent scalar calls exercise the scalar inverse, affine and cap
            # rules, not the batched implementation.
            amounts = np.array(
                [
                    sample_affine_capped_severity(
                        sampler,
                        np.array([rank]),
                        scale=law.multiplier,
                        post_multiplier=law.post_multiplier,
                    )[0]
                    for rank in ranks
                ]
            )
        terms.append(amounts)
    gross = np.array([math.fsum(row) for row in zip(*terms, strict=True)])
    welfare = np.array(
        [
            math.fsum(
                amount * component_view_weight(law, "welfare")
                for amount, law in zip(
                    row, kernel.member_outcomes[0].components, strict=True
                )
            )
            for row in zip(*terms, strict=True)
        ]
    )
    np.testing.assert_allclose(draws.component_totals, gross, rtol=4e-15)
    np.testing.assert_allclose(actual["gross"], gross, rtol=4e-15)
    np.testing.assert_array_equal(actual["oep_gross"], actual["gross"])
    np.testing.assert_allclose(actual["welfare"], welfare, rtol=4e-15)
    lab = kernel.campaign_lab_losses[0]
    sampler = native_severity_sampler(lab.severity)
    ranks = _rank(years, ordinals, "aggregate-lab-mark::lab")
    aggregate_lab = np.array(
        [sample_affine_capped_severity(sampler, np.array([rank]))[0] for rank in ranks]
    )
    # The member component's lab allocation adds to the episode's aggregate
    # lab loss, and the episode cap bounds their sum.
    uncapped = terms[1] * 0.5 + aggregate_lab
    expected_lab = np.minimum(uncapped, 150)
    np.testing.assert_allclose(actual["lab_loss::lab"], expected_lab, rtol=4e-15)
    np.testing.assert_array_equal(actual["lab_loss::other"], np.full(years.shape, 7))
    assert 0 < np.count_nonzero(aggregate_lab == 0) < len(aggregate_lab)
    assert 0 < np.count_nonzero(uncapped > 150) < len(uncapped)
    assert draws.member.all()


def test_vectorized_marks_are_identical_across_chunking_reordering_and_view_selection():
    views = ("welfare", "lab_loss::lab", "lab_loss::other")
    kernel = _kernel()
    years = np.repeat(np.arange(4), 150)
    ordinals = np.tile(np.arange(150), 4)
    expected = _values(kernel, views, years, ordinals)
    for batch_size in (1, 71, 256):
        batches = [
            _values(
                kernel,
                views,
                years[start : start + batch_size],
                ordinals[start : start + batch_size],
            )
            for start in range(0, len(years), batch_size)
        ]
        for view in views:
            np.testing.assert_array_equal(
                np.concatenate([batch[view] for batch in batches]), expected[view]
            )
    order = np.random.default_rng(2).permutation(len(years))
    reordered = _values(kernel, views, years[order], ordinals[order])
    for view in views:
        np.testing.assert_array_equal(reordered[view], expected[view][order])
    np.testing.assert_array_equal(
        _values(kernel, ("welfare",), years, ordinals)["welfare"], expected["welfare"]
    )


def test_touched_views_are_exactly_those_an_event_can_make_nonzero():
    kernel = _kernel()
    law = single_member_law(kernel)
    assert law is not None
    projection = linear_projection(
        law,
        (
            "gross",
            "welfare",
            "lab_loss::lab",
            "lab_loss::other",
            "lab_loss::nobody",
            "scenario::scenario::welfare",
            "scenario::other::welfare",
            "source_status::public::welfare",
            "source_status::not_yet_public::lab_loss::lab",
        ),
        scenario_id="scenario",
        source_public_status="public",
    )
    assert projection is not None
    touched = projection.touched()
    # The aggregate lab amount alone touches its lab's view; another scenario,
    # another source status and a lab with no allocation are untouched.
    assert tuple(touched.weights) == (
        "gross",
        "welfare",
        "lab_loss::lab",
        "lab_loss::other",
        "scenario::scenario::welfare",
        "source_status::public::welfare",
    )
    years = np.repeat(np.arange(4), 50)
    ordinals = np.tile(np.arange(50), 4)
    draws = _draw(kernel, years, ordinals)
    every = linear_values(projection, draws)
    reduced = linear_values(touched, draws)
    assert reduced.keys() == touched.weights.keys()
    for view, values in every.items():
        if view in reduced:
            assert values.tobytes() == reduced[view].tobytes(), view
        else:
            # +0.0 for every event, so adding the view changes no annual sum.
            assert not np.any(values) and not np.any(np.signbit(values)), view


def test_scenario_and_relation_views_use_exact_component_coefficients():
    payload = _kernel().model_dump(mode="json")
    for component in payload["member_outcomes"][0]["components"]:
        component.update(
            bearer_id="victim",
            contributor_ids=["model"],
            attribution_weights={"model": 1},
            lab_loss_weights={"allocation": 0.25},
        )
    kernel = MarkKernel.model_validate(payload)
    views = (
        "gross",
        "welfare",
        "bearer::victim",
        "contributor::model",
        "attribution::model",
        "lab_loss::allocation",
        "scenario::scenario::welfare",
        "scenario::other::welfare",
    )
    result = _values(
        kernel, views, np.zeros(8, dtype=int), np.arange(8), scenario_id="scenario"
    )
    for view in ("bearer::victim", "contributor::model", "attribution::model"):
        np.testing.assert_array_equal(result[view], result["gross"])
    np.testing.assert_array_equal(
        result["lab_loss::allocation"], result["gross"] * 0.25
    )
    np.testing.assert_array_equal(
        result["scenario::scenario::welfare"], result["welfare"]
    )
    assert not np.any(result["scenario::other::welfare"])


def test_annual_vectorized_stream_is_stable_across_execution_partition_and_unrelated_origins(
    monkeypatch,
):
    from copy import deepcopy
    from dataclasses import replace

    from catmodel.contracts.incident import ExecutionPlan
    from catmodel.simulation import incident_runtime as runtime
    from test_support.study import study_payload

    payload = study_payload()
    payload["entities"].extend(
        [
            {
                "entity_id": identifier,
                "kind": "lab",
                "definition": "Synthetic lab",
                "display_name": identifier,
                "status": "hypothetical",
            }
            for identifier in ("lab", "other")
        ]
    )
    law = payload["joint_states"][0]["laws"][0]
    segment = law["segments"][0]
    segment.update(
        kernel=_kernel().model_dump(mode="json"),
        start=0,
        end=0.5,
        reference_exposure_share=0.5,
        source_public_status="public",
        source_model_id="model",
        source_model_revision="r1",
    )
    second = deepcopy(segment)
    second.update(start=0.5, end=1, source_public_status="not_yet_public")
    law["segments"].append(second)
    other = deepcopy(law)
    other["origin_id"] = "other-origin"
    for item in other["segments"]:
        item["source_public_status"] = "not_yet_public"
    payload["joint_states"][0]["laws"].append(other)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    views = (
        "gross",
        "oep_gross",
        "welfare",
        "lab_loss::lab",
        "source_status::public::welfare",
    )
    plan = ExecutionPlan(
        plan_id="bulk",
        required_arms=("reference",),
        simulations=4,
        seed=42,
        requested_views=views,
        numerical_controls={"event_batch_size": 1},
    )
    counts = {"origin": np.array([0, 8, 4, 1]), "other-origin": np.array([3, 0, 5, 1])}
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: counts)
    baseline = runtime.simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )[0]
    for size in (7, 128):
        changed = plan.model_copy(
            update={
                "numerical_controls": {
                    **plan.numerical_controls,
                    "event_batch_size": size,
                }
            }
        )
        actual = runtime.simulate_incident_state(
            changed, compiled, response_sha256="a" * 64
        )[0]
        for view in views:
            np.testing.assert_array_equal(
                actual.views[view].array, baseline.views[view].array
            )
    reordered = replace(compiled, incidents=tuple(reversed(compiled.incidents)))
    actual = runtime.simulate_incident_state(plan, reordered, response_sha256="a" * 64)[
        0
    ]
    scope = "source_status::public::welfare"
    np.testing.assert_array_equal(
        actual.views[scope].array, baseline.views[scope].array
    )
    counts["origin"][3] = 30
    counts["other-origin"] *= 3
    changed = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)[
        0
    ]
    np.testing.assert_array_equal(
        changed.views[scope].array[:3], baseline.views[scope].array[:3]
    )


def test_shared_kernel_keeps_each_origins_scenario_projection(monkeypatch):
    from copy import deepcopy
    from dataclasses import replace

    from catmodel.contracts.incident import ExecutionPlan
    from catmodel.simulation import incident_runtime as runtime
    from test_support.study import study_payload

    payload = study_payload()
    first = payload["joint_states"][0]["laws"][0]
    first["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
        "scenario_ids"
    ] = []
    second = deepcopy(first)
    second.update(origin_id="other-origin", scenario_id="other")
    payload["joint_states"][0]["laws"].append(second)
    population = deepcopy(payload["populations"][0])
    population["scenario_id"] = "other"
    payload["populations"].append(population)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    first_incident, other_incident = compiled.incidents
    shared_kernel = first_incident.segments[0].kernel
    other_incident = replace(
        other_incident,
        segments=(replace(other_incident.segments[0], kernel=shared_kernel),),
    )
    compiled = replace(compiled, incidents=(first_incident, other_incident))
    monkeypatch.setattr(
        runtime,
        "_origin_counts",
        lambda *_: {"origin": np.array([1]), "other-origin": np.array([2])},
    )
    plan = ExecutionPlan(
        plan_id="shared-kernel",
        required_arms=("reference",),
        simulations=1,
        seed=42,
        requested_views=(
            "gross",
            "scenario::scenario::gross",
            "scenario::other::gross",
        ),
    )
    draw, reasons, _ = runtime.simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    assert reasons == {}
    np.testing.assert_array_equal(draw.views["gross"].array, [30])
    np.testing.assert_array_equal(draw.views["scenario::scenario::gross"].array, [10])
    np.testing.assert_array_equal(draw.views["scenario::other::gross"].array, [20])


def test_a_richer_origin_keeps_unrelated_native_origin_vectorized_and_paired(
    monkeypatch,
):
    from copy import deepcopy
    from dataclasses import replace

    from catmodel.contracts.incident import ExecutionPlan
    from catmodel.simulation import incident_runtime as runtime
    from test_support.study import study_payload

    payload = study_payload()
    kernel = _kernel(labs=False)
    law = payload["joint_states"][0]["laws"][0]
    law["segments"][0].update(
        kernel=kernel.model_dump(mode="json"),
        source_public_status="public",
        source_model_id="model",
        source_model_revision="r1",
    )
    other = deepcopy(law)
    other["origin_id"] = "rich"
    other["segments"][0]["source_public_status"] = "not_yet_public"
    payload["joint_states"][0]["laws"].append(other)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    counts = {"origin": np.array([20, 0, 10]), "rich": np.array([2, 1, 0])}
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: counts)
    scope = "source_status::public::welfare"
    plan = ExecutionPlan(
        plan_id="hybrid",
        required_arms=("reference",),
        simulations=3,
        seed=42,
        requested_views=("welfare", scope),
    )
    baseline = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)
    first, rich = compiled.incidents
    segment = rich.segments[0]
    physical_kernel = segment.kernel.model_copy(
        update={
            "campaign_components": (
                ComponentAmountLaw(
                    component_id="campaign", nature="transfer", amount=0
                ),
            )
        }
    )
    rich = replace(rich, segments=(replace(segment, kernel=physical_kernel),))
    compiled = replace(compiled, incidents=(first, rich))
    actual = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)
    np.testing.assert_array_equal(
        actual[0].views[scope].array, baseline[0].views[scope].array
    )
    # Both origins' events are drawn individually, the campaign's as episodes.
    assert actual.audit.vectorized_initiators == 33
    methods = actual[0].evaluator_provenance_by_risk_unit
    assert methods is not None
    vector_method, campaign_method = methods["origin"], methods["rich"]
    assert vector_method is not None and campaign_method is not None
    assert vector_method["method"] == "vectorized_native_incident"
    assert campaign_method["method"] == "vectorized_native_incident"


def _two_origin_state(kernel):
    from copy import deepcopy

    from test_support.study import study_payload

    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law["segments"][0].update(
        kernel=kernel.model_dump(mode="json"),
        source_public_status="public",
        source_model_id="model",
        source_model_revision="r1",
    )
    other = deepcopy(law)
    other["origin_id"] = "gated"
    other["segments"][0]["source_public_status"] = "not_yet_public"
    payload["joint_states"][0]["laws"].append(other)
    return compile_incident_state(setup_from_payload(payload), "state")


def test_a_gate_keeps_the_origins_ranks_and_unrelated_origin_streams(monkeypatch):
    from dataclasses import replace

    from catmodel.contracts.incident import ExecutionPlan
    from catmodel.simulation import incident_runtime as runtime

    compiled = _two_origin_state(_kernel(labs=False))
    counts = {"origin": np.array([20, 0, 10, 7]), "gated": np.array([12, 9, 0, 30])}
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: counts)
    public, other = (
        "source_status::public::welfare",
        "source_status::not_yet_public::welfare",
    )
    plan = ExecutionPlan(
        plan_id="gate",
        required_arms=("reference",),
        simulations=4,
        seed=42,
        requested_views=(public, other),
    )
    baseline = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)
    incidents = []
    for incident in compiled.incidents:
        if incident.origin_id == "gated":
            segment = incident.segments[0]
            thinned = segment.kernel.model_copy(update={"shared_probability": 0.5})
            incident = replace(incident, segments=(replace(segment, kernel=thinned),))
        incidents.append(incident)
    compiled = replace(compiled, incidents=tuple(incidents))
    actual = runtime.simulate_incident_state(plan, compiled, response_sha256="a" * 64)
    assert actual.audit.campaign_members == 0
    np.testing.assert_array_equal(
        actual[0].views[public].array, baseline[0].views[public].array
    )
    # Surviving members keep their event-addressed ranks, so thinning can only
    # remove whole member amounts from each year.
    reduced, full = actual[0].views[other].array, baseline[0].views[other].array
    assert np.all(reduced <= full)
    assert np.any(reduced < full) and np.any(reduced[full > 0] > 0)


@pytest.mark.parametrize("evaluator", ["vectorized", "campaign_sampler"])
def test_composed_member_gates_preserve_the_thinned_law(monkeypatch, evaluator):
    from catmodel.contracts.incident import ExecutionPlan
    from catmodel.simulation import incident_runtime as runtime

    kernel = MarkKernel(
        cluster=ClusterCount(
            family="finite",
            support=(
                UnitCountMass(count=0, probability=0.2),
                UnitCountMass(count=1, probability=0.8),
            ),
        ),
        shared_probability=0.6,
        unit_probability=0.5,
        gate_meaning="Three independent gates",
        selection_assumption="At most one member",
        displacement_assumption="No displacement",
        member_outcomes=(
            JointMarkOutcome(
                probability=1,
                components=(
                    ComponentAmountLaw(
                        component_id="harm", nature="resource_cost", amount=10
                    ),
                ),
            ),
        ),
    )
    compiled = _two_origin_state(kernel)
    simulations = 4_000
    counts = {
        "origin": np.full(simulations, 5),
        "gated": np.zeros(simulations, dtype=int),
    }
    monkeypatch.setattr(runtime, "_origin_counts", lambda *_: counts)
    if evaluator == "campaign_sampler":
        monkeypatch.setattr(runtime, "single_member_law", lambda _kernel: None)
    plan = ExecutionPlan(
        plan_id="thinned",
        required_arms=("reference",),
        simulations=simulations,
        seed=7,
        requested_views=("gross",),
    )
    draw, _, audit = runtime.simulate_incident_state(
        plan, compiled, response_sha256="a" * 64
    )
    probability = 0.6 * 0.8 * 0.5
    gross = draw.views["gross"].array
    expected = 10 * 5 * probability
    standard_error = 10 * math.sqrt(5 * probability * (1 - probability) / simulations)
    assert abs(gross.mean() - expected) < 6 * standard_error
    assert set(np.unique(gross)) <= {10.0 * count for count in range(6)}
    assert audit.realized_member_work == gross.sum() / 10
