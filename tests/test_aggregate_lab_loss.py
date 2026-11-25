"""Episode lab amounts add to component allocations once, under one episode cap."""

import math
from dataclasses import replace

import numpy as np
import pytest

from catmodel.contracts.components import AggregateLabLossLaw
from catmodel.contracts.incident import ExecutionPlan, MarkKernel
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.probability.distributions import Distribution
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_campaigns import campaign_law, draw_episodes
from catmodel.simulation.incident_moments import (
    analytic_known_zero_views,
    analytic_linear_eals,
)
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import _financial_study, study_payload

LAB = "lab_loss::lab"


def _payload(aggregate=None, *, members=3, weights=None):
    payload = study_payload()
    payload["entities"].append(
        {
            "entity_id": "lab",
            "kind": "lab",
            "definition": "Synthetic named lab",
            "display_name": "Synthetic lab",
            "status": "hypothetical",
        }
    )
    law = payload["joint_states"][0]["laws"][0]
    law["reference_annual_rate"] = 2
    kernel = law["segments"][0]["kernel"]
    kernel["cluster"]["support"][0]["count"] = members
    kernel["member_outcomes"][0]["components"][0]["lab_loss_weights"] = (
        {"lab": 1} if weights is None else weights
    )
    if aggregate is not None:
        kernel["campaign_lab_losses"] = [aggregate.model_dump(mode="json")]
    return payload


def _compiled(aggregate=None, *, members=3, weights=None):
    return compile_incident_state(
        setup_from_payload(_payload(aggregate, members=members, weights=weights)),
        "state",
    )


def _plan(*, views=("gross", "welfare", LAB), years=128):
    return ExecutionPlan(
        plan_id="synthetic-aggregate-oracle",
        required_arms=("reference",),
        simulations=years,
        seed=56,
        requested_views=views,
    )


def test_once_per_episode_adds_to_all_member_allocations_and_preserves_harm():
    before = _compiled()
    after = _compiled(AggregateLabLossLaw(lab_id="lab", amount=1000))
    plan = _plan()
    old, _, _ = simulate_incident_state(plan, before, response_sha256="a" * 64)
    new, reasons, audit = simulate_incident_state(plan, after, response_sha256="a" * 64)
    assert not reasons
    np.testing.assert_array_equal(
        new.annual_losses["gross"], old.annual_losses["gross"]
    )
    np.testing.assert_array_equal(
        new.annual_losses["welfare"], old.annual_losses["welfare"]
    )
    # Exactly three USD10 physical members per origin, all borne by the lab,
    # plus one USD1000 episode amount.
    np.testing.assert_allclose(
        new.annual_losses[LAB], new.annual_losses["gross"] / 30 * 1030, rtol=1e-15
    )
    np.testing.assert_array_equal(old.annual_losses[LAB], old.annual_losses["gross"])
    assert analytic_linear_eals(after, plan.requested_views).values == {
        "gross": 60,
        "welfare": 60,
        LAB: 2060,
    }
    total = audit.total_initiators
    assert isinstance(total, int) and total > 0


def test_episode_lab_loss_requires_identified_composition_and_allocations():
    payload = _payload(AggregateLabLossLaw(lab_id="lab", amount=500))
    outcome = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]
    outcome["components"][0].pop("lab_loss_weights")
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    draws, reasons, _ = simulate_incident_state(
        _plan(), compiled, response_sha256="a" * 64
    )
    assert LAB not in draws.annual_losses and LAB in reasons
    assert "welfare" in draws.annual_losses


def test_episode_cap_bounds_allocations_plus_amount_and_never_other_views():
    plan = _plan()
    uncapped = _compiled(AggregateLabLossLaw(lab_id="lab", amount=1000))
    binding = _compiled(
        AggregateLabLossLaw(lab_id="lab", amount=1000, episode_cap=1010)
    )
    slack = _compiled(AggregateLabLossLaw(lab_id="lab", amount=1000, episode_cap=1030))
    old, _, _ = simulate_incident_state(plan, uncapped, response_sha256="a" * 64)
    new, reasons, _ = simulate_incident_state(plan, binding, response_sha256="a" * 64)
    assert not reasons
    for view in ("gross", "welfare"):
        np.testing.assert_array_equal(new.annual_losses[view], old.annual_losses[view])
    np.testing.assert_array_equal(
        new.annual_losses[LAB], new.annual_losses["gross"] / 30 * 1010
    )
    # A cap that can bind makes the lab view nonlinear: no analytic mean.
    assert LAB not in analytic_linear_eals(binding, plan.requested_views).values
    assert new.views[LAB].native_mean is None
    assert analytic_linear_eals(slack, plan.requested_views).values[LAB] == 2060


def test_explicit_zero_and_unavailable_remain_distinct():
    compiled = _compiled(AggregateLabLossLaw(lab_id="lab", amount=0), weights={})
    draws, reasons, _ = simulate_incident_state(
        _plan(), compiled, response_sha256="a" * 64
    )
    assert not reasons
    assert np.all(draws.annual_losses[LAB] == 0)
    assert LAB in analytic_known_zero_views(compiled, (LAB,))
    unavailable = replace(
        compiled, unavailable_views={LAB: "Aggregate answer unavailable"}
    )
    draws, reasons, _ = simulate_incident_state(
        _plan(), unavailable, response_sha256="a" * 64
    )
    assert LAB not in draws.annual_losses
    assert reasons[LAB] == "Aggregate answer unavailable"
    assert LAB not in analytic_linear_eals(unavailable, (LAB,)).values


def test_lab_draws_do_not_shift_member_or_payment_draws():
    """An episode's aggregate lab amount has its own address, so adding one
    leaves every member count, amount and payment share unchanged."""
    study = _financial_study(shape=0.3)
    compiled = compile_incident_state(study, "low")
    incident = compiled.incidents[0]
    kernel = incident.segments[0].kernel
    aggregate = AggregateLabLossLaw(
        lab_id="lab",
        severity=native_severity_from_sampler(Distribution(100, 200, 300, 400, 500)),
    )
    expanded = MarkKernel.model_validate(
        {**kernel.model_dump(), "campaign_lab_losses": [aggregate.model_dump()]}
    )

    def draw(selected):
        return draw_episodes(
            campaign_law(selected),
            np.repeat(np.arange(8, dtype=np.int64), 8),
            np.tile(np.arange(8, dtype=np.int64), 8),
            root_seed=8,
            state_id=compiled.state_id,
            origin_id=incident.origin_id,
            payments=True,
            max_members=100,
        )

    old, new = draw(kernel), draw(expanded)
    assert not old.labs and np.all(new.labs["lab"] > 0)
    assert np.array_equal(old.potential, new.potential)
    assert np.array_equal(old.realized, new.realized)
    assert old.realized.sum() > 0
    assert len(old.marks) == len(new.marks)
    for before, after in zip(old.marks, new.marks, strict=True):
        assert before.law == after.law
        for name in ("event", "member", "totals"):
            assert np.array_equal(getattr(before, name), getattr(after, name))
        for left, right in zip(
            (*before.amounts, *before.payments),
            (*after.amounts, *after.payments),
            strict=True,
        ):
            assert (left is None and right is None) or np.array_equal(left, right)


@pytest.mark.parametrize("shape", [0.6, 1.2])
def test_native_mean_and_moment_admission_inspect_the_aggregate(shape):
    severity = native_severity_from_sampler(
        Distribution(1, 2, 3, 4, 5, gpd_shape=shape, gpd_scale=1)
    )
    compiled = _compiled(AggregateLabLossLaw(lab_id="lab", severity=severity))
    draw, reasons, _ = simulate_incident_state(
        _plan(), compiled, response_sha256="a" * 64
    )
    assert not reasons
    assert draw.views["gross"].variance_exists
    assert not draw.views[LAB].variance_exists
    assert draw.views[LAB].mean_exists == (shape < 1)
    # Two episodes a year, each with three USD10 lab-borne members.
    if shape < 1:
        assert analytic_linear_eals(compiled, (LAB,)).values[LAB] == pytest.approx(
            60
            + 2
            * Distribution(1, 2, 3, 4, 5, gpd_shape=shape, gpd_scale=1).analytic_mean()
        )
    else:
        assert LAB not in analytic_linear_eals(compiled, (LAB,)).values

    capped = native_severity_from_sampler(
        Distribution(1, 2, 3, 4, 5, gpd_shape=shape, gpd_scale=1, severity_cap=100)
    )
    capped_compiled = _compiled(AggregateLabLossLaw(lab_id="lab", severity=capped))
    draw, _, _ = simulate_incident_state(
        _plan(), capped_compiled, response_sha256="a" * 64
    )
    assert draw.views[LAB].mean_exists and draw.views[LAB].variance_exists
    assert LAB in analytic_linear_eals(capped_compiled, (LAB,)).values


def test_law_and_setup_refuse_ambiguous_laws_and_invalid_targets():
    for fields in (
        {},
        {
            "amount": 1,
            "severity": native_severity_from_sampler(Distribution(1, 2, 3, 4, 5)),
        },
    ):
        with pytest.raises(ValueError):
            AggregateLabLossLaw.model_validate({"lab_id": "lab", **fields})
    payload = _payload(AggregateLabLossLaw(lab_id="absent", amount=1))
    with pytest.raises(ValueError, match="declared lab entities"):
        setup_from_payload(payload)
    payload = _payload(AggregateLabLossLaw(lab_id="lab", amount=1))
    kernel = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    kernel["campaign_lab_losses"] *= 2
    with pytest.raises(ValueError, match="aggregate lab targets"):
        setup_from_payload(payload)


@pytest.mark.parametrize("gate", ["shared_probability", "unit_probability"])
def test_native_episode_lab_cost_precedes_member_gates(gate):
    # Native campaign costs belong to the initiating episode even when no
    # member succeeds. Contextual qualified-episode kernels use gates of one.
    payload = _payload(AggregateLabLossLaw(lab_id="lab", amount=500))
    kernel = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"]
    kernel[gate] = 0
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    draw, reasons, _ = simulate_incident_state(
        _plan(), compiled, response_sha256="a" * 64
    )
    assert not reasons
    assert np.all(draw.annual_losses["gross"] == 0)
    assert np.any(draw.annual_losses[LAB] > 0)
    assert analytic_linear_eals(compiled, ("gross", LAB)).values == {
        "gross": 0,
        LAB: 1000,
    }


def _status_aggregate_payload(*, mixed=False, shape=None):
    from tests.test_source_status_loss import _payload as source_status_payload

    payload = source_status_payload()
    for index, law in enumerate(payload["joint_states"][0]["laws"]):
        if mixed and index == 1:
            continue
        aggregate = AggregateLabLossLaw(lab_id="lab", amount=100 if index == 0 else 50)
        if shape is not None and index == 0:
            aggregate = AggregateLabLossLaw(
                lab_id="lab",
                severity=native_severity_from_sampler(
                    Distribution(1, 2, 3, 4, 5, gpd_shape=shape, gpd_scale=1)
                ),
            )
        kernel = law["segments"][0]["kernel"]
        kernel["campaign_lab_losses"] = [aggregate.model_dump(mode="json")]
        for component in kernel["member_outcomes"][0]["components"]:
            component["lab_loss_weights"] = {}
    return payload


def _status_views():
    return (
        "welfare",
        LAB,
        *(
            f"source_status::{status}::{measure}"
            for status in ("public", "not_yet_public")
            for measure in ("welfare", LAB)
        ),
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_aggregate_source_status_conserves_paired_rows_and_analytic_means(mixed):
    compiled = compile_incident_state(
        setup_from_payload(_status_aggregate_payload(mixed=mixed)), "state"
    )
    views = _status_views()
    draw, reasons, _ = simulate_incident_state(
        _plan(views=views), compiled, response_sha256="a" * 64
    )
    assert not reasons
    for measure in ("welfare", LAB):
        np.testing.assert_allclose(
            draw.annual_losses[measure],
            draw.annual_losses[f"source_status::public::{measure}"]
            + draw.annual_losses[f"source_status::not_yet_public::{measure}"],
        )
    means = analytic_linear_eals(compiled, views).values
    expected_private = 21 if mixed else 150
    assert means[LAB] == 200 + expected_private
    assert means[f"source_status::public::{LAB}"] == 200
    assert means[f"source_status::not_yet_public::{LAB}"] == expected_private
    assert means["welfare"] == 80
    assert not (set(views) & analytic_known_zero_views(compiled, views))
    for view, mean in means.items():
        native_mean = draw.views[view].native_mean
        assert native_mean is not None and native_mean.value == pytest.approx(mean)


def test_aggregate_moments_are_scoped_to_origin_status_and_unknown_is_unavailable():
    payload = _status_aggregate_payload(mixed=True, shape=1.2)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    views = _status_views()
    draw, reasons, _ = simulate_incident_state(
        _plan(views=views), compiled, response_sha256="a" * 64
    )
    assert not reasons
    public = f"source_status::public::{LAB}"
    private = f"source_status::not_yet_public::{LAB}"
    assert not draw.views[LAB].mean_exists and not draw.views[public].mean_exists
    assert not draw.views[public].variance_exists
    assert draw.views[private].mean_exists and draw.views[private].variance_exists
    means = analytic_linear_eals(compiled, views).values
    assert public not in means and LAB not in means and means[private] == 21

    segment = payload["joint_states"][0]["laws"][0]["segments"][0]
    for field in ("source_public_status", "source_model_id", "source_model_revision"):
        segment.pop(field)
    compiled = compile_incident_state(setup_from_payload(payload), "state")
    draw, reasons, _ = simulate_incident_state(
        _plan(views=views), compiled, response_sha256="a" * 64
    )
    assert LAB in draw.annual_losses
    assert all(view in reasons and view not in draw.annual_losses for view in views[2:])
    assert all(
        view not in analytic_linear_eals(compiled, views).values for view in views[2:]
    )


def _single_member_kernel(aggregate, components, *, cluster=None, shared=1.0):
    return MarkKernel.model_validate(
        {
            "cluster": cluster
            or {"family": "finite", "support": [{"count": 1, "probability": 1}]},
            "shared_probability": shared,
            "unit_probability": 1,
            "campaign_lab_losses": [aggregate.model_dump(mode="json")],
            "member_outcomes": [{"probability": 1, "components": components}],
            "gate_meaning": "One qualified episode.",
            "selection_assumption": "Fixed.",
            "displacement_assumption": "None.",
        }
    )


def _partitioned_total(cap=None):
    return native_severity_from_sampler(
        Distribution(
            1e8,
            1e9,
            5e9,
            2e10,
            1e11,
            gpd_shape=0.7,
            gpd_tail_ratio=10,
            severity_cap=cap,
        )
    ).model_dump(mode="json")


def test_capped_episode_lab_mean_matches_high_precision_simulation():
    from catmodel.contracts.native_severity import native_severity_sampler
    from catmodel.simulation.incident_moments import capped_episode_lab_mean

    total = _partitioned_total(1e14)
    other = native_severity_from_sampler(
        Distribution(1e9, 1e10, 5e10, 2e11, 5e11, gpd_shape=0.7, gpd_tail_ratio=4)
    )
    cap = 1e12
    components = [
        {
            "component_id": share_id,
            "nature": "resource_cost",
            "severity": total,
            "rank_group_id": "shared_total",
            "post_multiplier": share,
            "lab_loss_weights": weights,
        }
        for share_id, share, weights in (
            ("provider", 0.4, {"lab": 1}),
            ("victim", 0.6, {}),
        )
    ] + [
        {
            "component_id": "fine",
            "nature": "transfer",
            "amount": 2e9,
            "lab_loss_weights": {"lab": 1},
        }
    ]
    rng = np.random.default_rng(11)
    draws = 4_000_000
    shared = native_severity_sampler(
        native_severity_from_sampler(
            Distribution(
                1e8,
                1e9,
                5e9,
                2e10,
                1e11,
                gpd_shape=0.7,
                gpd_tail_ratio=10,
                severity_cap=1e14,
            )
        )
    ).sample_from_uniforms(rng.random(draws))
    sampled_other = native_severity_sampler(other).sample_from_uniforms(
        rng.random(draws)
    )
    for aggregate, amounts in (
        (
            AggregateLabLossLaw(lab_id="lab", severity=other, episode_cap=cap),
            sampled_other,
        ),
        (AggregateLabLossLaw(lab_id="lab", amount=3e11, episode_cap=cap), 3e11),
        (AggregateLabLossLaw(lab_id="lab", amount=0, episode_cap=cap), 0.0),
    ):
        result = capped_episode_lab_mean(
            _single_member_kernel(aggregate, components), aggregate
        )
        simulated = np.minimum(0.4 * shared + 2e9 + amounts, cap)
        error = simulated.std() / math.sqrt(draws)
        assert result is not None
        mean, bound = result
        assert abs(mean - simulated.mean()) < 4 * error + bound
        assert 0 < mean < cap


def _integrated_lab_state():
    """Two single-member episodes a year whose capped lab mean is integrated.

    The provider bears the whole GPD-tailed episode loss, and the aggregate lab
    loss's p95 lies far below the USD 1 trillion cap.
    """
    other = native_severity_from_sampler(
        Distribution(
            1e3, 1e4, 3e4, 1e5, 5e5, gpd_shape=0.7, gpd_tail_ratio=4, severity_cap=1e12
        )
    )
    total = native_severity_from_sampler(
        Distribution(1e3, 1e4, 3e4, 1e5, 3e5, gpd_shape=0.5, gpd_tail_ratio=4)
    )
    aggregate = AggregateLabLossLaw(lab_id="lab", severity=other, episode_cap=1e12)
    components = [
        {
            "component_id": "provider",
            "nature": "resource_cost",
            "severity": total.model_dump(mode="json"),
            "rank_group_id": "episode_total",
            "lab_loss_weights": {"lab": 1},
        }
    ]
    payload = _payload()
    payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"] = (
        _single_member_kernel(aggregate, components).model_dump(mode="json")
    )
    return compile_incident_state(setup_from_payload(payload), "state")


def test_replay_accepts_last_bit_platform_differences_and_refuses_real_ones(
    monkeypatch,
):
    """Replay compares certified intervals, and closed forms to 1e-9.

    A last-bit change in each distribution-function value, as between NumPy
    builds, moves the integrated mean's refinement and so its bound by far
    more than 1e-12, but the captured and recomputed certified intervals still
    overlap.
    """
    from catmodel.results.execution import _Claims, _validate_native_analytic_eals
    from catmodel.simulation import incident_moments
    from catmodel.simulation.capability_mixture import NativeMean

    compiled = _integrated_lab_state()
    draw, reasons, _ = simulate_incident_state(
        _plan(years=16), compiled, response_sha256="a" * 64
    )
    assert not reasons
    native = draw.views[LAB].native_mean
    assert native is not None and native.precision is not None
    bound = native.precision["integration_error_bound"]
    assert isinstance(bound, float) and 0 < bound <= 1e-3 * native.value
    _validate_native_analytic_eals(compiled, _Claims.of(draw))

    def captured(view, value, error_bound=None):
        row = draw.views[view].native_mean
        assert row is not None and row.precision is not None
        precision = dict(row.precision)
        if error_bound is not None:
            precision["integration_error_bound"] = error_bound
        return replace(
            draw,
            views={
                **draw.views,
                view: replace(
                    draw.views[view],
                    native_mean=NativeMean(value, row.estimator, precision),
                ),
            },
        )

    original = Distribution.cdf
    rng = np.random.default_rng(0)

    def last_bit(law, values, **options):
        # Move about two thirds of the values one ulp up or down.
        result = original(law, values, **options)
        direction = rng.choice([-np.inf, 0.0, np.inf], size=np.shape(result))
        moved = np.where(direction == 0, result, np.nextafter(result, direction))
        return np.clip(moved, 0, 1)

    monkeypatch.setattr(Distribution, "cdf", last_bit)
    incident_moments._capped_branch_mean.cache_clear()
    try:
        replayed = analytic_linear_eals(compiled, (LAB,))
        assert not math.isclose(
            replayed.error_bounds[LAB], bound, rel_tol=1e-12, abs_tol=0
        )
        _validate_native_analytic_eals(compiled, _Claims.of(draw))
    finally:
        incident_moments._capped_branch_mean.cache_clear()
    monkeypatch.setattr(Distribution, "cdf", original)
    gross = draw.views["gross"].native_mean
    assert gross is not None and (gross.precision or {})["integration_error_bound"] == 0
    _validate_native_analytic_eals(
        compiled, _Claims.of(captured("gross", gross.value * (1 + 1e-10)))
    )
    for changed in (
        # Disjoint certified intervals.
        captured(LAB, native.value + 3 * bound),
        # A captured bound wider than the quadrature tolerance.
        captured(LAB, native.value, 2e-3 * native.value),
        # A closed-form mean beyond floating-point agreement.
        captured("gross", gross.value * (1 + 1e-8)),
        # A closed-form mean captured with an integration bound.
        captured("gross", gross.value, 1e-6 * gross.value),
    ):
        with pytest.raises(ValueError, match="differs from its law-derived"):
            _validate_native_analytic_eals(compiled, _Claims.of(changed))


def test_provider_cells_alone_beyond_the_cap_give_exactly_the_cap():
    from catmodel.simulation.incident_moments import capped_episode_lab_mean

    aggregate = AggregateLabLossLaw(lab_id="lab", amount=0, episode_cap=500)
    components = [
        {
            "component_id": "fine",
            "nature": "transfer",
            "amount": 800,
            "lab_loss_weights": {"lab": 1},
        }
    ]
    kernel = _single_member_kernel(aggregate, components)
    assert capped_episode_lab_mean(kernel, aggregate) == (500, 0.0)
    # Simulated episodes of one USD800 fine each pay the cap alone.
    payload = _payload(aggregate, members=1)
    component = payload["joint_states"][0]["laws"][0]["segments"][0]["kernel"][
        "member_outcomes"
    ][0]["components"][0]
    component.update(amount=800, nature="transfer")
    draws, reasons, _ = simulate_incident_state(
        _plan(),
        compile_incident_state(setup_from_payload(payload), "state"),
        response_sha256="a" * 64,
    )
    assert not reasons and np.any(draws.annual_losses["gross"] > 0)
    np.testing.assert_array_equal(
        draws.annual_losses[LAB], draws.annual_losses["gross"] / 800 * 500
    )


def test_upper_support_covers_explicit_branches_and_cluster_supports():
    from catmodel.simulation.incident_moments import _episode_lab_upper_support

    component = {
        "component_id": "fine",
        "nature": "transfer",
        "amount": 10,
        "lab_loss_weights": {"lab": 1},
    }
    aggregate = AggregateLabLossLaw(lab_id="lab", amount=5, episode_cap=1000)
    single = _single_member_kernel(aggregate, [component])
    assert _episode_lab_upper_support(single, "lab") == 15
    several = _single_member_kernel(
        aggregate,
        [component],
        cluster={
            "family": "finite",
            "support": [
                {"count": 2, "probability": 0.5},
                {"count": 3, "probability": 0.5},
                {"count": 9, "probability": 0},
            ],
        },
    )
    assert _episode_lab_upper_support(several, "lab") == 3 * 10 + 5
    explicit = MarkKernel.model_validate(
        {
            "shared_probability": 1,
            "unit_probability": 1,
            "campaign_lab_losses": [aggregate.model_dump(mode="json")],
            "explicit_joint_members": [
                {
                    "probability": 0.5,
                    "members": [
                        {"probability": 1, "components": [component]},
                        {"probability": 1, "components": [component]},
                    ],
                },
                {
                    "probability": 0.5,
                    "members": [{"probability": 1, "components": [component]}],
                },
            ],
            "gate_meaning": "Explicit members.",
            "selection_assumption": "Fixed.",
            "displacement_assumption": "None.",
        }
    )
    # The larger branch: two members of USD 10 each plus the episode amount.
    assert _episode_lab_upper_support(explicit, "lab") == 25
    gated = _single_member_kernel(aggregate, [component], shared=0)
    assert _episode_lab_upper_support(gated, "lab") == 5


def test_clustered_episodes_keep_a_simulated_lab_mean_with_finite_variance():
    compiled = _compiled(
        AggregateLabLossLaw(lab_id="lab", amount=1000, episode_cap=1010)
    )
    draw, reasons, _ = simulate_incident_state(
        _plan(), compiled, response_sha256="a" * 64
    )
    assert not reasons
    assert draw.views[LAB].native_mean is None
    # A finite cap bounds every moment, so results report sampling precision.
    assert draw.views[LAB].variance_exists
