"""Event-addressed campaign episodes: their law, ranks and addresses."""

import math

import numpy as np
import pytest
from scipy.stats import binom

from catmodel.contracts.incident import ClusterCount, MarkKernel
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.probability.distributions import Distribution
from catmodel.simulation.incident_campaigns import (
    campaign_law,
    cluster_counts,
    draw_episodes,
    episode_members,
)

SEED, STATE, ORIGIN = 11, "state", "origin"
SEVERITY = native_severity_from_sampler(Distribution(1, 3, 10, 40, 100)).model_dump(
    mode="json"
)


def _component(identifier, *, scope="member", group="rank", amount=None):
    if amount is not None:
        return {"component_id": identifier, "amount": amount, "nature": "transfer"}
    return {
        "component_id": identifier,
        "severity": SEVERITY,
        "rank_group_id": group,
        "rank_scope": scope,
        "nature": "transfer",
    }


def _kernel(**fields):
    return MarkKernel.model_validate(
        {
            "shared_probability": 1,
            "unit_probability": 1,
            "gate_meaning": "Fixture gates.",
            "selection_assumption": "Fixture selection.",
            "displacement_assumption": "No displacement.",
            **fields,
        }
    )


def _events(count, *, years=1):
    """``count`` events in each of ``years`` years, by year and ordinal."""
    return (
        np.repeat(np.arange(years, dtype=np.int64), count),
        np.tile(np.arange(count, dtype=np.int64), years),
    )


def _draw(kernel, count, *, payments=False, max_members=10**6, **events):
    years, ordinals = _events(count, **events)
    return draw_episodes(
        campaign_law(kernel),
        years,
        ordinals,
        payments=payments,
        max_members=max_members,
        root_seed=SEED,
        state_id=STATE,
        origin_id=ORIGIN,
    )


def test_cluster_members_follow_the_count_law_thinned_by_the_unit_gate():
    kernel = _kernel(
        cluster={
            "family": "finite",
            "support": [
                {"count": 0, "probability": 0.2},
                {"count": 3, "probability": 0.5},
                {"count": 5, "probability": 0.3},
            ],
        },
        shared_probability=0.8,
        unit_probability=0.4,
        member_outcomes=[{"probability": 1, "components": [_component("harm")]}],
    )
    count = 100_000
    potential, realized = episode_members(
        campaign_law(kernel),
        *_events(count),
        root_seed=SEED,
        state_id=STATE,
        origin_id=ORIGIN,
    )
    # The shared gate closes a fifth of the episodes; the count law then gives
    # 0, 3 or 5 potential members, each kept with probability 0.4.
    support = {0: 0.2 + 0.8 * 0.2, 3: 0.8 * 0.5, 5: 0.8 * 0.3}
    for value, probability in support.items():
        share = np.mean(potential == value)
        assert abs(share - probability) < 5 * math.sqrt(probability / count)
    for trials in (3, 5):
        kept = realized[potential == trials]
        expected = binom.pmf(np.arange(trials + 1), trials, 0.4)
        observed = np.bincount(kept, minlength=trials + 1) / len(kept)
        assert np.all(np.abs(observed - expected) < 5 * np.sqrt(expected / len(kept)))


def test_a_finite_cluster_count_is_inverted_in_increasing_count_order():
    uniforms = np.random.default_rng(3).random(200_000)
    cluster = ClusterCount.model_validate(
        {
            "family": "finite",
            "support": [
                {"count": 4, "probability": 0.4},
                {"count": 3, "probability": 0.6},
            ],
        }
    )
    # The least count whose CDF reaches the uniform.
    np.testing.assert_array_equal(
        cluster_counts(cluster, uniforms), np.where(uniforms < 0.6, 3, 4)
    )


def test_member_outcomes_are_chosen_by_their_probabilities():
    kernel = _kernel(
        cluster={"family": "finite", "support": [{"count": 4, "probability": 1}]},
        member_outcomes=[
            {"probability": 0.25, "components": [_component("small", amount=1)]},
            {"probability": 0.0, "components": [_component("never", amount=7)]},
            {"probability": 0.75, "components": [_component("large", amount=2)]},
        ],
    )
    episodes = _draw(kernel, 20_000)
    law = campaign_law(kernel)
    rows = {draws.law: len(draws.event) for draws in episodes.marks}
    small, never, large = law.outcomes
    assert never not in rows
    share = rows[small] / (rows[small] + rows[large])
    assert abs(share - 0.25) < 5 * math.sqrt(0.25 * 0.75 / 80_000)
    assert np.all(episodes.realized == 4)


def test_campaign_ranks_are_shared_and_member_ranks_are_each_members_own():
    kernel = _kernel(
        cluster={"family": "finite", "support": [{"count": 3, "probability": 1}]},
        campaign_components=[_component("response", scope="campaign", group="wave")],
        member_outcomes=[
            {
                "probability": 1,
                "components": [
                    _component("shared", scope="campaign", group="wave"),
                    _component("own", scope="member", group="unit"),
                ],
            }
        ],
    )
    episodes = _draw(kernel, 50)
    campaign, members = episodes.marks
    assert np.all(campaign.member == -1)
    shared, own = members.amounts
    # Every member reads its episode's campaign rank, as the campaign does.
    np.testing.assert_array_equal(shared, campaign.amounts[0][members.event])
    # Members' own ranks differ within an episode.
    by_event = own.reshape(50, 3)
    assert np.all(np.ptp(by_event, axis=1) > 0)


def test_an_episodes_draws_do_not_depend_on_its_batch():
    kernel = _kernel(
        explicit_joint_members=[
            {
                "probability": 0.3,
                "members": [{"probability": 1, "components": [_component("a")]}],
                "campaign_components": [_component("fixed", amount=5)],
            },
            {
                "probability": 0.7,
                "members": [
                    {"probability": 1, "components": [_component("b")]},
                    {"probability": 1, "components": [_component("c")]},
                ],
            },
        ],
        unit_probability=0.6,
    )
    law = campaign_law(kernel)
    years, ordinals = _events(40, years=3)
    whole = draw_episodes(
        law,
        years,
        ordinals,
        max_members=10,
        root_seed=SEED,
        state_id=STATE,
        origin_id=ORIGIN,
    )
    order = np.random.default_rng(1).permutation(len(years))
    part = order[:50]
    alone = draw_episodes(
        law,
        years[part],
        ordinals[part],
        max_members=10,
        root_seed=SEED,
        state_id=STATE,
        origin_id=ORIGIN,
    )
    for kind in {draws.law for draws in alone.marks}:
        left = next(draws for draws in whole.marks if draws.law == kind)
        right = next(draws for draws in alone.marks if draws.law == kind)
        expected = {
            (int(e), int(m)): tuple(float(a[i]) for a in left.amounts)
            for i, (e, m) in enumerate(zip(left.event, left.member, strict=True))
        }
        for i, (e, m) in enumerate(zip(right.event, right.member, strict=True)):
            assert expected[(int(part[e]), int(m))] == tuple(
                float(a[i]) for a in right.amounts
            )
    np.testing.assert_array_equal(alone.realized, whole.realized[part])


def test_joint_branch_members_pass_independent_unit_gates():
    kernel = _kernel(
        explicit_joint_members=[
            {
                "probability": 1,
                "members": [
                    {"probability": 1, "components": [_component(name, amount=1)]}
                    for name in ("a", "b", "c", "d")
                ],
            }
        ],
        unit_probability=0.5,
    )
    episodes = _draw(kernel, 40_000)
    # A binomial thinning of exchangeable members: each member position is
    # kept half the time, and an episode keeps Binomial(4, 1/2) members.
    observed = np.bincount(episodes.realized, minlength=5) / 40_000
    expected = binom.pmf(np.arange(5), 4, 0.5)
    assert np.all(np.abs(observed - expected) < 5 * np.sqrt(expected / 40_000))
    positions = np.concatenate([draws.member for draws in episodes.marks])
    assert np.all(np.abs(np.bincount(positions) / 40_000 - 0.5) < 0.01)


def test_an_episode_over_the_member_cap_is_refused_before_its_members_are_drawn():
    kernel = _kernel(
        cluster={"family": "finite", "support": [{"count": 10**9, "probability": 1}]},
        member_outcomes=[{"probability": 1, "components": [_component("harm")]}],
    )
    with pytest.raises(ValueError, match="exceed max_members_per_campaign"):
        _draw(kernel, 2, max_members=10**6)
