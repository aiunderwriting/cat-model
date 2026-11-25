"""The respondent-sampling standard error (``NUM-RESPONDENT-SAMPLING``).

Every expected value below is computed by hand from the stated state EALs and
weights; the jackknife is an independent brute-force check of the same
variance.
"""

import math
from types import SimpleNamespace

import pytest

from catmodel.contracts.incident import JointState, counts_lab_loss
from catmodel.results.respondent_sampling import (
    NO_STANDARD_ERROR,
    SampleGroups,
    borrowed_reason,
    quadrature_sampling,
    respondent_sampling,
    row_sampling,
    sample_groups,
    standard_error,
)

# Two nodes, two models, three samples each. A model's share of a node is a
# half, split equally among its samples.
NODE_WEIGHTS = {"n1": 0.4, "n2": 0.6}
EALS = {
    ("n1", "A"): (1.0, 2.0, 6.0),  # mean 3, s^2 7
    ("n1", "B"): (10.0, 10.0, 16.0),  # mean 12, s^2 12
    ("n2", "A"): (3.0, 5.0, 7.0),  # mean 5, s^2 4
    ("n2", "B"): (0.0, 3.0, 9.0),  # mean 4, s^2 21
}
GROUPS = {
    f"{node}:{model}:{index}": f"{node}:{model}"
    for (node, model), values in EALS.items()
    for index in range(len(values))
}
VALUES = {
    f"{node}:{model}:{index}": value
    for (node, model), values in EALS.items()
    for index, value in enumerate(values)
}
SAMPLES = SampleGroups(GROUPS)
"""The groups with no state that reads a donor's answers."""


def _weights(node_weights, kept=VALUES):
    """Each kept state's weight: its node's weight, a half for its model,
    split equally among the model's kept samples there."""
    counts = {}
    for state in kept:
        counts[GROUPS[state]] = counts.get(GROUPS[state], 0) + 1
    return {
        state: node_weights[GROUPS[state].split(":")[0]] * 0.5 / counts[GROUPS[state]]
        for state in kept
    }


def _terms(weights, values=VALUES):
    return {state: (weight, values[state]) for state, weight in weights.items()}


def _jackknife(weights, values=VALUES):
    """The stratified delete-one jackknife variance of sum(weight * value):
    each sample left out in turn, its group's weight renormalized over the
    rest as an exclusion does (POOL-EXCLUDED-SAMPLES)."""
    members = {}
    for state in weights:
        members.setdefault(GROUPS[state], []).append(state)
    total = 0.0
    for states in members.values():
        mass = sum(weights[state] for state in states)
        n = len(states)
        estimates = []
        for left_out in states:
            rest = [state for state in states if state != left_out]
            estimates.append(mass * sum(values[state] for state in rest) / (n - 1))
        mean = sum(estimates) / n
        total += (n - 1) / n * sum((value - mean) ** 2 for value in estimates)
    return total


def test_level_standard_error_matches_hand_calculation_and_jackknife():
    weights = _weights(NODE_WEIGHTS)
    error, reason = standard_error(SAMPLES, _terms(weights), {})
    assert reason is None
    # E = 0.2*3 + 0.2*12 + 0.3*5 + 0.3*4 = 5.7, and
    # SE^2 = (0.2^2*7 + 0.2^2*12 + 0.3^2*4 + 0.3^2*21) / 3 = 3.01 / 3.
    # A weight to the first power gives 11.3 / 3, the population variance
    # (ddof 0) 3.01 * 2/3 / 3.
    assert error == pytest.approx(math.sqrt(3.01 / 3), rel=1e-12)
    assert error is not None
    assert error**2 == pytest.approx(_jackknife(weights), rel=1e-12)


def test_paired_difference_of_a_node_reweighting():
    reference = _weights(NODE_WEIGHTS)
    forecast = _weights({"n1": 0.7, "n2": 0.3})
    error, reason = standard_error(SAMPLES, _terms(forecast), _terms(reference))
    assert reason is None
    # Each group's weight moves by +-0.15, so SE^2 = 0.15^2 (7+12+4+21) / 3.
    # Unpaired, the arms' variances would add to (3.01 + 2.89) / 3, the
    # forecast's at group weights 0.35 and 0.15.
    assert error == pytest.approx(math.sqrt(0.0225 * 44 / 3), rel=1e-12)


def test_paired_difference_of_a_law_change_on_the_same_states():
    reference = _weights(NODE_WEIGHTS)
    doubled = {
        state: value * (2 if GROUPS[state] == "n1:A" else 1)
        for state, value in VALUES.items()
    }
    error, reason = standard_error(
        SAMPLES, _terms(reference, doubled), _terms(reference)
    )
    assert reason is None
    # Only n1:A changes, by its own EALs: SE^2 = 0.2^2 * 7 / 3.
    assert error == pytest.approx(math.sqrt(0.04 * 7 / 3), rel=1e-12)


def test_an_arm_identical_to_its_control_has_zero_difference_and_error():
    reference = _terms(_weights(NODE_WEIGHTS))
    assert standard_error(SAMPLES, reference, dict(reference)) == (0.0, None)


def test_a_group_only_one_arm_weights_enters_as_its_own_group():
    reference = _weights(NODE_WEIGHTS)
    # Like the no-progress forecast: every weight on node n2.
    point = _weights({"n1": 0.0, "n2": 1.0})
    point = {state: weight for state, weight in point.items() if weight > 0}
    error, reason = standard_error(SAMPLES, _terms(point), _terms(reference))
    assert reason is None
    # n1's groups lose 0.2 each; n2's move from 0.3 to 0.5.
    assert error == pytest.approx(
        math.sqrt((0.04 * 7 + 0.04 * 12 + 0.04 * 4 + 0.04 * 21) / 3), rel=1e-12
    )


def test_exclusion_reweighting_uses_the_kept_samples():
    # n1:A's third sample is excluded; the model keeps its half of n1,
    # split between the two kept samples.
    kept = {state: value for state, value in VALUES.items() if state != "n1:A:2"}
    weights = _weights(NODE_WEIGHTS, kept)
    assert weights["n1:A:0"] == pytest.approx(0.1)
    error, reason = standard_error(SAMPLES, _terms(weights), {})
    assert reason is None
    # n1:A: mean 1.5, s^2 0.5, n 2: SE^2 = 0.2^2*0.5/2 + (0.2^2*12 + 0.3^2*4
    # + 0.3^2*21) / 3 = 0.01 + 0.91. Counting the roster's three samples
    # instead of the two kept gives 0.2^2*0.5/3 for n1:A.
    assert error == pytest.approx(math.sqrt(0.92), rel=1e-12)
    assert error is not None
    assert error**2 == pytest.approx(_jackknife(weights, kept), rel=1e-12)


def test_fewer_than_two_samples_unequal_weights_and_unpaired_samples_are_unavailable():
    weights = _weights(NODE_WEIGHTS)
    single = {state: weights[state] for state in ("n1:A:0", "n2:B:0", "n2:B:1")}
    error, reason = standard_error(SAMPLES, _terms(single), {})
    assert error is None
    assert reason == (
        "Group 'n1:A' holds one weighted sample; a sample variance needs at least two."
    )
    unequal = dict(weights)
    unequal["n1:A:0"] *= 1.5
    assert standard_error(SAMPLES, _terms(unequal), {})[1] == (
        "Group 'n1:A': its samples carry unequal weights, so the EAL is not an "
        "equal-weight mean of independent samples."
    )
    other = {state: weight for state, weight in weights.items() if state != "n2:B:2"}
    assert standard_error(SAMPLES, _terms(other), _terms(weights))[1] == (
        "Group 'n2:B': the arms weight different samples, so their difference does not pair."
    )
    assert standard_error(SampleGroups({}), _terms(weights), {})[1] is not None


def _draws(values):
    return {
        state: SimpleNamespace(
            views={"welfare": SimpleNamespace(native_mean=SimpleNamespace(value=value))}
        )
        for state, value in values.items()
    }


def _metric(weights, values=VALUES):
    return {
        "status": "available",
        "expected_annual_loss": math.fsum(
            weight * values[state] for state, weight in weights.items()
        ),
    }


def test_the_exhibit_reports_each_eal_and_each_paired_change():
    reference = _weights(NODE_WEIGHTS)
    forecast = _weights({"n1": 0.7, "n2": 0.3})
    single = dict.fromkeys(reference, 0.0) | {"n1:A:0": 0.5, "n2:A:0": 0.5}
    report = {
        "arms": {
            "reference": {"metrics": {"welfare": _metric(reference)}},
            "forecast": {"metrics": {"welfare": _metric(forecast)}},
            "single": {"metrics": {"welfare": _metric(single)}},
        },
        "comparison_reference_arms": {
            "forecast": "reference",
            "single": "reference",
        },
    }
    draws = _draws(VALUES)
    exhibit = respondent_sampling(
        report,
        {"reference": draws, "forecast": draws, "single": draws},
        {"reference": reference, "forecast": forecast, "single": single},
        SAMPLES,
        reference_arm="reference",
    )
    assert exhibit["status"] == "available"
    assert (exhibit["groups"], exhibit["samples"], exhibit["smallest_group"]) == (
        4,
        12,
        3,
    )
    level = exhibit["views"]["welfare"]
    # The exhibit's EAL is the reported EAL itself.
    assert (
        level["expected_annual_loss"]
        == report["arms"]["reference"]["metrics"]["welfare"]["expected_annual_loss"]
    )
    assert level["expected_annual_loss"] == pytest.approx(5.7, rel=1e-12)
    assert level["sampling_standard_error"] == pytest.approx(
        math.sqrt(3.01 / 3), rel=1e-12
    )
    assert level["relative_sampling_standard_error"] == pytest.approx(
        math.sqrt(3.01 / 3) / 5.7, rel=1e-12
    )
    change = exhibit["comparisons"]["forecast"]["welfare"]
    assert change["control_arm"] == "reference"
    # 0.35*(3+12) + 0.15*(5+4) - 5.7 = 0.9.
    assert change["absolute_difference"] == pytest.approx(0.9, rel=1e-12)
    error = math.sqrt(0.0225 * 44 / 3)
    assert change["sampling_standard_error"] == pytest.approx(error, rel=1e-12)
    assert change["sampling_z"] == pytest.approx(0.9 / error, rel=1e-12)
    # An arm weighting one sample of a group the reference weights fully does
    # not pair with it.
    assert exhibit["comparisons"]["single"]["welfare"] == {
        "status": "unavailable",
        "control_arm": "reference",
        "reason": "Group 'n1:A': the arms weight different samples, so their difference does not pair.",
    }


def test_the_exhibit_withholds_simulated_means_and_non_respondent_setups():
    reference = _weights(NODE_WEIGHTS)
    report = {
        "arms": {"reference": {"metrics": {"welfare": _metric(reference)}}},
        "comparison_reference_arms": {},
    }
    draws = _draws(VALUES)
    draws["n1:A:0"].views["welfare"].native_mean = None
    exhibit = respondent_sampling(
        report,
        {"reference": draws},
        {"reference": reference},
        SAMPLES,
        reference_arm="reference",
    )
    assert exhibit["views"]["welfare"]["status"] == "unavailable"
    assert "not an analytic mean" in exhibit["views"]["welfare"]["reason"]
    assert respondent_sampling(
        report,
        {"reference": draws},
        {"reference": reference},
        SampleGroups(dict.fromkeys(GROUPS)),
        reference_arm="reference",
    ) == {
        "register_id": "NUM-RESPONDENT-SAMPLING",
        "status": "unavailable",
        "reason": "No state of the setup is a respondent sample.",
    }


# A state left out of the views that count the lab's own loss, the lab-loss
# views and the gross views (POOL-EXCLUDED-SAMPLES): the third sample of model
# A at node n1.
LAB = "lab_loss::lab"
COUNTING = (
    LAB,
    "source_status::public::lab_loss::lab",
    "gross",
    "scenario::x::gross",
)
NOT_COUNTING = (
    "welfare",
    "net_resource_cost",
    "oep_gross",
    "scenario::x::welfare",
    "scenario::lab_loss::welfare",
    "scenario::gross::net_resource_cost",
    "source_status::public::welfare",
)
LEFT_OUT = SampleGroups(GROUPS, left_out=frozenset({"n1:A:2"}))
NO_STATE_LEFT_REASON = (
    "Every state with weight in sampling group 'n1:A' is left out of the views "
    "that count the lab's own loss, because a case it reads declares that loss "
    "unavailable (POOL-EXCLUDED-SAMPLES), so no state carries the group's weight "
    "in them."
)


def test_the_views_that_count_the_lab_loss_are_the_lab_loss_and_gross_views():
    assert all(map(counts_lab_loss, COUNTING))
    assert not any(map(counts_lab_loss, NOT_COUNTING))


def test_a_view_that_counts_the_lab_loss_weighs_a_left_out_state_zero():
    weights = _weights(NODE_WEIGHTS)
    by_view = LEFT_OUT.view_weights(weights)
    # Every view that counts no lab's loss keeps the arm's weights themselves.
    for view in NOT_COUNTING:
        assert by_view.of(view) is weights
    # The lab-loss views and the gross views share one set of weights.
    lab = by_view.of(LAB)
    assert lab is not None and all(by_view.of(view) is lab for view in COUNTING)
    # n1:A's 0.2 is split between its two other samples, 0.1 each, as an
    # exclusion would split it; no other state's weight moves at all.
    assert lab["n1:A:2"] == 0.0
    kept = {state: value for state, value in VALUES.items() if state != "n1:A:2"}
    assert {state: weight for state, weight in lab.items() if weight} == (
        pytest.approx(_weights(NODE_WEIGHTS, kept), rel=1e-15)
    )
    assert {state: lab[state] for state in lab if GROUPS[state] != "n1:A"} == {
        state: weights[state] for state in weights if GROUPS[state] != "n1:A"
    }
    assert math.fsum(lab.values()) == pytest.approx(1.0, rel=1e-15)
    assert by_view.differ
    assert by_view.shared(NOT_COUNTING) and by_view.shared(COUNTING)
    assert not by_view.shared(["welfare", LAB])
    assert not by_view.shared(["welfare", "gross"])
    # Where no state with weight is left out, those views use the arm's
    # weights themselves.
    assert SAMPLES.view_weights(weights).of(LAB) is weights
    elsewhere = _weights({"n1": 0.0, "n2": 1.0})
    unmoved = LEFT_OUT.view_weights(elsewhere)
    assert unmoved.of("gross") is elsewhere and not unmoved.differ
    assert unmoved.shared(["welfare", LAB, "gross"])


def test_the_societal_estimate_keeps_a_sample_whose_lab_loss_is_unavailable():
    """Ten samples of one group, one with a societal EAL of 100 and nine of
    1, have an equal-weight mean of 10.9. With the first sample's lab loss
    unavailable, the welfare view still weighs all ten; only the views that
    count the lab's loss weigh the nine others, whose societal mean would be
    1.0."""
    states = [f"s{index}" for index in range(10)]
    groups = SampleGroups(dict.fromkeys(states, "n:A"), left_out=frozenset({"s0"}))
    weights = dict.fromkeys(states, 0.1)
    welfare = dict.fromkeys(states, 1.0) | {"s0": 100.0}
    by_view = groups.view_weights(weights)
    societal, lab = by_view.of("welfare"), by_view.of(LAB)
    assert societal is weights and lab is not None and by_view.of("gross") is lab
    assert math.fsum(societal[state] * welfare[state] for state in states) == (
        pytest.approx(10.9, rel=1e-12)
    )
    assert lab == pytest.approx(dict.fromkeys(states, 1 / 9) | {"s0": 0.0}, rel=1e-12)
    assert math.fsum(lab[state] * welfare[state] for state in states) == (
        pytest.approx(1.0, rel=1e-12)
    )


def test_the_lab_loss_weights_follow_whatever_weights_an_arm_has():
    weights = _weights(NODE_WEIGHTS)
    # Unequal weights in the group: the others share its 0.2 in proportion.
    authored = weights | {"n1:A:0": 0.05, "n1:A:1": 0.10, "n1:A:2": 0.05}
    lab = LEFT_OUT.view_weights(authored).of(LAB)
    assert lab is not None
    assert (lab["n1:A:0"], lab["n1:A:1"], lab["n1:A:2"]) == pytest.approx(
        (0.2 / 3, 0.4 / 3, 0.0), rel=1e-12
    )
    # An arm that weighs only the left-out state of the group has no state to
    # carry the group's weight: its lab-loss and gross views are unavailable,
    # with the reason, and its other views keep its weights.
    alone = weights | {"n1:A:0": 0.0, "n1:A:1": 0.0, "n1:A:2": 0.2}
    by_view = LEFT_OUT.view_weights(alone)
    for view in COUNTING:
        assert by_view.of(view) is None
        assert by_view.reason(view) == NO_STATE_LEFT_REASON
    assert by_view.of("welfare") is alone and by_view.reason("welfare") is None
    assert by_view.differ and not by_view.shared(["welfare", LAB])
    assert not by_view.shared([LAB, "gross"])
    # Two groups with a left-out state each are reweighted each within itself.
    both = SampleGroups(GROUPS, left_out=frozenset({"n1:A:2", "n2:B:0"}))
    lab = both.view_weights(weights).of(LAB)
    assert lab is not None
    assert [lab[f"n2:B:{index}"] for index in range(3)] == pytest.approx(
        [0.0, 0.15, 0.15], rel=1e-12
    )
    assert lab["n1:A:0"] == pytest.approx(0.1, rel=1e-12)
    assert lab["n2:A:0"] == weights["n2:A:0"]


def test_a_state_without_the_lab_loss_that_is_not_left_out_withholds_gross():
    """Where leaving samples out would breach an import guard, a state keeps
    its weight without a law of the lab's loss. Its gross loss would omit the
    lab's annual business loss, so the arm reports no gross view; its
    lab-loss views keep the arm's weights and are unavailable where the state
    does not hold them."""
    weights = _weights(NODE_WEIGHTS)
    groups = SampleGroups(GROUPS, without_lab_loss={"n1:A:2": "node:a:2"})
    by_view = groups.view_weights(weights)
    reason = (
        "State 'node:a:2' has weight and holds no law of the lab's own loss, so "
        "its gross loss would omit the lab's annual business loss, and it is not "
        "left out of the views that count that loss (POOL-EXCLUDED-SAMPLES)."
    )
    for view in ("gross", "scenario::x::gross"):
        assert by_view.of(view) is None and by_view.reason(view) == reason
    for view in (*NOT_COUNTING, LAB, "source_status::public::lab_loss::lab"):
        assert by_view.of(view) is weights and by_view.reason(view) is None
    assert not by_view.differ
    assert not by_view.shared(["welfare", "gross"])
    assert by_view.shared(["welfare", LAB])
    # An arm that gives the state no weight reports gross.
    elsewhere = _weights({"n1": 0.0, "n2": 1.0})
    assert groups.view_weights(elsewhere).of("gross") is elsewhere


def test_sample_groups_reads_the_states_left_out_and_those_without_the_lab_loss():
    def state(state_id: str, **fields) -> JointState:
        return JointState.model_construct(
            state_id=state_id,
            sampling_group_id="n:a",
            borrowed_leaves=(),
            **fields,
        )

    def law(rate: float, **unavailable: str) -> SimpleNamespace:
        return SimpleNamespace(
            reference_annual_rate=rate, unavailable_lab_losses=unavailable
        )

    groups = sample_groups(
        [
            state("s0", lab_loss_left_out=True, laws=(law(1.0, lab="No basis"),)),
            state("s1", lab_loss_left_out=False, laws=(law(1.0),)),
            # Not left out, and a law with incidents holds no lab's loss.
            state("s2", lab_loss_left_out=False, laws=(law(1.0), law(2.0, lab="x"))),
            # A law without incidents omits nothing.
            state("s3", lab_loss_left_out=False, laws=(law(0.0, lab="x"),)),
        ],
        lambda state_id: state_id.upper(),
    )
    assert groups.left_out == {"S0"}
    assert groups.group == dict.fromkeys(("S0", "S1", "S2", "S3"), "n:a")
    assert groups.without_lab_loss == {"S2": "s2"}
    # A state left out of the views that count the lab's loss names the group
    # that takes its weight there.
    with pytest.raises(ValueError, match="names the sampling group"):
        JointState(
            state_id="s",
            epistemic_group_id="g",
            weight=1.0,
            entity_paths=(),
            lab_loss_left_out=True,
        )


def _two_view_draws(values=VALUES):
    return {
        state: SimpleNamespace(
            views={
                view: SimpleNamespace(native_mean=SimpleNamespace(value=value))
                for view in ("welfare", "gross", LAB)
            }
        )
        for state, value in values.items()
    }


def test_the_exhibit_gives_a_view_that_counts_the_lab_loss_its_own_standard_error():
    reference = _weights(NODE_WEIGHTS)
    lab = LEFT_OUT.view_weights(reference).of(LAB)
    assert lab is not None
    alone = reference | {"n1:A:0": 0.0, "n1:A:1": 0.0, "n1:A:2": 0.2}
    report = {
        "arms": {
            "reference": {
                "metrics": {
                    "welfare": _metric(reference),
                    "gross": _metric(lab),
                    LAB: _metric(lab),
                }
            },
            "alone": {
                "metrics": dict.fromkeys(("welfare", "gross", LAB), _metric(alone))
            },
        },
        "comparison_reference_arms": {"alone": "reference"},
    }
    draws = _two_view_draws()
    exhibit = respondent_sampling(
        report,
        {"reference": draws, "alone": draws},
        {"reference": reference, "alone": alone},
        LEFT_OUT,
        reference_arm="reference",
    )
    # The welfare view weighs every sample: SE^2 = 3.01 / 3, as above.
    welfare = exhibit["views"]["welfare"]
    assert welfare["expected_annual_loss"] == pytest.approx(5.7, rel=1e-12)
    assert welfare["sampling_standard_error"] == pytest.approx(
        math.sqrt(3.01 / 3), rel=1e-12
    )
    # The lab-loss view and the gross view hold two samples of n1:A at 0.1
    # each: mean 1.5, s^2 0.5, so E = 0.2*1.5 + 0.2*12 + 0.3*5 + 0.3*4 = 5.4
    # and SE^2 = 0.2^2*0.5/2 + (0.2^2*12 + 0.3^2*4 + 0.3^2*21) / 3 = 0.92.
    kept = {state: value for state, value in VALUES.items() if state != "n1:A:2"}
    for view in (LAB, "gross"):
        level = exhibit["views"][view]
        assert level["expected_annual_loss"] == pytest.approx(5.4, rel=1e-12)
        assert level["sampling_standard_error"] == pytest.approx(
            math.sqrt(0.92), rel=1e-12
        )
        assert level["sampling_standard_error"] ** 2 == pytest.approx(
            _jackknife(_weights(NODE_WEIGHTS, kept), kept), rel=1e-12
        )
    # The counts are the arm's samples, not those of the views that count
    # the lab's loss, and the interpretation says so.
    assert (exhibit["groups"], exhibit["samples"]) == (4, 12)
    assert (
        "groups, samples and smallest_group count every sample the reference arm "
        "weighs" in exhibit["interpretation"]
    )
    # An arm with no weights in those views has no standard error of its
    # change in them, with the reason; its welfare change does not pair, since
    # it weighs one sample of a group the reference weighs in full.
    for view in (LAB, "gross"):
        assert exhibit["comparisons"]["alone"][view] == {
            "status": "unavailable",
            "control_arm": "reference",
            "reason": NO_STATE_LEFT_REASON,
        }
    assert exhibit["comparisons"]["alone"]["welfare"]["reason"] == (
        "Group 'n1:A': the arms weight different samples, so their difference "
        "does not pair."
    )


# One node, two models, two samples each. The borrower's model does not
# answer leaf "x", so each of its samples reads a donor sample's answers to it;
# here "x" is the only leaf, and each borrowing state's EAL copies its donor's.
BORROWED_EALS = {"d0": 1.0, "d1": 3.0, "b0": 1.0, "b1": 3.0}
BORROWED_GROUPS = {
    "d0": "n:donor",
    "d1": "n:donor",
    "b0": "n:borrower",
    "b1": "n:borrower",
}
BORROWING = SampleGroups(BORROWED_GROUPS, {"b0": ("b0", ("x",)), "b1": ("b1", ("x",))})
BORROWED_REASON = (
    "State 'b0' reads a donor's answers to leaf 'x' (POOL-UNANSWERED-SCENARIOS), "
    "so the sampling groups are not independent and no standard error is reported."
)
POOLED = dict.fromkeys(BORROWED_EALS, 0.25)
DONORS_ONLY = {"d0": 0.5, "d1": 0.5, "b0": 0.0, "b1": 0.0}


def _positive(weights):
    return _terms(
        {state: weight for state, weight in weights.items() if weight > 0},
        BORROWED_EALS,
    )


def test_a_borrowed_answer_withholds_the_standard_error():
    """The four states hold two independent answers, not four: the EAL is
    0.5 * (1 + 3), the mean of two independent draws with sample variance 2,
    whose standard error is sqrt(2 / 2) = 1.0. The sum over groups treats the
    borrower's copies as a second independent group and gives
    sqrt(2 * 0.5^2 * 2 / 2) = 0.7071, as if copying an answer added samples.
    """
    terms = _positive(POOLED)
    unmarked, reason = standard_error(SampleGroups(BORROWED_GROUPS), terms, {})
    assert reason is None and unmarked is not None
    assert unmarked == pytest.approx(math.sqrt(0.5), rel=1e-12)
    assert round(unmarked, 4) == 0.7071
    assert standard_error(BORROWING, terms, {}) == (None, BORROWED_REASON)
    # The donors alone are two independent draws at a half each.
    assert standard_error(BORROWING, _positive(DONORS_ONLY), {}) == (1.0, None)
    # Several borrowed leaves are all named, under the first borrowing state.
    several = SampleGroups(
        BORROWED_GROUPS, {"b1": ("b1", ("x", "y")), "b0": ("b0", ("x", "y"))}
    )
    assert standard_error(several, terms, {})[1] == (
        "State 'b0' reads a donor's answers to leaves 'x', 'y' "
        "(POOL-UNANSWERED-SCENARIOS), so the sampling groups are not independent "
        "and no standard error is reported."
    )


def test_every_withheld_diagnostic_finds_the_borrowing_state_one_way():
    """``SampleGroups.borrower`` names the first borrowing state among those
    it is given, and ``borrowed_reason`` says what is not reported for it."""
    assert BORROWING.borrower(["d0", "b1", "b0"]) == ("b0", ("x",))
    assert BORROWING.borrower(["d1", "b1"]) == ("b1", ("x",))
    assert BORROWING.borrower(["d0", "d1", "absent"]) is None
    assert SampleGroups(BORROWED_GROUPS).borrower(BORROWED_GROUPS) is None
    assert borrowed_reason("b0", ("x",)) == BORROWED_REASON
    assert borrowed_reason("b0", ("x",), NO_STANDARD_ERROR) == BORROWED_REASON
    assert borrowed_reason("b0", ("x", "y"), "nothing is reported.") == (
        "State 'b0' reads a donor's answers to leaves 'x', 'y' "
        "(POOL-UNANSWERED-SCENARIOS), so nothing is reported."
    )


def test_a_paired_change_is_withheld_where_either_arm_weights_a_borrowing_state():
    pooled, donors = _positive(POOLED), _positive(DONORS_ONLY)
    # The arm weights the borrowing states, or its control does.
    assert standard_error(BORROWING, pooled, donors) == (None, BORROWED_REASON)
    assert standard_error(BORROWING, donors, pooled) == (None, BORROWED_REASON)
    assert standard_error(BORROWING, pooled, dict(pooled)) == (None, BORROWED_REASON)
    # Neither does: the donors' doubled EALs against their own.
    doubled = {state: (weight, 2 * value) for state, (weight, value) in donors.items()}
    assert standard_error(BORROWING, doubled, donors) == (1.0, None)
    # The quadrature indicator's fields carry the same reason.
    assert quadrature_sampling(BORROWING, pooled, donors, 0.0) == {
        "sampling_standard_error": None,
        "sampling_z": None,
        "sampling_reason": BORROWED_REASON,
    }


def _borrowing_exhibit(reference, arms):
    """The exhibit of ``arms`` (weights by arm) against ``reference``."""
    draws = _draws(BORROWED_EALS)
    weights = {"reference": reference, **arms}
    report = {
        "arms": {
            arm: {"metrics": {"welfare": _metric(row, BORROWED_EALS)}}
            for arm, row in weights.items()
        },
        "comparison_reference_arms": dict.fromkeys(arms, "reference"),
    }
    return respondent_sampling(
        report,
        dict.fromkeys(weights, draws),
        weights,
        BORROWING,
        reference_arm="reference",
    )


def test_a_borrowing_state_at_zero_weight_withholds_nothing():
    exhibit = _borrowing_exhibit(DONORS_ONLY, {"pooled": POOLED})
    # The reference weights the donors alone, at a half each.
    assert exhibit["views"]["welfare"] == {
        "status": "available",
        "expected_annual_loss": 2.0,
        "sampling_standard_error": 1.0,
        "relative_sampling_standard_error": 0.5,
    }
    # The arm that weights the borrowing states has no standard error of its
    # change, and a tornado row takes the reason.
    assert exhibit["comparisons"]["pooled"]["welfare"] == {
        "status": "unavailable",
        "control_arm": "reference",
        "reason": BORROWED_REASON,
    }
    assert row_sampling(exhibit, "pooled", "reference", "welfare", 0.0) == {
        "sampling_standard_error": None,
        "sampling_z": None,
        "sampling_reason": BORROWED_REASON,
    }


def test_the_exhibit_withholds_a_level_and_a_change_that_read_borrowed_answers():
    exhibit = _borrowing_exhibit(POOLED, {"donors": DONORS_ONLY})
    assert exhibit["status"] == "available"
    assert (exhibit["groups"], exhibit["samples"]) == (2, 4)
    assert exhibit["views"]["welfare"] == {
        "status": "unavailable",
        "reason": BORROWED_REASON,
    }
    # The arm weights the donors alone; its control weights the borrowers.
    assert exhibit["comparisons"]["donors"]["welfare"] == {
        "status": "unavailable",
        "control_arm": "reference",
        "reason": BORROWED_REASON,
    }


def _state(state_id, group, borrowed_leaves=()):
    return JointState(
        state_id=state_id,
        epistemic_group_id=state_id,
        sampling_group_id=group,
        borrowed_leaves=borrowed_leaves,
        weight=0.0,
        entity_paths=(),
    )


def test_sample_groups_reads_each_states_group_and_borrowed_leaves():
    states = [
        _state("d0", "n:donor"),
        _state("b0", "n:borrower", ("x",)),
        _state("fixed", None),
    ]
    # A state that reads only its own answers serializes as it did without
    # the field, and a leaf is listed once.
    assert [
        state.model_dump(mode="json").get("borrowed_leaves") for state in states
    ] == [None, ["x"], None]
    with pytest.raises(ValueError, match="borrowed leaves must be unique"):
        _state("b1", "n:borrower", ("x", "x"))
    assert sample_groups(states) == SampleGroups(
        {"d0": "n:donor", "b0": "n:borrower", "fixed": None}, {"b0": ("b0", ("x",))}
    )
    # Under another key the reason still names the state by its ID.
    keyed = sample_groups(states, key=str.upper)
    assert keyed.group == {"D0": "n:donor", "B0": "n:borrower", "FIXED": None}
    assert keyed.borrowers == {"B0": ("b0", ("x",))}
