"""The public forecast integrated over its frontier progress by a fixed rule."""

from datetime import UTC, datetime
from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import fisk, truncnorm

from catmodel.contracts.public_frontier import (
    PublicFrontierQuadrature,
    forecast_times,
    progress_elapsed_years,
    quadrature_node_ids,
)
from catmodel.forecast.public_frontier import fit_public_trend
from catmodel.forecast.public_frontier_quadrature import (
    PROGRESS_CHUNK,
    RATE_SPAN,
    _laws,
    _log_pace_cdf,
    _log_pace_density,
    base_progress_cdf,
    base_progress_quantile,
    fejer_second_kind,
    public_frontier_quadrature,
)
from test_support.public_frontier import (
    SYNTHETIC_RELEASES,
    synthetic_forecast,
    synthetic_release,
)

START = datetime(2027, 1, 1, tzinfo=UTC)
END = datetime(2028, 1, 1, tzinfo=UTC)


def _forecast(**update):
    return synthetic_forecast().model_copy(update={"pace_shape": 5.0, **update})


def _quadrature(forecast=None, nodes=7):
    return public_frontier_quadrature(
        forecast or _forecast(), period_start=START, period_end=END, nodes=nodes
    )


def _middle(forecast):
    return progress_elapsed_years(forecast.as_of, forecast_times(START, END))


def _fixed_rate_forecast():
    """The synthetic forecast with its trend rate fixed at 10 points a year."""
    return _forecast(
        rate_prior=_forecast().rate_prior.model_copy(update={"sd": 0}),
        trend_window_start=_forecast().as_of,
    )


def _sampled_progress(forecast, size, seed, shape=None, cap=None):
    """Independent Monte Carlo of the frontier progress: draw the pooled rate
    and each lab's own pace, then take the frontier's gain.

    A pace is a log-logistic draw kept only if it lies within the cap, which is
    the capped law by its definition and uses none of its formulas. Returns
    the progress, the leading lab and the rate of each path.
    """
    posterior = fit_public_trend(forecast)
    rng = np.random.default_rng(seed)
    sd = np.sqrt(posterior.rate_variance)
    rates = truncnorm.rvs(
        -posterior.rate_location / sd,
        np.inf,
        loc=posterior.rate_location,
        scale=sd,
        size=size,
        random_state=rng,
    )
    anchors = np.asarray(
        [posterior.labs[row.lab_id].as_of_value for row in forecast.coordinates]
    )
    shape, cap = shape or forecast.pace_shape, cap or forecast.pace_cap
    pace = np.full((size, anchors.size), np.inf)
    while (beyond := pace > cap).any():
        uniforms = rng.random(int(beyond.sum()))
        pace[beyond] = np.exp((np.log(uniforms) - np.log1p(-uniforms)) / shape)
    paths = anchors + rates[:, None] * pace * _middle(forecast)
    return paths.max(axis=1) - anchors.max(), paths.argmax(axis=1), rates


def _reference_density(law, progress):
    """The density of the frontier progress by adaptive integration over the
    trend rate, written from the capped law's definition: given the rate, lab
    ``k`` leads at ``progress`` with the capped density of its pace there
    times every other lab's capped CDF."""
    location = law.posterior.rate_location
    sd = law.posterior.rate_variance**0.5
    gaps = np.asarray(law._gaps)
    pace = fisk(c=law.pace_shape)
    below_cap = pace.cdf(law.pace_cap)

    def integrand(rate):
        paces = (progress + gaps) / (rate * law.elapsed)
        cdf = np.minimum(pace.cdf(paces) / below_cap, 1)
        density = np.where(paces <= law.pace_cap, pace.pdf(paces) / below_cap, 0)
        leading = sum(
            density[k] / (rate * law.elapsed) * np.prod(np.delete(cdf, k))
            for k in range(gaps.size)
        )
        return leading * truncnorm.pdf(
            rate, -location / sd, np.inf, loc=location, scale=sd
        )

    # The integrand jumps where a lab's pace meets the cap.
    jumps = (progress + gaps) / (law.pace_cap * law.elapsed)
    edges = sorted(
        {
            max(location - 30 * sd, 1e-9),
            location + 30 * sd,
            *(location + shift * sd for shift in (-5, -1, 1, 5)),
            *jumps,
        }
    )
    edges = [edge for edge in edges if edges[0] <= edge <= location + 30 * sd]
    return sum(
        quad(integrand, low, high, epsabs=0, epsrel=1e-11, limit=200)[0]
        for low, high in pairwise(edges)
    )


def test_fejer_rule_is_interpolatory_and_nested():
    for count in (1, 3, 7, 15):
        nodes, weights = fejer_second_kind(count)
        assert weights.sum() == pytest.approx(1, abs=1e-14)
        for degree in range(count):
            assert weights @ nodes**degree == pytest.approx(1 / (degree + 1))
    seven, three = fejer_second_kind(7)[0], fejer_second_kind(3)[0]
    np.testing.assert_allclose(seven[1::2], three, rtol=0, atol=1e-15)


def test_nested_rules_share_node_ids_paths_and_trajectory_identities():
    assert quadrature_node_ids(3) == ("node-00", "node-1-4", "node-1-2", "node-3-4")
    rules = [_quadrature(nodes=count) for count in (3, 7, 15, 31)]
    for coarse, fine in pairwise(rules):
        # Interior node k of the coarser rule is node 2k of the finer one.
        shared = fine.nodes[::2]
        assert [node.draw_id for node in shared] == [
            node.draw_id for node in coarse.nodes
        ]
        for old, new in zip(coarse.nodes, shared, strict=True):
            assert old == new
            assert coarse.trajectory_sha256(old) == fine.trajectory_sha256(new)
        # The rule's own digest still tells the rules apart.
        assert coarse.digest() != fine.digest()
    finest = rules[-1]
    assert len({node.draw_id for node in finest.nodes}) == 32
    assert len({finest.trajectory_sha256(node) for node in finest.nodes}) == 32
    payload = finest.model_dump(mode="json")
    for index, node in enumerate(payload["nodes"]):
        node["draw_id"] = f"node-{index:02d}"
    with pytest.raises(ValueError, match="Fejér angle"):
        PublicFrontierQuadrature.model_validate(payload)


def test_trajectory_identity_covers_the_path_but_not_the_rule_weights():
    quadrature = _quadrature()
    node = quadrature.nodes[4]
    identity = quadrature.trajectory_sha256(node)
    # Laws and weights that do not move the node leave its identity.
    reweighted = quadrature.model_copy(
        update={
            "base": quadrature.base.model_copy(
                update={"weights": (0.0, *reversed(quadrature.base.weights[1:]))}
            ),
            "alternatives": quadrature.alternatives[:1],
        }
    )
    assert reweighted.trajectory_sha256(node) == identity
    assert reweighted.digest() != quadrature.digest()
    # The forecast, period, covering law and stretch are part of it.
    others = [
        _quadrature(_forecast(pace_shape=4.0)),
        _quadrature(_forecast(pace_cap=3.0)),
        public_frontier_quadrature(
            _forecast(),
            period_start=START,
            period_end=datetime(2027, 7, 1, tzinfo=UTC),
            nodes=7,
        ),
        quadrature.model_copy(update={"tail_stretch": 2.0}),
        quadrature.model_copy(update={"covering_pace_shapes": (5.0,)}),
    ]
    assert identity not in {other.trajectory_sha256(other.nodes[4]) for other in others}
    with pytest.raises(ValueError, match="not one of"):
        quadrature.trajectory_sha256(others[0].nodes[4])
    assert quadrature.covering_pace_shapes == (5.0, 3.0)
    payload = quadrature.model_dump(mode="json")
    payload["covering_pace_shapes"] = [3.0, 5.0]
    with pytest.raises(ValueError, match="covering law"):
        PublicFrontierQuadrature.model_validate(payload)


@pytest.mark.parametrize("shape,cap", [(5, 4), (3, 4), (8, 4), (5, 2), (0.8, 1.5)])
def test_capped_pace_law_is_the_log_logistic_law_truncated_at_the_cap(shape, cap):
    law = fisk(c=shape)
    pace = np.array([1e-3, 0.2, 0.9, 1.0, 1.4, cap * (1 - 1e-12), cap])
    np.testing.assert_allclose(
        np.exp(_log_pace_cdf(pace, shape, cap)), law.cdf(pace) / law.cdf(cap)
    )
    np.testing.assert_allclose(
        np.exp(_log_pace_density(pace, shape, cap)), law.pdf(pace) / law.cdf(cap)
    )
    # Beyond the cap the CDF is one and the density zero: no pace lies there
    # and none sits on the cap.
    beyond = np.array([cap * (1 + 1e-12), 2 * cap, 1e6])
    np.testing.assert_array_equal(np.exp(_log_pace_cdf(beyond, shape, cap)), 1)
    np.testing.assert_array_equal(np.exp(_log_pace_density(beyond, shape, cap)), 0)
    np.testing.assert_array_equal(np.exp(_log_pace_cdf(np.array([0.0]), shape, cap)), 0)
    total = quad(
        lambda value: np.exp(_log_pace_density(np.array([value]), shape, cap))[0],
        0,
        cap,
        points=[min(1.0, cap / 2)],
        epsabs=0,
        epsrel=1e-10,
    )[0]
    assert total == pytest.approx(1, abs=1e-8)


@pytest.mark.parametrize(
    "label,shape,cap",
    [("baseline", 5, 4), ("lower_pace_shape", 3, 4), ("lower_pace_cap", 5, 2)],
)
def test_progress_law_matches_sampling_of_independent_capped_lab_paths(
    label, shape, cap
):
    forecast = _forecast()
    posterior = fit_public_trend(forecast)
    base, alternatives, _ = _laws(forecast, posterior, _middle(forecast))
    law = {row.label: row for row in (base, *alternatives)}[label]
    assert (law.pace_shape, law.pace_cap) == (shape, cap)
    progress, _, _ = _sampled_progress(forecast, 400_000, seed=5, shape=shape, cap=cap)
    probabilities = (0.05, 0.25, 0.5, 0.75, 0.95, 0.999)
    points = np.quantile(progress, probabilities)
    np.testing.assert_allclose(law.cdf(points), probabilities, atol=3e-3)
    # The cap bounds a path's progress by its rate, so the sampled maximum
    # lies where the law has all but a sliver of its mass.
    assert law.cdf(np.array([progress.max()]))[0] > 1 - 1e-4
    grid = np.linspace(1e-3, progress.max() * 1.5, 20_001)
    assert np.trapezoid(law.density(grid), grid) == pytest.approx(1, abs=2e-3)
    for point in (points[0], points[2], points[4], points[5]):
        assert law.density(np.array([point]))[0] == pytest.approx(
            _reference_density(law, point), rel=1e-8
        )
    if label != "baseline":
        return
    elapsed = _middle(forecast)
    median = base_progress_quantile(5, posterior, elapsed, 0.5, pace_cap=4)
    assert median == pytest.approx(np.median(progress), rel=5e-3)
    assert base_progress_cdf(
        5, posterior, elapsed, median, pace_cap=4
    ) == pytest.approx(0.5, abs=1e-9)
    assert base_progress_cdf(5, posterior, elapsed, 0.0, pace_cap=4) == 0
    assert _quadrature(forecast).median_progress == pytest.approx(median, rel=1e-9)
    # The retained parameters include the cap: another cap is another law.
    assert base_progress_cdf(5, posterior, elapsed, points[4], pace_cap=2) > (
        base_progress_cdf(5, posterior, elapsed, points[4], pace_cap=4) + 0.01
    )
    with pytest.raises(ValueError, match="exceed the median pace"):
        base_progress_cdf(5, posterior, elapsed, median, pace_cap=1)


def test_a_fixed_rate_bounds_the_progress_by_the_cap():
    """With one trend rate the frontier cannot gain more than the cap times
    that rate times the elapsed time, and the law says so exactly."""
    fixed = _fixed_rate_forecast()
    posterior = fit_public_trend(fixed)
    elapsed = _middle(fixed)
    base = _laws(fixed, posterior, elapsed)[0]
    assert (posterior.rate_location, posterior.rate_variance) == (10, 0)
    bound = 4 * 10 * elapsed
    inside, outside = np.array([bound * (1 - 1e-9)]), np.array([bound * (1 + 1e-9)])
    assert base.cdf(inside)[0] < 1 and base.density(inside)[0] > 0
    assert base.cdf(outside)[0] == 1 and base.density(outside)[0] == 0
    # Below the bound the law is the product of the two labs' capped paces:
    # lab_b starts 10 points behind lab_a.
    point = 0.6 * bound
    pace = fisk(c=5)
    expected = (
        pace.cdf(point / (10 * elapsed))
        * pace.cdf((point + 10) / (10 * elapsed))
        / pace.cdf(4) ** 2
    )
    assert base.cdf(np.array([point]))[0] == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="beyond what the pace cap allows"):
        base.representative(bound * 1.01, "probe")


def test_a_law_evaluates_many_progress_values_in_bounded_chunks():
    """Values evaluated together, across chunk boundaries and in any shape,
    match each value evaluated alone, up to the summation order of the
    weighted sum over rate points."""
    forecast = _forecast()
    base = _laws(forecast, fit_public_trend(forecast), _middle(forecast))[0]
    grid = np.linspace(0.5, 40.0, 2_500)
    together = base.cdf(grid), base.density(grid)
    chunk = PROGRESS_CHUNK
    assert 2 * chunk < grid.size
    for index in (0, chunk - 1, chunk, 2 * chunk - 1, 2 * chunk, grid.size - 1):
        point = grid[index : index + 1]
        assert together[0][index] == pytest.approx(base.cdf(point)[0], rel=1e-13)
        assert together[1][index] == pytest.approx(base.density(point)[0], rel=1e-13)
    np.testing.assert_array_equal(
        base.cdf(grid.reshape(50, 50)), together[0].reshape(50, 50)
    )


def test_node_weights_integrate_each_law_on_the_shared_nodes():
    forecast = _forecast()
    quadrature = _quadrature(forecast)
    progress = np.asarray([node.progress for node in quadrature.nodes])
    posterior = fit_public_trend(forecast)
    base, alternatives, _ = _laws(forecast, posterior, _middle(forecast))
    laws = {law.label: law for law in (base, *alternatives)}
    grid = np.exp(np.linspace(np.log(1e-2), np.log(progress[-1] * 40), 20_001))
    for law in (quadrature.base, *quadrature.alternatives):
        if law.variant == "no_progress":
            assert law.weights[0] == law.embedded_weights[0] == 1
            continue
        density = laws[law.label].density(grid)
        mean = np.trapezoid(grid * density, grid) / np.trapezoid(density, grid)
        assert np.asarray(law.weights) @ progress == pytest.approx(mean, rel=0.02)
        assert law.weights[0] == 0


def test_quadrature_nodes_carry_representative_joint_paths():
    quadrature = _quadrature()
    first = quadrature.nodes[0]
    assert (first.progress, first.baseline_rate) == (0, None)
    assert first.rates == first.pace_multipliers == (0, 0)
    assert quadrature.base.weights[0] == 0
    assert {law.label for law in quadrature.alternatives} == {
        "lower_pace_shape",
        "higher_pace_shape",
        "lower_pace_cap",
        "no_progress",
    }
    assert set(quadrature.unavailable_alternatives) == {
        "trend_discrepancy",
        "higher_pace_cap",
    }
    # The no-progress node holds every lab at its as-of frontier.
    assert all(values == (100, 90) for values in first.values)
    middle = progress_elapsed_years(quadrature.as_of, quadrature.times)
    for node in quadrature.nodes[1:]:
        rates = np.asarray(node.rates)
        np.testing.assert_allclose(
            rates, node.baseline_rate * np.asarray(node.pace_multipliers), rtol=1e-12
        )
        assert max(np.asarray([100, 90]) + rates * middle) - 100 == pytest.approx(
            node.progress, rel=1e-9
        )
    payload = quadrature.model_dump(mode="json")
    payload["nodes"][2]["values"][-1][0] += 1
    with pytest.raises(ValueError, match="grow at its lab rates"):
        PublicFrontierQuadrature.model_validate(payload)
    payload = quadrature.model_dump(mode="json")
    payload["nodes"][2]["progress"] *= 1.01
    with pytest.raises(ValueError, match="gain at the period's middle"):
        PublicFrontierQuadrature.model_validate(payload)
    with pytest.raises(ValueError, match="odd number"):
        _quadrature(nodes=4)


@pytest.mark.parametrize("cap", [4.0, 2.5, 1.2])
def test_no_node_path_outruns_the_pace_cap(cap):
    """Above its as-of score no lab gains more than the cap times the node's
    trend rate times the elapsed time, at any time of any node."""
    forecast = _forecast(pace_cap=cap)
    quadrature = _quadrature(forecast, nodes=31)
    assert quadrature.base.pace_cap == cap
    elapsed = np.asarray(
        [
            (at - quadrature.as_of).total_seconds() / (365.25 * 86400)
            for at in quadrature.times
        ]
    )
    highest = 0.0
    for node in quadrature.nodes[1:]:
        gain = np.asarray(node.values) - np.asarray([100, 90])
        limit = cap * node.baseline_rate * elapsed[:, None]
        assert np.all(gain <= limit * (1 + 1e-12))
        assert max(node.pace_multipliers) <= cap
        highest = max(highest, max(node.pace_multipliers))
    # The outer nodes come close to the cap, so the bound is not idle.
    assert highest > 0.85 * cap
    payload = quadrature.model_dump(mode="json")
    node = payload["nodes"][-1]
    leader = int(np.argmax(node["pace_multipliers"]))
    node["pace_multipliers"][leader] = cap * 1.001
    node["rates"][leader] = node["baseline_rate"] * cap * 1.001
    with pytest.raises(ValueError, match="within the base pace cap"):
        PublicFrontierQuadrature.model_validate(payload)


def test_every_law_carries_its_cap_and_the_lower_cap_is_an_alternative():
    quadrature = _quadrature()
    laws = {law.label: law for law in (quadrature.base, *quadrature.alternatives)}
    assert quadrature.base.pace_cap == 4
    assert (laws["lower_pace_cap"].pace_shape, laws["lower_pace_cap"].pace_cap) == (
        5,
        2,
    )
    assert {
        laws[label].pace_cap for label in ("lower_pace_shape", "higher_pace_shape")
    } == {4}
    assert laws["no_progress"].pace_cap is laws["no_progress"].pace_shape is None
    for law in laws.values():
        assert sum(law.weights) == pytest.approx(1, abs=1e-12)
        assert sum(law.embedded_weights) == pytest.approx(1, abs=1e-12)
    # A higher cap has paths beyond every node, so it is never integrated.
    assert "higher cap" in quadrature.unavailable_alternatives["higher_pace_cap"]
    # The lower cap moves weight from the fast nodes to the slow ones.
    progress = np.asarray([node.progress for node in quadrature.nodes])
    assert np.asarray(laws["lower_pace_cap"].weights) @ progress < (
        np.asarray(quadrature.base.weights) @ progress
    )
    payload = quadrature.model_dump(mode="json")
    payload["base"]["pace_cap"] = None
    with pytest.raises(ValueError, match="a pace shape and a pace cap"):
        PublicFrontierQuadrature.model_validate(payload)
    # A forecast whose own cap is not above the alternative's has no lower cap.
    for cap in (2.0, 1.5):
        low = _quadrature(_forecast(pace_cap=cap))
        assert "lower_pace_cap" not in {law.label for law in low.alternatives}
        assert "not a lower cap" in low.unavailable_alternatives["lower_pace_cap"]
        assert "higher_pace_cap" in low.unavailable_alternatives


def test_the_lower_cap_has_no_weight_on_nodes_beyond_its_reach():
    """Under a fixed rate of 10 the lower cap of 2 cannot bring the frontier
    past 20 points a year, so the nodes beyond get exactly no weight. Under an
    uncertain rate a node gets none where no rate within the integrated span
    reaches it, and a positive weight otherwise."""

    def lower_and_progress(forecast):
        quadrature = _quadrature(forecast, nodes=15)
        lower = next(
            law for law in quadrature.alternatives if law.label == "lower_pace_cap"
        )
        assert sum(lower.weights) == pytest.approx(1, abs=1e-12)
        assert all(weight > 0 for weight in quadrature.base.weights[1:])
        return (
            np.asarray(lower.weights)[1:],
            np.asarray([node.progress for node in quadrature.nodes[1:]]),
        )

    fixed = _fixed_rate_forecast()
    weights, progress = lower_and_progress(fixed)
    beyond = progress > 2 * 10 * _middle(fixed)
    assert 2 <= beyond.sum() <= progress.size - 4
    assert np.all(weights[beyond] == 0) and np.all(weights[~beyond] > 0)

    tight = _forecast(
        rate_prior=_forecast().rate_prior.model_copy(update={"mean": 10, "sd": 0.2}),
        trend_window_start=_forecast().as_of,
    )
    posterior = fit_public_trend(tight)
    assert posterior.rate_location == 10
    assert posterior.rate_variance == pytest.approx(0.04)
    weights, progress = lower_and_progress(tight)
    typical = progress > 2 * 10 * _middle(tight)
    beyond = progress > 2 * (10 + RATE_SPAN * 0.2) * _middle(tight)
    assert beyond.sum() >= 2 and (typical & ~beyond).sum() >= 1
    assert np.all(weights[beyond] == 0) and np.all(weights[~beyond] > 0)
    # Between the two bounds only rates far above the kernel's location reach
    # the node, so the weight is positive and small.
    assert np.all(weights[typical & ~beyond] < 1e-3)


def test_the_frontier_leader_differs_across_nodes_of_a_close_race():
    releases = tuple(
        {**row, "value": 99.5} if row["release_id"] == "b-2" else row
        for row in SYNTHETIC_RELEASES
    )
    forecast = synthetic_forecast(releases=releases).model_copy(
        update={"pace_shape": 5.0}
    )
    quadrature = _quadrature(forecast, nodes=15)
    middle = progress_elapsed_years(quadrature.as_of, quadrature.times)
    anchors = np.asarray([100, 99.5])
    leaders = {
        int(np.argmax(anchors + np.asarray(node.rates) * middle))
        for node in quadrature.nodes[1:]
    }
    assert leaders == {0, 1}


def test_representative_paths_follow_the_law_given_progress():
    """Over many node identities, the representative draws at one progress
    reproduce the share of paths each lab leads among sampled paths that
    reach about that progress, and keep every other lab below the frontier."""
    forecast = _forecast()
    posterior = fit_public_trend(forecast)
    base = _laws(forecast, posterior, _middle(forecast))[0]
    progress, leader, sampled_rates = _sampled_progress(forecast, 1_000_000, seed=11)
    middle = _middle(forecast)
    anchors = np.asarray([100, 90])
    # The 70% point is ordinary progress; at the 99.9% point the leader's pace
    # is near the cap, so only the higher trend rates can reach it.
    for level, probes in ((0.7, 3000), (0.999, 1500)):
        target = float(np.quantile(progress, level))
        near = np.abs(progress / target - 1) < (0.01 if level < 0.9 else 0.05)
        expected = float(np.mean(leader[near] == 1))
        leads, rates = [], []
        for index in range(probes):
            rate, pace = base.representative(target, f"probe-{index}")
            heights = anchors + rate * pace * middle
            assert heights.max() - 100 == pytest.approx(target, rel=1e-9)
            assert pace.max() <= base.pace_cap
            leads.append(int(np.argmax(heights)))
            rates.append(rate)
        observed = float(np.mean(np.asarray(leads) == 1))
        tolerance = 5 * np.sqrt(
            max(expected * (1 - expected), 1e-3) * (1 / probes + 1 / near.sum())
        )
        assert observed == pytest.approx(expected, abs=tolerance)
        # The trend rate follows its law given the progress.
        spread = np.std(sampled_rates[near])
        assert np.mean(rates) == pytest.approx(
            np.mean(sampled_rates[near]),
            abs=5 * spread * np.sqrt(1 / probes + 1 / near.sum()),
        )
        assert np.std(rates) == pytest.approx(spread, rel=0.15)


def test_estimated_discrepancy_adds_its_interval_ends_and_a_fixed_rate_integrates():
    releases = tuple(
        synthetic_release(lab, f"{lab}-{index}", date, value)
        for lab, rows in (
            (
                "lab_a",
                (
                    ("2026-01-15T00:00:00Z", 90),
                    ("2026-04-01T00:00:00Z", 95),
                    ("2026-08-15T00:00:00Z", 100),
                ),
            ),
            (
                "lab_b",
                (
                    ("2026-02-01T00:00:00Z", 84),
                    ("2026-05-01T00:00:00Z", 87),
                    ("2026-09-01T00:00:00Z", 90),
                ),
            ),
        )
        for index, (date, value) in enumerate(rows)
    )
    estimated = synthetic_forecast(releases=releases).model_copy(
        update={"pace_shape": 5.0, "trend_discrepancy_sd": None}
    )
    labels = {law.label for law in _quadrature(estimated).alternatives}
    assert {"lower_trend_discrepancy", "upper_trend_discrepancy"} <= labels
    quadrature = _quadrature(_fixed_rate_forecast())
    assert sum(quadrature.base.weights) == pytest.approx(1)
    assert {node.baseline_rate for node in quadrature.nodes[1:]} == {10}
