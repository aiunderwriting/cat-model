"""Exceedance curves sum each tail exactly, as ``math.fsum`` does."""

import math

import numpy as np
import pytest

from catmodel.results.results_export import _curve
from catmodel.simulation.capability_mixture import effective_sample_size


class _Draw:
    def __init__(self, losses, weights):
        self.annual_losses = {"gross": np.asarray(losses, dtype=float)}
        self._weights = np.asarray(weights, dtype=float)

    def normalized_weights(self):
        return self._weights / np.sum(self._weights)


def _expected(draws, weights, thresholds):
    rows = [(draws[key], mass) for key, mass in sorted(weights.items()) if mass > 0]
    values = np.concatenate([draw.annual_losses["gross"] for draw, _ in rows])
    masses = np.concatenate([mass * draw.normalized_weights() for draw, mass in rows])
    values, masses = values[masses > 0], masses[masses > 0]
    points = []
    for threshold in thresholds:
        tail = masses[values > threshold]
        points.append(
            (
                float(threshold),
                min(1.0, math.fsum(float(value) for value in tail)),
                len(tail),
                0.0 if len(tail) == 0 else effective_sample_size(tail),
            )
        )
    return points


@pytest.mark.parametrize("uniform", [True, False])
def test_each_tail_probability_is_the_exactly_rounded_sum(uniform):
    rng = np.random.default_rng(20261103)
    draws, weights = {}, {}
    for index in range(7):
        years = int(rng.integers(500, 3000))
        losses = rng.pareto(1.2, years) * (rng.random(years) < 0.3)
        likelihood = np.ones(years) if uniform else rng.random(years) ** 3
        draws[f"state-{index}"] = _Draw(losses, likelihood)
        # Masses spread over many binary orders, one of them zero.
        weights[f"state-{index}"] = (
            0.0 if index == 3 else float(10 ** -rng.uniform(0, 9))
        )
    thresholds = np.unique(np.concatenate(([0.0], rng.pareto(1.2, 40))))
    points = _curve(draws, weights, "gross", thresholds)
    assert [
        (row.loss, row.probability, row.useful_exceedances, row.tail_weight_ess)
        for row in points
    ] == _expected(draws, weights, thresholds)
    # Quantile thresholds, chosen by the curve itself, too.
    default = _curve(draws, weights, "gross", None)
    assert [(row.loss, row.probability) for row in default] == [
        row[:2] for row in _expected(draws, weights, [r.loss for r in default])
    ]
