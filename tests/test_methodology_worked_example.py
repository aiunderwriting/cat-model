"""The methodology's illustrative capability-response cell matches the engine.

methodology.md's worked example states the count laws' rank means, two rank
reads, the severity law's GPD scale and mean, and the resulting EALs. These
checks recompute each stated figure at the precision the text prints it.
"""

import math

import numpy as np
import pytest

from catmodel.contracts.elicitation import Quantiles
from catmodel.elicitation.epistemic import epistemic_mean, epistemic_quantiles
from catmodel.probability.distributions import Distribution

COUNTS = {
    140: (0.5, 1, 2, 4, 10),
    150: (0.75, 1.5, 3, 6, 15),
    160: (1, 2, 4, 8, 20),
}
STATED_MEANS = {140: 3.35, 150: 5.02, 160: 6.70}
STATED_EALS = {140: 10.1, 150: 15.2, 160: 20.2}


def _count(eci: int) -> Quantiles:
    return Quantiles(**dict(zip(("p5", "p25", "p50", "p75", "p95"), COUNTS[eci])))


def _severity() -> Distribution:
    return Distribution(0.1, 0.5, 1, 3, 10, gpd_shape=0.5, gpd_tail_ratio=3)


def test_interpolated_anchor_is_the_average_of_its_neighbours():
    for low, mid, high in zip(COUNTS[140], COUNTS[150], COUNTS[160], strict=True):
        assert mid == pytest.approx((low + high) / 2)


@pytest.mark.parametrize("eci", sorted(COUNTS))
def test_count_rank_means(eci):
    assert round(epistemic_mean(_count(eci)), 2) == STATED_MEANS[eci]


def test_rank_reads_at_eci_150():
    median, ninetieth = epistemic_quantiles(_count(150), np.asarray([0.5, 0.9]))
    assert median == 3.0
    assert round(float(ninetieth), 2) == 12.10


def test_severity_scale_mean_and_p99():
    law = _severity()
    assert round(0.5 * 20 / (math.sqrt(5) - 1), 2) == 8.09
    assert law.effective_gpd_scale == pytest.approx(0.5 * 20 / (math.sqrt(5) - 1))
    assert law.quantile(0.99) == pytest.approx(30)
    assert round(law.analytic_mean(), 2) == 3.02


@pytest.mark.parametrize("eci", sorted(COUNTS))
def test_eal_is_rank_mean_times_severity_mean(eci):
    eal = epistemic_mean(_count(eci)) * _severity().analytic_mean()
    assert round(eal, 1) == STATED_EALS[eci]
