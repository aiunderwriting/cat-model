"""Mixtures with component-intrinsic cap atoms keep exact pooled laws and means."""

import numpy as np
import pytest

from catmodel.probability.distributions import Distribution, MixtureSeverityDistribution


@pytest.mark.parametrize("currency", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("outer_cap", [None, 70.0, 60.0, 40.0])
def test_intrinsic_cap_atoms_have_consistent_native_means_and_quantiles(
    currency, reverse, outer_cap
):
    components = [
        Distribution(100, 100, 100, 100, 100, severity_cap=50),
        Distribution(80, 80, 80, 80, 80),
    ]
    if reverse:
        components.reverse()
    mixture = MixtureSeverityDistribution(
        [component.with_currency_scale(currency) for component in components],
        np.array([0.5, 0.5]),
        severity_cap=None if outer_cap is None else outer_cap * currency,
    )
    low = 50 if outer_cap is None else min(50, outer_cap)
    high = 80 if outer_cap is None else min(80, outer_cap)
    assert mixture.analytic_mean() == pytest.approx(
        (low + high) / 2 * currency, rel=2e-15
    )
    assert mixture.quantile(0.25) == pytest.approx(low * currency, rel=2e-15)
    assert mixture.quantile(0.75) == pytest.approx(high * currency, rel=2e-15)
    assert mixture.analytic_mean(severity_cap_override=60 * currency) == pytest.approx(
        min(low, 60) * currency / 2 + min(high, 60) * currency / 2
    )
