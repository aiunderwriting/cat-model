"""Loss-level counts: implied annual counts at or above loss levels."""

import json
import math
from types import SimpleNamespace

import pytest

from catmodel.elicitation.loss_level_counts import (
    exceedance_bounds,
    loss_level_counts,
)

LOSS = SimpleNamespace(p5=1e5, p25=2e5, p50=5e5, p75=1e6, p95=1e7, p99=5e7)
MEAN = 85.617


@pytest.mark.parametrize(
    ("level", "bounds"),
    [
        (5e4, (0.95, 1.0)),
        (1e5, (0.95, 1.0)),
        (1e7, (0.05, 0.25)),
        (2e7, (0.01, 0.05)),
        (5e7, (0.01, 0.05)),
        (1e9, (0.0, 0.01)),
    ],
)
def test_bounds_follow_the_stated_quantiles(level, bounds):
    assert exceedance_bounds(LOSS, level) == bounds


def _quantile(values: list[float], p: float) -> float:
    """inf{x : F(x) >= p} of an equally weighted sample."""
    ordered = sorted(values)
    return ordered[math.ceil(round(p * len(ordered), 9)) - 1]


def test_bounds_contain_the_true_exceedance_of_a_law_with_atoms():
    # Atoms at the admission floor and at the stated p95.
    values = [1e5] * 120 + [3e5] * 40 + [2e6] * 20 + [1e7] * 15 + [6e7] * 4 + [5e8] * 1
    total = SimpleNamespace(
        **{
            name: _quantile(values, p)
            for name, p in (
                ("p5", 0.05),
                ("p25", 0.25),
                ("p50", 0.5),
                ("p75", 0.75),
                ("p95", 0.95),
                ("p99", 0.99),
            )
        }
    )
    assert (total.p5, total.p50, total.p75, total.p95) == (1e5, 1e5, 3e5, 1e7)
    for level in (5e4, 1e5, 2e5, 3e5, 1e6, 2e6, 5e6, 1e7, 3e7, 6e7, 1e8, 5e8, 1e9):
        true = sum(value >= level for value in values) / len(values)
        low, high = exceedance_bounds(total, level)
        assert low <= true <= high
        # One minus p at an untied stated quantile never exceeds the truth.
        named = [
            tail
            for name, tail in (("p75", 0.25), ("p95", 0.05))
            if getattr(total, name) == level
        ]
        if named:
            assert named[0] <= true
    # At an atom the stated 1 - p is only a lower bound on the exceedance.
    assert sum(value >= 1e7 for value in values) / len(values) == 0.1


def test_a_stated_quantile_has_one_value_and_a_fixed_level_bounds():
    counts = loss_level_counts(MEAN, LOSS)
    assert [row.level for row in counts] == [5e5, 1e6, 1e7, 5e7, 1e8]
    assert [row.basis for row in counts] == ["p50", "p75", "p95", "p99", "fixed"]
    at = {row.level: row for row in counts}
    # The fixed USD 1 million level equals the stated p75, so it is one value.
    for level, tail in ((5e5, 0.5), (1e6, 0.25), (1e7, 0.05), (5e7, 0.01)):
        assert at[level].single
        assert at[level].annual_count_low == float(f"{MEAN * tail:.4g}")
    # Above p99 only the one-in-a-hundred bound follows, rounded outward.
    above = at[1e8]
    assert not above.single
    assert above.annual_count_low == 0.0
    assert above.annual_count_high == 0.8562


def test_a_reporting_level_between_quantiles_has_bounds_rounded_outward():
    total = SimpleNamespace(p5=1e5, p25=2e5, p50=5e5, p75=2e6, p95=1e7, p99=5e7)
    at = {row.level: row for row in loss_level_counts(MEAN, total)}
    assert at[1e6].basis == "fixed" and not at[1e6].single
    # Exact bounds 21.40425 and 42.8085: nearest rounding would narrow both.
    assert at[1e6].annual_count_low == 21.4 <= MEAN * 0.25
    assert at[1e6].annual_count_high == 42.81 >= MEAN * 0.5


def test_tied_quantiles_have_bounds():
    tied = SimpleNamespace(p5=1e5, p25=1e5, p50=5e5, p75=1e7, p95=1e7, p99=5e7)
    at = {row.level: row for row in loss_level_counts(100.0, tied)}
    assert at[1e7].basis == "p75=p95"
    assert not at[1e7].single
    assert (at[1e7].annual_count_low, at[1e7].annual_count_high) == (25.0, 50.0)


def test_records_hold_plain_python_scalars():
    for row in loss_level_counts(MEAN, LOSS):
        record = row.record()
        for key in ("level", "annual_count_low", "annual_count_high"):
            assert type(record[key]) is float
        json.dumps(record, allow_nan=False)
