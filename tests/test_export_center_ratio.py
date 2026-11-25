"""The chain summary reads the node nearest the median and checks its ratio."""

import math
from datetime import UTC, datetime

import pytest

from catmodel.forecast.public_frontier_quadrature import public_frontier_quadrature
from catmodel.results.results_export import _center_node, _require_center_ratio
from test_support.public_frontier import synthetic_forecast


def test_the_center_node_is_the_interior_node_nearest_the_median_progress():
    quadrature = public_frontier_quadrature(
        synthetic_forecast().model_copy(update={"pace_shape": 5.0}),
        period_start=datetime(2027, 1, 1, tzinfo=UTC),
        period_end=datetime(2028, 1, 1, tzinfo=UTC),
        nodes=7,
    )
    center = _center_node(quadrature)
    gaps = [
        abs(math.log(node.progress / quadrature.median_progress))
        for node in quadrature.nodes[1:]
    ]
    assert center == quadrature.nodes[1 + gaps.index(min(gaps))]


@pytest.mark.parametrize("ratio", [1.09, 1.09 * (1 + 5e-5), 1.09 * (1 - 5e-5)])
def test_a_ratio_within_rounding_of_its_node_is_accepted(ratio):
    _require_center_ratio(ratio, 1.09)


@pytest.mark.parametrize("ratio", [None, 1.0, 1.2])
def test_a_ratio_away_from_its_node_is_refused(ratio):
    with pytest.raises(ValueError, match="node's ratio"):
        _require_center_ratio(ratio, 1.09)
