"""Severity amounts: admission at the evidence boundary, a wide numerical domain inside."""

import numpy as np
import pytest
from pydantic import ValidationError

from catmodel.contracts.components import ComponentAmountLaw
from catmodel.contracts.contextual_risk import ContextualComponentAnswer
from catmodel.contracts.elicitation import SeverityQuantiles
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.probability.distributions import (
    Distribution,
    SeverityDomainError,
)
from catmodel.simulation.incident_moments import component_amount_mean
from test_support.components import sample_component_amounts


def _component(
    law: Distribution, *, multiplier: float = 1.0, post_multiplier: float = 1.0
) -> ComponentAmountLaw:
    return ComponentAmountLaw(
        component_id="loss",
        severity=native_severity_from_sampler(law),
        rank_group_id="severity",
        multiplier=multiplier,
        post_multiplier=post_multiplier,
    )


def _amount(component: ComponentAmountLaw, rank: float) -> float:
    return sample_component_amounts(
        (component,), np.random.default_rng(0), rank_uniforms={"severity": rank}
    )[0].amount


def test_a_post_scale_above_the_largest_stress_cap_keeps_an_analytic_mean():
    law = Distribution(1e9, 1e10, 1e11, 1e12, 1e13, gpd_shape=0.8, gpd_tail_ratio=5)
    capped = Distribution(
        1e9, 1e10, 1e11, 1e12, 1e13, gpd_shape=0.8, gpd_tail_ratio=5, severity_cap=5e14
    )
    component = _component(capped, post_multiplier=3)
    mean = component_amount_mean(component)
    assert mean == pytest.approx(3 * law.analytic_mean(severity_cap_override=5e14))
    assert _amount(component, 0.999999) == pytest.approx(3 * 5e14)


def test_a_pre_scale_on_a_catastrophe_capped_law_keeps_its_tail():
    law = Distribution(
        1e10, 1e11, 1e12, 5e12, 2e13, gpd_shape=0.8, gpd_scale=4e13, severity_cap=1e14
    )
    component = _component(law, multiplier=20)
    mean = component_amount_mean(component)
    assert mean is not None
    assert 20 * law.p50 < mean < 1e14
    assert _amount(component, 0.5) == pytest.approx(min(20 * law.p50, 1e14))


def test_a_tiny_composition_share_of_a_one_dollar_ladder_is_evaluable():
    law = Distribution(1, 2, 4, 8, 16, gpd_shape=0.5, gpd_tail_ratio=3)
    component = _component(law, post_multiplier=1e-10)
    assert component_amount_mean(component) == pytest.approx(
        1e-10 * law.analytic_mean()
    )
    assert _amount(component, 0.5) == pytest.approx(4e-10)


def test_a_transform_leaving_the_numerical_domain_is_refused():
    law = Distribution(1e95, 1e96, 1e97, 1e98, 1e99)
    with pytest.raises(ValueError, match="leaves the numerical"):
        law.with_excess_scale(100)
    with pytest.raises(ValueError, match="leaves the numerical"):
        law.with_currency_scale(100)


@pytest.mark.parametrize(
    "knots,tail",
    [
        ((1e101,) * 5, {}),
        ((1e-101,) * 5, {}),
        ((1, 2, 3, 4, 5), {"gpd_shape": 5.5, "gpd_scale": 1.0}),
        ((1, 2, 3, 4, 5), {"gpd_shape": 0.5, "gpd_scale": 1e101}),
    ],
)
def test_laws_outside_the_numerical_domain_are_refused(knots, tail):
    with pytest.raises(SeverityDomainError, match="numerical"):
        Distribution(*knots, **tail)


def test_evidence_outside_the_admission_domain_is_refused():
    with pytest.raises(ValidationError, match="admitted monetary domain"):
        SeverityQuantiles(p5=1, p25=2, p50=3, p75=5, p95=10, p99=1e19, p999=2e20)
    SeverityQuantiles(p5=1, p25=2, p50=3, p75=5, p95=10, p99=1e19, p999=1e20)
    with pytest.raises(ValidationError, match="admitted monetary domain"):
        ContextualComponentAnswer(component_id="loss", status="available", amount=2e20)
