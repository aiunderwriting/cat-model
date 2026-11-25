"""Scaled physical amounts stay exact inside the numerical monetary domain."""

from decimal import Decimal, localcontext

import pytest

from catmodel.contracts._setup_codec import encode_scientific
from catmodel.contracts.native_severity import native_severity_from_sampler
from catmodel.probability.distributions import Distribution
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.run_setup import setup_from_payload
from test_support.study import _plan, _study


@pytest.mark.parametrize(
    "native,amount,multiplier,post_multiplier,cap",
    [
        (native, amount, multiplier, post_multiplier, None)
        for native in (False, True)
        for amount, multiplier, post_multiplier in (
            (1e9, 1e-6, 1e-3),
            (1e-6, 1e-3, 1e9),
            (5.0, 0.37, 2.5),
        )
    ]
    + [(True, 1e9, 1e-6, 1e-3, 1e2)],
)
def test_scaled_components_preserve_decimal_amounts_and_native_means(
    native, amount, multiplier, post_multiplier, cap
):
    payload = encode_scientific(_study(rate=100))
    payload["alternatives"] = []
    for state in payload["joint_states"]:
        kernel = state["laws"][0]["segments"][0]["kernel"]
        kernel["cluster"]["support"] = [{"count": 1, "probability": 1}]
        component = kernel["member_outcomes"][0]["components"][0]
        component.update(
            amount=amount, multiplier=multiplier, post_multiplier=post_multiplier
        )
        if native:
            component.update(
                amount=None,
                rank_group_id="severity",
                severity=native_severity_from_sampler(
                    Distribution(*(amount,) * 5, severity_cap=cap)
                ).model_dump(mode="json"),
            )
    compiled = compile_incident_state(setup_from_payload(payload), "low")
    direct_plan = _plan(20).model_copy(
        update={
            "requested_views": ("gross",),
            "required_arms": ("reference",),
        }
    )
    direct, _, direct_allocation = simulate_incident_state(
        direct_plan, compiled, response_sha256="a" * 64
    )
    with localcontext() as context:
        context.prec = 80
        before_post = Decimal.from_float(amount) * Decimal.from_float(multiplier)
        if cap is not None:
            before_post = min(before_post, Decimal.from_float(cap))
        mark = float(before_post * Decimal.from_float(post_multiplier))
    total_count = direct_allocation.total_initiators
    assert isinstance(total_count, int)
    assert sum(direct.annual_losses["gross"]) == pytest.approx(
        mark * total_count, rel=2e-13, abs=0
    )
    assert any(view.native_mean is not None for view in direct.views.values())
    assert any(
        view.native_mean is not None and view.native_mean.precision is not None
        for view in direct.views.values()
    )
    assert direct.views["gross"].native_mean is not None
    assert direct.views["gross"].native_mean.value == pytest.approx(
        100 * mark, rel=2e-13, abs=0
    )
    assert direct.views["gross"].native_mean is not None
    assert direct.views["gross"].native_mean.precision is not None
    assert direct.views["gross"].native_mean.precision["known_zero_law"] is False


@pytest.mark.parametrize(
    "update",
    [
        {"multiplier": 1e-200},
        {"post_multiplier": 1e200},
        {"amount": 1e-300},
        {"amount": 1e300},
    ],
)
def test_components_outside_the_monetary_domain_are_refused(update):
    from pydantic import ValidationError

    from catmodel.contracts.components import ComponentAmountLaw

    with pytest.raises(ValidationError, match="domain"):
        ComponentAmountLaw.model_validate(
            {"component_id": "loss", "amount": 1.0} | update
        )


@pytest.mark.parametrize(
    "knots,tail",
    [
        ((1e300,) * 5, {}),
        ((1e-300,) * 5, {}),
        ((1, 2, 3, 4, 5), {"gpd_shape": 10.0, "gpd_scale": 1.0}),
        ((1, 2, 3, 4, 5), {"gpd_shape": 0.5, "gpd_scale": 1e300}),
        ((1, 2, 3, 4, 5), {"severity_cap": 1e-310}),
    ],
)
def test_laws_outside_the_monetary_domain_are_refused(knots, tail):
    with pytest.raises(ValueError, match="domain"):
        Distribution(*knots, **tail)


def test_scaled_product_distinguishes_exact_zero_from_representable_composition():
    from catmodel.probability.numerics import scaled_product

    assert scaled_product(0, 1e-200, 1e200) == 0
    assert scaled_product(1e300, 1e-200, 1e-200) == pytest.approx(
        1e-100, rel=2e-15, abs=0
    )
    assert scaled_product(1e-200, 1e-200, 1e300) == pytest.approx(
        1e-100, rel=2e-15, abs=0
    )


def test_product_comparison_preserves_positive_subnormal_boundary():
    from catmodel.probability.numerics import product_at_least

    assert not product_at_least(0.0, 1e200, 1e-200)
    assert not product_at_least(5e-324, 0.75, 5e-324)
    assert product_at_least(5e-324, 1.0, 5e-324)
