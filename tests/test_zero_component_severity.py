"""Unconditional zero-loss atoms survive elicitation, compilation and pooling."""

import json
import math

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from catmodel.contracts.contextual_risk import (
    ContextualRiskInstrument,
    LabAnnualBusinessLossAnswer,
)
from catmodel.contracts.elicitation import SeverityQuantiles
from catmodel.contracts.incident import IncidentLaw
from catmodel.contracts.native_severity import native_severity_sampler
from catmodel.elicitation.contextual_risk import (
    contextual_prompts,
    parse_contextual_answer,
)
from catmodel.frozen_calls import validation_feedback
from catmodel.probability.distributions import Distribution, MixtureSeverityDistribution
from catmodel.setup.contextual_node import compile_contextual_node
from catmodel.simulation.incident_moments import single_period_law_means
from test_support.contextual_risk import contextual_answer, contextual_instrument
from test_support.question_text import answer_schema_of


def _quantiles(last_zero):
    values = (0, 0, 0 if last_zero >= 0.5 else 2, 0 if last_zero == 0.75 else 3, 10)
    return dict(zip(("p5", "p25", "p50", "p75", "p95"), values, strict=True)) | {
        "p99": 50,
        "p999": 200,
    }


@pytest.mark.parametrize("last_zero", [0.25, 0.5, 0.75])
def test_zero_loss_atom_matches_schema_and_survives_native_compilation(last_zero):
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"][0]["components"][1]["gpd_shape"] = 0.5
    instrument = ContextualRiskInstrument.model_validate(payload)
    prompt = contextual_prompts(instrument)[0]
    answer = contextual_answer().model_dump(mode="json")
    quantiles = _quantiles(last_zero)
    answer["branches"][0]["components"][1] = {
        "component_id": "fine",
        "status": "available",
        "loss": quantiles,
    }
    answer["lab_annual_business_loss"] = {
        "status": "available",
        "positive_probability": 0,
    }
    schema = answer_schema_of(prompt.render())
    Draft202012Validator(schema).validate(answer)
    parsed = parse_contextual_answer(json.dumps(answer), prompt)
    assert parsed.branches[0].components[1].loss == SeverityQuantiles(**quantiles)
    law = compile_contextual_node(
        instrument,
        instrument.nodes[0],
        {prompt.anchor_score: parsed},
        prompt.anchor_score,
    )
    assert isinstance(law, IncidentLaw)
    fine = law.segments[0].kernel.member_outcomes[0].components[1]
    assert fine.severity is not None
    sampler = native_severity_sampler(fine.severity)
    assert sampler.cdf(np.array([0]))[0] == last_zero
    assert sampler.survival_probability(0) == 1 - last_zero
    np.testing.assert_allclose(
        sampler.sample_from_uniforms(np.array([0.25, 0.5, 0.75, 0.95, 0.99])),
        [quantiles[key] for key in ("p25", "p50", "p75", "p95", "p99")],
    )
    fine_mean = sampler.analytic_mean()
    means = single_period_law_means(law, ("gross", "welfare", "lab_loss::lab"))
    assert means["welfare"] == 100
    assert means["gross"] == pytest.approx(100 + fine_mean)
    assert means["lab_loss::lab"] == pytest.approx(fine_mean)


@pytest.mark.parametrize("last_zero", [0.25, 0.5, 0.75])
def test_lab_business_loss_still_requires_conditional_positive_quantiles(last_zero):
    with pytest.raises(ValidationError, match="conditional-positive"):
        LabAnnualBusinessLossAnswer.model_validate(
            {
                "status": "available",
                "positive_probability": 0.1,
                "loss": _quantiles(last_zero),
            }
        )


@pytest.mark.parametrize(
    "change",
    [
        {"p5": -1},
        {"p95": 0},
        {"p99": 10},
        {"p999": 50},
        {"p50": 4, "p75": 3},
    ],
)
def test_nonnegative_severity_retains_tail_order_and_domain_checks(change):
    with pytest.raises(ValidationError):
        SeverityQuantiles.model_validate(_quantiles(0.75) | change)


def test_tail_validation_feedback_does_not_require_a_positive_median():
    with pytest.raises(ValidationError) as caught:
        SeverityQuantiles.model_validate(_quantiles(0.75) | {"p99": 10})
    assert json.loads(validation_feedback(caught.value))["validation_errors"] == [
        {
            "location": [],
            "type": "severity_requires_p5_below_p95_and_p99_above_p95",
        }
    ]


@pytest.mark.parametrize("anchored", [False, True])
def test_zero_median_law_pools_with_a_zero_atom(anchored):
    # A zero-median component pooled with an exact zero atom, as authored
    # mixtures can do: the pooled atom at zero adds both masses, and the
    # pooled inverse agrees with the general CDF-root inverse.
    law = Distribution(
        0,
        0,
        0,
        3,
        10,
        gpd_shape=0.5,
        gpd_tail_ratio=5,
        severity_minimum=0,
        large_loss_amount=1e3 if anchored else None,
        large_loss_exceedance=1e-3 if anchored else None,
    )
    assert law.cdf(np.array([0.0]))[0] == 0.5
    assert math.isfinite(law.analytic_mean()) and law.analytic_mean() > 0
    if anchored:
        assert law.survival_probability(1e3) == pytest.approx(1e-3)
    mixture = MixtureSeverityDistribution(
        [Distribution(0, 0, 0, 0, 0, severity_minimum=0), law], np.array([0.4, 0.6])
    )
    assert mixture.analytic_mean() == pytest.approx(0.6 * law.analytic_mean())
    ranks = np.array([0.1, 0.69, 0.71, 0.9, 0.99, 0.999, 0.99999])
    pooled = mixture.sample_from_uniforms_uncapped(ranks)
    general = mixture.sample_from_uniforms_uncapped(ranks, _allow_zero_atom=False)
    assert np.all(np.isfinite(pooled))
    np.testing.assert_allclose(pooled, general, rtol=1e-9)
    assert pooled[0] == pooled[1] == 0
    assert pooled[2] > 0
