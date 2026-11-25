"""A node's lab business loss is one annual amount, drawn once per year with
an incident (``ACC-LAB-ANNUAL-BUSINESS-LOSS``).

Its analytic mean is the annual law's mean times the probability of at least
one incident in a year, integrated over the count's parameter rank.
"""

import math

import numpy as np
import pytest
from scipy.special import ndtri

from catmodel.contracts.contextual_risk import ContextualRiskInstrument
from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.incident import IncidentLaw, ParameterRankLaw
from catmodel.contracts.native_severity import native_severity_sampler
from catmodel.setup.contextual_node import CompileOptions, compile_contextual_node
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.rank_spread import REFERENCE_SPREAD_TAIL_FORM
from catmodel.simulation.incident import (
    CompiledIncident,
    compile_incident_state,
)
from catmodel.simulation.incident_moments import (
    at_least_one_probability,
    single_period_law_means,
)
from catmodel.simulation.incident_runtime import (
    score_multiplier,
    simulate_incident_state,
)
from catmodel.simulation.small_events import SIMULATED_RANK_RANGE
from test_support.contextual_risk import contextual_answer, contextual_instrument
from tests.test_contextual_study import _changed_artifact

LAB = "lab_loss::lab"
RATE = {"p5": 0.2, "p25": 0.6, "p50": 1.0, "p75": 1.8, "p95": 4.0}
QUANTILES = ("p5", "p25", "p50", "p75", "p95", "p99", "p999")
FIXED = {"status": "available", "positive_probability": 1, "amount": 1e6}
STOCHASTIC = {
    "status": "available",
    "positive_probability": 0.5,
    "loss": dict(zip(QUANTILES, (2e5, 5e5, 1e6, 2e6, 5e6, 8e6, 1.5e7), strict=True)),
}


def _instrument(family: str, dispersion: float) -> ContextualRiskInstrument:
    payload = contextual_instrument().model_dump(mode="json")
    payload["nodes"][0].update(count_family=family, count_variance_to_mean=dispersion)
    return ContextualRiskInstrument.model_validate(payload)


def _simulated(instrument, business, years):
    def change(case):
        case["answer"]["rate"] = RATE
        case["answer"]["lab_annual_business_loss"] = business

    setup = compile_contextual_setup(
        instrument,
        _changed_artifact(instrument, change),
        setup_id="annual-business-loss",
        years_per_state=1,
        seed=11,
    )
    state = max(setup.joint_states, key=lambda row: row.weight)
    compiled = compile_incident_state(setup, state.state_id)
    draw, reasons, _ = simulate_incident_state(
        setup.execution, compiled, response_sha256="0" * 64, years=years
    )
    assert LAB not in reasons
    (incident,) = compiled.incidents
    # Every fixture episode has USD 100 of welfare loss, so welfare counts them.
    episodes = draw.views["welfare"].array / 100
    np.testing.assert_allclose(episodes, np.round(episodes), atol=1e-9)
    return incident, draw.views[LAB], draw.views["gross"], np.round(episodes)


@pytest.mark.parametrize(
    ("family", "dispersion"), [("poisson", 1.0), ("negative_binomial", 3.0)]
)
def test_analytic_annual_mean_matches_simulation(family, dispersion):
    incident, lab, gross, episodes = _simulated(
        _instrument(family, dispersion), FIXED, 200_000
    )
    occurred = episodes > 0
    # The fixture's USD 20 fine per episode plus USD 1 million once in each
    # year with an episode, never once per episode; gross counts the same
    # business loss beside the episode's USD 120.
    np.testing.assert_allclose(lab.array, 20 * episodes + 1e6 * occurred, rtol=1e-12)
    np.testing.assert_allclose(gross.array, 120 * episodes + 1e6 * occurred, rtol=1e-12)
    assert np.all(lab.array <= gross.array)
    assert lab.native_mean.value <= gross.native_mean.value
    result = at_least_one_probability(incident)
    assert result is not None
    probability, bound = result
    assert bound <= 1e-3 * probability
    analytic = lab.native_mean.value
    assert analytic == pytest.approx(
        20 * incident.annual_rate + 1e6 * probability, rel=1e-12
    )
    assert gross.native_mean.value == pytest.approx(
        120 * incident.annual_rate + 1e6 * probability, rel=1e-12
    )
    # The simulated share of years with an episode is the integrated
    # probability within four standard errors.
    share = occurred.mean()
    error = math.sqrt(probability * (1 - probability) / occurred.size)
    assert abs(share - probability) < 4 * error + bound
    # Replacing the probability by the expected count would move the mean far
    # outside that noise.
    assert incident.annual_rate - probability > 50 * error


def test_stochastic_annual_law_compiles_and_simulates():
    incident, lab, gross, episodes = _simulated(
        _instrument("poisson", 1.0), STOCHASTIC, 200_000
    )
    (annual,) = incident.annual_lab_losses
    assert annual.severity is not None
    business = lab.array - 20 * episodes
    np.testing.assert_allclose(gross.array - 120 * episodes, business, atol=1e-6)
    occurred = episodes > 0
    assert np.all(business[~occurred] == 0)
    positive = business[occurred] > 0
    assert positive.mean() == pytest.approx(0.5, abs=0.01)
    assert np.median(business[occurred][positive]) == pytest.approx(1e6, rel=0.05)
    mean, error = business.mean(), business.std() / math.sqrt(business.size)
    result = at_least_one_probability(incident)
    assert result is not None
    probability = result[0]
    expected = lab.native_mean.value - 20 * incident.annual_rate
    assert abs(mean - expected) < 4 * error
    assert expected == pytest.approx(
        probability * native_severity_sampler(annual.severity).analytic_mean(),
        rel=1e-9,
    )


def _law(business, **options) -> IncidentLaw:
    instrument = contextual_instrument()
    answer = contextual_answer().model_dump(mode="json")
    answer.update(rate=RATE, lab_annual_business_loss=business)
    parsed = type(contextual_answer()).model_validate(answer)
    law = compile_contextual_node(
        instrument,
        instrument.nodes[0],
        {0: parsed, 2: parsed},
        1,
        options=CompileOptions(**options),
    )
    assert isinstance(law, IncidentLaw)
    return law


@pytest.mark.parametrize(
    ("family", "dispersion"), [("poisson", 1.0), ("negative_binomial", 2.5)]
)
@pytest.mark.parametrize("spread", [1.0, 1.6])
def test_probability_of_an_incident_is_integrated_over_the_rank(
    family, dispersion, spread
):
    law = _law(FIXED).model_copy(
        update={
            "count_family": family,
            "count_variance_to_mean": dispersion,
            "parameter_rank": ParameterRankLaw(
                group_id="node",
                count=Quantiles(**RATE),
                rank_spread=spread,
                tail_form=REFERENCE_SPREAD_TAIL_FORM,
            ),
        }
    )
    incident = CompiledIncident(
        origin_id="node",
        scenario_id=law.scenario_id,
        entity_id=law.entity_id,
        annual_rate=2.0,
        count_family=family,
        count_variance_to_mean=dispersion,
        segments=(),
        parameter_rank=law.parameter_rank,
    )
    result = at_least_one_probability(incident)
    assert result is not None
    probability, bound = result
    # A dense midpoint rule over the clipped rank, as simulation reads it.
    ranks = np.clip((np.arange(2_000_000) + 0.5) / 2_000_000, *SIMULATED_RANK_RANGE)
    assert law.parameter_rank is not None
    counts = 2.0 * score_multiplier(law.parameter_rank)(ndtri(ranks))
    decay = 1.0 if family == "poisson" else math.log(dispersion) / (dispersion - 1)
    direct = float(np.mean(-np.expm1(-decay * counts)))
    assert bound <= 1e-3 * probability
    assert probability == pytest.approx(direct, rel=1e-4)
    # Jensen: below the probability at the mean count, so it is not that.
    assert probability < 1 - math.exp(-decay * 2.0) - 1e-3


def test_fixed_rate_probability_is_closed_form():
    for family, dispersion, expected in (
        ("poisson", 1.0, 1 - math.exp(-0.7)),
        ("negative_binomial", 3.0, 1 - 3.0 ** (-0.7 / 2)),
    ):
        incident = CompiledIncident(
            origin_id="o",
            scenario_id="s",
            entity_id="e",
            annual_rate=0.7,
            count_family=family,
            count_variance_to_mean=dispersion,
            segments=(),
        )
        assert at_least_one_probability(incident) == (
            pytest.approx(expected, rel=1e-15),
            0.0,
        )


def test_provider_only_bound_and_cap_act_on_the_annual_loss():
    base = _law({**FIXED, "amount": 2e12})
    (annual,) = base.annual_lab_losses
    assert annual.amount == 1e12
    assert _law({**FIXED, "amount": 2e12}, frontier_lab_value=3e11).annual_lab_losses[
        0
    ].amount == pytest.approx(3e11)
    assert (
        _law({**FIXED, "amount": 2e12}, frontier_lab_value=None)
        .annual_lab_losses[0]
        .amount
        == 2e12
    )
    stochastic = _law(STOCHASTIC, frontier_lab_value=1e6)
    (capped,) = stochastic.annual_lab_losses
    assert capped.severity is not None
    assert native_severity_sampler(capped.severity).quantile(1) == 1e6
    assert _law(FIXED, lab_loss_provider_only=True).annual_lab_losses == ()
    means = single_period_law_means(
        _law(FIXED, lab_loss_provider_only=True), (LAB,), episodes=False
    )
    assert means[LAB] == 0


def test_business_loss_leaves_welfare_and_admission_unchanged():
    instrument = _instrument("poisson", 1.0)
    without = _simulated(
        instrument, {"status": "available", "positive_probability": 0}, 2_000
    )
    with_loss = _simulated(instrument, FIXED, 2_000)
    # Welfare counts the same episodes with or without the business loss.
    np.testing.assert_array_equal(without[3], with_loss[3])
    # Each incident's own marks, and so its admission against the qualifying
    # floor, are the same law: the business loss is outside every episode.
    plain = _law({"status": "available", "positive_probability": 0})
    loaded = _law(FIXED)
    assert loaded.segments == plain.segments
    assert loaded.annual_lab_losses and not plain.annual_lab_losses
