"""Per-year parameter scores: rank-free laws, state-keyed streams, exact EAL.

A contextual law's parameter rank moves only its expected count. Each law
records the count law's mean over the rank as its reference rate and carries
the count law itself; each simulated year reads one score per rank group from
its joint state's rank stream, the same in every arm of the state and
independent of every other state's unless states declare one stream.
"""

import math
from bisect import bisect_left
from dataclasses import replace
from itertools import pairwise
from typing import Any, cast

import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr, ndtri
from scipy.stats import norm, poisson
from scipy.stats import t as student_t

from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskArtifact,
)
from catmodel.contracts.elicitation import Quantiles
from catmodel.contracts.incident import (
    SPREAD_TAIL_FORMS,
    Alternative,
    IncidentLaw,
    LawOverride,
    PopulationAbsence,
    SpreadTailForm,
)
from catmodel.contracts.law_overrides import HeldOverrides
from catmodel.elicitation.contextual_frequency import (
    ContextualChainPrompt,
    chain_guardrail_retry_prompt,
)
from catmodel.elicitation.contextual_risk import contextual_prompts
from catmodel.elicitation.eal_crosscheck import pooled_answer
from catmodel.elicitation.epistemic import (
    epistemic_mean,
    epistemic_quantiles,
)
from catmodel.results.execution import run_setup
from catmodel.results.sensitivity_summary import (
    EXTRAPOLATION_NOTE,
    sensitivity_summary,
)
from catmodel.setup.contextual_compilation import _log_no_incident
from catmodel.setup.contextual_node import CompileOptions, compile_contextual_node
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.store import build_setup
from catmodel.simulation import incident_runtime as runtime
from catmodel.simulation.capability_mixture import (
    CapabilityMixture,
    ConditionalCapabilityDraws,
    ConditionalView,
    tail_estimates,
)
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.tail_gate import TailGateControls, view_gate
from test_support.contextual_risk import (
    contextual_answer,
    contextual_case,
    contextual_instrument,
    matched_chain_answer,
)
from test_support.count_law_oracle import CountLawOracle
from test_support.results import (
    available_metric,
    present,
    reduced_arm,
    view_diagnostics,
)

KEYS = ("p5", "p25", "p50", "p75", "p95")
SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817)
# USD per qualified incident in the fixture answer: a victim loss of 100 and a
# provider-borne fine of 20, which the lab bears.
PER_COUNT = {"welfare": 100.0, "gross": 120.0, "lab_loss::lab": 20.0}
# The fixture's lab business loss, once in a year with an incident.
ANNUAL = {"lab_loss::lab": 70.0, "gross": 70.0}
LAWS = (
    Quantiles(p5=0.05, p25=0.1, p50=0.2, p75=0.4, p95=1.0),
    Quantiles(p5=0, p25=0, p50=1, p75=3, p95=50),
    Quantiles(p5=1, p25=1, p50=1, p75=2, p95=2),
    Quantiles(p5=0.001, p25=0.5, p50=2, p75=30, p95=4000),
)
WIDE = Quantiles(p5=0.05, p25=0.2, p50=0.5, p75=1.5, p95=8)
# Zero below p25: ranks under 0.25 read an expected count of exactly zero.
ZERO_LOWER = Quantiles(p5=0, p25=0, p50=0.5, p75=2, p95=8)


def _rate(median, ratio):
    """Count quantiles log-symmetric about ``median`` with p95/p5 ``ratio``."""
    sigma = math.log(ratio) / (2 * -SCORES[0])
    return Quantiles(
        **dict(
            zip(
                KEYS,
                (median * math.exp(sigma * z) for z in (*SCORES, -SCORES[0])),
                strict=True,
            )
        )
    )


def _answer(rate):
    payload = contextual_answer().model_dump(mode="json")
    payload["rate"] = rate.model_dump()
    return ContextualRiskAnswer.model_validate(payload)


def _artifact(instrument, rate):
    """Every case answers ``rate``, with a chain implying the same count law,
    so pooling keeps ``rate``; a triggered EAL cross-check is declined."""
    cases = []
    for prompt in contextual_prompts(instrument):
        answer = _answer(rate)
        # The fixture chain's exposure is one unit, so its rate is its count.
        chain = matched_chain_answer(prompt.node, answer, 1).model_copy(
            update={"opportunity_rate": rate}
        )
        chain_prompt = ContextualChainPrompt(
            instrument, prompt.node, prompt.respondent, prompt.anchor_score
        )
        guardrail_retried = (
            chain
            if chain_guardrail_retry_prompt(chain_prompt, chain) is not None
            else None
        )
        cases.append(
            contextual_case(
                instrument,
                prompt,
                answer,
                chain=chain,
                guardrail_retried=guardrail_retried,
            )
        )
    return ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(cases),
    )


def _law(
    rate,
    *,
    spread=1.0,
    tail_scale=1.0,
    group=None,
    tail_form: SpreadTailForm = "pure_stretch",
):
    instrument = contextual_instrument()
    return compile_contextual_node(
        instrument,
        instrument.nodes[0],
        {0.5: _answer(rate)},
        0.5,
        rank_group=group,
        options=CompileOptions(
            epistemic_tail_scale=tail_scale,
            rank_spread=spread,
            spread_tail_form=tail_form,
        ),
    )


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("spread", [0.5, 1.0, 1.3, 2.0])
@pytest.mark.parametrize("tail_scale", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("tail_form", SPREAD_TAIL_FORMS)
def test_stretched_rank_mean_is_exact(law, spread, tail_scale, tail_form):
    oracle = CountLawOracle(
        law, tail_scale=tail_scale, spread=spread, tail_form=tail_form
    )
    assert epistemic_mean(
        law, epistemic_tail_scale=tail_scale, spread=spread, tail_form=tail_form
    ) == pytest.approx(oracle.mean(), rel=1e-12)


def test_stretched_rank_mean_matches_stretched_ranks_and_refuses_bad_spreads():
    law = LAWS[0]
    ranks = (np.arange(200_000) + 0.5) / 200_000
    brute = np.mean(epistemic_quantiles(law, ndtr(0.7 * ndtri(ranks))))
    assert epistemic_mean(law, spread=0.7) == pytest.approx(brute, rel=1e-4)
    for spread in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="rank spread"):
            epistemic_mean(law, spread=spread)


def test_each_law_carries_its_count_law_and_its_mean_as_the_rate():
    law = _law(LAWS[0], group="public::scenario")
    assert isinstance(law, IncidentLaw)
    assert law.reference_annual_rate == pytest.approx(
        epistemic_mean(LAWS[0]), rel=1e-15
    )
    rank = law.parameter_rank
    assert rank is not None
    assert (rank.group_id, rank.count, rank.rank_spread) == (
        "public::scenario",
        LAWS[0],
        1,
    )
    # Spread and tail-width arms keep the same rank and read it through their
    # own count law, so their reference rates are their own means.
    for spread, tail_scale in ((1.5, 1.0), (1.0, 1.5), (0.7, 0.5)):
        law = _law(LAWS[0], spread=spread, tail_scale=tail_scale)
        assert isinstance(law, IncidentLaw) and law.parameter_rank is not None
        assert law.reference_annual_rate == pytest.approx(
            epistemic_mean(LAWS[0], epistemic_tail_scale=tail_scale, spread=spread),
            rel=1e-15,
        )
        assert law.parameter_rank.rank_spread == spread
        assert law.parameter_rank.epistemic_tail_scale == tail_scale
    zero = _law(Quantiles(p5=0, p25=0, p50=0, p75=0, p95=0))
    assert isinstance(zero, PopulationAbsence)
    assert zero.disposition == "structural_zero"


def _compiled(rate=WIDE, *, seed=7):
    instrument = contextual_instrument()
    setup = compile_contextual_setup(
        instrument,
        _artifact(instrument, rate),
        setup_id="ranks",
        years_per_state=4096,
        seed=seed,
    )
    state = next(state for state in setup.joint_states if state.weight > 0)
    return setup, compile_incident_state(setup, state.state_id)


def _scores(incidents, seed, years, stream="state"):
    return runtime.parameter_scores(incidents, seed, years, stream)


def test_year_scores_depend_on_seed_stream_group_and_year_only():
    _, compiled = _compiled()
    incidents = compiled.incidents
    first = runtime.parameter_rank_multipliers(incidents, _scores(incidents, 11, 4096))
    again = runtime.parameter_rank_multipliers(incidents, _scores(incidents, 11, 1024))
    (origin,) = first
    # A year's rank does not depend on how many years are simulated.
    np.testing.assert_array_equal(first[origin][:1024], again[origin])
    for seed, stream in ((12, "state"), (11, "another state")):
        other = runtime.parameter_rank_multipliers(
            incidents, _scores(incidents, seed, 4096, stream)
        )
        assert not np.array_equal(first[origin], other[origin])
    # The multiplier's mean over the rank is one, so the years' counts have
    # the reference rate as their mean.
    assert np.mean(first[origin]) == pytest.approx(1.0, rel=0.05)


def test_laws_sharing_a_group_read_one_rank_and_groups_are_independent():
    _, compiled = _compiled()
    (incident,) = compiled.incidents
    assert incident.parameter_rank is not None
    same = replace(incident, origin_id="same-group")
    other = replace(
        incident,
        origin_id="other-group",
        parameter_rank=incident.parameter_rank.model_copy(update={"group_id": "other"}),
    )
    incidents = (incident, same, other)
    multipliers = runtime.parameter_rank_multipliers(
        incidents, _scores(incidents, 11, 20_000)
    )
    np.testing.assert_array_equal(
        multipliers[incident.origin_id], multipliers["same-group"]
    )
    correlation = np.corrcoef(
        np.argsort(np.argsort(multipliers[incident.origin_id])),
        np.argsort(np.argsort(multipliers["other-group"])),
    )[0, 1]
    assert abs(correlation) < 0.03


def test_every_states_ranks_are_uniform_and_states_are_independent():
    """The one-year predictive law needs each state's rank to be uniform on
    (0, 1) in every year; the precision of all state-years needs the states'
    ranks independent. Oracles are the uniform law's own: each of 20 equal
    bins holds a Binomial(n, 1/20) count, the mean is 1/2 with variance 1/12,
    two independent uniforms have correlation zero with standard error
    1/sqrt(n) and both exceed 0.9 with probability 1/100. Ranks shared by two
    states would give correlation one and a joint exceedance of 1/10."""
    _, compiled = _compiled()
    years, tolerance = 40_000, 5.0
    ranks = [
        ndtr(next(iter(_scores(compiled.incidents, 11, years, stream).values())))
        for stream in ("node-1:respondent-a", "node-2:respondent-a", "node-1:b", "z")
    ]
    bin_error = math.sqrt(years * 0.05 * 0.95)
    for rank in ranks:
        assert rank.min() > 0 and rank.max() < 1
        counts = np.histogram(rank, bins=20, range=(0, 1))[0]
        assert np.max(np.abs(counts - years / 20)) < tolerance * bin_error
        assert abs(rank.mean() - 0.5) < tolerance * math.sqrt(1 / 12 / years)
        # Consecutive years are independent draws.
        assert abs(np.corrcoef(rank[:-1], rank[1:])[0, 1]) < tolerance / math.sqrt(
            years
        )
    for index, left in enumerate(ranks):
        for right in ranks[index + 1 :]:
            assert abs(np.corrcoef(left, right)[0, 1]) < tolerance / math.sqrt(years)
            both = np.mean((left > 0.9) & (right > 0.9))
            assert abs(both - 0.01) < tolerance * math.sqrt(0.01 * 0.99 / years)


def _with_paired_states(setup, state_id):
    """``setup`` with two zero-weight copies of a state, ``variant`` declaring
    the state's rank stream and ``unpaired`` declaring none, and an arm ``more``
    that raises the state's rate by a tenth."""
    state = next(row for row in setup.joint_states if row.state_id == state_id)
    (law,) = state.laws
    copies = (
        state.model_copy(
            update={"state_id": "variant", "weight": 0.0, "rank_stream_id": state_id}
        ),
        state.model_copy(update={"state_id": "unpaired", "weight": 0.0}),
    )
    more = Alternative(
        alternative_id="more",
        purpose="scientific_alternative",
        description="A tenth more incidents in one state.",
    )
    raised = LawOverride(
        state_id=state_id,
        law=law.model_copy(
            update={"reference_annual_rate": 1.1 * law.reference_annual_rate}
        ),
    )
    return replace(
        setup,
        joint_states=(*setup.joint_states, *copies),
        alternatives=(more,),
        overrides=HeldOverrides({"more": (raised,)}),
    )


def test_arms_and_declared_states_stay_paired_while_other_states_are_independent():
    """Pairing is by rank stream. Every arm of a state compiles to the state's
    own stream, so a law-changing arm reads its control's ranks year by year;
    a state that declares another state's stream, as a variant state declares
    its headline counterpart's, reads that state's ranks; an undeclared copy
    of the same law reads independent ones."""
    setup, compiled = _compiled()
    # An undeclared state serializes without the field, so setups that declare
    # no stream keep their identity.
    assert all(
        "rank_stream_id" not in row.model_dump(mode="json")
        for row in setup.joint_states
    )
    extended = _with_paired_states(setup, compiled.state_id)
    years = 20_000

    def scores(state_id, arm="reference"):
        state = compile_incident_state(extended, state_id, arm)
        values = runtime.parameter_scores(
            state.incidents, extended.execution.seed, years, state.rank_stream_id
        )
        return state, next(iter(values.values()))

    control, control_scores = scores(compiled.state_id)
    assert control.rank_stream_id == compiled.state_id
    arm, arm_scores = scores(compiled.state_id, "more")
    assert arm.conditional_key != control.conditional_key
    np.testing.assert_array_equal(arm_scores, control_scores)
    variant, variant_scores = scores("variant")
    assert (variant.state_id, variant.rank_stream_id) == ("variant", compiled.state_id)
    np.testing.assert_array_equal(variant_scores, control_scores)
    unpaired, unpaired_scores = scores("unpaired")
    assert unpaired.rank_stream_id == "unpaired"
    assert abs(np.corrcoef(unpaired_scores, control_scores)[0, 1]) < 5 / math.sqrt(
        years
    )
    # A paired state's counts still invert its own uniforms: only the rank, and
    # through it each year's rate, is common to the pair.
    plan = extended.execution.model_copy(update={"simulations": years})
    (paired,) = runtime._origin_counts(variant, plan).values()
    (reference,) = runtime._origin_counts(control, plan).values()
    assert not np.array_equal(paired, reference)
    assert np.corrcoef(paired, reference)[0, 1] > 5 / math.sqrt(years)


def test_simulated_counts_vary_by_year_and_average_the_reference_rate():
    setup, compiled = _compiled(_rate(40.0, 20.0))
    plan = setup.execution.model_copy(update={"simulations": 20_000})
    counts = runtime._origin_counts(compiled, plan)
    (values,) = counts.values()
    (incident,) = compiled.incidents
    assert values.mean() == pytest.approx(incident.annual_rate, rel=0.03)
    # Poisson at a fixed rate has variance equal to its mean; the year ranks
    # add the count law's own variance.
    assert values.var() > 3 * values.mean()


def test_count_changing_arms_keep_every_years_count_paired_with_the_control():
    """Each year's count inverts one uniform addressed by the state, origin and
    year, which no count law changes. An arm that reads a tenth more rate or a
    wider rank spread changes only the year's rate, so the year's count moves
    with it: never below the control's where the rate is higher and never
    above where it is lower."""
    setup, compiled = _compiled(_rate(40.0, 20.0))
    plan = setup.execution.model_copy(update={"simulations": 20_000})
    (incident,) = compiled.incidents
    assert incident.parameter_rank is not None

    def counts_and_rates(variant):
        state = replace(compiled, incidents=(variant,))
        scores = runtime.parameter_scores(
            state.incidents, plan.seed, plan.simulations, state.rank_stream_id
        )
        (counts,) = runtime._origin_counts(state, plan, scores).values()
        multipliers = runtime.parameter_rank_multipliers(state.incidents, scores)
        multiplier = multipliers.get(variant.origin_id, np.ones(plan.simulations))
        return counts, variant.annual_rate * multiplier

    control, rate = counts_and_rates(incident)
    wider = incident.parameter_rank.model_copy(update={"rank_spread": 1.5})
    for arm in (
        replace(incident, annual_rate=1.1 * incident.annual_rate),
        replace(incident, parameter_rank=wider),
    ):
        counts, arm_rate = counts_and_rates(arm)
        assert np.all((counts - control) * np.sign(arm_rate - rate) >= 0)
        assert np.any(counts != control)
    # At a fixed rate, comonotone counts at 1.1 times the rate differ from the
    # control's by a tenth of its mean and a count or so; independent draws
    # would differ with about twice a count's variance.
    fixed = replace(incident, parameter_rank=None)
    plain, _ = counts_and_rates(fixed)
    more, _ = counts_and_rates(replace(fixed, annual_rate=1.1 * fixed.annual_rate))
    assert np.all(more >= plain)
    assert np.var(more - plain) < 0.05 * np.var(plain)


def _oracle(incident):
    """The independent oracle of a compiled law's five count quantiles, which
    pooling rounds, at the law's rank spread, tail scale and tail form."""
    law = incident.parameter_rank
    assert law is not None
    return CountLawOracle(
        law.count,
        tail_scale=law.epistemic_tail_scale,
        spread=law.rank_spread,
        tail_form=law.tail_form,
    )


def _rank_expectation(oracle, function):
    """``E[function(Q(Phi(Z)))]`` over the year's normal score, by adaptive
    quadrature split at the count law's kinks."""
    edges = [-9.0, *oracle.kinks(), 9.0]
    return math.fsum(
        integrate.quad(
            lambda z: function(oracle.at_score(z)) * norm.pdf(z),
            low,
            high,
            epsabs=1e-13,
            epsrel=1e-11,
        )[0]
        for low, high in pairwise(sorted(set(edges)))
    )


def _event_variance(oracle, event, *, states, years, shared):
    """The exact variance of the mixture's estimate of ``P(event)`` from
    ``states`` identical equal-mass states of ``years`` years, with
    ``event(rate)`` the event's probability in a year at expected count
    ``rate``: every state-year an independent Bernoulli draw when states read
    their own ranks, and, when they share a year's rank, the variance of the
    probability over the rank plus the states' average conditional variance."""
    mean = _rank_expectation(oracle, event)
    square = _rank_expectation(oracle, lambda rate: event(rate) ** 2)
    if not shared:
        return mean, mean * (1 - mean) / (states * years)
    return mean, ((square - mean**2) + (mean - square) / states) / years


def _blocks(compiled, plan, *, states, years, blocks, shared):
    """``blocks`` independent replicates of ``states`` copies of one compiled
    state at ``years`` years each: a year's draws depend only on the seed, the
    state and the year, so disjoint runs of years are independent replicates
    of the whole simulation. Returns the engine's draws by state."""
    draws = []
    for index in range(states):
        state = replace(
            compiled,
            state_id=f"state-{index}",
            rank_stream_id="one stream" if shared else f"state-{index}",
        )
        draws.append(
            runtime.simulate_incident_state(
                plan, state, response_sha256="fixture", years=years * blocks
            ).draws
        )
    return draws


@pytest.mark.stochastic
def test_loss_probability_has_the_precision_of_all_state_years():
    """24 identical states of 256 years, a count law whose rank moves the
    annual probability of an incident from near zero to near one: the welfare
    view is 100 per incident, so a year has a loss when its count is positive.

    The oracle is the count law read by ``CountLawOracle`` and the Poisson
    zero probability, with no engine code. With each state on its own rank
    stream the 6,144 state-years are independent draws, so the estimate's
    variance over 100 independent replicates is p(1 - p)/6,144 and the tail
    gate's batch-means half-width reports it. Declaring one stream for every
    state couples their years through one rank a year: it leaves the mean and
    each state's analytic EAL unchanged and multiplies the variance by the
    oracle's ratio, which state-keyed streams avoid."""
    states, years, blocks = 24, 256, 100
    rate = _rate(0.05, 400.0)
    setup, compiled = _compiled(rate)
    plan = setup.execution.model_copy(update={"simulations": years * blocks})
    (incident,) = compiled.incidents
    oracle = _oracle(incident)

    def positive(expected_count):
        return -math.expm1(-expected_count)

    # Five standard errors of a variance estimated from normal replicates, on
    # the log scale; the two couplings' variances differ by far more.
    factor = math.exp(5 * math.sqrt(2 / (blocks - 1)))
    variances = {}
    for shared in (False, True):
        probability, variance = _event_variance(
            oracle, positive, states=states, years=years, shared=shared
        )
        draws = _blocks(
            compiled, plan, states=states, years=years, blocks=blocks, shared=shared
        )
        # The analytic EAL does not depend on how ranks are addressed: 100
        # per incident times the count law's mean over its rank.
        for draw in draws:
            native = draw.views["welfare"].native_mean
            assert native is not None
            assert native.value == pytest.approx(100 * oracle.mean(), rel=1e-9)
        losses = np.stack([draw.views["welfare"].array for draw in draws])
        estimates = (losses > 0).reshape(states, blocks, years).mean(axis=(0, 2))
        assert abs(estimates.mean() - probability) < 5 * math.sqrt(variance / blocks)
        variances[shared] = (float(np.var(estimates, ddof=1)), variance)
        assert variance / factor < variances[shared][0] < variance * factor
        # The simulated mean estimates the analytic EAL under either coupling.
        means = losses.reshape(states, blocks, years).mean(axis=(0, 2))
        assert abs(means.mean() - 100 * oracle.mean()) < 5 * means.std(
            ddof=1
        ) / math.sqrt(blocks)
        if shared:
            continue
        # The reported half-width is Student's t times the batch-means
        # standard error, whose square is unbiased for the estimate's variance
        # with 15 degrees of freedom in each replicate.
        reported = []
        for block in range(blocks):
            window = slice(block * years, (block + 1) * years)
            mixture = CapabilityMixture(
                {
                    str(index): replace(
                        draw,
                        stratum_sha256=str(index),
                        views={
                            "welfare": replace(
                                draw.views["welfare"],
                                array=draw.views["welfare"].array[window],
                            )
                        },
                    )
                    for index, draw in enumerate(draws)
                },
                {str(index): 1 / states for index in range(states)},
            )
            gate = view_gate(
                mixture,
                "welfare",
                tail_estimates(mixture, "welfare", (0.9,)),
                TailGateControls(),
                mean_exists=True,
            )
            reported.append(gate["probability_positive"]["half_width"])
        t = float(student_t.ppf(0.975, 15))
        assert np.mean(np.square(reported)) / t**2 == pytest.approx(
            variance, rel=5 * math.sqrt(2 / (15 * blocks))
        )
    # The gain itself: sharing costs the oracle's ratio in variance, here
    # about tenfold, and the measured variances reproduce it.
    oracle_ratio = variances[True][1] / variances[False][1]
    assert oracle_ratio > 8
    measured = variances[True][0] / variances[False][0]
    ratio_factor = math.exp(5 * math.sqrt(4 / (blocks - 1)))
    assert oracle_ratio / ratio_factor < measured < oracle_ratio * ratio_factor


@pytest.mark.stochastic
def test_one_in_ten_loss_has_the_precision_of_all_state_years():
    """The 1-in-10 annual loss of 24 identical states of 256 years whose
    counts, about 85 a year on average, follow their rank closely.

    A sample quantile's variance is that of the estimated probability of
    staying at or below the true quantile, over the squared density there
    (the delta method), so the ratio of the two couplings' VaR variances is
    the ratio of those probabilities' variances, which the count law gives
    exactly: independent state-years against one shared rank a year. The
    replicates' variances reproduce that ratio, about twentyfold."""
    states, years, blocks, level = 24, 256, 100, 0.9
    rate = _rate(40.0, 20.0)
    setup, compiled = _compiled(rate)
    plan = setup.execution.model_copy(update={"simulations": years * blocks})
    (incident,) = compiled.incidents
    oracle = _oracle(incident)

    def at_most(count):
        return lambda expected_count: float(poisson.cdf(count, expected_count))

    # The least count whose probability of not being exceeded reaches 0.9.
    quantile = bisect_left(
        range(10_000),
        level,
        key=lambda count: _rank_expectation(oracle, at_most(count)),
    )

    measured, exact = {}, {}
    for shared in (False, True):
        _, exact[shared] = _event_variance(
            oracle, at_most(quantile), states=states, years=years, shared=shared
        )
        counts = []
        for index in range(states):
            state = replace(
                compiled,
                state_id=f"state-{index}",
                rank_stream_id="one stream" if shared else f"state-{index}",
            )
            (drawn,) = runtime._origin_counts(state, plan).values()
            counts.append(drawn)
        estimates = []
        for block in range(blocks):
            window = slice(block * years, (block + 1) * years)
            mixture = CapabilityMixture(
                {
                    str(index): ConditionalCapabilityDraws(
                        stratum_sha256=str(index),
                        response_sha256="fixture",
                        seed=0,
                        views={"welfare": ConditionalView(100.0 * drawn[window])},
                    )
                    for index, drawn in enumerate(counts)
                },
                {str(index): 1 / states for index in range(states)},
            )
            estimates.append(tail_estimates(mixture, "welfare", (level,))[level].var)
        # The estimates centre on the oracle's quantile within a count or two.
        assert np.median(estimates) == pytest.approx(100 * quantile, abs=200)
        measured[shared] = float(np.var(estimates, ddof=1))
    oracle_ratio = exact[True] / exact[False]
    assert oracle_ratio > 15
    factor = math.exp(5 * math.sqrt(4 / (blocks - 1)))
    assert (
        oracle_ratio / factor < measured[True] / measured[False] < oracle_ratio * factor
    )


def test_ranks_below_a_zero_quantile_give_years_with_no_incident():
    """The count law is zero below p25, so about a quarter of the years read
    an expected count of exactly zero and simulate no incident at all."""
    setup, compiled = _compiled(ZERO_LOWER)
    plan = setup.execution.model_copy(update={"simulations": 20_000})
    (incident,) = compiled.incidents
    multipliers = runtime.parameter_rank_multipliers(
        compiled.incidents,
        runtime.parameter_scores(
            compiled.incidents, plan.seed, plan.simulations, compiled.rank_stream_id
        ),
    )[incident.origin_id]
    zero = multipliers == 0
    assert zero.mean() == pytest.approx(0.25, abs=0.01)
    (counts,) = runtime._origin_counts(compiled, plan).values()
    assert not counts[zero].any()
    assert counts[~zero].mean() == pytest.approx(incident.annual_rate / 0.75, rel=0.05)


def _run(tmp_path, rate, seed):
    instrument = contextual_instrument()
    setup = compile_contextual_setup(
        instrument,
        _artifact(instrument, rate),
        setup_id="headline",
        years_per_state=512,
        seed=seed,
    )
    package = build_setup(
        destination=tmp_path / f"package-{seed}", setup=setup, code_commit="fixture"
    )
    return run_setup(
        setup_path=package.root, output=tmp_path / f"run-{seed}", code_commit="fixture"
    )


def _metric(report, view="welfare") -> dict[str, Any]:
    return cast(
        dict[str, Any],
        available_metric(reduced_arm(report["arms"]["reference"])["metrics"][view]),
    )


def test_headline_eal_is_the_exact_rank_mean_and_the_years_estimate_it(tmp_path):
    reports = [_run(tmp_path, ZERO_LOWER, seed) for seed in (3, 4)]
    instrument = contextual_instrument()
    pooled_rate = pooled_answer(_artifact(instrument, ZERO_LOWER).cases[0]).rate
    assert pooled_rate is not None
    ranks = (np.arange(1_000_000) + 0.5) / 1_000_000
    counts = epistemic_quantiles(pooled_rate, ranks)
    brute = float(np.mean(counts))
    # The probability of at least one incident, over the same ranks.
    occurrence = float(np.mean(-np.expm1(-counts)))
    exhibit = reports[0]["sensitivity_summary"]["frequency_estimate"]["views"]
    for view, per_count in PER_COUNT.items():
        metrics = [_metric(report, view) for report in reports]
        annual = ANNUAL.get(view, 0.0) * occurrence
        for report, metric in zip(reports, metrics, strict=True):
            eal = metric["expected_annual_loss"]
            assert eal - annual == pytest.approx(
                per_count * epistemic_mean(pooled_rate), rel=5e-4
            )
            assert eal == pytest.approx(per_count * brute + annual, rel=5e-4)
            assert metric["eal_estimator"] == "native_analytic_mean"
            assert metric["eal_precision"]["precision_basis"] == (
                "deterministic_integration"
            )
            # Every year reads its own rank, so the simulated mean estimates
            # the analytic one.
            mean = view_diagnostics(report, "reference", view)["mean_diagnostics"]
            error = present(mean["empirical_mean_standard_error"])
            assert abs(metric["eal_reconciliation_difference"]) < 5 * error
            assert "sampled_rank_expected_annual_loss" not in metric
        assert metrics[0]["expected_annual_loss"] == metrics[1]["expected_annual_loss"]
        rows = exhibit[view]
        assert rows["by_arm"]["final"]["expected_annual_loss"] == pytest.approx(
            metrics[0]["expected_annual_loss"], rel=1e-12
        )
        assert "rank_integrated" not in rows["by_arm"]


@pytest.mark.parametrize("tail_form", SPREAD_TAIL_FORMS)
def test_no_incident_reads_the_compiled_spread_tail_form(tail_form):
    law = _law(LAWS[1], spread=1.5, tail_form=tail_form)
    assert isinstance(law, IncidentLaw)
    rank = law.parameter_rank
    assert rank is not None
    assert rank.tail_form == tail_form
    assert "rank_spread_tail_form" not in law.conditioning_context
    survival = 0.01
    scores = ndtri((np.arange(4096) + 0.5) / 4096)
    oracle = CountLawOracle(
        rank.count,
        tail_scale=rank.epistemic_tail_scale,
        spread=rank.rank_spread,
        tail_form=tail_form,
    )
    quantiles = np.asarray([oracle.at_score(float(score)) for score in scores])
    expected = math.log(float(np.mean(np.exp(-survival * quantiles))))
    assert _log_no_incident(law, survival) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "spread_arm", ("epistemic_spread_widened", "epistemic_tail_widened")
)
def test_a_leading_widened_spread_arm_is_flagged_in_the_tornado(spread_arm):
    def contrast(difference):
        return {
            "gross": {
                "expected_annual_loss": {
                    "reference": 10.0,
                    "alternative": 10.0 + difference,
                    "absolute_difference": difference,
                    "relative_change": difference / 10,
                }
            }
        }

    report = {
        "arms": {
            "reference": {
                "metrics": {
                    "gross": {"status": "available", "expected_annual_loss": 10.0}
                }
            },
            spread_arm: {},
            "tail_shape_higher": {},
        },
        "comparisons": {
            spread_arm: contrast(40.0),
            "tail_shape_higher": contrast(5.0),
        },
        "coverage": {"comparisons": {"status": "complete"}},
    }
    tornado = sensitivity_summary(report, reference_arm="reference", purposes={})[
        "tornado"
    ]
    assert tornado["extrapolation_note"] == EXTRAPOLATION_NOTE
    assert [row["arm"] for row in tornado["rows"]] == [
        spread_arm,
        "tail_shape_higher",
    ]
    report["comparisons"][spread_arm] = contrast(1.0)
    tornado = sensitivity_summary(report, reference_arm="reference", purposes={})[
        "tornado"
    ]
    assert tornado["extrapolation_note"] is None
