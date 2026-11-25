"""Fixed-VMR negative-binomial annual counts have their declared closed-form law.

A count law with annual mean ``lam`` and variance-to-mean ratio ``d > 1`` has
PGF ``G(z) = (1 + beta (1 - z))^(-r)`` with ``beta = d - 1`` and
``r = lam / beta``. Its cumulants are

    k1 = r beta = lam
    k2 = r beta (1 + beta) = d lam
    k4 = r beta (1 + beta) (1 + 6 beta + 6 beta^2)

and ``P(N = 0) = G(0) = d^(-r)``. The oracle uses only these contract-level
formulas, not the engine's sampler parameterization. Each loss has a fixed
amount of one, so the origin's annual scenario loss equals its annual count and
the check runs through public compilation and the production annual simulator.

Tolerances are five standard errors of the estimator under the exact law:
``sqrt(k2 / n)`` for the sample mean, ``sqrt((k4 + 2 k2^2) / n)`` for the
sample variance and ``sqrt(p0 (1 - p0) / n)`` for the empirical zero
frequency.
"""

import copy
import math

import numpy as np
import pytest

from catmodel.contracts._setup_codec import decode_scientific
from catmodel.contracts.incident import ExecutionPlan
from catmodel.simulation.incident import compile_incident_state
from catmodel.simulation.incident_runtime import simulate_incident_state
from test_support.study import study_payload

RATE = 3.0
VARIANCE_TO_MEAN = 2.0
SIMULATIONS = 100_000


def _setup(arm_correlation: float | None):
    payload = study_payload()
    law = payload["joint_states"][0]["laws"][0]
    law.update(
        reference_annual_rate=RATE,
        count_family="negative_binomial",
        count_variance_to_mean=VARIANCE_TO_MEAN,
    )
    law["segments"][0]["kernel"]["member_outcomes"][0]["components"][0]["amount"] = 1
    required = ["reference"]
    if arm_correlation is not None:
        # The Gaussian count-rank experiment must keep each fixed NB marginal.
        # The coupled origin reports to its own scenario so the checked
        # scenario view holds exactly one origin's annual count.
        other = copy.deepcopy(law)
        other.update(origin_id="other", scenario_id="other")
        other["segments"][0]["kernel"]["member_outcomes"][0]["components"][0][
            "scenario_ids"
        ] = ["other"]
        payload["joint_states"][0]["laws"].append(other)
        payload["populations"].append(
            {**payload["populations"][0], "scenario_id": "other"}
        )
        experiment = {
            "kind": "fixed_marginal_gaussian",
            "origin_groups": {"origin": "all", "other": "all"},
        }
        payload["alternatives"] = [
            {
                "alternative_id": "control",
                "purpose": "scientific_alternative",
                "description": "Independent count-rank control.",
                "count_experiment": {**experiment, "correlation": 0},
                "matched_control_id": "control",
            },
            {
                "alternative_id": "coupled",
                "purpose": "scientific_alternative",
                "description": "Coupled count ranks with fixed marginals.",
                "count_experiment": {**experiment, "correlation": arm_correlation},
                "matched_control_id": "control",
            },
        ]
        required += ["control", "coupled"]
    plan = ExecutionPlan(
        plan_id="count-law-oracle",
        required_arms=tuple(required),
        requested_views=("scenario::scenario::gross",),
        simulations=SIMULATIONS,
        seed=8_675_309,
    )
    return decode_scientific(payload, plan, setup_id="count-law-oracle"), plan


def _annual_counts(arm: str, arm_correlation: float | None) -> np.ndarray:
    setup, plan = _setup(arm_correlation)
    draw, unavailable, _ = simulate_incident_state(
        plan, compile_incident_state(setup, "state", arm), response_sha256="a" * 64
    )
    assert not unavailable
    counts = np.asarray(draw.annual_losses["scenario::scenario::gross"])
    assert np.array_equal(counts, np.round(counts))
    return counts


@pytest.mark.stochastic
@pytest.mark.parametrize(
    ("arm", "arm_correlation"),
    [("reference", None), ("coupled", 0.6)],
    ids=["reference", "gaussian-rank-marginal"],
)
def test_negative_binomial_counts_have_declared_mean_variance_and_zero_mass(
    arm, arm_correlation
):
    counts = _annual_counts(arm, arm_correlation)
    n = counts.size
    beta = VARIANCE_TO_MEAN - 1
    size = RATE / beta
    k2 = size * beta * (1 + beta)
    k4 = size * beta * (1 + beta) * (1 + 6 * beta + 6 * beta**2)
    zero_mass = VARIANCE_TO_MEAN**-size

    assert abs(counts.mean() - RATE) <= 5 * math.sqrt(k2 / n)
    assert abs(counts.var(ddof=1) - k2) <= 5 * math.sqrt((k4 + 2 * k2**2) / n)
    assert abs(np.mean(counts == 0) - zero_mass) <= 5 * math.sqrt(
        zero_mass * (1 - zero_mass) / n
    )
