"""Seeded draws for tests of a severity law or a liability kernel.

The engine gives each law's inverse transform the keyed uniforms of an event;
these helpers give the same transforms a generator's uniforms.
"""

import numpy as np

from catmodel.probability.distributions import SeveritySampler
from catmodel.probability.liability import LiabilityAttenuation


def sample_severity(
    law: SeveritySampler, count: int, *, rng: np.random.Generator
) -> np.ndarray:
    """``count`` capped amounts of ``law`` at uniforms drawn from ``rng``."""
    return law.sample_from_uniforms(rng.uniform(0.0, 1.0, count))


def sample_liability_shares(
    attenuation: LiabilityAttenuation,
    severities: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """One developer-liability share per severity, at recovery uniforms and
    then share uniforms drawn from ``rng``."""
    count = len(severities)
    return attenuation.sample_conditional_from_uniforms(
        severities, rng.uniform(size=count), rng.uniform(size=count)
    )
