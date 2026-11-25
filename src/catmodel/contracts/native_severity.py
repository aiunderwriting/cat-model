"""Exact generative native severity laws for evidence and study bindings.

``native_severity_sampler`` and ``native_severity_from_sampler`` convert a law
to its evaluator and back without loss.
"""

from __future__ import annotations

import math
import sys
from dataclasses import fields
from functools import lru_cache
from typing import Literal, Self

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from catmodel.contracts.shared_fields import SharedFieldsSet
from catmodel.probability.distributions import (
    Distribution,
    MixtureSeverityDistribution,
    SeveritySampler,
)


class NativeSeverityComponent(SharedFieldsSet):
    model_config = ConfigDict(extra="forbid", frozen=True)

    p5: float = Field(allow_inf_nan=False)
    p25: float = Field(allow_inf_nan=False)
    p50: float = Field(allow_inf_nan=False)
    p75: float = Field(allow_inf_nan=False)
    p95: float = Field(allow_inf_nan=False)
    gpd_shape: float | None = Field(default=None, allow_inf_nan=False)
    severity_cap: float | None = Field(default=None, allow_inf_nan=False)
    severity_minimum: float | None = Field(default=None, allow_inf_nan=False)
    severity_minimum_strict: bool = False
    gpd_scale: float | None = Field(default=None, allow_inf_nan=False)
    gpd_tail_ratio: float | None = Field(default=None, allow_inf_nan=False)
    large_loss_amount: float | None = Field(default=None, allow_inf_nan=False)
    large_loss_exceedance: float | None = Field(default=None, allow_inf_nan=False)
    # The two-anchor tail's second anchor. A law without one is stored without
    # these fields, so it keeps the encoding, and the identities, it had
    # before they existed.
    very_large_loss_amount: float | None = Field(
        default=None, allow_inf_nan=False, exclude_if=lambda value: value is None
    )
    very_large_loss_exceedance: float | None = Field(
        default=None, allow_inf_nan=False, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def valid_native_law(self) -> Self:
        Distribution(**self.model_dump(mode="python"))
        return self


class NativeSeverityLaw(SharedFieldsSet):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["native_severity"] = "native_severity"
    # The continuations outside [p5, p95] and a continuity GPD scale never
    # take an end slope below the smaller of the end segment's probability
    # secant and the lognormal slope through it; a component may carry a
    # large-loss anchor, which replaces its tail above p99, and with it a
    # second anchor, which replaces its tail above the first; the
    # quantile-spline body interpolates the log excess over the floor on the
    # normal-score scale. A law recorded under another schema version is
    # refused.
    schema_version: Literal[1] = 1
    representation: Literal["distribution", "mixture"]
    components: tuple[NativeSeverityComponent, ...] = Field(min_length=1)
    weights: tuple[float, ...] = Field(min_length=1)
    severity_cap: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_representation(self) -> Self:
        if len(self.components) != len(self.weights):
            raise ValueError("native severity components and weights must align")
        if any(
            not math.isfinite(weight) or weight < 0.0 or weight > 1.0
            for weight in self.weights
        ):
            raise ValueError("native severity weights must be finite and nonnegative")
        if not any(weight > 0.0 for weight in self.weights):
            raise ValueError("native severity requires positive component mass")
        if not math.isclose(
            math.fsum(self.weights),
            1.0,
            rel_tol=0.0,
            abs_tol=4 * sys.float_info.epsilon * len(self.weights),
        ):
            raise ValueError(
                "native severity stores already normalized component weights"
            )
        if self.representation == "distribution" and (
            len(self.components) != 1
            or self.weights != (1.0,)
            or self.severity_cap is not None
        ):
            raise ValueError(
                "one native distribution requires one unit-weight component and no additional cap"
            )
        native_severity_sampler(self)
        return self


def native_severity_from_sampler(sampler: SeveritySampler) -> NativeSeverityLaw:
    """Capture every generative field without fitting diagnostic quantiles."""

    def component(law: Distribution) -> NativeSeverityComponent:
        return NativeSeverityComponent.model_validate(
            {
                field.name: getattr(law, field.name)
                for field in fields(Distribution)
                if field.init
            }
        )

    if isinstance(sampler, Distribution):
        return NativeSeverityLaw(
            representation="distribution",
            components=(component(sampler),),
            weights=(1.0,),
        )
    if isinstance(sampler, MixtureSeverityDistribution):
        return NativeSeverityLaw(
            representation="mixture",
            components=tuple(component(law) for law in sampler.components),
            weights=tuple(float(weight) for weight in sampler.weights),
            severity_cap=sampler.severity_cap,
        )
    raise ValueError(
        "native severity serialization requires a declared Distribution or MixtureSeverityDistribution"
    )


def native_severity_sampler(
    law: NativeSeverityLaw,
) -> Distribution | MixtureSeverityDistribution:
    """A new evaluator of ``law``, owned by the caller."""
    components = [
        Distribution(**component.model_dump(mode="python"))
        for component in law.components
    ]
    if law.representation == "distribution":
        return components[0]
    result = MixtureSeverityDistribution(
        components=components,
        weights=np.asarray(law.weights, dtype=float),
        severity_cap=law.severity_cap,
    )
    # Re-normalizing persisted weights can move their authored atom boundaries.
    result.weights = np.asarray(law.weights, dtype=float).copy()
    return result


@lru_cache(maxsize=128)
def _shared_sampler(specification: str) -> SeveritySampler:
    return native_severity_sampler(NativeSeverityLaw.model_validate_json(specification))


def shared_native_severity_sampler(law: NativeSeverityLaw) -> SeveritySampler:
    """``law``'s evaluator, built once per distinct law and shared by every
    caller, so a caller must not change it; ``native_severity_sampler``
    builds one the caller owns."""
    return _shared_sampler(law.model_dump_json())
