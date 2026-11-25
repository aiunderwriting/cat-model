"""Scalar drivers for one component's and one payment kernel's draws at given
ranks, computed as the event-addressed samplers compute each event's."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from catmodel.contracts.components import (
    ComponentAmountLaw,
    ComponentObligation,
    ComponentPayment,
    NativePaymentKernel,
)
from catmodel.contracts.native_severity import shared_native_severity_sampler
from catmodel.probability.distributions import sample_affine_capped_severity
from catmodel.probability.liability import LiabilityAttenuation
from catmodel.simulation.authored.payments import constant_native_payment_fraction


@dataclass(frozen=True)
class SampledAmount:
    """One component's realized amount."""

    component_id: str
    amount: float


def _rank(ranks, scope, group, rng):
    key = (scope, group)
    if key not in ranks:
        ranks[key] = ranks[group] if group in ranks else float(rng.random())
    return ranks[key]


def sample_component_amounts(
    laws: Sequence[ComponentAmountLaw],
    rng: np.random.Generator,
    *,
    rank_uniforms: Mapping[str | tuple[str, str], float] | None = None,
) -> tuple[SampledAmount, ...]:
    """Each law's amount at its rank group's uniform (``rank_uniforms``, by
    group or by scope and group; a group without one draws it from ``rng``):
    zero with a zero multiplier, a fixed amount scaled by both multipliers,
    or the native severity inverse at the rank under them."""
    ranks: dict = dict(rank_uniforms or {})
    sampled = []
    for law in laws:
        if law.multiplier == 0 or law.post_multiplier == 0:
            amount = 0.0
        elif law.amount is not None:
            amount = law.amount * law.multiplier * law.post_multiplier
        else:
            assert law.severity is not None
            uniform = _rank(ranks, law.rank_scope, law.rank_group_id, rng)
            amount = float(
                sample_affine_capped_severity(
                    shared_native_severity_sampler(law.severity),
                    np.array([uniform]),
                    scale=law.multiplier,
                    post_multiplier=law.post_multiplier,
                )[0]
            )
        sampled.append(SampledAmount(law.component_id, amount))
    return tuple(sampled)


def sample_native_payments(
    kernel: NativePaymentKernel,
    rng: np.random.Generator,
    *,
    rank_uniforms: Mapping[str | tuple[str, str], float],
) -> tuple[tuple[ComponentObligation, ...], tuple[ComponentPayment, ...]]:
    """A native payment kernel's obligations and payments: its liability
    share, constant at an endpoint and otherwise drawn at the severity of its
    rank group's uniform with recovery and share uniforms from ``rng``, for its
    liability components, and the whole of its direct components."""
    fraction = constant_native_payment_fraction(kernel)
    if fraction is None:
        assert kernel.liability is not None
        uniform = _rank(
            dict(rank_uniforms), kernel.rank_scope, kernel.rank_group_id, rng
        )
        severity = sample_affine_capped_severity(
            shared_native_severity_sampler(kernel.source_severity), np.array([uniform])
        )
        fraction = float(
            LiabilityAttenuation(
                **kernel.liability.model_dump()
            ).sample_conditional_from_uniforms(
                severity, np.array([rng.random()]), np.array([rng.random()])
            )[0]
        )
    amounts = [(identifier, fraction) for identifier in kernel.liability_component_ids]
    amounts.extend((identifier, 1.0) for identifier in kernel.direct_component_ids)
    return (
        tuple(
            ComponentObligation(
                component_id=identifier,
                entity_id=kernel.entity_id,
                fraction=value,
                basis="reduced_form_payment_proxy",
            )
            for identifier, value in amounts
        ),
        tuple(
            ComponentPayment(
                component_id=identifier,
                entity_id=kernel.entity_id,
                fraction=value,
                basis="reduced_form_payment_proxy",
            )
            for identifier, value in amounts
        ),
    )
