"""Replay-time results conditioned on representative public capability paths."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import numpy as np

from catmodel.forecast.public_frontier_quadrature import (
    base_progress_cdf,
    base_progress_quantile,
)
from catmodel.results.respondent_sampling import (
    NOT_ANALYTIC_REASON,
    SampleGroups,
    analytic_eal_terms,
    sample_groups,
    sampling_view,
    standard_error,
)
from catmodel.results.sensitivity_results import REWEIGHTED_EXHIBIT_REASON
from catmodel.setup.public_frontier import public_state_node
from catmodel.setup.public_frontier_comparisons import (
    CONTEXT_VARIANT_FAMILIES,
    SAFEGUARD_CONTROL,
)
from catmodel.simulation.capability_mixture import (
    ConditionalCapabilityDraws,
    known_zero_law,
)
from catmodel.simulation.incident_runtime import state_key
from catmodel.simulation.tail_gate import (
    TailGateControls,
    batch_half_width,
    batch_layout,
    batch_masses,
    batch_support_reason,
    minimum_batch_support,
    unconverged_reason,
)

REPRESENTATIVE_PERCENTILES = (0.1, 0.5, 0.9)
"""Base-law frontier-progress quantiles shown as representative paths
(``NUM-CAPABILITY-SCENARIOS``)."""

NODE_MATCH_RELATIVE_TOLERANCE = 1e-6
"""Relative log-progress tolerance within which a percentile path is a collected
node rather than an interpolation."""

CURVE_LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999)
"""Annual loss quantile display levels; the tail gate still decides each point's
status."""

CURVE_THRESHOLDS = (1e9, 1e10, 2.5e10, 5e10, 1e11, 2e11, 5e11, 1e12)
"""Annual loss exceedance thresholds in the native monetary unit, USD."""

CURRENT_FRONTIER_ARM = "public_forecast::no_progress"
_ZERO_NODE = "node-00"
_CURRENT_FRONTIER = (
    "current_frontier",
    "Current frontier",
    "Every lab's capability stays at its level at the start of the study year (the no-progress counterfactual).",
)
_PERCENTILE_COPY = {
    0.1: (
        "Slower progress (10th percentile)",
        "Capability grows along a path at the base forecast's 10th-percentile frontier progress.",
    ),
    0.5: (
        "Expected progress (median)",
        "Capability grows along a path at the base forecast's median frontier progress.",
    ),
    0.9: (
        "Faster progress (90th percentile)",
        "Capability grows along a path at the base forecast's 90th-percentile frontier progress.",
    ),
}
_INTERPRETATION = (
    "Annual loss laws and EALs conditional on the no-progress counterfactual or "
    "one representative percentile of the base forecast's frontier progress, "
    "the public frontier's gain over its as-of level at the middle of the study "
    "year. Percentile paths are single conditioning paths, not "
    "probability-weighted forecasts."
)
_PERCENTILE_PROBABILITY_NOTE = (
    "A percentile path is a single conditioning path with no probability mass "
    "of its own; it is a representative point, not a band."
)
_CURRENT_PROBABILITY_NOTE = (
    "This is a counterfactual arm, not a forecast scenario; its forecast "
    "probability is zero."
)
_UNKNOWN_GROWTH_MULTIPLE = object()
SAFEGUARD_FAMILY = CONTEXT_VARIANT_FAMILIES["safeguard_level"]


@dataclass(frozen=True, slots=True)
class _StepLaw:
    """A discrete weighted loss law as unique losses and cumulative mass."""

    losses: np.ndarray
    cumulative: np.ndarray
    cumulative_mass: np.ndarray
    total_mass: float

    @classmethod
    def from_samples(
        cls, losses: Sequence[float] | np.ndarray, masses: Sequence[float] | np.ndarray
    ) -> _StepLaw:
        values = np.asarray(losses, dtype=float)
        weights = np.asarray(masses, dtype=float)
        if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
            raise ValueError("step-law losses and masses must be aligned vectors")
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
        ):
            raise ValueError(
                "step-law losses and masses must be finite and nonnegative"
            )
        positive = weights > 0
        if not np.any(positive):
            raise ValueError("step-law masses must have positive total weight")
        values, weights = values[positive], weights[positive]
        order = np.argsort(values)
        sorted_values, sorted_weights = values[order], weights[order]
        cumulative_mass = np.cumsum(sorted_weights)
        total = float(cumulative_mass[-1])
        if not math.isfinite(total) or total <= 0:
            raise ValueError("step-law masses must have finite positive total weight")
        unique, starts = np.unique(sorted_values, return_index=True)
        ends = np.concatenate((starts[1:], [len(sorted_weights)])) - 1
        cumulative_mass = cumulative_mass[ends]
        cumulative = cumulative_mass / total
        cumulative = np.maximum.accumulate(np.clip(cumulative, 0.0, 1.0))
        cumulative[-1] = 1.0
        unique.flags.writeable = False
        cumulative.flags.writeable = False
        cumulative_mass.flags.writeable = False
        return cls(unique, cumulative, cumulative_mass, total)

    def quantile(self, probabilities: float | Sequence[float] | np.ndarray):
        """The weighted lower quantile, matching ``weighted_var_tvar``."""
        values = np.asarray(probabilities, dtype=float)
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("step-law probabilities must lie in [0, 1]")
        scalar = values.ndim == 0
        targets = values.reshape(-1) * self.total_mass
        indexes = np.searchsorted(self.cumulative_mass, targets, side="left")
        indexes = np.minimum(indexes, len(self.losses) - 1)
        result = self.losses[indexes].reshape(values.shape)
        return float(result) if scalar else result

    def exceedance(self, threshold: float) -> float:
        """Read strict ``P(loss > threshold)`` from the exact step function."""
        if not math.isfinite(threshold):
            raise ValueError("loss thresholds must be finite")
        if threshold < 0:
            return 1.0
        index = int(np.searchsorted(self.losses, threshold, side="right"))
        return float(1.0 - (0.0 if index == 0 else self.cumulative[index - 1]))


def _interpolate_losses(lower, upper, weight: float) -> np.ndarray:
    """Vectorized log1p interpolation that preserves shared loss atoms."""
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    if weight == 0:
        return lower_values.copy()
    if weight == 1:
        return upper_values.copy()
    interpolated = np.expm1(
        (1 - weight) * np.log1p(lower_values) + weight * np.log1p(upper_values)
    )
    interpolated = np.where(lower_values == upper_values, lower_values, interpolated)
    return np.clip(
        interpolated,
        np.minimum(lower_values, upper_values),
        np.maximum(lower_values, upper_values),
    )


def _interpolated_step_law(lower: _StepLaw, upper: _StepLaw, weight: float) -> _StepLaw:
    """Build one Vincentized law from the merged breakpoints of two step laws."""
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Vincentization weight must lie in [0, 1]")
    if weight == 0:
        return lower
    if weight == 1:
        return upper
    breaks = np.unique(
        np.concatenate(([0.0], lower.cumulative, upper.cumulative, [1.0]))
    )
    left, right = breaks[:-1], breaks[1:]
    keep = right > left
    left, right = left[keep], right[keep]
    midpoints = left + (right - left) / 2
    low_values = lower.quantile(midpoints)
    high_values = upper.quantile(midpoints)
    losses = _interpolate_losses(low_values, high_values, weight)
    return _StepLaw.from_samples(losses, right - left)


def vincentized_quantile(
    lower_losses: Sequence[float] | np.ndarray,
    lower_masses: Sequence[float] | np.ndarray,
    upper_losses: Sequence[float] | np.ndarray,
    upper_masses: Sequence[float] | np.ndarray,
    weight: float,
    probability: float,
) -> float:
    """Pure log1p-Vincentized lower quantile of two weighted step laws."""
    lower = _StepLaw.from_samples(lower_losses, lower_masses)
    upper = _StepLaw.from_samples(upper_losses, upper_masses)
    return float(_interpolated_step_law(lower, upper, weight).quantile(probability))


def vincentized_exceedance_probability(
    lower_losses: Sequence[float] | np.ndarray,
    lower_masses: Sequence[float] | np.ndarray,
    upper_losses: Sequence[float] | np.ndarray,
    upper_masses: Sequence[float] | np.ndarray,
    weight: float,
    threshold: float,
) -> float:
    """Pure strict exceedance reader for two log1p-Vincentized step laws."""
    lower = _StepLaw.from_samples(lower_losses, lower_masses)
    upper = _StepLaw.from_samples(upper_losses, upper_masses)
    return _interpolated_step_law(lower, upper, weight).exceedance(threshold)


@dataclass(frozen=True, slots=True)
class _NodeState:
    state_id: str
    key: str
    weight: float
    draw: ConditionalCapabilityDraws | None


@dataclass(slots=True)
class _NodeView:
    measure: str
    source_status: str
    status: str
    reason: str | None
    expected_annual_loss: float | None
    sampling_standard_error: float | None
    sampling_reason: str | None
    years_per_state: int | None
    law: _StepLaw | None
    batch_laws: list[_StepLaw] | None
    batch_samples: list[tuple[np.ndarray, np.ndarray]] | None
    batch_reason: str | None
    var_reason: str | None
    structural_zero: bool


@dataclass(slots=True)
class _Node:
    node_id: str
    progress: float
    progress_cdf: float
    arm: str
    base_weight: float | None
    eci_by_lab: dict[str, dict[str, float]]
    states: list[_NodeState]
    by_view: dict[str, _NodeView]
    controls: TailGateControls


def _state_draw(state: _NodeState) -> ConditionalCapabilityDraws:
    if state.draw is None:
        raise AssertionError("a missing state draw was already rejected")
    return state.draw


def _trajectory_record(package) -> Mapping[str, Any] | None:
    for evidence in package.evidence.records:
        acquisition = evidence.acquisition
        if acquisition and "trajectory_quadrature" in acquisition:
            return acquisition["trajectory_quadrature"]
    return None


def _respondent_spread() -> dict[str, object]:
    """The spread k and common level bias b every setup's count laws use
    (``POOL-SPREAD``, ``POOL-COMMON-BIAS``)."""
    return {"k": 1.0, "basis": "face_value", "level_bias": 1.0}


def _disclosures(max_growth_multiple) -> list[str]:
    if max_growth_multiple is _UNKNOWN_GROWTH_MULTIPLE:
        trajectory = (
            "The setup lacks usable trajectory-quadrature evidence, so its "
            "capability and deployment-scale path cannot be reported."
        )
    elif max_growth_multiple is None:
        trajectory = (
            "The deployment scales are fixed authored text and do not move with "
            "the capability trajectory. A scenario is a world, not a model swap."
        )
    else:
        trajectory = (
            "Capability and the lab's path-linked deployment scale move together "
            "along a trajectory, saturating at "
            f"{max_growth_multiple:g}× the median path's growth. A scenario is a "
            "world, not a model swap."
        )
    return [
        trajectory,
        "The capability index is the single general ECI; there are no domain-specific capability levels.",
        "EALs are conditional on respondent spread k = 1 (face value).",
        "Each percentile scenario rests on the answers at about two trajectory nodes, so it is noisier than the integrated headline.",
        "Interpolation in log R between collected paths is a smoothness assumption. Curves are Vincentized in log1p(loss), which keeps the zero-loss atom continuous; EAL is interpolated linearly in log R and is not the mean of the interpolated curve.",
        "Deep-tail points need more simulated state-years. Withheld points are unconverged at the run's years and are not to be quoted without their status.",
        "A scenario conditions on one path; it is not a probability-weighted forecast, and the headline EAL integrates over all paths.",
    ]


def _unavailable_report(reason: str, max_growth_multiple=_UNKNOWN_GROWTH_MULTIPLE):
    return {
        "status": "unavailable",
        "reason": reason,
        "interpretation": _INTERPRETATION,
        "disclosures": _disclosures(max_growth_multiple),
        "respondent_spread": _respondent_spread(),
        "percentiles": {
            "assumption_id": "NUM-CAPABILITY-SCENARIOS",
            "values": list(REPRESENTATIVE_PERCENTILES),
            "law": "base forecast law of frontier progress at the middle of the study year",
        },
        "levels": list(CURVE_LEVELS),
        "thresholds": list(CURVE_THRESHOLDS),
        "scenarios": [],
        "nodes": [],
        "reconciliation": {},
    }


def _measure_and_source_status(view: str) -> tuple[str, str]:
    if view.startswith("source_status::"):
        _, source_status, measure = view.split("::", 2)
        return measure, source_status
    return view, "total"


def _capture_support(capture, arm: str, key: str):
    captured = capture.arms.get(arm, {}).get(key)
    if captured is None:
        return None
    return capture.supports.get(captured.support_sha256)


def _support_reason(capture, arm: str, key: str, view: str) -> str | None:
    support = _capture_support(capture, arm, key)
    if support is None:
        return None
    return support.unavailable_views.get(view)


def _functional_reason(
    capture, arm: str, key: str, view: str, functional: str
) -> str | None:
    support = _capture_support(capture, arm, key)
    if support is None:
        return None
    return support.unavailable_functionals.get(view, {}).get(functional)


def _joined_reasons(reasons: Sequence[str]) -> str | None:
    return "; ".join(dict.fromkeys(reason for reason in reasons if reason)) or None


def _unavailable_node_view(
    view: str,
    state_reasons: Sequence[str],
    years_per_state: int | None,
) -> _NodeView:
    measure, source_status = _measure_and_source_status(view)
    reason = (
        _joined_reasons(state_reasons)
        or "No positive-weight conditional state was captured for this node."
    )
    return _NodeView(
        measure=measure,
        source_status=source_status,
        status="unavailable",
        reason=reason,
        expected_annual_loss=None,
        sampling_standard_error=None,
        sampling_reason=reason,
        years_per_state=years_per_state,
        law=None,
        batch_laws=None,
        batch_samples=None,
        batch_reason=None,
        var_reason=reason,
        structural_zero=False,
    )


def _node_view(
    node: _Node,
    view: str,
    capture,
    groups: SampleGroups,
    batches: int,
) -> _NodeView:
    """One view of a node's states at their weights in that view: a view
    that counts the lab's loss, a lab-loss view or a gross view, gives a
    state left out of those views weight zero and its sampling group's weight
    in the node to the group's other states (``SampleGroups.view_weights``)."""
    measure, source_status = _measure_and_source_status(view)
    states = [state for state in node.states if state.weight > 0]
    years = [
        _state_draw(state).n_simulations for state in states if state.draw is not None
    ]
    years_per_state = max(years, default=None)
    if not states:
        return _unavailable_node_view(view, (), years_per_state)
    by_view = groups.view_weights({state.key: state.weight for state in states})
    weights = by_view.of(view)
    if weights is None:
        return _unavailable_node_view(
            view, (by_view.reason(view) or "",), years_per_state
        )
    if weights is not by_view.arm:
        states = [
            replace(state, weight=weights[state.key])
            for state in states
            if weights[state.key] > 0
        ]
    missing = []
    for state in states:
        if state.draw is None:
            missing.append(
                f"State {state.state_id!r} has no captured draws in arm {node.arm!r}."
            )
        elif view not in state.draw.annual_losses:
            missing.append(
                _support_reason(capture, node.arm, state.key, view)
                or f"State {state.state_id!r} did not retain view {view!r}."
            )
    if missing:
        return _unavailable_node_view(view, missing, years_per_state)

    active = [state for state in states if state.draw is not None]
    assert len(active) == len(states)
    functional_reasons = [
        reason
        for state in active
        if (
            reason := _functional_reason(
                capture,
                node.arm,
                state.key,
                view,
                "expected_annual_loss",
            )
        )
    ]
    no_finite_mean = [
        state for state in active if not _state_draw(state).views[view].mean_exists
    ]
    if no_finite_mean and not functional_reasons:
        functional_reasons.append(
            "The transformed conditional loss law has no finite mean."
        )
    if functional_reasons:
        eal = None
        eal_status = "unavailable"
        eal_reason = _joined_reasons(functional_reasons)
    else:
        native = all(
            _state_draw(state).views[view].native_mean is not None for state in active
        )
        state_eals = {}
        for state in active:
            draw = _state_draw(state)
            found = draw.views[view]
            state_eals[state.key] = (
                found.native_mean.value
                if native and found.native_mean is not None
                else float(np.dot(found.array, draw.normalized_weights()))
            )
        eal = math.fsum(state.weight * state_eals[state.key] for state in active)
        eal_status, eal_reason = "available", None

    node_draws = {state.key: _state_draw(state) for state in active}
    node_weights = {state.key: state.weight for state in active}
    if eal_status == "available" and all(
        _state_draw(state).views[view].native_mean is not None for state in active
    ):
        terms = analytic_eal_terms(node_draws, node_weights, view)
        assert terms is not None
        error, sampling_reason = standard_error(groups, terms, {})
    elif eal_status == "available":
        error, sampling_reason = None, NOT_ANALYTIC_REASON
    else:
        error, sampling_reason = None, eal_reason

    losses = np.concatenate(
        [_state_draw(state).annual_losses[view] for state in active]
    )
    masses = np.concatenate(
        [state.weight * _state_draw(state).normalized_weights() for state in active]
    )
    positive_mass = masses > 0
    losses, masses = losses[positive_mass], masses[positive_mass]
    law = _StepLaw.from_samples(losses, masses)
    structural_zero_states = {
        state.key: known_zero_law(_state_draw(state), view) for state in active
    }
    structural_zero = all(structural_zero_states.values())
    var_reasons = [
        reason
        for state in active
        if (reason := _functional_reason(capture, node.arm, state.key, view, "var"))
    ]
    var_reason = _joined_reasons(var_reasons)

    nonzero_states = [
        state for state in active if not structural_zero_states[state.key]
    ]
    batch_laws: list[_StepLaw] | None = None
    batch_samples: list[tuple[np.ndarray, np.ndarray]] | None = None
    batch_reason = None
    if not nonzero_states:
        batch_laws = [law] * batches
        # An exact zero atom: no simulated years, so it needs no tail support.
        batch_samples = []
    else:
        layout = batch_layout(
            [(_state_draw(state), state.weight) for state in nonzero_states],
            batches,
        )
        if isinstance(layout, str):
            batch_reason = layout
        else:
            sample_losses = np.concatenate(
                [_state_draw(state).annual_losses[view] for state in nonzero_states]
            )
            sample_masses = np.concatenate(
                [
                    state.weight * _state_draw(state).normalized_weights()
                    for state in nonzero_states
                ]
            )
            parts = batch_masses(sample_masses, layout, batches)
            if isinstance(parts, str):
                batch_reason = parts
            else:
                zero_mass = math.fsum(
                    state.weight
                    for state in active
                    if structural_zero_states[state.key]
                )
                batch_laws, batch_samples = [], []
                for selected, batch_weights in parts:
                    # The simulated years support the batch's statistics; its
                    # law adds the structurally zero states' exact atom.
                    batch_loss = sample_losses[selected]
                    batch_samples.append((batch_loss, batch_weights))
                    if zero_mass > 0:
                        batch_loss = np.concatenate((batch_loss, np.zeros(1)))
                        batch_weights = np.concatenate((batch_weights, [zero_mass]))
                    batch_laws.append(_StepLaw.from_samples(batch_loss, batch_weights))

    return _NodeView(
        measure=measure,
        source_status=source_status,
        status=eal_status,
        reason=eal_reason,
        expected_annual_loss=eal,
        sampling_standard_error=error,
        sampling_reason=sampling_reason,
        years_per_state=years_per_state,
        law=law,
        batch_laws=batch_laws,
        batch_samples=batch_samples,
        batch_reason=batch_reason,
        var_reason=var_reason,
        structural_zero=structural_zero,
    )


def _curve_point(
    fields: Mapping[str, object],
    value_name: str,
    value: float | None,
    *,
    functional: str,
    controls: TailGateControls | None = None,
    batch_values: Sequence[float] | None = None,
    minimum_batch_exceedances: int | None = None,
    years_per_state: int | None = None,
    unavailable_reason: str | None = None,
    structural_zero: bool = False,
    zero_empirical_tail: bool = False,
) -> dict[str, object]:
    """Construct a curve point and decide its status from one set of inputs."""
    status = "unavailable"
    reason = unavailable_reason
    half_width = relative = None
    if unavailable_reason is None and structural_zero:
        status, half_width, relative = "available", 0.0, 0.0
    elif unavailable_reason is None and (
        controls is None or batch_values is None or minimum_batch_exceedances is None
    ):
        reason = (
            "No non-structural state-years were available to assess the curve point."
        )
    elif unavailable_reason is None:
        assert controls is not None
        assert batch_values is not None
        assert minimum_batch_exceedances is not None
        half_width = batch_half_width(batch_values)
        relative = (
            half_width / abs(value) if value else 0.0 if half_width == 0 else None
        )
        converged = (
            relative is not None
            and relative <= controls.tolerance
            and batch_support_reason(
                minimum_batch_exceedances, controls.batches, functional
            )
            is None
        )
        gate_record = {
            functional: {
                "status": "converged" if converged else "unconverged",
                "relative_half_width": relative,
            },
            "minimum_batch_exceedances": minimum_batch_exceedances,
        }
        reason = (
            None
            if converged
            else unconverged_reason(
                gate_record,
                functional,
                {"batches": controls.batches, "tolerance": controls.tolerance},
                years_per_state or 0,
            )
        )
        status = "available" if converged else "withheld"
        if zero_empirical_tail:
            status = "withheld"
            reason = reason or (
                "A zero empirical exceedance probability with non-structural loss "
                "support does not establish an impossible loss."
            )
    return {
        **fields,
        value_name: value,
        "status": status,
        "reason": reason,
        "half_width": half_width,
        "relative_half_width": relative,
        "minimum_batch_exceedances": minimum_batch_exceedances,
    }


def _unavailable_curve_points(
    reason: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    quantiles = [
        _curve_point(
            {
                "non_exceedance_probability": level,
                "return_period_years": 1 / (1 - level),
            },
            "annual_loss",
            None,
            functional="var",
            unavailable_reason=reason,
        )
        for level in CURVE_LEVELS
    ]
    exceedance = [
        _curve_point(
            {"loss": threshold},
            "probability",
            None,
            functional="probability",
            unavailable_reason=reason,
        )
        for threshold in CURVE_THRESHOLDS
    ]
    return quantiles, exceedance


def _minimum_batch_exceedances(
    node_views: Sequence[_NodeView], thresholds: Sequence[float], functional: str
) -> int | None:
    """The sparsest batch's supporting years (``minimum_batch_support``) over
    both nodes, each at its own quantile of the point's level.

    A wholly structural-zero node is an exact atom with no simulated years,
    so it needs no support.
    """
    counts = []
    for data, threshold in zip(node_views, thresholds, strict=True):
        if data.batch_samples is None:
            return None
        if data.batch_samples:
            counts.append(
                minimum_batch_support(data.batch_samples, threshold, functional)
            )
    return min(counts) if counts else None


def _scenario_curve(
    lower: _Node,
    upper: _Node,
    weight: float,
    view: str,
    controls: TailGateControls,
    years_per_state: int | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lower_data, upper_data = lower.by_view[view], upper.by_view[view]
    var_reason = lower_data.var_reason or upper_data.var_reason
    if var_reason is not None:
        return _unavailable_curve_points(var_reason)
    assert lower_data.law is not None and upper_data.law is not None
    full_law = _interpolated_step_law(lower_data.law, upper_data.law, weight)
    if lower_data.structural_zero and upper_data.structural_zero:
        return (
            [
                _curve_point(
                    {
                        "non_exceedance_probability": level,
                        "return_period_years": 1 / (1 - level),
                    },
                    "annual_loss",
                    0.0,
                    functional="var",
                    structural_zero=True,
                )
                for level in CURVE_LEVELS
            ],
            [
                _curve_point(
                    {"loss": threshold},
                    "probability",
                    0.0,
                    functional="probability",
                    structural_zero=True,
                )
                for threshold in CURVE_THRESHOLDS
            ],
        )
    batch_reason = lower_data.batch_reason or upper_data.batch_reason
    if batch_reason is None:
        assert lower_data.batch_laws is not None
        assert upper_data.batch_laws is not None
        if len(lower_data.batch_laws) != len(upper_data.batch_laws):
            batch_reason = (
                "The two trajectory nodes do not have the same tail-gate batch count."
            )
    if batch_reason is None:
        assert lower_data.batch_laws is not None
        assert upper_data.batch_laws is not None
        batch_laws = [
            _interpolated_step_law(low, high, weight)
            for low, high in zip(
                lower_data.batch_laws, upper_data.batch_laws, strict=True
            )
        ]
    else:
        batch_laws = None

    quantiles = []
    for level in CURVE_LEVELS:
        value = float(full_law.quantile(level))
        minimum = (
            None
            if batch_reason is not None
            else _minimum_batch_exceedances(
                (lower_data, upper_data),
                (
                    float(lower_data.law.quantile(level)),
                    float(upper_data.law.quantile(level)),
                ),
                "var",
            )
        )
        quantiles.append(
            _curve_point(
                {
                    "non_exceedance_probability": level,
                    "return_period_years": 1 / (1 - level),
                },
                "annual_loss",
                value,
                functional="var",
                controls=controls,
                batch_values=None
                if batch_laws is None
                else [float(law.quantile(level)) for law in batch_laws],
                minimum_batch_exceedances=minimum,
                years_per_state=years_per_state,
                unavailable_reason=batch_reason,
            )
        )

    exceedance = []
    for threshold in CURVE_THRESHOLDS:
        probability = full_law.exceedance(threshold)
        minimum = (
            None
            if batch_reason is not None
            else _minimum_batch_exceedances(
                (lower_data, upper_data),
                (
                    float(lower_data.law.quantile(1 - probability)),
                    float(upper_data.law.quantile(1 - probability)),
                ),
                "probability",
            )
        )
        exceedance.append(
            _curve_point(
                {"loss": threshold},
                "probability",
                probability,
                functional="probability",
                controls=controls,
                batch_values=None
                if batch_laws is None
                else [law.exceedance(threshold) for law in batch_laws],
                minimum_batch_exceedances=minimum,
                years_per_state=years_per_state,
                unavailable_reason=batch_reason,
                zero_empirical_tail=probability == 0,
            )
        )
    return quantiles, exceedance


def _controls(capture, arm: str) -> TailGateControls:
    plan = capture.execution_plan
    base = plan.get("numerical_controls", {})
    per_arm = plan.get("numerical_controls_by_arm", {}).get(arm, {})
    return TailGateControls.from_controls({**base, **per_arm}.get("tail_gate"))


def _node_eci(node_eci, node_id: str) -> dict[str, dict[str, float]]:
    row = node_eci[node_id]
    start = row["period_start"]
    end = row["period_end"]
    return {
        lab: {"period_start": float(start[lab]), "period_end": float(end[lab])}
        for lab in start
    }


def _frontier_eci(eci_by_lab: Mapping[str, Mapping[str, float]]):
    if not eci_by_lab:
        return None, None
    return (
        max(value["period_start"] for value in eci_by_lab.values()),
        max(value["period_end"] for value in eci_by_lab.values()),
    )


def _interpolated_eci(lower: _Node, upper: _Node, weight: float):
    labs = {}
    for lab in lower.eci_by_lab:
        low, high = lower.eci_by_lab[lab], upper.eci_by_lab[lab]
        labs[lab] = {
            "period_start": low["period_start"]
            + weight * (high["period_start"] - low["period_start"]),
            "period_end": low["period_end"]
            + weight * (high["period_end"] - low["period_end"]),
        }
    start, end = _frontier_eci(labs)
    return start, end, labs


def _node_record(node: _Node) -> dict[str, object]:
    start, end = _frontier_eci(node.eci_by_lab)
    return {
        "node_id": node.node_id,
        "progress": node.progress,
        "progress_cdf": node.progress_cdf,
        "eci_period_start": start,
        "eci_period_end": end,
        "base_weight": node.base_weight,
        "arm": node.arm,
        "samples": sum(state.weight > 0 for state in node.states),
        "by_view": {
            view: {
                "measure": data.measure,
                "source_status": data.source_status,
                "expected_annual_loss": data.expected_annual_loss,
                "status": data.status,
                "reason": data.reason,
                "sampling_standard_error": data.sampling_standard_error,
                "sampling_reason": data.sampling_reason,
            }
            for view, data in node.by_view.items()
        },
    }


def _node_reference(node: _Node) -> dict[str, object]:
    return {
        "node_id": node.node_id,
        "progress": node.progress,
        "progress_cdf": node.progress_cdf,
    }


def _unavailable_scenario(
    scenario: str,
    label: str,
    description: str,
    percentile: float | None,
    reason: str,
    views: Sequence[str],
    *,
    progress: float | None = None,
    progress_cdf: float | None = None,
    arm: str | None = None,
    interpolation: Mapping[str, object] | None = None,
):
    by_view = {}
    for view in views:
        quantiles, exceedance = _unavailable_curve_points(reason)
        by_view[view] = _view_record(
            view,
            "unavailable",
            reason,
            None,
            None,
            reason,
            None,
            quantiles,
            exceedance,
        )
    return {
        "scenario": scenario,
        "label": label,
        "description": description,
        "status": "unavailable",
        "reason": reason,
        "forecast_probability": 0.0 if scenario == "current_frontier" else None,
        "forecast_probability_note": _CURRENT_PROBABILITY_NOTE
        if scenario == "current_frontier"
        else _PERCENTILE_PROBABILITY_NOTE,
        "percentile": percentile,
        "progress": progress,
        "progress_cdf": progress_cdf,
        "eci_period_start": None,
        "eci_period_end": None,
        "eci_by_lab": {},
        "arm": arm,
        "interpolation": interpolation
        or {
            "lower_node": None,
            "upper_node": None,
            "weight": None,
            "exact_node": False,
        },
        "by_view": by_view,
    }


def _view_record(
    view: str,
    status: str,
    reason: str | None,
    expected_annual_loss: float | None,
    sampling_standard_error: float | None,
    sampling_reason: str | None,
    years_per_state: int | None,
    quantiles: list[dict[str, object]],
    exceedance: list[dict[str, object]],
) -> dict[str, object]:
    measure, source_status = _measure_and_source_status(view)
    return {
        "view": view,
        "measure": measure,
        "source_status": source_status,
        "status": status,
        "reason": reason,
        "expected_annual_loss": expected_annual_loss,
        "sampling_standard_error": sampling_standard_error,
        "sampling_reason": sampling_reason,
        "years_per_state": years_per_state,
        "quantiles": quantiles,
        "exceedance": exceedance,
    }


def _scenario_from_nodes(
    scenario: str,
    label: str,
    description: str,
    percentile: float | None,
    progress: float,
    progress_cdf: float,
    lower: _Node,
    upper: _Node,
    weight: float,
    exact: bool,
    views: Sequence[str],
):
    eci_weight = (
        0.0
        if lower is upper
        else (progress - lower.progress) / (upper.progress - lower.progress)
    )
    start, end, eci_by_lab = _interpolated_eci(lower, upper, eci_weight)
    controls = lower.controls
    interpolation = {
        "lower_node": _node_reference(lower),
        "upper_node": _node_reference(upper),
        "weight": weight,
        "exact_node": exact,
    }
    by_view = {}
    for view in views:
        low_data, high_data = lower.by_view[view], upper.by_view[view]
        years_per_state = max(
            (
                years
                for years in (low_data.years_per_state, high_data.years_per_state)
                if years is not None
            ),
            default=None,
        )
        if low_data.status != "available" or high_data.status != "available":
            reason = (
                _joined_reasons([low_data.reason or "", high_data.reason or ""])
                or "A bracketing trajectory node is unavailable for this view."
            )
            eal, eal_status, eal_reason = None, "unavailable", reason
        else:
            assert low_data.expected_annual_loss is not None
            assert high_data.expected_annual_loss is not None
            eal = (1 - weight) * low_data.expected_annual_loss + (
                weight * high_data.expected_annual_loss
            )
            eal_status, eal_reason = "available", None
        if (
            low_data.sampling_standard_error is None
            or high_data.sampling_standard_error is None
        ):
            sampling_error = None
            sampling_reason = (
                _joined_reasons(
                    [low_data.sampling_reason or "", high_data.sampling_reason or ""]
                )
                or "A bracketing node's respondent-sampling standard error is unavailable."
            )
        else:
            sampling_error = math.sqrt(
                ((1 - weight) * low_data.sampling_standard_error) ** 2
                + (weight * high_data.sampling_standard_error) ** 2
            )
            sampling_reason = None
        quantiles, exceedance = _scenario_curve(
            lower,
            upper,
            weight,
            view,
            controls,
            years_per_state,
        )
        by_view[view] = _view_record(
            view,
            eal_status,
            eal_reason,
            eal,
            sampling_error,
            sampling_reason,
            years_per_state,
            quantiles,
            exceedance,
        )
    return {
        "scenario": scenario,
        "label": label,
        "description": description,
        "status": "available",
        "reason": None,
        "forecast_probability": 0.0 if percentile is None else None,
        "forecast_probability_note": _CURRENT_PROBABILITY_NOTE
        if percentile is None
        else _PERCENTILE_PROBABILITY_NOTE,
        "percentile": percentile,
        "progress": progress,
        "progress_cdf": progress_cdf,
        "eci_period_start": start,
        "eci_period_end": end,
        "eci_by_lab": eci_by_lab,
        "arm": lower.arm,
        "interpolation": interpolation,
        "by_view": by_view,
    }


def _current_scenario(node: _Node | None, views: Sequence[str]):
    scenario, label, description = _CURRENT_FRONTIER
    if node is None:
        reason = f"The setup has no required {CURRENT_FRONTIER_ARM!r} arm."
        return _unavailable_scenario(
            scenario,
            label,
            description,
            None,
            reason,
            views,
            progress=0.0,
            progress_cdf=0.0,
            arm=CURRENT_FRONTIER_ARM,
            interpolation={
                "lower_node": {
                    "node_id": _ZERO_NODE,
                    "progress": 0.0,
                    "progress_cdf": 0.0,
                },
                "upper_node": {
                    "node_id": _ZERO_NODE,
                    "progress": 0.0,
                    "progress_cdf": 0.0,
                },
                "weight": 0.0,
                "exact_node": True,
            },
        )
    return _scenario_from_nodes(
        scenario,
        label,
        description,
        None,
        node.progress,
        node.progress_cdf,
        node,
        node,
        0.0,
        True,
        views,
    )


def _reconciliation(current, views: Sequence[str], report):
    arm_metrics = (
        report.get("arms", {}).get(CURRENT_FRONTIER_ARM, {}).get("metrics", {})
    )
    reconciliation = {}
    for view in views:
        current_eal = current["by_view"][view]["expected_annual_loss"]
        reported_eal = arm_metrics.get(view, {}).get("expected_annual_loss")
        matches = (
            current_eal is not None
            and reported_eal is not None
            and math.isclose(current_eal, reported_eal, rel_tol=1e-9, abs_tol=0)
        )
        reconciliation[view] = {
            "scenario": "current_frontier",
            "arm": CURRENT_FRONTIER_ARM,
            "status": "available" if matches else "unavailable",
            "reason": None
            if matches
            else current["by_view"][view]["reason"]
            or arm_metrics.get(view, {}).get("reason")
            or "Node-00 EAL does not reproduce the no-progress arm's reported EAL.",
            "current_frontier_expected_annual_loss": current_eal,
            "no_progress_arm_expected_annual_loss": reported_eal,
        }
    return reconciliation


def _bracket(nodes: Sequence[_Node], progress: float):
    interior = [node for node in nodes if node.progress > 0]
    if not interior:
        return None
    matches = [
        node
        for node in interior
        if abs(math.log(progress) - math.log(node.progress))
        <= NODE_MATCH_RELATIVE_TOLERANCE
    ]
    if matches:
        exact = min(
            matches, key=lambda node: abs(math.log(progress) - math.log(node.progress))
        )
        return exact, exact, 0.0, True
    if progress < interior[0].progress or progress > interior[-1].progress:
        return None
    for lower, upper in pairwise(interior):
        if lower.progress < progress < upper.progress:
            weight = (math.log(progress) - math.log(lower.progress)) / (
                math.log(upper.progress) - math.log(lower.progress)
            )
            return lower, upper, weight, False
    return None


def _scenario_for_progress(
    scenario: str,
    label: str,
    description: str,
    percentile: float,
    progress: float,
    progress_cdf: float,
    nodes: Sequence[_Node],
    views: Sequence[str],
):
    found = _bracket(nodes, progress)
    if found is None:
        reason = (
            "The percentile progress lies outside the collected interior trajectory "
            "nodes; capability scenarios do not extrapolate."
        )
        return _unavailable_scenario(
            scenario,
            label,
            description,
            percentile,
            reason,
            views,
            progress=progress,
            progress_cdf=progress_cdf,
            arm="reference",
        )
    lower, upper, weight, exact = found
    return _scenario_from_nodes(
        scenario,
        label,
        description,
        percentile,
        progress,
        progress_cdf,
        lower,
        upper,
        weight,
        exact,
        views,
    )


def _percentile_scenario(percentile: float) -> tuple[str, str, str]:
    name = format(100 * percentile, ".12g")
    scenario_id = f"p{name.replace('.', '_')}"
    copy = _PERCENTILE_COPY.get(percentile)
    if copy is not None:
        label, description = copy
    else:
        label = f"{name}th percentile path"
        description = (
            "Capability grows along a path at the base forecast's "
            f"{name}th-percentile frontier progress."
        )
    return scenario_id, label, description


@dataclass(frozen=True, slots=True)
class _NodeContext:
    """What every node of every arm is reduced against."""

    node_progress: Mapping[str, float]
    node_eci: Mapping[str, Any]
    pace_shape: float
    pace_cap: float
    posterior: Mapping[str, Any]
    progress_elapsed: float
    draws_by_arm: Mapping[str, Mapping[str, ConditionalCapabilityDraws]]
    capture: Any
    groups: SampleGroups
    views: tuple[str, ...]


def _build_node(
    node_id: str,
    arm: str,
    state_ids: Sequence[str],
    node_weights: Mapping[str, float],
    base_weight: float | None,
    context: _NodeContext,
) -> _Node:
    """One arm's states at one trajectory node, normalized to the node."""
    members = []
    for state_id in state_ids:
        key = state_key(state_id)
        weight = node_weights.get(key, 0.0)
        if weight <= 0:
            continue
        draw = context.draws_by_arm.get(arm, {}).get(key)
        members.append(_NodeState(state_id, key, weight, draw))
    total = math.fsum(state.weight for state in members)
    if total > 0:
        members = [
            _NodeState(state.state_id, state.key, state.weight / total, state.draw)
            for state in members
        ]
    else:
        members = []
    progress = float(context.node_progress[node_id])
    row = _Node(
        node_id=node_id,
        progress=progress,
        progress_cdf=float(
            base_progress_cdf(
                context.pace_shape,
                context.posterior,
                context.progress_elapsed,
                progress,
                pace_cap=context.pace_cap,
            )
        ),
        arm=arm,
        base_weight=base_weight,
        eci_by_lab=_node_eci(context.node_eci, node_id),
        states=members,
        by_view={},
        controls=_controls(context.capture, arm),
    )
    row.by_view = {
        view: _node_view(
            row, view, context.capture, context.groups, row.controls.batches
        )
        for view in context.views
    }
    return row


SAFEGUARD_GRID_LEVELS = (
    ("weaker", f"{SAFEGUARD_FAMILY.family_id}::weaker"),
    ("current", SAFEGUARD_CONTROL),
    ("stronger", f"{SAFEGUARD_FAMILY.family_id}::stronger"),
)
"""Safeguard rows of the grid, weakest first, with the arm each reduces.

``current`` is the variants' matched control: the headline's own captures at the
stipulated safeguards, reweighted to the embedded rule's nodes.
"""

VARIANT_NO_PROGRESS_REASON = (
    "The safeguard-level variant study did not collect the no-progress node "
    "(node-00), so this safeguard level has no current-frontier states; the "
    "cell is not estimated from other nodes."
)
_GRID_INTERPRETATION = (
    "Each safeguard level's annual loss laws and EALs conditional on the "
    "current-frontier counterfactual or one representative percentile path of "
    "the base forecast's frontier progress, reduced from that level's "
    "own node-conditioned states by the same interpolation, sampling error "
    "and tail gate as the reference scenarios."
)
_GRID_DISCLOSURES = (
    "Safeguard levels other than the current one come from separately collected variant studies on the headline's embedded rule, so each cell rests on about half the headline's interior nodes and on the variant studies' own respondents.",
    "A percentile whose progress lies outside a level's collected node range is clamped to the nearest node rather than extrapolated; such a cell is marked capability_clamped with the node and the log-progress gap.",
    "The current safeguard level's current-frontier cell is the reference no-progress scenario. A weaker or stronger level's current-frontier cell is reduced from its variant study's own no-progress (node-00) states the same way, and is unavailable when the variant study did not collect that node.",
)


def _variant_node(state_id: str, level: str) -> str | None:
    """The trajectory node of one safeguard variant's state, or None for another arm's."""
    prefix = SAFEGUARD_FAMILY.state_id(level, "")
    if not state_id.startswith(prefix):
        return None
    return public_state_node(state_id.removeprefix(prefix))


def _clamp_fields(
    clamped: bool, node: _Node | None = None, progress: float | None = None
) -> dict[str, object]:
    return {
        "capability_clamped": clamped,
        "clamped_node": None if node is None else _node_reference(node),
        "clamp_log_progress_gap": None
        if node is None or progress is None
        else math.log(progress) - math.log(node.progress),
    }


def _grid_cell(
    scenario: str,
    label: str,
    description: str,
    percentile: float,
    progress: float,
    progress_cdf: float,
    nodes: Sequence[_Node],
    views: Sequence[str],
    arm: str,
):
    found = _bracket(nodes, progress)
    if found is not None:
        lower, upper, weight, exact = found
        return {
            **_scenario_from_nodes(
                scenario,
                label,
                description,
                percentile,
                progress,
                progress_cdf,
                lower,
                upper,
                weight,
                exact,
                views,
            ),
            **_clamp_fields(False),
        }
    interior = [node for node in nodes if node.progress > 0]
    if not interior:
        return {
            **_unavailable_scenario(
                scenario,
                label,
                description,
                percentile,
                "This safeguard level has no collected interior trajectory node.",
                views,
                progress=progress,
                progress_cdf=progress_cdf,
                arm=arm,
            ),
            **_clamp_fields(False),
        }
    nearest = interior[0] if progress < interior[0].progress else interior[-1]
    return {
        **_scenario_from_nodes(
            scenario,
            label,
            description,
            percentile,
            progress,
            progress_cdf,
            nearest,
            nearest,
            0.0,
            False,
            views,
        ),
        **_clamp_fields(True, nearest, progress),
    }


def _grid_level_nodes(
    level: str,
    arm: str,
    package,
    targets,
    state_draws: Mapping[str, str],
    embedded_weights: Mapping[str, float],
    context: _NodeContext,
) -> list[_Node]:
    """One safeguard level's interior nodes from its own positive-weight states."""
    weights = targets.get(arm, {})
    by_node: dict[str, list[str]] = {}
    for state in package.joint_states:
        state_id = state.state_id
        if weights.get(state_key(state_id), 0.0) <= 0:
            continue
        node_id = (
            state_draws.get(state_id)
            if arm == SAFEGUARD_CONTROL
            else _variant_node(state_id, level)
        )
        if node_id is None or node_id == _ZERO_NODE:
            continue
        if node_id not in context.node_progress:
            raise ValueError(
                f"safeguard state {state_id!r} names node {node_id!r}, which is "
                "not a headline trajectory node"
            )
        by_node.setdefault(node_id, []).append(state_id)
    nodes = [
        _build_node(
            node_id,
            arm,
            state_ids,
            weights,
            embedded_weights.get(node_id),
            context,
        )
        for node_id, state_ids in by_node.items()
    ]
    return sorted(nodes, key=lambda node: (node.progress, node.node_id))


def pooled_node_weights(states) -> dict[str, float]:
    """Within-node weights of one node's respondent states (``POOL-WEIGHTS``).

    Each sampling group, one respondent model at the node, takes an equal
    share split equally among its kept samples; a model whose samples were all
    excluded has no state, so the kept models share its weight. This is the
    weighting the no-progress arm gives the headline's node-00 states once
    normalized within the node, and it needs no arm weight.
    """
    sizes = Counter(state.sampling_group_id for state in states)
    return {
        state_key(state.state_id): 1 / len(sizes) / sizes[state.sampling_group_id]
        for state in states
    }


VARIANT_NOT_SIMULATED_REASON = (
    "The safeguard-level variant study's no-progress (node-00) states were not "
    "simulated in any required arm, so this safeguard level has no "
    "current-frontier draws; the cell is not estimated from other nodes."
)
VARIANT_PARTLY_SIMULATED_REASON = (
    "Only some of the safeguard-level variant study's no-progress (node-00) states "
    "were simulated; reweighting over those would change the cell's respondent "
    "pool (POOL-WEIGHTS), so the cell is unavailable."
)


def _variant_zero_node(level: str, package, targets, context: _NodeContext):
    """A variant's no-progress node from its own ``node-00`` states.

    A required arm that weights the states positively, the variant's own first,
    weights them and supplies their draws, as the no-progress arm does for the
    reference's node-00. Otherwise, as when they carry no weight in any arm,
    they are weighted within the node by ``pooled_node_weights`` on the draws
    of the first required arm that captured every one. Returns None when the
    variant study has no such states, and an unavailable reason when their draws
    are missing: a partly simulated node is not reweighted over the simulated
    states, because that would change its respondent pool.
    """
    states = [
        state
        for state in package.joint_states
        if _variant_node(state.state_id, level) == _ZERO_NODE
    ]
    if not states:
        return None
    variant_arm = f"{SAFEGUARD_FAMILY.family_id}::{level}"
    arms = (
        variant_arm,
        *(row for row in package.execution.required_arms if row != variant_arm),
    )
    state_ids = [state.state_id for state in states]
    keys = {state_key(state_id) for state_id in state_ids}

    def captured(arm: str, wanted: set[str]) -> set[str]:
        return wanted & set(context.draws_by_arm.get(arm, {}))

    for arm in arms:
        weights = targets.get(arm, {})
        positive = {key for key in keys if weights.get(key, 0.0) > 0}
        if not positive:
            continue
        found = captured(arm, positive)
        if found != positive:
            return (
                VARIANT_PARTLY_SIMULATED_REASON
                if found
                else VARIANT_NOT_SIMULATED_REASON
            )
        return _build_node(_ZERO_NODE, arm, state_ids, weights, None, context)
    for arm in arms:
        if captured(arm, keys) == keys:
            return _build_node(
                _ZERO_NODE,
                arm,
                state_ids,
                pooled_node_weights(states),
                None,
                context,
            )
    if any(captured(arm, keys) for arm in arms):
        return VARIANT_PARTLY_SIMULATED_REASON
    return VARIANT_NOT_SIMULATED_REASON


def _unavailable_grid(reason: str) -> dict[str, object]:
    return {
        "status": "unavailable",
        "reason": reason,
        "interpretation": _GRID_INTERPRETATION,
        "disclosures": list(_GRID_DISCLOSURES),
        "levels": [],
        "cells": [],
    }


def _safeguard_grid(
    package,
    targets,
    state_draws: Mapping[str, str],
    embedded_weights: Mapping[str, float],
    context: _NodeContext,
    current: Mapping[str, object],
    percentile_progress: Sequence[tuple[float, float, float]],
) -> dict[str, object]:
    """The capability-by-safeguard grid over each safeguard level's own nodes."""
    views = context.views
    required = package.execution.required_arms
    levels, cells = [], []
    for level, arm in SAFEGUARD_GRID_LEVELS:
        cell_fields = {"safeguard_level": level, "safeguard_arm": arm}
        if arm not in required:
            reason = f"The setup has no required {arm!r} arm."
            levels.append(
                {**cell_fields, "status": "unavailable", "reason": reason, "nodes": []}
            )
            scenario_rows = [
                _unavailable_scenario(*_CURRENT_FRONTIER, None, reason, views, arm=arm)
            ]
            for percentile, progress, cdf in percentile_progress:
                scenario, label, description = _percentile_scenario(percentile)
                scenario_rows.append(
                    _unavailable_scenario(
                        scenario,
                        label,
                        description,
                        percentile,
                        reason,
                        views,
                        progress=progress,
                        progress_cdf=cdf,
                        arm=arm,
                    )
                )
            cells.extend(
                {**cell_fields, **row, **_clamp_fields(False)} for row in scenario_rows
            )
            continue
        nodes = _grid_level_nodes(
            level, arm, package, targets, state_draws, embedded_weights, context
        )
        zero = (
            None
            if arm == SAFEGUARD_CONTROL
            else _variant_zero_node(level, package, targets, context)
        )
        zero_node = zero if isinstance(zero, _Node) else None
        levels.append(
            {
                **cell_fields,
                "status": "available",
                "reason": None,
                "nodes": [
                    _node_record(node)
                    for node in ([] if zero_node is None else [zero_node]) + nodes
                ],
            }
        )
        if arm == SAFEGUARD_CONTROL:
            current_cell = dict(current)
        elif zero_node is not None:
            current_cell = _current_scenario(zero_node, views)
        else:
            current_cell = _unavailable_scenario(
                *_CURRENT_FRONTIER,
                None,
                VARIANT_NO_PROGRESS_REASON if zero is None else zero,
                views,
                progress=0.0,
                progress_cdf=0.0,
                arm=arm,
            )
        cells.append({**cell_fields, **current_cell, **_clamp_fields(False)})
        for percentile, progress, cdf in percentile_progress:
            scenario, label, description = _percentile_scenario(percentile)
            cells.append(
                {
                    **cell_fields,
                    **_grid_cell(
                        scenario,
                        label,
                        description,
                        percentile,
                        progress,
                        cdf,
                        nodes,
                        views,
                        arm,
                    ),
                }
            )
    return {
        "status": "available",
        "reason": None,
        "interpretation": _GRID_INTERPRETATION,
        "disclosures": list(_GRID_DISCLOSURES),
        "levels": levels,
        "cells": cells,
    }


def capability_scenarios(
    package,
    draws_by_arm,
    capture,
    report,
    targets,
    *,
    reweighted: bool,
) -> dict[str, object]:
    """Reduce stored conditional annual draws to representative ECI paths.

    A setup with safeguard-level variants also gets ``safeguard_grid``: the same
    scenarios over each safeguard level's own states.
    """
    with_grid = SAFEGUARD_CONTROL in package.execution.required_arms

    def unavailable(reason: str, *args) -> dict[str, object]:
        record = _unavailable_report(reason, *args)
        if with_grid:
            record["safeguard_grid"] = _unavailable_grid(reason)
        return record

    quadrature = _trajectory_record(package)
    if quadrature is None:
        return unavailable("The setup has no trajectory_quadrature acquisition record.")
    if reweighted:
        return unavailable(
            REWEIGHTED_EXHIBIT_REASON,
            quadrature.get("max_growth_multiple", _UNKNOWN_GROWTH_MULTIPLE),
        )
    if not all(
        key in quadrature
        for key in (
            "node_progress",
            "draw_weights",
            "state_draws",
            "base_law",
            "node_eci",
            "max_growth_multiple",
        )
    ):
        return unavailable(
            "The trajectory_quadrature record lacks base-law or node-path evidence needed for replay."
        )

    views = tuple(
        view for view in package.execution.requested_views if sampling_view(view)
    )
    base_law = quadrature["base_law"]
    pace_shape = base_law["pace_shape"]
    pace_cap = base_law["pace_cap"]
    posterior = base_law["posterior"]
    elapsed = base_law["progress_elapsed_years"]
    node_progress = quadrature["node_progress"]
    reference = package.execution.reference_arm
    state_draws = quadrature["state_draws"]
    states_by_id = {state.state_id: state for state in package.joint_states}
    groups = sample_groups(package.joint_states, state_key)
    full_weights = quadrature["draw_weights"].get(reference, {}).get("full", {})
    context = _NodeContext(
        node_progress=node_progress,
        node_eci=quadrature["node_eci"],
        pace_shape=pace_shape,
        pace_cap=pace_cap,
        posterior=posterior,
        progress_elapsed=elapsed,
        draws_by_arm=draws_by_arm,
        capture=capture,
        groups=groups,
        views=views,
    )
    nodes = []
    for node_id, _progress in sorted(
        node_progress.items(), key=lambda row: (row[1], row[0])
    ):
        arm = CURRENT_FRONTIER_ARM if node_id == _ZERO_NODE else reference
        if node_id == _ZERO_NODE and arm not in targets:
            node_weights = {}
        else:
            node_weights = targets.get(arm, {})
        nodes.append(
            _build_node(
                node_id,
                arm,
                [
                    state_id
                    for state_id, mapped_node in state_draws.items()
                    if mapped_node == node_id and state_id in states_by_id
                ],
                node_weights,
                full_weights.get(node_id),
                context,
            )
        )
    zero_node = next((node for node in nodes if node.node_id == _ZERO_NODE), None)
    if CURRENT_FRONTIER_ARM not in package.execution.required_arms:
        zero_node = None
    current = _current_scenario(zero_node, views)
    reconciliation = _reconciliation(current, views, report)
    scenarios = [current]
    percentile_progress = []
    for percentile in REPRESENTATIVE_PERCENTILES:
        percentile = float(percentile)
        scenario, label, description = _percentile_scenario(percentile)
        progress = float(
            base_progress_quantile(
                pace_shape, posterior, elapsed, percentile, pace_cap=pace_cap
            )
        )
        cdf = float(
            base_progress_cdf(
                pace_shape, posterior, elapsed, progress, pace_cap=pace_cap
            )
        )
        percentile_progress.append((percentile, progress, cdf))
        scenarios.append(
            _scenario_for_progress(
                scenario,
                label,
                description,
                percentile,
                progress,
                cdf,
                nodes,
                views,
            )
        )

    record: dict[str, object] = {
        "status": "available",
        "reason": None,
        "interpretation": _INTERPRETATION,
        "disclosures": _disclosures(quadrature["max_growth_multiple"]),
        "respondent_spread": _respondent_spread(),
        "percentiles": {
            "assumption_id": "NUM-CAPABILITY-SCENARIOS",
            "values": list(REPRESENTATIVE_PERCENTILES),
            "law": "base forecast law of frontier progress at the middle of the study year",
        },
        "levels": list(CURVE_LEVELS),
        "thresholds": list(CURVE_THRESHOLDS),
        "scenarios": scenarios,
        "nodes": [_node_record(node) for node in nodes],
        "reconciliation": reconciliation,
    }
    if with_grid:
        record["safeguard_grid"] = _safeguard_grid(
            package,
            targets,
            state_draws,
            quadrature["draw_weights"].get(reference, {}).get("embedded", {}),
            context,
            current,
            percentile_progress,
        )
    return record
