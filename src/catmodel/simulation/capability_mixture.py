"""Stratified integration over complete conditional annual portfolios.

Each stratum is already a complete conditional annual portfolio: ownership,
allocation and financial terms run before this module combines distributions.
A stratum's capability conditioning is fixed, while a count law with a
parameter rank reads a fresh rank in each of its simulated years.
"""

import json
import math
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Literal, TypeVar

import numpy as np

from catmodel.contracts.results import (
    StatisticStatus,
    TvarCoefficientBasis,
    TvarEstimator,
)
from catmodel.probability.percentiles import SortedSample

DETERMINISTIC_EAL_PRECISION_BASIS = "deterministic_integration"
# Why a control-variate TVaR's coefficient has its value (``tail_estimates``):
# estimated by regression, one because a positive-mass state's law has no
# finite second moment, or zero because the view has no analytic mean.
REGRESSION_COEFFICIENT: TvarCoefficientBasis = "regression"
INFINITE_VARIANCE_COEFFICIENT: TvarCoefficientBasis = "infinite_variance"
NO_EXACT_MEAN_COEFFICIENT: TvarCoefficientBasis = "no_exact_mean"

T = TypeVar("T")


def _immutable_vector(values: np.ndarray) -> np.ndarray:
    base = values
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    if isinstance(base, bytes):
        return values
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True)
class NativeMean:
    """A separate conditional mean estimator, distinct from the sampled mean."""

    value: float
    estimator: str
    precision: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
            or self.value < 0
        ):
            raise ValueError("conditional EAL estimates must be finite and nonnegative")
        if (
            not isinstance(self.estimator, str)
            or not self.estimator.strip()
            or self.estimator == "normalized_empirical_mean"
        ):
            raise ValueError("each separate EAL view requires its named estimator")
        if self.precision is not None:
            json.dumps(dict(self.precision), allow_nan=False)
            for field in (
                "numerical_standard_error",
                "standard_error_upper_bound",
                "integration_error_bound",
            ):
                value = self.precision.get(field)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(
                        "conditional EAL precision must be finite and nonnegative"
                    )
            object.__setattr__(
                self, "precision", MappingProxyType(dict(self.precision))
            )

    def __reduce__(self):
        # A copy, such as a worker process's result, is rebuilt through the
        # constructor, so it is validated and immutable like the original.
        precision = None if self.precision is None else dict(self.precision)
        return type(self), (self.value, self.estimator, precision)


@dataclass(frozen=True)
class ConditionalView:
    """One aligned observable and the moment claims belonging to that observable."""

    array: np.ndarray
    mean_exists: bool = True
    variance_exists: bool = True
    native_mean: NativeMean | None = None

    def __post_init__(self) -> None:
        if (
            self.array.ndim != 1
            or not np.all(np.isfinite(self.array))
            or np.any(self.array < 0)
        ):
            raise ValueError("annual losses must be finite nonnegative vectors")
        if not self.mean_exists and self.native_mean is not None:
            raise ValueError(
                "a nonexisting target mean cannot have a finite native EAL estimate"
            )


@dataclass(frozen=True)
class ConditionalCapabilityDraws:
    """Aligned conditional views, reusable under pure reweighting."""

    stratum_sha256: str
    response_sha256: str
    seed: int
    views: Mapping[str, ConditionalView]
    joint_law_semantics: Literal[
        "physical_joint", "marginal_only", "mixed_or_unknown"
    ] = "mixed_or_unknown"
    evaluator_provenance_by_risk_unit: Mapping[str, Mapping[str, object]] | None = None
    # Each year's equal mass, derived once (``normalized_weights``).
    _normalized_weights: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.joint_law_semantics not in {
            "physical_joint",
            "marginal_only",
            "mixed_or_unknown",
        }:
            raise ValueError("unsupported captured joint-law semantics")
        if (
            self.joint_law_semantics == "physical_joint"
            and not self.evaluator_provenance_by_risk_unit
        ):
            raise ValueError("physical joint captures require evaluator provenance")
        if self.joint_law_semantics == "physical_joint" and any(
            record.get("joint_law_semantics") != "physical_joint"
            for record in (self.evaluator_provenance_by_risk_unit or {}).values()
        ):
            raise ValueError(
                "physical joint capture disagrees with a constituent evaluator"
            )
        if self.evaluator_provenance_by_risk_unit is not None:
            json.dumps(
                {
                    key: dict(value)
                    for key, value in self.evaluator_provenance_by_risk_unit.items()
                },
                allow_nan=False,
            )
        if not self.views:
            raise ValueError("conditional draws require at least one annual loss view")
        if any(not name.strip() for name in self.views):
            raise ValueError("capture view names must be nonempty")
        lengths = {len(view.array) for view in self.views.values()}
        if len(lengths) != 1 or next(iter(lengths)) == 0:
            raise ValueError("conditional views must share nonempty aligned years")
        weights = np.ones(self.n_simulations) / self.n_simulations
        weights.flags.writeable = False
        # Derived once: every reduction statistic reads the same masses.
        object.__setattr__(self, "_normalized_weights", weights)
        immutable = {}

        def freeze(values: np.ndarray) -> np.ndarray:
            key = id(values)
            if key not in immutable:
                immutable[key] = _immutable_vector(values)
            return immutable[key]

        object.__setattr__(
            self,
            "views",
            MappingProxyType(
                {
                    name: replace(view, array=freeze(view.array))
                    for name, view in self.views.items()
                }
            ),
        )

    def __reduce__(self):
        # A copy, such as a worker process's result, is rebuilt through the
        # constructor, so its arrays are validated and immutable like the
        # original's.
        return type(self), tuple(
            dict(self.views) if row.name == "views" else getattr(self, row.name)
            for row in fields(self)
            if row.init
        )

    @property
    def annual_losses(self) -> Mapping[str, np.ndarray]:
        """Project the shared annual vectors without duplicating per-view metadata."""
        return MappingProxyType({name: view.array for name, view in self.views.items()})

    @property
    def n_simulations(self) -> int:
        return len(next(iter(self.annual_losses.values())))

    def normalized_weights(self) -> np.ndarray:
        """Each year's mass within the state, one over the years (read-only)."""
        return self._normalized_weights


@dataclass(frozen=True)
class CapabilityMixture:
    """A non-IID annual law; every positive stratum has its declared exact mass."""

    draws: Mapping[str, ConditionalCapabilityDraws]
    stratum_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if (
            not self.stratum_weights
            or any(not math.isfinite(w) or w < 0 for w in self.stratum_weights.values())
            or not math.isclose(
                math.fsum(self.stratum_weights.values()), 1, abs_tol=1e-10
            )
        ):
            raise ValueError(
                "path probabilities must be finite, nonnegative and sum to one"
            )
        if not set(self.draws).issubset(self.stratum_weights):
            raise ValueError("stored draws contain paths outside the forecast support")
        views = None
        for path, weight in self.stratum_weights.items():
            draw = self.draws.get(path)
            if draw is None:
                if weight > 0:
                    raise ValueError(
                        "positive reweighting requires stored conditional draws"
                    )
                continue
            if draw.stratum_sha256 != path:
                raise ValueError(
                    "stored conditional path identity differs from its key"
                )
            if views is None:
                views = set(draw.annual_losses)
            elif views != set(draw.annual_losses):
                raise ValueError(
                    "conditional strata must expose identical annual loss views"
                )

    def positive_strata(self) -> list[tuple[ConditionalCapabilityDraws, float]]:
        """Stored strata with positive mass, in the order ``samples`` concatenates."""
        return [(self.draws[p], w) for p, w in self.stratum_weights.items() if w > 0]

    def derived(self, key: Hashable, compute: Callable[[], T]) -> T:
        """``compute()``, computed once per ``key`` for this immutable mixture,
        so statistics that share an intermediate (a sample, its sort, its
        batches) compute it once. The result must not be modified."""
        cache = self.__dict__.setdefault("_derived", {})
        if key not in cache:
            cache[key] = compute()
        return cache[key]

    def samples(self, view: str) -> tuple[np.ndarray, np.ndarray]:
        """Every positive stratum's years of ``view`` and their masses,
        concatenated (read-only)."""

        def concatenate():
            strata = self.positive_strata()
            losses = np.concatenate([draw.annual_losses[view] for draw, _ in strata])
            masses = np.concatenate(
                [mass * draw.normalized_weights() for draw, mass in strata]
            )
            losses.flags.writeable = masses.flags.writeable = False
            return losses, masses

        return self.derived(("samples", view), concatenate)

    def sorted_sample(self, view: str) -> SortedSample:
        """``samples(view)`` sorted once for its percentiles."""
        return self.derived(("sorted", view), lambda: SortedSample(*self.samples(view)))

    def metrics(
        self,
        view: str,
        *,
        levels: tuple[float, ...] = (0.99, 0.995, 0.999),
        estimates: Mapping[float, "TailMeanEstimate"] | None = None,
    ) -> dict[str, object]:
        losses, masses = self.samples(view)
        empirical_mean = float(np.dot(losses, masses))
        active = self.positive_strata()
        mean_exists = all(draw.views[view].mean_exists for draw, _ in active)
        mean_reason = (
            None
            if mean_exists
            else "The transformed conditional loss law has no finite mean."
        )
        native_eal = native_mixture_mean(active, (view,))
        separate = native_eal is not None
        eal = empirical_mean if native_eal is None else native_eal
        second_moment_exists = all(
            draw.views[view].variance_exists for draw, _ in active
        )
        positive_mass = masses > 0
        supported_losses, supported_masses = (
            losses[positive_mass],
            masses[positive_mass],
        )
        if np.all(supported_losses == supported_losses[0]):
            empirical_variance = 0.0
            standard_deviation = 0.0
        else:
            scale = float(np.max(supported_losses))
            scaled = supported_losses / scale
            centered = scaled - float(np.dot(scaled, supported_masses))
            standard_deviation = scale * math.sqrt(
                float(np.dot(centered * centered, supported_masses))
            )
            empirical_variance = standard_deviation * standard_deviation
        variance_reason = (
            None
            if second_moment_exists
            else "The transformed conditional loss law has no admitted finite second moment."
        )
        empirical_variance_reason = None
        if not math.isfinite(empirical_variance):
            empirical_variance_reason = (
                "The empirical variance exceeds finite numeric range."
            )
        elif standard_deviation > 0 and empirical_variance == 0:
            empirical_variance_reason = (
                "The empirical variance falls below finite positive numeric range."
            )
        statistics = {
            "expected_annual_loss": statistic_status(unavailable=mean_reason),
            "annual_probability_positive": statistic_status(),
            "annual_loss_median": statistic_status(),
            "annual_loss_variance": statistic_status(
                unavailable=variance_reason, withheld=empirical_variance_reason
            ),
            "normalized_empirical_variance": statistic_status(
                withheld=empirical_variance_reason
            ),
        }
        variance_reason = variance_reason or empirical_variance_reason
        if any(not math.isfinite(level) or not 0 <= level < 1 for level in levels):
            raise ValueError("tail probability levels must be finite in [0, 1)")
        if estimates is None:
            estimates = tail_estimates(self, view, levels)
        tail_metrics = {}
        for level in levels:
            estimate = estimates[level]
            tail = {
                "var": estimate.var,
                "tvar": estimate.tvar if mean_exists else None,
                "tvar_estimator": estimate.estimator if mean_exists else None,
                "exceedance_count": estimate.exceedance_count,
            }
            tail_metrics[str(level)] = tail
            statistics[f"var::{level}"] = statistic_status()
            statistics[f"tvar::{level}"] = statistic_status(unavailable=mean_reason)
        unavailable_precision = {
            "numerical_standard_error": None,
            "standard_error_upper_bound": None,
            "precision_basis": "unavailable_functional",
            "precision_suppression_reason": mean_reason,
        }
        estimator: object = (
            {
                draw.stratum_sha256: native.estimator
                for draw, _ in active
                if (native := draw.views[view].native_mean) is not None
            }
            if separate
            else "normalized_empirical_mean"
        )
        analytic = analytic_mixture_mean(active, (view,))
        coherence = (
            {
                "eal_tail_coherence": eal_tail_coherence(
                    eal,
                    supported_losses,
                    tail_metrics,
                    error_bound=0.0 if analytic is None else analytic.error_bound,
                )
            }
            if separate and mean_exists
            else {}
        )
        return {
            "expected_annual_loss": eal if mean_exists else None,
            "eal_estimator": estimator,
            "eal_precision": self._eal_precision(view, active, separate, "native_mean")
            if mean_exists
            else unavailable_precision,
            **coherence,
            "normalized_empirical_mean": empirical_mean,
            "eal_reconciliation_difference": empirical_mean - eal
            if mean_exists
            else None,
            "annual_loss_variance": empirical_variance
            if variance_reason is None
            else None,
            "annual_loss_variance_estimator": "weighted_empirical_variance",
            "normalized_empirical_variance": empirical_variance
            if empirical_variance_reason is None
            else None,
            "annual_probability_positive": float(np.sum(masses[losses > 0])),
            "annual_loss_median": self.sorted_sample(view).percentile(50.0),
            "tail_metrics": tail_metrics,
            "statistics": statistics,
            "sampling_method": "stratified_conditional_capability_mixture",
            "stratum_diagnostics": {
                path: self._diagnostics(draw, view, self.stratum_weights[path])
                for path, draw in self.draws.items()
            },
            "unsimulated_zero_mass_strata": sorted(
                set(self.stratum_weights) - set(self.draws)
            ),
        }

    @staticmethod
    def _eal_precision(
        view: str,
        active: list[tuple[ConditionalCapabilityDraws, float]],
        separate: bool,
        field: str,
    ) -> dict[str, object]:
        if not separate:
            return {
                "numerical_standard_error": None,
                "standard_error_upper_bound": None,
                "precision_basis": "sampling_design_aware_empirical_mean_precision_required",
                "precision_suppression_reason": "See the batch-means empirical-mean standard error in mean_diagnostics; conditional benchmark precision does not estimate this mean.",
                "stratum_estimators": {},
            }
        metadata = {
            draw.stratum_sha256: dict(native.precision)
            if (native := getattr(draw.views[view], field)) is not None
            and native.precision is not None
            else {
                "numerical_standard_error": None,
                "precision_suppression_reason": "Conditional EAL estimator precision was not supplied.",
            }
            for draw, _ in active
        }
        return {
            **aggregate_eal_precision(
                metadata, {draw.stratum_sha256: weight for draw, weight in active}
            ),
            "stratum_estimators": metadata,
        }

    @staticmethod
    def _diagnostics(
        draw: ConditionalCapabilityDraws, view: str, probability: float
    ) -> dict[str, object]:
        weights = draw.normalized_weights()
        return {
            "probability": probability,
            "sample_mass": float(np.sum(probability * weights)),
            "n_simulations": draw.n_simulations,
            "positive_loss_samples": int(
                np.count_nonzero(draw.annual_losses[view] > 0)
            ),
            "seed": draw.seed,
        }


def weighted_var_tvar(
    losses: np.ndarray,
    masses: np.ndarray,
    level: float,
    *,
    ordered: SortedSample | None = None,
) -> tuple[float, float]:
    """Left-inverse VaR and sample tail mean TVaR of a discrete law with unit
    mass (``tail_parts``): the TVaR at control-variate coefficient zero."""
    parts = tail_parts(losses, masses, level, ordered=ordered)
    return parts.var, control_variate_tvar(parts, level, None, 0.0)


@dataclass(frozen=True)
class TailParts:
    """A sample's VaR at a level and the two means its TVaR is built from:
    the expected excess ``E[(S - var)+]`` and the limited expected value
    ``E[min(S, var)]``."""

    var: float
    excess: float
    limited: float


def tail_parts(
    losses: np.ndarray,
    masses: np.ndarray,
    level: float,
    *,
    ordered: SortedSample | None = None,
) -> TailParts:
    """The left-inverse VaR of a discrete law with unit mass, and its expected
    excess and limited expected value at that VaR.

    At ``v = VaR_p``, ``TVaR_p = v + E[(S - v)+] / (1 - p)`` (Acerbi and Tasche,
    2002, *Journal of Banking and Finance* 26:1487-1503). The sample's expected
    excess counts the strict exceedances of VaR; the part of the VaR atom that
    fills the upper ``1 - level`` probability adds no excess, so for unit mass
    this is the fractional-atom tail mean. ``ordered``, the same sample already
    sorted, saves sorting it again.
    """
    if ordered is None:
        ordered = SortedSample(losses, masses)
    var = ordered.quantile(level)
    strict = losses > var
    return TailParts(
        var,
        float(np.dot(losses[strict] - var, masses[strict])),
        float(np.dot(np.minimum(losses, var), masses)),
    )


def control_variate_tvar(
    parts: TailParts, level: float, mean: float | None, coefficient: float
) -> float:
    """TVaR at ``level`` from a sample's ``parts`` and control-variate
    ``coefficient`` b, with ``mean`` the law's analytic mean.

    ``T = v + ((1 - b) E[(S - v)+] + b (mean - E[min(S, v)])) / (1 - level)``,
    which is the sample tail mean plus ``b (mean - E[S]) / (1 - level)``,
    written so that no large year enters twice and cancels. At ``b = 0`` it is
    the sample tail mean; at ``b = 1`` only the bounded limited expected value
    comes from the sample. TVaR is at least VaR, so a negative excess, from a
    sample whose limited expected value exceeds the analytic mean, is raised
    to VaR. Without an analytic mean only ``b = 0`` exists. An error of at
    most ``e`` in the mean moves the TVaR by at most ``b e / (1 - level)``.
    """
    excess = (1 - coefficient) * parts.excess
    if mean is not None:
        excess += coefficient * (mean - parts.limited)
    elif coefficient != 0:
        raise ValueError("a nonzero control-variate coefficient needs an analytic mean")
    return parts.var + max(excess, 0.0) / (1 - level)


def statistic_status(
    *, unavailable: str | None = None, withheld: str | None = None
) -> StatisticStatus:
    """Status and reason of one reported number.

    ``unavailable``: the law gives the statistic no finite value. ``withheld``:
    the value exists, but this run cannot represent or support it. The first
    applicable reason wins.
    """
    if unavailable is not None:
        return {"status": "unavailable", "reason": unavailable}
    if withheld is not None:
        return {"status": "withheld", "reason": withheld}
    return {"status": "available", "reason": None}


@dataclass(frozen=True)
class AnalyticMean:
    """A mean integrated from the laws rather than averaged over simulated
    years, with its certified integration error bound: the exact mean lies
    within ``error_bound`` of ``value``. The bound is zero for a closed form
    and positive for a certified numerical integral."""

    value: float
    error_bound: float


@dataclass(frozen=True)
class TailMeanEstimate:
    """A view's VaR and control-variate TVaR at one level (``tail_estimates``).

    ``parts`` are the view's VaR, expected excess and limited expected value
    on all its years, ``mean`` its analytic mean, None without one, and
    ``coefficient`` the control-variate coefficient, with its
    ``coefficient_basis``.
    """

    level: float
    parts: TailParts
    mean: AnalyticMean | None
    coefficient: float
    coefficient_basis: TvarCoefficientBasis
    exceedance_count: int

    @property
    def var(self) -> float:
        return self.parts.var

    @property
    def tvar(self) -> float:
        return self.tvar_at(self.parts, self.coefficient)

    @property
    def sample_tvar(self) -> float:
        """The sample tail mean: the TVaR at coefficient zero."""
        return self.tvar_at(self.parts, 0.0)

    @property
    def estimator(self) -> TvarEstimator:
        return {
            "method": "control_variate",
            "coefficient": self.coefficient,
            "coefficient_basis": self.coefficient_basis,
        }

    def tvar_at(self, parts: TailParts, coefficient: float) -> float:
        """The TVaR of ``parts``, this view's on all years or on one batch of
        them, at ``coefficient``."""
        return control_variate_tvar(
            parts,
            self.level,
            None if self.mean is None else self.mean.value,
            coefficient,
        )

    def integration_margin_at(self, coefficient: float) -> float:
        """The most the analytic mean's certified integration error can move
        this view's TVaR at ``coefficient``: ``coefficient * error_bound /
        (1 - level)`` (``control_variate_tvar``). No number of years reduces
        it. Zero for a closed-form mean and without a mean."""
        if self.mean is None:
            return 0.0
        return coefficient * self.mean.error_bound / (1 - self.level)

    @property
    def integration_margin(self) -> float:
        """``integration_margin_at`` the reported coefficient, which the tail
        gate adds to the TVaR's batch-means half-width."""
        return self.integration_margin_at(self.coefficient)

    def admits(self, coefficient: float) -> bool:
        """Whether this view's TVaR can take ``coefficient``: only zero without
        an analytic mean, only one without a finite second moment, otherwise
        any in [0, 1]."""
        if self.coefficient_basis == NO_EXACT_MEAN_COEFFICIENT:
            return coefficient == 0
        if self.coefficient_basis == INFINITE_VARIANCE_COEFFICIENT:
            return coefficient == 1
        return 0 <= coefficient <= 1


def common_coefficient(
    control: TailMeanEstimate, alternative: TailMeanEstimate
) -> float | None:
    """The control-variate coefficient an arm effect on TVaR applies to both
    arms, or None when none suits both and each keeps its own.

    One coefficient keeps the effect a difference of one estimator, so the
    control term, which common years make nearly the same in both arms,
    cancels: the control's coefficient when the alternative admits it, else
    the alternative's when the control admits it.
    """
    for coefficient in (control.coefficient, alternative.coefficient):
        if control.admits(coefficient) and alternative.admits(coefficient):
            return coefficient
    return None


def control_variate_coefficient(
    mixture: CapabilityMixture, view: str, var: float
) -> float:
    """The regression coefficient ``Cov((L - var)+, L) / Var(L)``, clipped to
    [0, 1]: the coefficient of the exact-mean control that minimizes the
    TVaR's sampling variance.

    The covariances are those of the estimators, so they are taken over the
    independent years, each year's values weighted by its mass. When every
    state has the same number of years, a year's values are summed over the
    states, since states can share a year's draws; otherwise each state's
    years count apart. Structurally zero states add nothing. The excess and
    the limited loss ``min(L, var)`` are both non-decreasing in ``L``, so for
    any law the true coefficient lies in [0, 1]; the estimate is clipped to
    that range, and it is zero when the loss does not vary.
    """
    strata = [
        (draw, mass)
        for draw, mass in mixture.positive_strata()
        if not known_zero_law(draw, view)
    ]
    scale = max(
        (float(np.max(draw.annual_losses[view])) for draw, _ in strata), default=0.0
    )
    if scale == 0:
        return 0.0
    excess, loss = [], []
    for draw, mass in strata:
        values = draw.annual_losses[view] / scale
        weights = mass * draw.normalized_weights()
        excess.append(weights * np.maximum(values - var / scale, 0.0))
        loss.append(weights * values)
    if len({len(years) for years in loss}) == 1:
        excess, loss = [np.sum(excess, axis=0)], [np.sum(loss, axis=0)]
    # Rescale so that tiny masses cannot underflow when squared.
    size = max(float(np.max(np.abs(years))) for years in loss)
    if size == 0:
        return 0.0
    covariance = variance = 0.0
    for x, y in zip(excess, loss, strict=True):
        x, y = x / size, y / size
        centered = y - float(np.mean(y))
        covariance += float(np.dot(x - float(np.mean(x)), centered))
        variance += float(np.dot(centered, centered))
    if not variance > 0:
        return 0.0
    return min(max(covariance / variance, 0.0), 1.0)


def tail_estimates(
    mixture: CapabilityMixture, view: str, levels: Iterable[float]
) -> dict[float, TailMeanEstimate]:
    """The VaR and control-variate TVaR of ``view`` at each level.

    The sample tail mean estimates TVaR's expected excess from the few years
    beyond the VaR, and for heavy tails it is usually below the truth. When
    every positive-mass state has an analytic native mean of the same
    variable, in closed form or as a certified numerical integral
    (``analytic_mixture_mean``), the years' empirical mean is a control
    variate with a known mean, and the TVaR is the sample tail mean plus
    ``b (analytic mean - empirical mean) / (1 - level)``
    (``control_variate_tvar``). The coefficient ``b`` is zero without an
    analytic mean (``NO_EXACT_MEAN_COEFFICIENT``), so the TVaR is the sample
    tail mean; one when a positive-mass state's law has no finite second
    moment (``INFINITE_VARIANCE_COEFFICIENT``), where the excess has infinite
    variance and the control removes it; and otherwise the regression
    coefficient at the VaR (``control_variate_coefficient``,
    ``REGRESSION_COEFFICIENT``). It is set once on all years, and every batch
    of the tail gate uses it. A certified integral's error bound can move the
    TVaR by at most its ``integration_margin``, which the tail gate adds to
    the TVaR's half-width.
    """
    active = mixture.positive_strata()
    mean = (
        analytic_mixture_mean(active, (view,))
        if all(draw.views[view].mean_exists for draw, _ in active)
        else None
    )
    infinite_variance = any(not draw.views[view].variance_exists for draw, _ in active)
    losses, masses = mixture.samples(view)
    ordered = mixture.sorted_sample(view)
    estimates = {}
    for level in levels:
        parts = tail_parts(losses, masses, level, ordered=ordered)
        if mean is None:
            coefficient, basis = 0.0, NO_EXACT_MEAN_COEFFICIENT
        elif infinite_variance:
            coefficient, basis = 1.0, INFINITE_VARIANCE_COEFFICIENT
        else:
            coefficient = control_variate_coefficient(mixture, view, parts.var)
            basis = REGRESSION_COEFFICIENT
        estimates[level] = TailMeanEstimate(
            level,
            parts,
            mean,
            coefficient,
            basis,
            int(np.count_nonzero((losses > parts.var) & (masses > 0))),
        )
    return estimates


def summed_view(
    draw: ConditionalCapabilityDraws, views: Iterable[str]
) -> ConditionalView:
    """One state's views summed year by year, with their joint moment claims.

    The sum has a mean only if every view does, and a finite second moment only
    if every view does. Its native mean is the sum of the views' analytic
    native means when each has one, with the sum of their certified
    integration error bounds (``analytic_mixture_mean``).
    """
    views = tuple(views)
    members = [draw.views[name] for name in views]
    analytic = analytic_mixture_mean([(draw, 1.0)], views)
    return ConditionalView(
        sum((member.array for member in members), start=np.zeros(draw.n_simulations)),
        mean_exists=all(member.mean_exists for member in members),
        variance_exists=all(member.variance_exists for member in members),
        native_mean=None
        if analytic is None
        else NativeMean(
            analytic.value,
            "sum_of_native_analytic_means",
            {
                "numerical_standard_error": 0.0,
                "standard_error_upper_bound": 0.0,
                "precision_basis": DETERMINISTIC_EAL_PRECISION_BASIS,
                "integration_error_bound": analytic.error_bound,
                "known_zero_law": all(known_zero_law(draw, name) for name in views),
            },
        ),
    )


def native_mixture_mean(
    strata: Iterable[tuple[ConditionalCapabilityDraws, float]],
    views: Iterable[str],
) -> float | None:
    """Mean of the summed views under the strata masses from the states'
    native means, or None when a state has none.

    This is the EAL a view reports (``CapabilityMixture.metrics``): with a
    native mean in every state it does not come from the simulated years,
    whatever each native estimator's precision; otherwise the EAL is the
    years' empirical mean. ``analytic_mixture_mean`` is the same mean when
    every native mean is analytic.
    """
    views = tuple(views)
    terms = []
    for draw, mass in strata:
        for view in views:
            native = draw.views[view].native_mean
            if native is None:
                return None
            terms.append(mass * native.value)
    return math.fsum(terms)


def analytic_mixture_mean(
    strata: Iterable[tuple[ConditionalCapabilityDraws, float]],
    views: Iterable[str],
) -> AnalyticMean | None:
    """Analytic mean of the summed views under the strata masses, with its
    certified integration error bound, if any.

    Every state's native mean must be analytic: deterministic-integration
    provenance with an integration error bound, zero in closed form and
    positive for a certified numerical integral. Each lies within its bound
    of the exact mean of the simulated variable in its state, so mixing the
    means and the bounds with the stored state masses gives a value within
    the mixed bound of the exact mean of the mixture law whose samples are
    reduced. Any missing or sampled native mean returns None.

    This one mean is the control-variate TVaR's (``tail_estimates``), the one
    the tail gate reconciles the simulated years with and checks them against
    (``tail_gate.view_gate``), and a summed view's (``summed_view``).
    """
    views = tuple(views)
    values, bounds = [], []
    for draw, mass in strata:
        for view in views:
            native = draw.views[view].native_mean
            precision = {} if native is None else native.precision or {}
            bound = precision.get("integration_error_bound")
            if (
                native is None
                or precision.get("precision_basis") != DETERMINISTIC_EAL_PRECISION_BASIS
                or isinstance(bound, bool)
                or not isinstance(bound, int | float)
            ):
                return None
            values.append(mass * native.value)
            bounds.append(mass * float(bound))
    value, bound = math.fsum(values), math.fsum(bounds)
    if not (math.isfinite(value) and math.isfinite(bound)):
        return None
    return AnalyticMean(value, bound)


def eal_tail_coherence(
    eal: float,
    losses: np.ndarray,
    tail_metrics: Mapping[str, Mapping[str, object]],
    *,
    error_bound: float = 0.0,
) -> dict[str, object]:
    """Whether the analytic EAL fits the simulated years beside it.

    A distribution's TVaR at any level is at least its mean, and no mean
    exceeds its largest value. An EAL above the largest simulated annual loss
    or a reported TVaR, by more than its certified integration error bound
    ``error_bound``, means the simulated years miss the upper parameter ranks
    or severities that carry the mean, so the simulated tail statistics
    understate them.
    """
    # The least the exact mean can be.
    floor = eal - error_bound
    largest = float(np.max(losses)) if losses.size else 0.0
    levels = sorted(
        level
        for level, tail in tail_metrics.items()
        if isinstance(tvar := tail.get("tvar"), int | float) and floor > tvar
    )
    exceeds_maximum = floor > largest
    if not exceeds_maximum and not levels:
        return {"status": "coherent", "reason": None}
    parts = (["the largest simulated annual loss"] if exceeds_maximum else []) + (
        [f"the TVaR at level {', '.join(levels)}"] if levels else []
    )
    return {
        "status": "incoherent",
        "reason": "The analytic EAL exceeds "
        + " and ".join(parts)
        + ": the simulated years miss the upper parameter ranks or severities that carry the mean, so the simulated tail statistics understate them; simulate more years.",
        "exceeds_sampled_maximum": exceeds_maximum,
        "exceeds_tvar_levels": levels,
    }


def known_zero_law(draw: ConditionalCapabilityDraws, view: str | None) -> bool:
    """Require a captured structural proof, rather than observed zero losses.

    The state's native analytic mean must be a proven zero: every count law is
    zero at every parameter rank, or the view has no positive support.
    """
    views = (view,) if view is not None else tuple(draw.annual_losses)
    return all(
        (native := draw.views[name].native_mean) is not None
        and native.value == 0
        and (metadata := native.precision or {}).get("precision_basis")
        == DETERMINISTIC_EAL_PRECISION_BASIS
        and metadata.get("known_zero_law") is True
        and not metadata.get("precision_suppression_reason")
        for name in views
    )


def useful_support_unavailable_reason(
    mixture: CapabilityMixture, view: str
) -> str | None:
    """Why the simulated years cannot support zero-loss precision, if so.

    A state whose law is a proven zero needs no positive sample: its zeros are
    exact draws of the simulated mixture.
    """
    strata = mixture.positive_strata()
    for draw, _ in strata:
        if np.any(draw.annual_losses[view] > 0) or known_zero_law(draw, view):
            continue
        return (
            "No positive loss samples were captured for a positive-mass "
            "conditional state; zero-loss precision is unsupported."
        )
    return None


def design_precision_unavailable_reason(
    draws: Iterable[ConditionalCapabilityDraws], *, view: str | None = None
) -> str | None:
    """Why no design-based variance over independent years exists for these
    strata."""
    if any(draw.n_simulations < 2 and not known_zero_law(draw, view) for draw in draws):
        return "Native sampling has fewer than two independent years."
    return None


def aggregate_eal_precision(
    metadata: Mapping[str, Mapping[str, object]], weights: Mapping[str, float]
) -> dict[str, object]:
    """Bound a sum's estimated Monte Carlo SE without asserting independence.

    The weighted sum of the native standard errors bounds the sum's standard
    error under any covariance; missing or suppressed precision is never treated
    as zero. Certified deterministic integration error bounds
    (``integration_error_bound``) add by weight; an estimator without one has no
    deterministic integration error.
    """
    if set(metadata) != set(weights) or any(
        not math.isfinite(weight) or weight < 0 for weight in weights.values()
    ):
        raise ValueError(
            "EAL precision weights must match its components and be nonnegative"
        )
    bounds = []
    integration = []
    unavailable = []
    deterministic = True
    for key, record in metadata.items():
        if weights[key] == 0:
            continue
        error = record.get("integration_error_bound")
        integration.append(
            weights[key] * float(error) if isinstance(error, (int, float)) else 0.0
        )
        bound = record.get("standard_error_upper_bound")
        if bound is None:
            bound = record.get("numerical_standard_error")
        if (
            record.get("precision_suppression_reason")
            or isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(bound)
            or bound < 0
        ):
            unavailable.append(key)
            continue
        deterministic = (
            deterministic
            and bound == 0
            and record.get("precision_basis") == "deterministic_integration"
        )
        bounds.append(weights[key] * float(bound))
    if unavailable or not bounds:
        return {
            "numerical_standard_error": None,
            "standard_error_upper_bound": None,
            "precision_basis": "unavailable_component_estimator_precision",
            "precision_suppression_reason": "Missing or suppressed conditional estimator precision: "
            + ", ".join(unavailable or metadata),
        }
    return {
        "numerical_standard_error": 0.0 if deterministic else None,
        "standard_error_upper_bound": math.fsum(bounds),
        "integration_error_bound": math.fsum(integration),
        "precision_basis": "deterministic_integration"
        if deterministic
        else "sum_of_weighted_standard_error_bounds_no_independence_assumed",
        "precision_suppression_reason": None,
    }


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish ESS without squaring weights at their potentially tiny original scale."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("ESS requires finite nonnegative weights")
    maximum = float(np.max(values)) if len(values) else 0.0
    if maximum == 0:
        return 0.0
    scaled = values / maximum
    return float(np.sum(scaled) ** 2 / np.dot(scaled, scaled))
