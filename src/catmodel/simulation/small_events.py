"""Aggregate an origin's small incidents per year by their exact first two moments.

An event-addressed origin whose gross loss is one shared severity draw split by
fixed shares has two independent per-event marks that the views read: its
gross loss ``Y`` (the shared draw behind the member gate, times the summed
shares) and its aggregate lab loss ``L``. Every linear view is a fixed
multiple of ``Y``, plus ``L`` for the lab's own view. Given an origin-year's
event count, a multinomial draw splits its events into the rectangle
``Y < H``, ``L < H_L`` and two large cells. Large-cell events are drawn
individually with every mark; enough rectangle events are represented by two
normal sums with their exact conditional mean and variance, and no cap binds
in the rectangle, so every view's share of those sums is exact and only their
law beyond two moments is approximated. The split points come from the
compiled law and the frozen controls, never from the requested views, once for
each range of the year's parameter score (``SmallEventBuckets``,
``NUM-SMALL-EVENTS``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtri

from catmodel.probability.distributions import (
    SeveritySampler,
)
from catmodel.simulation.components import component_view_weight
from catmodel.simulation.incident_linear import EventCell, SingleMemberLaw

SMALL_EVENT_TARGET = 16.0
"""Expected individually drawn events per origin-year (kappa).

An INITIAL_ESTIMATE numerical judgment; range 8/16/32.

It mainly sets speed: each mark's split point is the loss whose expected annual
exceedances equal this target, or half of it when the origin has a stochastic
aggregate-lab law, at the score bucket's representative rate, unless
``SMALL_EVENT_MAX_SKEWNESS`` forces it lower (``NUM-SMALL-EVENTS``). Every
summed year keeps its exact mean and variance, but the split decides which
losses the normal law approximates, so it also moves the approximate tail
within the reported TVaR envelope.
"""

SMALL_EVENT_MAX_SKEWNESS = 0.1
"""Largest skewness of a year's small-event sum given its count.

An INITIAL_ESTIMATE numerical judgment; range 0.05/0.1/0.2.

A year's ``n`` small events have skewness ``gamma / sqrt(n)``, where ``gamma``
is one small event's skewness; they are summed by a normal law only when that
is at most 0.1 for both the gross and the aggregate-lab sum, and ``n`` is at
least ``min_summed_count``. Years with fewer are drawn event by event. At 0.1
the first-order Cornish-Fisher skewness correction to the sum's 99.9th
percentile is at most 0.14 of its standard deviation, an approximation rather
than a bound (``NUM-SMALL-EVENTS``).
"""

SMALL_EVENT_MIN_SUMMED = 100
"""Fewest small events a year's normal sum may represent.

An INITIAL_ESTIMATE numerical judgment; range 50/100/200.

The skewness bound alone admits a single event of a nearly symmetric law; a
normal law is used only for sums of at least this many events.
"""

SMALL_EVENT_BUCKET_RATE_RATIO = 2.0
"""Ratio of highest to lowest expected count in a score bucket (rho).

An INITIAL_ESTIMATE numerical judgment; range 1.5/2/4.

A year reads its count law at a fresh normal score, so its expected count can
range over orders of magnitude. Each ranked origin's scores are cut, downward
from ``SMALL_EVENT_TOP_RANK_SCORE``, where the expected count falls by this
ratio, and every year in a bucket uses the split points of the bucket's lowest
expected count. A year then expects at most this ratio times its split's
individually drawn events at that count: ``kappa * rho`` where the target sets
the split (``NUM-SMALL-EVENTS``).
"""

SMALL_EVENT_TOP_RANK_SCORE = 5.0
"""Normal score of the top score bucket's lowest score (z_max).

An INITIAL_ESTIMATE numerical judgment; range 4/5/6.

Scores above it, about 2.9e-7 of a rank group's years, share the top bucket,
whose expected count has no upper bound (``NUM-SMALL-EVENTS``).
"""

SMALL_EVENT_MAX_RANK_BUCKETS = 64
"""Most rank buckets one origin may have; a law needing more is refused."""

SMALL_EVENT_BODIES = ("normal", "individual")
"""``normal`` sums small events by their moments; ``individual`` draws each one
(an audit mode with the same large events, exact in law)."""

QUADRATURE_RELATIVE_TOLERANCE = 1e-9
_ORDERS = (24, 48)
_NODES = {order: leggauss(order) for order in _ORDERS}


@dataclass(frozen=True, slots=True)
class SmallEventControls:
    target: float = SMALL_EVENT_TARGET
    max_skewness: float = SMALL_EVENT_MAX_SKEWNESS
    body: Literal["normal", "individual"] = "normal"
    rate_ratio: float = SMALL_EVENT_BUCKET_RATE_RATIO
    top_rank_score: float = SMALL_EVENT_TOP_RANK_SCORE
    min_summed: int = SMALL_EVENT_MIN_SUMMED

    @classmethod
    def from_controls(cls, raw: object) -> SmallEventControls:
        if raw is None:
            return cls()
        allowed = {
            "target_large_events_per_year",
            "max_body_skewness",
            "body",
            "bucket_rate_ratio",
            "top_rank_score",
            "min_summed_count",
        }
        if not isinstance(raw, Mapping) or set(raw) - allowed:
            raise ValueError(f"small_events controls take only {sorted(allowed)}")
        raw = cast(Mapping[str, Any], raw)
        target = raw.get("target_large_events_per_year", SMALL_EVENT_TARGET)
        skewness = raw.get("max_body_skewness", SMALL_EVENT_MAX_SKEWNESS)
        body = raw.get("body", "normal")
        ratio = raw.get("bucket_rate_ratio", SMALL_EVENT_BUCKET_RATE_RATIO)
        top = raw.get("top_rank_score", SMALL_EVENT_TOP_RANK_SCORE)
        minimum = raw.get("min_summed_count", SMALL_EVENT_MIN_SUMMED)
        for name, value in (
            ("target_large_events_per_year", target),
            ("max_body_skewness", skewness),
            ("bucket_rate_ratio", ratio),
            ("top_rank_score", top),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"small_events {name} must be positive and finite")
        if ratio <= 1:
            raise ValueError("small_events bucket_rate_ratio must exceed one")
        if body not in SMALL_EVENT_BODIES:
            raise ValueError(f"small_events body must be one of {SMALL_EVENT_BODIES}")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("small_events min_summed_count must be a positive integer")
        return cls(
            float(target), float(skewness), body, float(ratio), float(top), minimum
        )

    def record(self) -> dict[str, object]:
        return {
            "target_large_events_per_year": self.target,
            "max_body_skewness": self.max_skewness,
            "body": self.body,
            "bucket_rate_ratio": self.rate_ratio,
            "top_rank_score": self.top_rank_score,
            "min_summed_count": self.min_summed,
        }


SIMULATED_RANK_RANGE = (float(np.nextafter(0.0, 1.0)), float(np.nextafter(1.0, 0.0)))
"""The range of a simulated parameter rank, to which a year's uniform is clipped."""
SIMULATED_SCORE_RANGE: tuple[float, float] = (
    float(ndtri(SIMULATED_RANK_RANGE[0])),
    float(ndtri(SIMULATED_RANK_RANGE[1])),
)
"""Normal scores of the simulated rank range after its uniform is clipped."""

_SHARE_NODES, _SHARE_WEIGHTS = leggauss(16)


def bucket_mean_shares(
    multiplier, scores: np.ndarray, breaks: tuple[float, ...]
) -> tuple[float, ...]:
    """Each score bucket's share ``E[m(Z); Z in b]`` of a per-score multiplier.

    ``m`` maps scores to multipliers and is nondecreasing; ``scores`` are the
    buckets' inner edges and ``breaks`` the normal scores where ``m`` has a
    kink. Sixteen-point Gauss-Legendre in the normal score ``z`` over the
    simulated scores' range, on unit pieces split at both, where
    ``m(z) phi(z)`` is smooth.
    """
    low, high = SIMULATED_SCORE_RANGE
    points = sorted(
        {
            low,
            high,
            *map(float, range(math.ceil(low), math.floor(high) + 1)),
            *(float(score) for score in (*scores, *breaks) if low < score < high),
        }
    )
    pieces = np.array(list(pairwise(points)))
    half = (pieces[:, 1] - pieces[:, 0])[:, None] / 2
    nodes = ((pieces[:, 1] + pieces[:, 0])[:, None] / 2 + half * _SHARE_NODES).ravel()
    weights = (half * _SHARE_WEIGHTS).ravel() * np.exp(-nodes * nodes / 2)
    weights /= math.sqrt(2 * math.pi)
    values = multiplier(nodes)
    buckets = np.searchsorted(scores, nodes, side="right")
    return tuple(
        float(share)
        for share in np.bincount(
            buckets, weights=weights * values, minlength=len(scores) + 1
        )
    )


@dataclass(frozen=True, slots=True)
class SmallEventSplit:
    """One origin's split point and its small events' exact moments.

    Gross uniforms below ``rank_gross`` and aggregate-lab uniforms below
    ``rank_lab`` form the small rectangle. ``gross_moments`` are raw moments
    1-3 of an event's gross loss there (gated-out events count as zero), and
    ``lab_moments`` of its aggregate lab loss.
    """

    split_point: float
    lab_split_point: float
    gross_per_raw: float
    gross_per_rank_value: float
    sampler: SeveritySampler
    rank_gross: float
    rank_lab: float
    member_probability: float
    small_member_probability: float
    gross_moments: tuple[float, float, float]
    lab_moments: tuple[float, float, float]
    minimum_count: int

    @property
    def large_gross_probability(self) -> float:
        return self.member_probability * (1.0 - self.rank_gross)

    @property
    def large_lab_probability(self) -> float:
        return (1.0 - self.large_gross_probability) * (1.0 - self.rank_lab)

    @property
    def small_probability(self) -> float:
        return (1.0 - self.large_gross_probability) * self.rank_lab

    def cells(self) -> dict[str, EventCell]:
        """The three cells' addresses and rank intervals, large cells first."""
        return {
            "large_gross": EventCell(
                "small-events::large-gross::", 1.0, (self.rank_gross, 1.0), (0.0, 1.0)
            ),
            "large_lab": EventCell(
                "small-events::large-lab::",
                self.small_member_probability,
                (0.0, self.rank_gross),
                (self.rank_lab, 1.0),
            ),
            "small": EventCell(
                "small-events::small::",
                self.small_member_probability,
                (0.0, self.rank_gross),
                (0.0, self.rank_lab),
            ),
        }

    def body_maximum(self, counts: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
        """The largest gross loss among each year's ``counts`` small events.

        Exact in law: the maximum of ``n`` draws from the rectangle's gross law
        is its quantile at ``uniform ** (1 / n)``, zero below the gated-out mass.
        """
        result = np.zeros(len(counts))
        positive = counts > 0
        level = np.ones(len(counts))
        level[positive] = uniforms[positive] ** (1.0 / counts[positive])
        gated = 1.0 - self.small_member_probability
        member = positive & (level > gated)
        if np.any(member):
            ranks = (level[member] - gated) / self.small_member_probability
            ranks = np.minimum(
                ranks * self.rank_gross, np.nextafter(self.rank_gross, 0.0)
            )
            result[member] = self.gross_per_rank_value * (
                self.sampler.sample_from_uniforms_uncapped(ranks)
            )
        return result


@dataclass(frozen=True, slots=True)
class SmallEventBuckets:
    """One origin's split points in each parameter-score bucket.

    A year whose score satisfies ``edges[b - 1] <= z < edges[b]`` is in bucket
    ``b``; a law without a parameter rank has no edges and one bucket.
    ``rates`` holds the lowest expected count a year in each bucket can have,
    at which the bucket's split was found, and ``mean_shares`` each bucket's
    share of the expected count, ``E[m(Z); Z in b]`` for the year's count
    multiplier ``m``, which weights its small-event variance. ``splits`` holds
    each bucket's split, or why its years are drawn event by event.
    """

    segment: int
    gross_per_raw: float
    edges: tuple[float, ...]
    rates: tuple[float, ...]
    mean_shares: tuple[float, ...]
    splits: tuple[SmallEventSplit | str, ...]

    def assign(self, scores: np.ndarray | None, simulations: int) -> np.ndarray:
        """Each year's bucket, from its parameter score alone."""
        if not self.edges:
            return np.zeros(simulations, dtype=np.int64)
        assert scores is not None
        return np.searchsorted(np.asarray(self.edges), scores, side="right")

    @property
    def largest_rate_ratio(self) -> float | None:
        """The largest ratio of a split bucket's highest to lowest expected
        count, over buckets with an upper edge; the top bucket has none."""
        ratios = [
            self.rates[bucket + 1] / self.rates[bucket]
            for bucket, split in enumerate(self.splits[:-1])
            if not isinstance(split, str)
        ]
        return max(ratios) if ratios else None

    def record(self) -> dict[str, object]:
        """One column entry per bucket; a bucket drawn event by event has null
        split fields and is listed under its reason. A lab split point is
        null where no lab cap bounds it: a capture's JSON holds no infinity,
        so replay re-derives the same null."""
        reasons: dict[str, list[int]] = {}
        for bucket, split in enumerate(self.splits):
            if isinstance(split, str):
                reasons.setdefault(split, []).append(bucket)

        def column(read):
            return [
                None if isinstance(split, str) else read(split) for split in self.splits
            ]

        return {
            "lowest_score": [SIMULATED_SCORE_RANGE[0], *self.edges]
            if self.edges
            else [None],
            "representative_rate": list(self.rates),
            "split_point_usd": column(lambda split: split.split_point),
            "lab_split_point_usd": column(
                lambda split: (
                    split.lab_split_point
                    if math.isfinite(split.lab_split_point)
                    else None
                )
            ),
            "minimum_summed_count": column(lambda split: split.minimum_count),
            "event_by_event_reasons": reasons,
            "largest_rate_ratio": self.largest_rate_ratio,
        }


def fixed_rate_buckets(
    prepared: SmallEventLaw, segment: int, rate: float
) -> SmallEventBuckets | str:
    """A law without a parameter rank: one bucket at its rate."""
    split = prepared.split(rate)
    if isinstance(split, str):
        return split
    return SmallEventBuckets(
        segment=segment,
        gross_per_raw=prepared.gross_per_raw,
        edges=(),
        rates=(rate,),
        mean_shares=(1.0,),
        splits=(split,),
    )


def _lowest_scores_reaching(multiplier, targets: np.ndarray, low: float, high: float):
    """For each target, the lowest normal score in ``[low, high]`` whose
    multiplier reaches it, by bisection; the multiplier at ``high`` must."""
    below = np.full(len(targets), low)
    above = np.full(len(targets), high)
    for _ in range(64):
        middle = 0.5 * (below + above)
        reached = multiplier(middle) >= targets
        above = np.where(reached, middle, above)
        below = np.where(reached, below, middle)
    return above


def ranked_buckets(
    prepared: SmallEventLaw,
    segment: int,
    reference_rate: float,
    multiplier,
    breaks: tuple[float, ...],
) -> SmallEventBuckets | str:
    """A ranked law's buckets, cut downward from the top score by rate ratio.

    ``multiplier`` maps year normal scores to the count multiplier, which does
    not decrease with the score; ``breaks`` are the normal scores where it has
    a kink. The top bucket holds scores above ``top_rank_score``. Below it,
    edges fall at the lowest scores whose expected count reaches the top score's
    divided by ``rate_ratio``, its square and so on; each bucket's split is
    found at its lowest score's count. Cutting stops at the first bucket whose
    split is refused for a reason every lower count shares (``LOW_RATE_REFUSALS``):
    that bucket and every lower score form one bucket drawn event by event.
    A bucket refused for another reason is drawn event by event and cutting
    continues below it. Returns the reason when no bucket has a split.
    """
    controls = prepared.controls
    low, high = SIMULATED_SCORE_RANGE
    top = min(controls.top_rank_score, high)
    top_multiplier = float(multiplier(np.array([top]))[0])
    floor = float(multiplier(np.array([low]))[0])
    powers = controls.rate_ratio ** np.arange(1, SMALL_EVENT_MAX_RANK_BUCKETS + 1)
    targets = top_multiplier / powers
    targets = targets[targets > floor]
    scores = _lowest_scores_reaching(multiplier, targets, low, top)
    # Each cut's lowest score, from the top bucket down, then the lowest score.
    lowest = [top, *(float(score) for score in scores)]
    found: list[SmallEventSplit | str] = []
    rates: list[float] = []
    merged = None
    for score in lowest:
        rate = reference_rate * float(multiplier(np.array([score]))[0])
        split = prepared.split(rate)
        if isinstance(split, str) and split in LOW_RATE_REFUSALS:
            merged = split
            break
        found.append(split)
        rates.append(rate)
    # Every bucket so far starts at its cut; the lowest bucket holds every
    # score below the last, refused if cutting stopped, and otherwise spans less
    # than one ratio and takes its own split.
    edges = lowest[: len(found)]
    bottom_rate = reference_rate * floor
    found.append(merged if merged is not None else prepared.split(bottom_rate))
    rates.append(bottom_rate)
    if len(found) > SMALL_EVENT_MAX_RANK_BUCKETS:
        raise ValueError(
            f"an origin needs more than {SMALL_EVENT_MAX_RANK_BUCKETS} rank buckets "
            f"at a bucket rate ratio of {controls.rate_ratio}; raise the ratio"
        )
    if all(isinstance(split, str) for split in found):
        return next(split for split in found if isinstance(split, str))
    # Ascending in score: the lowest bucket first, the top bucket last.
    edges.reverse()
    found.reverse()
    rates.reverse()
    shares = bucket_mean_shares(multiplier, np.asarray(edges), breaks)
    return SmallEventBuckets(
        segment=segment,
        gross_per_raw=prepared.gross_per_raw,
        edges=tuple(edges),
        rates=tuple(rates),
        mean_shares=shares,
        splits=tuple(found),
    )


def _cdf(sampler: SeveritySampler, value: float) -> float:
    """P(raw <= value) before the sampler's outer cap."""
    return float(sampler.cdf(np.array([value], dtype=float), apply_cap=False)[0])


_TOP_RANK = 1.0 - 1e-12


def _pieces(upper: float, breaks: tuple[float, ...]) -> list[tuple[float, float]]:
    points = {0.0, upper, *(rank for rank in breaks if 0.0 < rank < upper)}
    edge = 0.05
    while 1.0 - edge < upper:
        points.add(1.0 - edge)
        edge /= 2
    # Merge points that differ only by rounding (a knot can coincide with a
    # geometric point); a sliver piece would sit on a law's boundary band.
    merged: list[float] = []
    for point in sorted(points):
        if not merged or point - merged[-1] > 1e-9 * (1.0 - merged[-1]):
            merged.append(point)
    merged[-1] = upper
    return list(pairwise(merged))


class _RankLaw:
    """Raw moments 1-3 of ``scale * Q(U)`` given ``U < upper``, for any upper.

    Gauss-Legendre nodes on fixed rank pieces split at the law's knots,
    splice, p99 break, any large-loss anchor, and geometrically toward rank
    one, evaluated in one call. ``approximate`` interpolates their cumulative
    sums for searching; ``certified`` integrates the piece holding ``upper``
    on its own at two orders. The integrand is bounded below any ``upper < 1``.
    """

    def __init__(self, sampler: SeveritySampler, scale: float):
        self.sampler, self.scale = sampler, scale
        pieces = _pieces(_TOP_RANK, sampler.rank_breaks())
        self.edges = np.array([0.0, *(high for _, high in pieces)])
        nodes, weights = _NODES[48]
        ranks = np.concatenate(
            [0.5 * (high - low) * nodes + 0.5 * (high + low) for low, high in pieces]
        )
        mass = np.concatenate([0.5 * (high - low) * weights for low, high in pieces])
        values = scale * sampler.sample_from_uniforms_uncapped(ranks)
        terms = np.stack([mass * values**power for power in (1, 2, 3)])
        self.ranks = ranks
        self.mass = mass
        self.cumulative = np.concatenate(
            [np.zeros((3, 1)), np.cumsum(terms, axis=1)], axis=1
        )
        self.terms = terms
        piece = np.searchsorted(self.edges, ranks, side="right") - 1
        self.piece_sums = np.stack(
            [np.bincount(piece, weights=row, minlength=len(pieces)) for row in terms]
        )

    def approximate(self, upper: float) -> np.ndarray:
        """Moments with the node holding ``upper`` counted pro rata."""
        index = int(np.searchsorted(self.ranks, upper))
        total = self.cumulative[:, index].copy()
        if index < len(self.ranks):
            low = self.ranks[index - 1] if index else 0.0
            fraction = (upper - low) / (self.ranks[index] - low)
            total += min(max(fraction, 0.0), 1.0) * self.terms[:, index]
        return total / upper

    def _piece(self, low: float, high: float, order: int) -> np.ndarray:
        nodes, weights = _NODES[order]
        ranks = 0.5 * (high - low) * nodes + 0.5 * (high + low)
        mass = 0.5 * (high - low) * weights
        values = self.scale * self.sampler.sample_from_uniforms_uncapped(ranks)
        return np.array([np.dot(mass, values**power) for power in (1, 2, 3)])

    def certified(self, upper: float) -> tuple[float, float, float] | None:
        """The moments, if two quadrature orders agree to the tolerance."""
        index = int(np.searchsorted(self.edges, upper, side="right")) - 1
        below = self.piece_sums[:, :index].sum(axis=1)
        low = self.edges[index]
        found = [
            (below + (self._piece(low, upper, order) if upper > low else 0.0)) / upper
            for order in _ORDERS
        ]
        coarse, fine = found
        if not np.all(np.isfinite(fine)) or np.any(
            np.abs(fine - coarse) > QUADRATURE_RELATIVE_TOLERANCE * np.abs(fine)
        ):
            return None
        return (float(fine[0]), float(fine[1]), float(fine[2]))


def _gated(moments, member: float) -> tuple[float, float, float]:
    """Raw moments of ``G * X`` where ``P(G = 1) = member``."""
    return (member * moments[0], member * moments[1], member * moments[2])


def skewness(moments) -> float:
    """Absolute skewness from raw moments 1-3 (zero for a point mass)."""
    m1, m2, m3 = moments
    variance = m2 - m1 * m1
    if variance <= 1e-12 * m2:
        return 0.0
    return abs(m3 - 3 * m1 * m2 + 2 * m1**3) / variance**1.5


def _highest(admissible, low: float, high: float) -> float | None:
    """The largest rank in ``(low, high]`` that ``admissible`` accepts.

    Truncating lower makes a law less skewed, so acceptable ranks reach down
    toward ``low``; a descending scan in ``1 - rank`` finds one before
    bisecting upward.
    """
    if high <= low:
        return None
    if admissible(high):
        return high
    above = high
    for point in 1.0 - (1.0 - high) * ((1.0 - low) / (1.0 - high)) ** (
        np.arange(1, 49) / 48
    ):
        point = max(float(point), low + (high - low) * 1e-12)
        if admissible(point):
            found = point
            break
        above = point
    else:
        return None
    for _ in range(50):
        middle = 0.5 * (found + above)
        if admissible(middle):
            found = middle
        else:
            above = middle
    return found


BELOW_TARGET = "Its expected count does not exceed the individually drawn target."
CAP_AT_FLOOR = "A cap lies at or below the smallest admitted loss."
TOO_FEW_SMALL_EVENTS = "Too few small events remain to meet the skewness bound."
NO_LAB_SPLIT = "No aggregate-lab split meets the skewness bound."
NO_GROSS_SPLIT = (
    "No gross split above the smallest admitted loss meets the skewness bound."
)
LOW_RATE_REFUSALS = frozenset(
    {BELOW_TARGET, CAP_AT_FLOOR, TOO_FEW_SMALL_EVENTS, NO_LAB_SPLIT, NO_GROSS_SPLIT}
)
"""Refusals that every lower expected count shares: a lower count only lowers
the target's rank and the small-event count that the skewness bound needs, and
a cap refusal does not depend on the count. Other refusals, such as moments
that cannot be certified, can occur at one count and not at a lower one. The
two split refusals are monotone up to ``_highest``'s scan, which can miss a
narrow admissible interval and stop cutting a step or two early; those years
are then drawn event by event, which is exact."""


class SmallEventLaw:
    """The rate-independent part of an origin's split points.

    Eligibility, caps and the two marks' rank laws depend only on the law and
    the frozen controls, so one origin's rank buckets share them; ``split``
    finds the split points at one expected count.
    """

    def __init__(self, law: SingleMemberLaw, controls: SmallEventControls):
        self.controls = controls
        self.refusal: str | None = None
        self.cap_refusal: str | None = None
        outcome = law.outcome
        if outcome.memberships or outcome.payment_kernel is not None:
            self.refusal = "The outcome carries policy terms."
            return
        stochastic = [
            (component, sampler)
            for component, sampler in zip(outcome.components, law.samplers, strict=True)
            if component.multiplier != 0 and component.post_multiplier != 0
        ]
        shared = "The gross loss is not one severity draw split by fixed shares."
        if not stochastic:
            self.refusal = shared
            return
        first, sampler = stochastic[0]
        if sampler is None or any(
            component.amount is not None
            or other is not sampler
            or component.multiplier != first.multiplier
            or (component.rank_scope, component.rank_group_id)
            != (first.rank_scope, first.rank_group_id)
            for component, other in stochastic
        ):
            self.refusal = shared
            return
        if len(law.labs) > 1:
            self.refusal = "The outcome carries several aggregate-lab laws."
            return
        self.sampler = sampler
        self.gross_per_raw = math.fsum(
            component.post_multiplier for component, _ in stochastic
        )
        self.scale = self.gross_per_raw * first.multiplier
        self.lab = law.labs[0] if law.labs else None
        self.lab_sampler = law.lab_samplers[0] if law.labs else None
        self.stochastic_lab = self.lab is not None and self.lab.amount is None
        gross_cap = (
            math.inf
            if sampler.severity_cap is None
            else self.gross_per_raw * sampler.severity_cap
        )
        self.lab_cap = math.inf
        lab = self.lab
        if lab is not None and lab.episode_cap is not None:
            try:
                share = math.fsum(
                    component.post_multiplier
                    * component_view_weight(component, f"lab_loss::{lab.lab_id}")
                    for component, _ in stochastic
                )
            except ValueError:
                share = self.gross_per_raw
            # Half the episode cap to each mark: share / gross * H + H_L <= cap.
            if share > 0:
                gross_cap = min(
                    gross_cap, lab.episode_cap * self.gross_per_raw / (2 * share)
                )
            self.lab_cap = lab.episode_cap / 2
            if lab.amount is not None and lab.amount >= self.lab_cap:
                self.cap_refusal = CAP_AT_FLOOR
        self.member_probability = law.member_probability
        # Ranks below these keep each mark under its caps.
        self.gross_ceiling = (
            _TOP_RANK
            if not math.isfinite(gross_cap)
            else min(_TOP_RANK, _cdf(sampler, gross_cap * (1 - 1e-9) / self.scale))
        )
        if self.stochastic_lab:
            assert self.lab_sampler is not None
            self.lab_ceiling = (
                _TOP_RANK
                if not math.isfinite(self.lab_cap)
                else min(_TOP_RANK, _cdf(self.lab_sampler, self.lab_cap * (1 - 1e-9)))
            )
            self.lab_floor = _cdf(self.lab_sampler, 0.0)
        self._laws: tuple[_RankLaw, _RankLaw | None] | None = None

    def _rank_laws(self) -> tuple[_RankLaw, _RankLaw | None]:
        if self._laws is None:
            self._laws = (
                _RankLaw(self.sampler, self.scale),
                _RankLaw(self.lab_sampler, 1.0)
                if self.stochastic_lab and self.lab_sampler is not None
                else None,
            )
        return self._laws

    def split(self, rate: float) -> SmallEventSplit | str:
        """The split points at expected count ``rate``, or why there are none.

        Each mark has its own rank threshold, the gross draw's and the
        aggregate lab loss's, and so its own split point. Each mark targets half
        the individually drawn events when the origin has a stochastic
        aggregate-lab law, all of them otherwise. Each threshold is then lowered
        where needed so that, at the small-event count a year at ``rate`` leaves
        less five Poisson standard deviations, both sums' skewness is at most the
        bound.
        ``rate`` sets only speed and the skewness margin. Caps bound both: a
        small event's capped gross stays below the outer cap, and its lab view,
        at most ``share / gross * H + H_L``, below the episode lab cap.
        """
        if self.refusal is not None:
            return self.refusal
        controls = self.controls
        if not math.isfinite(rate) or rate <= controls.target:
            return BELOW_TARGET
        if self.cap_refusal is not None:
            return self.cap_refusal
        q = self.member_probability
        stochastic_lab = self.stochastic_lab
        target = controls.target / 2 if stochastic_lab else controls.target
        gross_speed = min(self.gross_ceiling, 1.0 - target / (rate * q))
        if stochastic_lab:
            lab_floor = self.lab_floor
            lab_speed = min(self.lab_ceiling, 1.0 - target / rate)
        if gross_speed <= 0.0 or (stochastic_lab and lab_speed <= lab_floor):
            return CAP_AT_FLOOR
        gross_law, lab_law = self._rank_laws()

        def gross_skew(rank: float) -> float:
            member = q * rank / (1.0 - q + q * rank)
            return skewness(_gated(gross_law.approximate(rank), member))

        # Both sums share each year's small-event count, and lowering a
        # threshold shrinks the count a year leaves, so repeat until it holds.
        expected = rate - controls.target
        for _ in range(8):
            floor_count = expected - 5 * math.sqrt(max(expected, 0.0))
            if floor_count < controls.min_summed:
                return TOO_FEW_SMALL_EVENTS
            # One percent of slack so the iteration settles strictly inside.
            bound = controls.max_skewness * math.sqrt(0.99 * floor_count)
            if stochastic_lab:
                assert lab_law is not None
                lab_rank = _highest(
                    lambda rank, bound=bound: (
                        skewness(lab_law.approximate(rank)) <= bound
                    ),
                    lab_floor,
                    lab_speed,
                )
                if lab_rank is None:
                    if lab_floor <= 0.0:
                        return NO_LAB_SPLIT
                    # Only the zero atom stays small: every event with positive
                    # aggregate lab loss is drawn, and the small events'
                    # aggregate-lab sum is exactly zero.
                    lab_rank = lab_floor
            else:
                lab_rank = 1.0
            gross_rank = _highest(
                lambda rank, bound=bound: gross_skew(rank) <= bound, 0.0, gross_speed
            )
            if gross_rank is None:
                return NO_GROSS_SPLIT
            expected = rate * (1.0 - q + q * gross_rank) * lab_rank
            need = max(
                controls.min_summed,
                (gross_skew(gross_rank) / controls.max_skewness) ** 2,
                (skewness(lab_law.approximate(lab_rank)) / controls.max_skewness) ** 2
                if lab_law is not None and lab_rank > lab_floor
                else 0.0,
            )
            if expected - 5 * math.sqrt(expected) >= need:
                break
        else:
            return "No split pair meets the skewness bound."
        gross_moments = gross_law.certified(gross_rank)
        if gross_moments is None:
            return "The small events' moments cannot be integrated to the tolerance."
        member = q * gross_rank / (1.0 - q + q * gross_rank)
        gross_moments = _gated(gross_moments, member)
        if stochastic_lab and lab_rank == lab_floor:
            lab_moments, lab_split = (0.0, 0.0, 0.0), 0.0
        elif stochastic_lab:
            assert lab_law is not None and self.lab_sampler is not None
            lab_moments = lab_law.certified(lab_rank)
            if lab_moments is None:
                return (
                    "The small events' moments cannot be integrated to the tolerance."
                )
            lab_split = float(
                self.lab_sampler.sample_from_uniforms_uncapped(np.array([lab_rank]))[0]
            )
        else:
            lab = self.lab
            fixed = 0.0 if lab is None or lab.amount is None else float(lab.amount)
            lab_moments = (fixed, fixed**2, fixed**3)
            lab_split = self.lab_cap
        minimum = max(
            controls.min_summed,
            math.ceil((skewness(gross_moments) / controls.max_skewness) ** 2),
            math.ceil((skewness(lab_moments) / controls.max_skewness) ** 2),
        )
        return SmallEventSplit(
            split_point=self.scale
            * float(
                self.sampler.sample_from_uniforms_uncapped(np.array([gross_rank]))[0]
            ),
            lab_split_point=lab_split,
            gross_per_raw=self.gross_per_raw,
            gross_per_rank_value=self.scale,
            sampler=self.sampler,
            rank_gross=gross_rank,
            rank_lab=lab_rank,
            member_probability=q,
            small_member_probability=member,
            gross_moments=gross_moments,
            lab_moments=lab_moments,
            minimum_count=minimum,
        )


def body_variance(
    split: SmallEventSplit,
    gross_coefficient: float,
    lab_coefficient: float,
    rate: float,
    variance_to_mean: float,
) -> float:
    """Variance of one view's small-event sum: ``E[N]Var(Y) + Var(N)E[Y]^2``.

    ``N`` thins the origin's count (variance-to-mean ratio ``variance_to_mean``)
    to the rectangle; ``Y = a G + c L`` with gross ``G`` and lab loss ``L``
    independent on it.
    """
    p = split.small_probability
    g1, g2, _ = split.gross_moments
    l1, l2, _ = split.lab_moments
    a, c = gross_coefficient, lab_coefficient
    mean = a * g1 + c * l1
    second = a * a * g2 + 2 * a * c * g1 * l1 + c * c * l2
    count_mean = rate * p
    count_variance = p * p * variance_to_mean * rate + p * (1 - p) * rate
    return count_mean * (second - mean * mean) + count_variance * mean * mean


def tvar_envelope(standard_deviation: float, level: float) -> float:
    """Distribution-free bound on a mean-zero noise's TVaR (Cauchy-Schwarz).

    Any ``D`` with mean zero and standard deviation ``s`` has
    ``TVaR_p(D) <= s * sqrt(p / (1 - p))``. The small-event sums and their
    normal stand-ins share each state's conditional mean and variance, so the
    annual totals' TVaRs differ by at most this.
    """
    return standard_deviation * math.sqrt(level / (1.0 - level))


SMALL_EVENT_LIMITATION = "Incidents below a per-origin split point are summed per year by a normal law with their exact conditional mean and variance; larger incidents are drawn individually with every mark. The reported small_event_aggregation envelope bounds the difference between the TVaRs of the summed and event-by-event laws, excluding Monte Carlo error (NUM-SMALL-EVENTS)."


def small_event_results(
    executions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    weights: Mapping[str, Mapping[str, float]],
    reference_arm: str,
    levels: tuple[float, ...],
    *,
    view_weights: Mapping[str, Callable[[str], Mapping[str, float] | None]]
    | None = None,
) -> dict[str, Any]:
    """Controls, split points and each view's TVaR envelope, from the capture.

    ``executions`` holds each arm's per-state execution records and ``weights``
    the arm's state weights. ``events`` counts every arm's individually drawn
    and summed events and its fallback years, origin-years in a bucket with a
    split whose small events were too few to sum. A view's envelope is
    ``sqrt(sum_z w_z Var_z) * sqrt(p / (1 - p))`` over positive-weight states
    (``tvar_envelope``): the variance of the states' small-event sums mixed by
    weight, each centred on its own mean and already mixed over each origin's
    rank buckets. ``view_weights`` gives, by arm, a view's weights where some
    view's are not the arm's, as a lab-loss or gross view's are where a state
    is left out of the views that count the lab's loss; a view the arm has no
    weights for (None) has no envelope.
    """
    controls: dict[str, Any] | None = None
    envelopes: dict[str, Any] = {}
    events = {"individually_drawn": 0, "summed": 0, "fallback_years": 0}
    for arm, rows in executions.items():
        positive = {
            state: weight for state, weight in weights[arm].items() if weight > 0
        }
        total = math.fsum(positive.values())
        # The positive weight of each set of weights a view reads, by identity.
        totals = {id(weights[arm]): total}
        variances: dict[str, list[float]] = {}
        reasons: set[str] = set()
        for state, execution in rows.items():
            record: Mapping[str, Any] = execution["small_events"]
            controls = controls or dict(record["controls"])
            events["individually_drawn"] += int(execution["vectorized_initiators"])
            events["summed"] += int(execution["summed_small_initiators"])
            events["fallback_years"] += sum(
                int(row.get("fallback_years", 0))
                for row in record["by_origin"].values()
            )
            if state not in positive:
                continue
            if record["envelope_reason"]:
                reasons.add(record["envelope_reason"])
            for view, variance in record["body_variance_by_view"].items():
                of = weights[arm] if view_weights is None else view_weights[arm](view)
                if of is None or of.get(state, 0.0) <= 0:
                    continue
                if id(of) not in totals:
                    totals[id(of)] = math.fsum(
                        weight for weight in of.values() if weight > 0
                    )
                variances.setdefault(view, []).append(
                    of[state] / totals[id(of)] * variance
                )
        envelopes[arm] = {
            "reason": "; ".join(sorted(reasons)) or None,
            "views": {
                view: {
                    "body_standard_deviation": (
                        deviation := math.sqrt(math.fsum(parts))
                    ),
                    "tvar_bound": {
                        str(level): None if reasons else tvar_envelope(deviation, level)
                        for level in levels
                    },
                }
                for view, parts in sorted(variances.items())
            },
        }
    splits: dict[str, dict[str, Any]] = {}
    for execution in executions.get(reference_arm, {}).values():
        record = execution["small_events"]
        for row in record["by_origin"].values():
            if row["status"] != "summed_small_events":
                continue
            key = f"{row['scenario_id']}::{row['source_public_status']}"
            entry = splits.setdefault(key, {"states": 0, "split_points_usd": []})
            entry["states"] += 1
            entry["split_points_usd"].extend(
                point
                for point in row["rank_buckets"]["split_point_usd"]
                if point is not None
            )
    return {
        "method": "Poisson-thinned large cells drawn individually; small events summed by a normal law with exact conditional mean and variance, at split points found per parameter-rank bucket (NUM-SMALL-EVENTS).",
        "controls": controls,
        "events": events,
        "split_points_by_origin": {
            key: {
                "states": entry["states"],
                "min_usd": min(entry["split_points_usd"]),
                "max_usd": max(entry["split_points_usd"]),
            }
            for key, entry in sorted(splits.items())
        },
        "error_envelope": envelopes,
        "envelope_basis": "Distribution-free Cauchy-Schwarz bound on the TVaR of the small-event sums' noise, which the exact and the normal sums share in mean and variance; it bounds the difference between the two laws' TVaRs, not Monte Carlo error.",
    }
