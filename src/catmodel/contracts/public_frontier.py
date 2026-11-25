"""Observed-anchored lab general-ECI forecasts with independent lab paces.

Each lab's path starts at the frontier it had published by the forecast as-of
date and grows at its own positive constant rate: a Bayesian trend rate pooled
across the labs' frontier-raising releases, with lab-specific levels, times a
median-one pace drawn independently for each lab. Labs therefore progress at
different rates and the frontier leader can change. The pace shape is an
explicit forward assumption whose default, 5, is a conservative judgment
informed by a backtest of releases.
"""

from datetime import datetime
from fractions import Fraction
from itertools import pairwise
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmodel.contracts.capability_evidence import CapabilityEvidenceRef, UTCInstant
from catmodel.contracts.capability_paths import (
    Finite,
    JointPathDraw,
    Probability,
    Vector,
)
from catmodel.contracts.governance import (
    SourceScaleAnchor,
    SourceScaleCorrespondence,
)
from catmodel.contracts.primitives import (
    RequiredIdentifier,
    RequiredString,
    Sha256Digest,
)
from catmodel.hashing import canonical_json_sha256
from catmodel.json_io import require_marker

type PublicFrontierVariant = Literal["baseline", "no_progress"]
VARIANTS: tuple[PublicFrontierVariant, ...] = ("baseline", "no_progress")
YEAR_SECONDS = 365.25 * 24 * 60 * 60


def forecast_times(start: datetime, end: datetime) -> tuple[datetime, ...]:
    """Period start, each UTC month start strictly inside the period, and its end."""
    for value in (start, end):
        offset = value.utcoffset()
        if offset is None:
            raise ValueError("forecast period bounds must include a timezone")
        if offset.total_seconds() != 0:
            raise ValueError("forecast period bounds must be UTC instants")
    if not start < end:
        raise ValueError("forecast period must end after it starts")
    times = [start]
    at = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while True:
        at = (
            at.replace(year=at.year + 1, month=1)
            if at.month == 12
            else at.replace(month=at.month + 1)
        )
        if at >= end:
            break
        times.append(at)
    times.append(end)
    return tuple(times)


def progress_elapsed_years(as_of: datetime, times: tuple[datetime, ...]) -> float:
    """Years from as-of to the middle of the period ``times`` spans.

    Frontier progress, the public frontier's gain over its as-of level, is
    measured at this instant: it orders trajectories, sets their node weights
    and links path-linked exposure to capability.
    """
    middle = times[0] + (times[-1] - times[0]) / 2
    return (middle - as_of).total_seconds() / YEAR_SECONDS


def frontier_progress(
    anchors: np.ndarray, rates: np.ndarray, elapsed: float
) -> np.ndarray:
    """The frontier's gain over its as-of level after ``elapsed`` years.

    ``rates`` has one row per path and one column per lab, in ``anchors``'
    order; each lab path is ``anchor + rate * elapsed``.
    """
    rates = np.atleast_2d(rates)
    return (anchors[None, :] + rates * elapsed).max(axis=1) - anchors.max()


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def digest(self) -> str:
        return canonical_json_sha256(self.model_dump(mode="json"))


# Diffuse default rate prior, in ECI points per 365.25-day year, before
# positive-rate conditioning. Wide enough that the records, not the prior,
# set the posterior; an authored prior needs its own rationale.
DIFFUSE_RATE_PRIOR_MEAN = 0.0
DIFFUSE_RATE_PRIOR_SD = 100.0


class PublicRatePrior(_Record):
    """Gaussian prior on the pooled trend rate, conditioned on a positive rate."""

    mean: Finite
    sd: Annotated[Finite, Field(ge=0)]
    rationale: RequiredString

    @model_validator(mode="after")
    def positive_support(self) -> Self:
        if self.sd == 0 and self.mean <= 0:
            raise ValueError("fixed baseline rate must be positive")
        return self


class PublicFrontierCoordinate(_Record):
    lab_id: RequiredIdentifier
    domain_id: Literal["general"] = "general"
    scale_id: RequiredIdentifier
    units: Literal["ECI points"]
    evidence_refs: tuple[CapabilityEvidenceRef, ...] = Field(min_length=1)
    rationale: RequiredString
    limitations: tuple[RequiredString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def named_lab(self) -> Self:
        if self.lab_id == "public_frontier":
            raise ValueError("public_frontier is a derived maximum, not a fitted lab")
        return self


class PublicFrontierRelease(_Record):
    """One eligible release on the forecast's frozen current ECI scale.

    Availability is supplied evidence or an explicit proxy, not inferred from
    the fact that the release appears in a current score table.
    """

    release_id: RequiredIdentifier
    lab_id: RequiredIdentifier
    available_at: UTCInstant
    value: Finite
    benchmark_error_sd: Annotated[Finite, Field(ge=0)]
    evidence_refs: tuple[CapabilityEvidenceRef, ...] = Field(min_length=1)
    availability_basis: Literal[
        "documented_public_release", "documented_known_by", "release_date_proxy"
    ]
    rationale: RequiredString


class PublicLabRecords(_Record):
    """One lab's likelihood records and observed as-of frontier."""

    likelihood_release_ids: tuple[RequiredIdentifier, ...]
    as_of_release_id: RequiredIdentifier
    as_of_value: Finite

    @model_validator(mode="after")
    def unique_records(self) -> Self:
        if len(set(self.likelihood_release_ids)) != len(self.likelihood_release_ids):
            raise ValueError("a release can enter the trend likelihood only once")
        return self

    @property
    def observation_count(self) -> int:
        return len(self.likelihood_release_ids)


class PublicTrendPosterior(_Record):
    """Gaussian kernel of the pooled rate before positive-rate conditioning.

    Lab levels have flat priors and are integrated out; they do not set paths.
    The location and variance are not moments of the truncated distribution.
    """

    rate_location: Finite
    rate_variance: Annotated[Finite, Field(ge=0)]
    trend_discrepancy_sd: Annotated[Finite, Field(gt=0)]
    trend_discrepancy_basis: Literal["pooled_reml_estimate", "authored"]
    trend_discrepancy_interval: tuple[Finite, Finite] | None = Field(
        default=None,
        description="90% profile-likelihood interval of an estimated discrepancy.",
    )
    labs: dict[RequiredIdentifier, PublicLabRecords] = Field(min_length=1)
    method: Literal["pooled_gaussian_rate_lab_fixed_levels_positive"] = (
        "pooled_gaussian_rate_lab_fixed_levels_positive"
    )

    @model_validator(mode="after")
    def positive_support(self) -> Self:
        if self.rate_variance == 0 and self.rate_location <= 0:
            raise ValueError("posterior has no positive baseline-rate support")
        if (self.trend_discrepancy_interval is None) != (
            self.trend_discrepancy_basis == "authored"
        ):
            raise ValueError("only an estimated discrepancy carries an interval")
        if self.trend_discrepancy_interval is not None:
            lower, upper = self.trend_discrepancy_interval
            if not 0 <= lower <= self.trend_discrepancy_sd <= upper:
                raise ValueError("discrepancy interval must contain its estimate")
        return self


class PublicFrontierForecast(_Record):
    """Observed lab frontiers, a pooled positive record trend and independent lab paces."""

    schema_version: Literal[1] = 1
    forecast_id: RequiredIdentifier
    as_of: UTCInstant
    frontier_definition: RequiredString = Field(
        description="Eligible public universe and coverage of the named lab roster on one frozen general-ECI scale."
    )
    coordinates: tuple[PublicFrontierCoordinate, ...] = Field(min_length=1)
    trend_window_start: UTCInstant = Field(
        description="Frontier-raising releases available on or after this instant enter the trend likelihood."
    )
    trend_discrepancy_sd: Annotated[Finite, Field(gt=0)] | None = Field(
        default=None,
        description="SD of a frontier-raising release's score around its lab's linear trend, excluding benchmark error; shared by the labs. Omit it to estimate it from the window records by restricted maximum likelihood; an authored value is a sensitivity override.",
    )
    rate_prior: PublicRatePrior = PublicRatePrior(
        mean=DIFFUSE_RATE_PRIOR_MEAN,
        sd=DIFFUSE_RATE_PRIOR_SD,
        rationale="Diffuse default: the frontier-raising releases, not the prior, determine the pooled rate.",
    )
    releases: tuple[PublicFrontierRelease, ...] = Field(min_length=1)
    scale_anchors: tuple[SourceScaleAnchor, ...] = Field(
        default=(),
        description="The models whose fixed scores define the scale, as the score table's provenance records them; questions quote them as the scale's reference.",
    )
    # Left out of a forecast that has none, so a forecast written before the
    # field existed keeps its digest, and with it its trajectories' identities.
    scale_correspondence: SourceScaleCorrespondence | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="What the score table's publisher reports a gain on the scale to have corresponded to in another measure, as the table's provenance records it; questions quote it to say what a point means.",
    )
    # INITIAL_ESTIMATE between wider near-term dispersion and shape 8, the
    # backtest's lowest-loss shape (docs/capability-frontiers.md,
    # OBS-PUBLIC-JOINT).
    pace_shape: Annotated[Finite, Field(gt=0)] = 5.0
    # INITIAL_ESTIMATE: no lab progresses faster than this multiple of the
    # pooled trend rate, about twice the fastest twelve months of any roster
    # lab in the captured record (docs/capability-frontiers.md,
    # OBS-PUBLIC-JOINT). The pace law is truncated at it.
    pace_cap: Annotated[Finite, Field(gt=1)] = 4.0
    pace_rationale: RequiredString
    pooling_rationale: RequiredString
    likelihood_rationale: RequiredString
    limitations: tuple[RequiredString, ...] = Field(min_length=1)
    parameter_origin: Literal[
        "bayesian_pooled_record_trend_with_independent_lab_paces"
    ] = "bayesian_pooled_record_trend_with_independent_lab_paces"
    frontier_scope: Literal["publicly_available_models"] = "publicly_available_models"

    @model_validator(mode="after")
    def coherent_forecast(self) -> Self:
        labs = {row.lab_id for row in self.coordinates}
        if len(labs) != len(self.coordinates):
            raise ValueError("public forecast lab IDs must be unique")
        if len({row.scale_id for row in self.coordinates}) != 1:
            raise ValueError("lab forecasts must share one frozen general-ECI scale")
        if self.trend_window_start > self.as_of:
            raise ValueError("the trend window cannot start after the forecast as-of")
        identities = [(row.lab_id, row.release_id) for row in self.releases]
        if len(set(identities)) != len(identities):
            raise ValueError("each lab's release ID must be unique")
        if not {row.lab_id for row in self.releases} <= labs:
            raise ValueError("releases must name forecast labs")
        if any(row.available_at > self.as_of for row in self.releases):
            raise ValueError("a release cannot follow the forecast as-of")
        if {row.lab_id for row in self.releases} != labs:
            raise ValueError(
                "every forecast lab needs a published release at or before as-of to anchor its path"
            )
        return self


class PublicPathDraw(JointPathDraw):
    baseline_rate: Annotated[Finite, Field(gt=0)]
    pace_multipliers: Vector
    rates: Vector


class PublicFrontierEnsemble(_Record):
    schema_version: Literal[1] = 1
    forecast_sha256: Sha256Digest
    as_of: UTCInstant
    coordinates: tuple[PublicFrontierCoordinate, ...] = Field(min_length=1)
    times: tuple[UTCInstant, ...] = Field(min_length=2)
    draws: tuple[PublicPathDraw, ...] = Field(min_length=1)
    posterior: PublicTrendPosterior
    seed: int = Field(strict=True, ge=0)
    variant: PublicFrontierVariant
    interpolation: Literal["linear"] = "linear"
    source_references: tuple[CapabilityEvidenceRef, ...] = Field(min_length=1)
    assumptions: tuple[RequiredString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_paths(self) -> Self:
        labs = {row.lab_id for row in self.coordinates}
        if len(labs) != len(self.coordinates) or set(self.posterior.labs) != labs:
            raise ValueError("posterior must cover each unique forecast lab")
        if len({row.scale_id for row in self.coordinates}) != 1:
            raise ValueError("lab forecasts must share one frozen general-ECI scale")
        if self.times != forecast_times(self.times[0], self.times[-1]):
            raise ValueError(
                "public forecast times must be the period bounds and every month start between them"
            )
        if self.as_of > self.times[0]:
            raise ValueError("public forecast as-of cannot follow the period start")
        if len({row.draw_id for row in self.draws}) != len(self.draws):
            raise ValueError("public path draw IDs must be unique")
        if not np.isclose(sum(row.weight for row in self.draws), 1, rtol=0, atol=1e-10):
            raise ValueError("public path weights must sum to one")
        elapsed = np.asarray(
            [(at - self.as_of).total_seconds() / YEAR_SECONDS for at in self.times]
        )
        anchors = np.asarray(
            [self.posterior.labs[row.lab_id].as_of_value for row in self.coordinates]
        )
        size = len(self.coordinates)
        for row in self.draws:
            if (
                len(row.values) != len(self.times)
                or any(len(values) != size for values in row.values)
                or any(
                    len(values) != size for values in (row.pace_multipliers, row.rates)
                )
            ):
                raise ValueError("each public draw must contain all times and labs")
            baseline = row.baseline_rate
            pace = np.asarray(row.pace_multipliers)
            if self.variant == "no_progress":
                valid_pace = np.all(pace == 0)
            else:
                valid_pace = np.all(pace > 0)
            if not valid_pace:
                raise ValueError("pace multipliers differ from the declared variant")
            with np.errstate(over="ignore", invalid="ignore"):
                expected_rates = baseline * pace
                expected_values = anchors + elapsed[:, None] * expected_rates
            if not np.isfinite(expected_rates).all() or not np.allclose(
                row.rates, expected_rates, rtol=1e-12, atol=1e-12
            ):
                raise ValueError("public rates must equal the baseline rate times pace")
            if not np.isfinite(expected_values).all() or not np.allclose(
                row.values, expected_values, rtol=1e-12, atol=1e-12
            ):
                raise ValueError(
                    "each lab path must start at its observed as-of frontier and keep one constant ECI rate"
                )
        return self


type QuadratureVariant = Literal["baseline", "no_progress"]

NO_PROGRESS_NODE_ID = "node-00"
"""The no-progress node's ID; it lies outside the Fejér rule, so every study has it."""


def quadrature_node_ids(interior_nodes: int) -> tuple[str, ...]:
    """The no-progress node's ID, then each interior node's, in ascending progress.

    Fejér's second rule with ``n`` nodes puts node ``k`` at the angle
    ``kπ/(n + 1)``, and the reduced fraction ``k/(n + 1)`` names it: node 1 of
    7 and node 2 of 15 are both ``node-1-8``. A node shared by nested rules
    therefore keeps its ID whatever the node count.
    """
    if isinstance(interior_nodes, bool) or interior_nodes < 1:
        raise ValueError("a quadrature rule needs at least one interior node")
    angles = (Fraction(k, interior_nodes + 1) for k in range(1, interior_nodes + 1))
    return (
        NO_PROGRESS_NODE_ID,
        *(f"node-{angle.numerator}-{angle.denominator}" for angle in angles),
    )


class PublicQuadratureNode(_Record):
    """One collected trajectory: a joint draw of every lab's linear path.

    An interior node sits at one of the rule's frontier-progress values and
    carries one representative joint draw from the base forecast's law given
    that progress: the pooled trend-rate draw ``baseline_rate``, each lab's
    independent pace multiplier and its rate, ``baseline_rate`` times the
    pace. The no-progress node holds every lab at its as-of frontier, with no
    rate draw and zero paces, rates and progress.
    """

    draw_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]
    progress: Annotated[Finite, Field(ge=0)]
    baseline_rate: Annotated[Finite, Field(gt=0)] | None
    pace_multipliers: Vector
    rates: Vector
    values: tuple[Vector, ...]


class PublicQuadratureLaw(_Record):
    """One forecast law of frontier progress integrated on the shared nodes.

    ``weights`` is the full rule and ``embedded_weights`` the nested lower-order
    rule on a subset of the same nodes; both are normalized node probabilities.
    """

    label: RequiredIdentifier
    variant: QuadratureVariant
    pace_shape: Annotated[Finite, Field(gt=0)] | None
    pace_cap: Annotated[Finite, Field(gt=1)] | None
    trend_discrepancy_sd: Annotated[Finite, Field(gt=0)]
    weights: tuple[Probability, ...]
    embedded_weights: tuple[Probability, ...]

    @model_validator(mode="after")
    def normalized(self) -> Self:
        if any(
            (value is None) != (self.variant != "baseline")
            for value in (self.pace_shape, self.pace_cap)
        ):
            raise ValueError("only a baseline law has a pace shape and a pace cap")
        for weights in (self.weights, self.embedded_weights):
            if not np.isclose(sum(weights), 1, rtol=0, atol=1e-10):
                raise ValueError("quadrature law weights must sum to one")
        return self


class PublicFrontierQuadrature(_Record):
    """Deterministic trajectory nodes for the public forecast and its alternatives.

    A trajectory's frontier progress P is the public frontier's gain over its
    as-of level at the middle of the period (``progress_elapsed_years``). The
    interior nodes follow Fejér's second rule on a normal-score scale,
    stretched by ``tail_stretch``, of P under an equal mixture of the base law
    and its widest supported pace alternative; ``covering_pace_shapes`` lists
    the mixture's pace shapes, the base's first, each with the base trend
    kernel and the base pace cap, so every node lies inside the support of the
    capped pace. Each interior node carries one representative joint path drawn
    from the base law given its progress, and the no-progress node holds every
    lab at its as-of frontier. Every law, the base included, is integrated on
    the same nodes by its exact density ratio of P to the mixture.
    ``median_progress`` is the base law's median P. Node IDs follow
    ``quadrature_node_ids``.
    """

    schema_version: Literal[1] = 1

    @classmethod
    def from_saved(cls, payload: object) -> Self:
        """The quadrature a saved file's decoded JSON holds, which states the
        current schema version."""
        require_marker(
            payload,
            "schema_version",
            1,
            "public frontier quadrature schema version",
            "plan the study again with the current software",
        )
        return cls.model_validate(payload)

    forecast_sha256: Sha256Digest
    as_of: UTCInstant
    coordinates: tuple[PublicFrontierCoordinate, ...] = Field(min_length=1)
    times: tuple[UTCInstant, ...] = Field(min_length=2)
    posterior: PublicTrendPosterior
    rule: Literal["stretched_fejer_frontier_progress"] = (
        "stretched_fejer_frontier_progress"
    )
    tail_stretch: Annotated[Finite, Field(gt=0)]
    covering_pace_shapes: tuple[Annotated[Finite, Field(gt=0)], ...] = Field(
        min_length=1
    )
    median_progress: Annotated[Finite, Field(gt=0)]
    interior_nodes: int = Field(strict=True, ge=3)
    nodes: tuple[PublicQuadratureNode, ...]
    base: PublicQuadratureLaw
    alternatives: tuple[PublicQuadratureLaw, ...]
    unavailable_alternatives: dict[RequiredIdentifier, RequiredString]
    interpolation: Literal["linear"] = "linear"
    source_references: tuple[CapabilityEvidenceRef, ...] = Field(min_length=1)
    assumptions: tuple[RequiredString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent_rule(self) -> Self:
        if self.interior_nodes % 2 == 0:
            raise ValueError("the nested rule needs an odd interior node count")
        if len(self.nodes) != self.interior_nodes + 1:
            raise ValueError("nodes are the no-progress node and every interior node")
        progress = [node.progress for node in self.nodes]
        if progress[0] != 0 or any(b <= a for a, b in pairwise(progress)):
            raise ValueError("node progress must start at zero and increase")
        if tuple(node.draw_id for node in self.nodes) != quadrature_node_ids(
            self.interior_nodes
        ):
            raise ValueError(
                "node IDs must name the no-progress node and each interior node's Fejér angle"
            )
        labels = [law.label for law in (self.base, *self.alternatives)]
        if len(set(labels) | set(self.unavailable_alternatives)) != len(labels) + len(
            self.unavailable_alternatives
        ):
            raise ValueError("forecast law labels must be unique")
        if self.base.variant != "baseline" or self.base.weights[0] != 0:
            raise ValueError("the base forecast gives the no-progress node no weight")
        if self.covering_pace_shapes[0] != self.base.pace_shape or len(
            set(self.covering_pace_shapes)
        ) != len(self.covering_pace_shapes):
            raise ValueError(
                "the covering law mixes the base pace shape with distinct wider shapes"
            )
        for law in (self.base, *self.alternatives):
            if len(law.weights) != len(self.nodes) or len(law.embedded_weights) != len(
                self.nodes
            ):
                raise ValueError("every law weighs every node")
            if any(
                weight
                for index, weight in enumerate(law.embedded_weights)
                if index > 0 and index % 2 == 1
            ):
                raise ValueError("the embedded rule uses every other interior node")
        if self.times != forecast_times(self.times[0], self.times[-1]):
            raise ValueError(
                "public forecast times must be the period bounds and every month start between them"
            )
        if self.as_of > self.times[0]:
            raise ValueError("public forecast as-of cannot follow the period start")
        if len({row.lab_id for row in self.coordinates}) != len(self.coordinates) or (
            set(self.posterior.labs) != {row.lab_id for row in self.coordinates}
        ):
            raise ValueError("posterior must cover each unique forecast lab")
        elapsed = np.asarray(
            [(at - self.as_of).total_seconds() / YEAR_SECONDS for at in self.times]
        )
        anchors = np.asarray(
            [self.posterior.labs[row.lab_id].as_of_value for row in self.coordinates]
        )
        middle = progress_elapsed_years(self.as_of, self.times)
        cap = self.base.pace_cap
        assert cap is not None  # a baseline law has one
        for index, node in enumerate(self.nodes):
            pace = np.asarray(node.pace_multipliers)
            rates = np.asarray(node.rates)
            if pace.shape != anchors.shape or rates.shape != anchors.shape:
                raise ValueError("each node needs one pace and one rate per lab")
            if index == 0:
                if node.baseline_rate is not None or np.any(pace != 0):
                    raise ValueError(
                        "the no-progress node has no rate draw and zero paces"
                    )
            elif (
                node.baseline_rate is None
                or not np.all(pace > 0)
                or np.any(pace > cap * (1 + 1e-12))
            ):
                raise ValueError(
                    "an interior node has a rate draw and, per lab, a positive "
                    "pace within the base pace cap"
                )
            expected_rates = (
                np.zeros_like(anchors) if index == 0 else node.baseline_rate * pace
            )
            if not np.allclose(rates, expected_rates, rtol=1e-12, atol=1e-12):
                raise ValueError("node rates must equal the rate draw times each pace")
            expected = anchors[None, :] + elapsed[:, None] * rates[None, :]
            if np.shape(node.values) != expected.shape or not np.allclose(
                node.values, expected, rtol=1e-12, atol=1e-12
            ):
                raise ValueError(
                    "each node path must start at the observed as-of frontiers and grow at its lab rates"
                )
            if not np.isclose(
                node.progress,
                frontier_progress(anchors, rates, middle)[0],
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ValueError(
                    "node progress must be its frontier's gain at the period's middle"
                )
        return self

    def trajectory_sha256(self, node: PublicQuadratureNode) -> str:
        """The identity of one node's trajectory, shared by every nested rule.

        It covers what determines the node's path: the forecast, the period,
        the base and covering laws, the stretch and the node itself, with its
        joint draw. The node count, the weights and the alternatives are left out, so
        refining the rule from ``n`` to ``2n + 1`` nodes keeps each shared
        node's identity, and its questions' prompts. ``digest`` identifies the
        whole rule, weights included.
        """
        if node not in self.nodes:
            raise ValueError("the node is not one of this quadrature's nodes")
        return canonical_json_sha256(
            {
                **self.model_dump(
                    mode="json",
                    include={
                        "forecast_sha256",
                        "times",
                        "rule",
                        "tail_stretch",
                        "covering_pace_shapes",
                    },
                ),
                "base": self.base.model_dump(
                    mode="json", exclude={"weights", "embedded_weights"}
                ),
                "node": node.model_dump(mode="json"),
            }
        )


type NonnegativeFinite = Annotated[Finite, Field(ge=0)]
type Fraction01 = Annotated[Finite, Field(ge=0, le=1)]


class PublicBacktestCandidate(_Record):
    """Rolling-origin scores of one dispersion candidate."""

    candidate_id: RequiredIdentifier
    variant: Literal["baseline", "trend_only", "no_progress"]
    pace_shape: Annotated[Finite, Field(gt=0)] | None
    mean_pinball_loss: NonnegativeFinite
    mean_pinball_loss_by_horizon: dict[str, NonnegativeFinite] = Field(min_length=1)
    central_50_coverage: Fraction01
    central_90_coverage: Fraction01
    mean_upper_tail_pinball_loss: NonnegativeFinite = Field(
        description="Mean pinball loss at the upper-tail levels, which the pace shape mainly controls."
    )
    upper_exceedance_rates: dict[str, Fraction01] = Field(
        description="Share of cases whose realized frontier exceeds each upper forecast quantile."
    )
    mean_upper_tail_pinball_loss_by_horizon: dict[str, NonnegativeFinite] = Field(
        min_length=1
    )
    central_90_coverage_by_horizon: dict[str, Fraction01] = Field(min_length=1)
    upper_exceedance_rates_by_horizon: dict[str, dict[str, Fraction01]] = Field(
        min_length=1
    )
    loss_difference_vs_lowest: Finite
    loss_difference_se: NonnegativeFinite = Field(
        description="Paired moving-block bootstrap SE of the mean loss difference from the lowest-loss candidate."
    )

    @model_validator(mode="after")
    def shape_matches_variant(self) -> Self:
        if (self.pace_shape is None) != (self.variant != "baseline"):
            raise ValueError("only a baseline candidate has a pace shape")
        return self


class PublicFrontierBacktest(_Record):
    """Pinball scores of anchored frontier forecasts made at past origins."""

    schema_version: Literal[1] = 1
    forecast_sha256: Sha256Digest
    target: Literal["public_frontier_at_horizon"] = "public_frontier_at_horizon"
    quantile_levels: tuple[float, ...]
    upper_tail_levels: tuple[float, ...]
    horizons_months: tuple[Annotated[int, Field(strict=True, ge=1)], ...] = Field(
        min_length=1
    )
    origins: tuple[UTCInstant, ...] = Field(min_length=1)
    scored_cases: int = Field(strict=True, ge=1)
    pace_cap: Annotated[Finite, Field(gt=1)] = Field(
        description="The forecast's pace cap, at which every pace-shape candidate's pace law is truncated."
    )
    n_draws: int = Field(strict=True, ge=1)
    seed: int = Field(strict=True, ge=0)
    bootstrap_block_origins: int = Field(strict=True, ge=1)
    bootstrap_replicates: int = Field(strict=True, ge=1)
    candidates: tuple[PublicBacktestCandidate, ...] = Field(min_length=2)
    lowest_loss_candidate: RequiredIdentifier
    limitations: tuple[RequiredString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent_backtest(self) -> Self:
        identities = [row.candidate_id for row in self.candidates]
        if len(set(identities)) != len(identities):
            raise ValueError("backtest candidate IDs must be unique")
        if tuple(sorted(set(self.origins))) != self.origins:
            raise ValueError("backtest origins must be unique and ordered")
        best = min(self.candidates, key=lambda row: row.mean_pinball_loss)
        if self.lowest_loss_candidate != best.candidate_id:
            raise ValueError("lowest-loss candidate must have the lowest mean loss")
        if best.loss_difference_vs_lowest != 0 or best.loss_difference_se != 0:
            raise ValueError("the lowest-loss candidate differs from itself by zero")
        return self
