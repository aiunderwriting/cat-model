"""How far an EAL could move if the respondents were sampled again.

A report-only exhibit (``NUM-RESPONDENT-SAMPLING``; methodology.md gives the
estimator and what it covers). In a context-bound setup each state is one
respondent sample, and its sampling group (``JointState.sampling_group_id``) is
its trajectory node or fixed-anchor coordinate and its respondent's model; the
run gives a group's kept samples equal weights. For a change between an arm and
its control on the same samples, y_s = w_alt,s * a_s - w_ctl,s * c_s is
independent and identically distributed within a group, so the change
D = sum_g n_g * mean_g(y) has variance sum_g n_g Var(y), estimated by
sum_g n_g s_g^2(y). With no control this is sum_g W_g^2 s_g^2 / n_g, the
variance of the EAL itself. A group only one arm weights enters as its own
independent group; a group both arms weight must hold the same samples in both.

A state whose model does not answer a leaf reads a donor's answers to it, a
sample of another model at the same node (``POOL-UNANSWERED-SCENARIOS``;
``JointState.borrowed_leaves``). The donor's answers are then in two groups,
so the groups are not independent and the sum over groups understates the
variance. Two donor states with EALs 1 and 3 and two borrowing states that
copy them, all at weight 0.25, hold two independent answers, whose EAL has
standard error 1.0; the sum over the two groups gives 0.7071. No standard
error or z is reported for an EAL or a change in which a state with weight
reads a donor's answers (``sample_groups``, ``standard_error``). The EAL
parameter band and a trajectory node's standard deviation of state EALs count
the same answers as independent, and are withheld on the same test
(``SampleGroups.borrower``, ``borrowed_reason``).

A view that counts the lab's own loss is computed on the samples that give
it (``POOL-EXCLUDED-SAMPLES``). Those views are the lab-loss views and the
gross views, which add the lab's annual business loss
(``contracts.incident.counts_lab_loss``). A state is left out of them when a
case it reads declares the lab's loss unavailable
(``JointState.lab_loss_left_out``). ``SampleGroups.view_weights`` gives an
arm's weights in each view (``ViewWeights``): such a view gives a left-out
state weight zero and splits its sampling group's weight in the arm among the
group's other states in proportion to their weights
(``excluded_samples.lab_loss_weights``), and every other view uses the arm's
weights unchanged. Everything in the results that weighs states in a view
reads its weights there. The standard error of a view that counts the lab's
loss therefore has the samples that are not left out as its groups, at those
weights.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from catmodel.contracts.incident import JointState, counts_lab_loss, gross_view
from catmodel.setup.excluded_samples import (
    gross_withheld_reason,
    lab_loss_weights,
    lacks_lab_loss,
)

NOISE_Z = 2.0
"""|z| below which an effect is reported as within respondent-sampling noise."""

INTERPRETATION = (
    "Respondent-sampling standard error of each EAL and each arm's EAL change "
    "against its control, the reference unless the arm has a matched control. "
    "The kept samples of one sampling group (a trajectory node or fixed-anchor "
    "coordinate and one respondent model) are treated as independent draws "
    "from that model's answers there, and the groups as independent; each "
    "group's weight times its sample mean is the EAL, so its variance is the "
    "sum over groups of the squared weight times the unbiased sample variance "
    "over the group's sample count. A change pairs each sample's weighted EAL "
    "in the two arms. z is the change over its standard error; |z| below 2 is "
    "within respondent-sampling noise. Where a state with weight in the arm "
    "or its control reads another model's answers to a scenario its own model "
    "does not answer (POOL-UNANSWERED-SCENARIOS), the groups are not "
    "independent and no standard error or z is reported. A view that counts "
    "the lab's own loss, a lab-loss view or a gross view, has as its groups "
    "the samples that are not left out of those views (POOL-EXCLUDED-SAMPLES), "
    "at their weights in that view; groups, samples and smallest_group count "
    "every sample the reference arm weighs. It does not cover "
    "the choice or weighting of respondent models, a bias shared by every "
    "sample of a model, or quadrature, simulation or integration error, and "
    "it changes no result."
)

NOT_ANALYTIC_REASON = (
    "A weighted state's EAL in this view is not an analytic mean, so it carries "
    "simulation noise this standard error does not separate."
)

type Terms = Mapping[str, tuple[float, float]]
"""Each positive-weight state's (weight, EAL) in one arm and view."""


def sampling_view(view: str) -> bool:
    """The views with a respondent-sampling standard error: welfare, gross,
    lab loss and the source-status views."""
    return view in {"welfare", "gross"} or view.startswith(
        ("lab_loss::", "source_status::")
    )


def sample_variance(values: list[float]) -> float:
    """The unbiased sample variance of at least two values."""
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


@dataclass(frozen=True)
class ViewWeights:
    """An arm's state weights in each view (``SampleGroups.view_weights``)."""

    arm: Mapping[str, float]
    """The arm's weights, which every view that counts no lab's loss uses."""
    counting_lab_loss: Mapping[str, float] | None
    """The arm's weights in the views that count the lab's own loss, its
    lab-loss views and its gross views (``excluded_samples.lab_loss_weights``):
    ``arm`` itself where no state with weight is left out of them, and None
    where they are unavailable."""
    counting_lab_loss_reason: str | None = None
    """Why the arm has no weights in those views, where it has none."""
    gross_withheld: str | None = None
    """Why the arm's gross views are unavailable at those weights, where
    they are: a state with weight would omit the lab's annual business loss
    (``excluded_samples.gross_withheld_reason``)."""

    @classmethod
    def same(cls, weights: Mapping[str, float]) -> ViewWeights:
        """An arm whose every view uses ``weights``."""
        return cls(weights, weights)

    def of(self, view: str) -> Mapping[str, float] | None:
        """The weights ``view`` is reported at, or None where it is
        unavailable in the arm (``reason``)."""
        if not counts_lab_loss(view):
            return self.arm
        if self.gross_withheld is not None and gross_view(view):
            return None
        return self.counting_lab_loss

    def reason(self, view: str) -> str | None:
        """Why ``view`` has no weights in the arm, or None where it has."""
        if self.of(view) is not None:
            return None
        return self.counting_lab_loss_reason or self.gross_withheld

    @property
    def differ(self) -> bool:
        """Whether the views that count the lab's loss weigh the states
        otherwise than the arm's other views do."""
        return self.counting_lab_loss is not self.arm

    def shared(self, views: Iterable[str]) -> bool:
        """Whether ``views`` are all reported, at one set of weights, so that
        a year's losses in them can be read together."""
        found = [self.of(view) for view in views]
        return all(weights is not None and weights is found[0] for weights in found)


@dataclass(frozen=True)
class SampleGroups:
    """What the results read of a setup's states as respondent samples, each
    under the key the weights and terms give it (``sample_groups``)."""

    group: Mapping[str, str | None]
    """Each state's sampling group; None where it is not a respondent sample."""
    borrowers: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType({})
    """The ID and borrowed leaves of each state that reads a donor's answers
    (``JointState.borrowed_leaves``)."""
    left_out: frozenset[str] = frozenset()
    """The states left out of the views that count the lab's loss
    (``JointState.lab_loss_left_out``)."""
    without_lab_loss: Mapping[str, str] = MappingProxyType({})
    """The ID of each state that holds no law of the lab's loss without being
    left out (``excluded_samples.lacks_lab_loss``)."""

    def borrower(self, states: Iterable[str]) -> tuple[str, tuple[str, ...]] | None:
        """The ID and borrowed leaves of the first of ``states``, by state
        ID, that reads a donor's answers, or None where none does. Every
        diagnostic that counts the states' answers as independent asks here
        (``borrowed_reason``)."""
        found = sorted(
            self.borrowers[state] for state in states if state in self.borrowers
        )
        return found[0] if found else None

    def view_weights(self, weights: Mapping[str, float]) -> ViewWeights:
        """The weights of each view in an arm that weighs the states by
        ``weights``, whichever they are: the setup's, an alternative's own
        or a reweighting's targets."""
        found, reason = lab_loss_weights(weights, self.group, self.left_out)
        return ViewWeights(
            weights,
            found,
            reason,
            None
            if found is None
            else gross_withheld_reason(found, self.without_lab_loss),
        )


def sample_groups(
    states: Iterable[JointState], key: Callable[[str], str] = str
) -> SampleGroups:
    """The sampling groups of a setup's ``states``, its borrowing states,
    those left out of the views that count the lab's loss and those that
    hold no law of it without being left out, each under ``key`` of its ID."""
    states = tuple(states)
    return SampleGroups(
        {key(state.state_id): state.sampling_group_id for state in states},
        {
            key(state.state_id): (state.state_id, state.borrowed_leaves)
            for state in states
            if state.borrowed_leaves
        },
        frozenset(key(state.state_id) for state in states if state.lab_loss_left_out),
        {
            key(state.state_id): state.state_id
            for state in states
            if lacks_lab_loss(state)
        },
    )


NO_STANDARD_ERROR = (
    "the sampling groups are not independent and no standard error is reported."
)
"""What a borrowing state with weight does to a standard error
(``borrowed_reason``)."""


def borrowed_reason(
    state_id: str, leaves: tuple[str, ...], consequence: str = NO_STANDARD_ERROR
) -> str:
    """Why a diagnostic that counts answers as independent is not reported
    where a state reads ``leaves`` from a donor. ``consequence`` says what is
    then not independent and what is not reported: the standard error by
    default, the EAL parameter band (``eal_parameter_band.NO_BAND``) or a
    trajectory node's standard deviation (``sensitivity_results.NO_NODE_SD``)."""
    listed = ", ".join(repr(leaf) for leaf in leaves)
    return (
        f"State {state_id!r} reads a donor's answers to "
        f"{'leaf' if len(leaves) == 1 else 'leaves'} {listed} "
        f"(POOL-UNANSWERED-SCENARIOS), so {consequence}"
    )


def standard_error(
    groups: SampleGroups, alternative: Terms, control: Terms
) -> tuple[float | None, str | None]:
    """The respondent-sampling standard error of ``sum(alternative) -
    sum(control)``, each the sum of its states' weight times EAL, or why it
    is unavailable.

    ``groups`` gives each state's sampling group and the states that read a
    donor's answers. An empty ``control`` gives the standard error of the
    alternative's EAL itself. A borrowing state with weight in either makes
    it unavailable: its donor's answers are in two groups, which the sum over
    groups would count as independent.
    """
    borrowing = groups.borrower(
        state
        for terms in (alternative, control)
        for state, (weight, _) in terms.items()
        if weight > 0
    )
    if borrowing is not None:
        return None, borrowed_reason(*borrowing)
    members: dict[str, tuple[dict[str, tuple[float, float]], ...]] = {}
    for side, terms in enumerate((alternative, control)):
        for state, term in terms.items():
            group = groups.group.get(state)
            if group is None:
                return (
                    None,
                    f"State {state!r} is weighted but is not a respondent sample.",
                )
            members.setdefault(group, ({}, {}))[side][state] = term
    total = []
    for group, (alt, ctl) in sorted(members.items()):
        if alt and ctl and set(alt) != set(ctl):
            return (
                None,
                f"Group {group!r}: the arms weight different samples, so their difference does not pair.",
            )
        for side in (alt, ctl):
            weights = [weight for weight, _ in side.values()]
            if any(
                not math.isclose(weight, weights[0], rel_tol=1e-12)
                for weight in weights
            ):
                return (
                    None,
                    f"Group {group!r}: its samples carry unequal weights, so the EAL is not an equal-weight mean of independent samples.",
                )
        samples = sorted(alt.keys() | ctl.keys())
        if len(samples) < 2:
            return (
                None,
                f"Group {group!r} holds one weighted sample; a sample variance needs at least two.",
            )
        contributions = [
            (alt[state][0] * alt[state][1] if state in alt else 0.0)
            - (ctl[state][0] * ctl[state][1] if state in ctl else 0.0)
            for state in samples
        ]
        total.append(len(samples) * sample_variance(contributions))
    return math.sqrt(math.fsum(total)), None


def _z(difference: float, error: float) -> float | None:
    """The change over its standard error; None at a zero standard error."""
    return difference / error if error > 0 else None


def analytic_eal_terms(draws, weights: Mapping[str, float], view: str) -> Terms | None:
    """Each positive state's weight and analytic EAL, or None if some
    state's EAL in ``view`` is not an analytic mean."""
    terms = {}
    for state, weight in weights.items():
        if weight <= 0:
            continue
        native = draws[state].views[view].native_mean
        if native is None:
            return None
        terms[state] = (weight, native.value)
    return terms


def _eal_reason(metric: Mapping[str, Any] | None) -> str | None:
    """Why a reduced view has no finite EAL, or None."""
    if metric is None or metric.get("status") != "available":
        return "The view is unavailable in this arm."
    if metric.get("expected_annual_loss") is None:
        status = (metric.get("statistics") or {}).get("expected_annual_loss") or {}
        return status.get("reason") or "The view has no finite EAL."
    return None


def respondent_sampling(
    report: Mapping[str, Any],
    draws_by_arm: Mapping[str, Mapping[str, Any]],
    weights_by_arm: Mapping[str, Mapping[str, float]],
    groups: SampleGroups,
    *,
    reference_arm: str,
) -> dict[str, Any]:
    """The ``respondent_sampling`` exhibit of a reduced report: each sampling
    view's reference EAL with its standard error, and each arm's change
    against its control with the change's standard error and z.

    ``draws_by_arm`` and ``weights_by_arm`` are the reduction's own, keyed by
    state, and ``groups`` are the setup's under the same keys
    (``sample_groups``). Each view's terms carry its own weights in the arm
    (``SampleGroups.view_weights``), so the standard error of a view that
    counts the lab's loss is that of the samples it holds."""
    if not any(groups.group.values()):
        return {
            "register_id": "NUM-RESPONDENT-SAMPLING",
            "status": "unavailable",
            "reason": "No state of the setup is a respondent sample.",
        }
    sizes: dict[str, int] = {}
    for state, weight in weights_by_arm[reference_arm].items():
        if weight > 0 and (group := groups.group.get(state)) is not None:
            sizes[group] = sizes.get(group, 0) + 1
    metrics = {arm: row.get("metrics", {}) for arm, row in report["arms"].items()}
    views = [view for view in metrics[reference_arm] if sampling_view(view)]
    view_weights = {
        arm: groups.view_weights(weights) for arm, weights in weights_by_arm.items()
    }

    def terms(arm: str, view: str) -> tuple[Terms | None, str | None]:
        reason = _eal_reason(metrics.get(arm, {}).get(view))
        if reason is not None:
            return None, reason
        weights = view_weights[arm].of(view)
        if weights is None:
            return None, view_weights[arm].reason(view)
        found = analytic_eal_terms(draws_by_arm[arm], weights, view)
        return (found, None) if found is not None else (None, NOT_ANALYTIC_REASON)

    levels = {}
    for view in views:
        found, reason = terms(reference_arm, view)
        error = None
        if found is not None:
            error, reason = standard_error(groups, found, {})
        if found is None or error is None:
            levels[view] = {"status": "unavailable", "reason": reason}
            continue
        eal = math.fsum(weight * value for weight, value in found.values())
        levels[view] = {
            "status": "available",
            "expected_annual_loss": eal,
            "sampling_standard_error": error,
            "relative_sampling_standard_error": error / eal if eal else None,
        }
    comparisons: dict[str, dict[str, Any]] = {}
    for arm, control in sorted(report["comparison_reference_arms"].items()):
        if arm == control:
            continue
        rows = comparisons[arm] = {}
        for view in views:
            alternative, reason = terms(arm, view)
            baseline, control_reason = terms(control, view)
            error = None
            if alternative is not None and baseline is not None:
                error, reason = standard_error(groups, alternative, baseline)
            if alternative is None or baseline is None or error is None:
                rows[view] = {
                    "status": "unavailable",
                    "control_arm": control,
                    "reason": reason or control_reason,
                }
                continue
            after = math.fsum(weight * value for weight, value in alternative.values())
            before = math.fsum(weight * value for weight, value in baseline.values())
            rows[view] = {
                "status": "available",
                "control_arm": control,
                "control_eal": before,
                "alternative_eal": after,
                "absolute_difference": after - before,
                "sampling_standard_error": error,
                "sampling_z": _z(after - before, error),
            }
    return {
        "register_id": "NUM-RESPONDENT-SAMPLING",
        "status": "available",
        "groups": len(sizes),
        "samples": sum(sizes.values()),
        "smallest_group": min(sizes.values(), default=0),
        "noise_z": NOISE_Z,
        "interpretation": INTERPRETATION,
        "views": levels,
        "comparisons": comparisons,
    }


def row_sampling(
    exhibit: Mapping[str, Any], arm: str, control: str, view: str, difference: float
) -> dict[str, Any]:
    """A tornado row's sampling fields: the standard error of its change from
    ``exhibit``, its z, and the reason where they are unavailable."""
    row = (exhibit.get("comparisons") or {}).get(arm, {}).get(view)
    if exhibit.get("status") != "available":
        reason = exhibit.get("reason")
    elif row is None or row.get("control_arm") != control:
        reason = "No respondent-sampling standard error is reported for this change."
    elif row["status"] != "available":
        reason = row["reason"]
    else:
        error = row["sampling_standard_error"]
        return {
            "sampling_standard_error": error,
            "sampling_z": _z(difference, error),
            "sampling_reason": None,
        }
    return {
        "sampling_standard_error": None,
        "sampling_z": None,
        "sampling_reason": reason,
    }


def quadrature_sampling(
    groups: SampleGroups, full: Terms, embedded: Terms, difference: float
) -> dict[str, Any]:
    """The trajectory quadrature indicator's sampling fields: the standard
    error of the full rule's EAL minus the embedded rule's on the same states,
    its z, and the reason where they are unavailable."""
    error, reason = standard_error(groups, full, embedded)
    return {
        "sampling_standard_error": error,
        "sampling_z": None if error is None else _z(difference, error),
        "sampling_reason": reason,
    }
