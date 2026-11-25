"""Plan, collect and analyse the conditioning pilot: do respondents' answers move with the capability trajectory, and with what in it?

A public study asks every question once per trajectory node, and no
collection has measured whether the answers respond to a node's path
(docs/limitations.md, item 6). The pilot asks a few scenarios on a small node
rule in three arms:

- ``reference``: each question as ``capability-forecast public-study`` plans
  it.
- ``fixed_scale``: every path-linked deployment scale at its median-path
  level on every node. A node's scales move with its progress, so the
  reference arm confounds the capability path with stated use; in this arm a
  question differs between nodes only in its path. Every other word is the
  reference question's, including the sentence that describes the scales as
  linked to the trajectory, so the two arms differ only in the stated numbers.
- ``placebo``: the trajectory table's lab columns in reverse order. The labs,
  dates and values are the same, so the change carries no information and
  should not move an answer beyond resampling noise.

``plan`` builds the templates, the forecast and every arm's instruments
without calling a provider. ``collect`` requires exactly one of ``--dry-run``,
which freezes every question, and ``--live``, which asks the respondent
models; it resumes an interrupted collection. ``analyse`` reads the captures
and writes ``pilot.md`` and ``pilot.csv``. At the default three-node rule and
the ``example`` profile a pilot asks 720 initial questions per scenario.
Runbook section 2, "Conditioning pilot", gives the cost and how to read the
output.

    uv run --frozen python scripts/conditioning_pilot.py plan ../pilot --source-bundle ../hfl-dry-run/source-bundle --context config/studies/hypothetical-frontier-lab/context.json --scenario fraud_individuals --scenario output_reliance_software
    uv run --frozen python scripts/conditioning_pilot.py collect ../pilot --dry-run
    uv run --frozen python scripts/conditioning_pilot.py collect ../pilot --live
    uv run --frozen python scripts/conditioning_pilot.py analyse ../pilot --output ../pilot-analysis
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from catmodel.linear_algebra_threads import use_one_linear_algebra_thread

# Before NumPy loads: planning fits the forecast and the analysis replays
# captures, so this process takes the one linear-algebra thread every catmodel
# command runs with.
use_one_linear_algebra_thread()

from catmodel.artifacts import freeze_json  # noqa: E402
from catmodel.collection.contextual_collection import run_collections  # noqa: E402
from catmodel.config import load_model_profile  # noqa: E402
from catmodel.contracts.contextual_risk import (  # noqa: E402
    ContextualRiskCase,
    ContextualRiskInstrument,
    render_exposure,
)
from catmodel.contracts.elicitation import Respondent  # noqa: E402
from catmodel.contracts.public_frontier import (  # noqa: E402
    PublicFrontierForecast,
    PublicFrontierQuadrature,
)
from catmodel.elicitation.epistemic import epistemic_mean  # noqa: E402
from catmodel.json_io import strict_json_loads  # noqa: E402
from catmodel.output_paths import require_new_evidence_output  # noqa: E402
from catmodel.resources import data_path  # noqa: E402
from catmodel.results.respondent_sampling import NOISE_Z, sample_variance  # noqa: E402
from catmodel.setup.contextual_inputs import (  # noqa: E402
    contextual_requests,
    load_contextual_collection,
)
from catmodel.setup.public_study import (  # noqa: E402
    prepare_public_forecast_file,
    public_study_templates,
    write_public_frontier_study,
)

ARMS: dict[str, str] = {
    "reference": "Each question as the public study plans it.",
    "fixed_scale": (
        "Every path-linked deployment scale at its median-path level on every "
        "node, so between nodes a question differs only in its capability "
        "path. Every other word is the reference question's, including the "
        "sentence that describes the scales as linked to the trajectory."
    ),
    "placebo": (
        "The trajectory table's lab columns in reverse order: the same labs, "
        "dates and values, so the change carries no information."
    ),
}
"""Each arm and what its questions change."""

NOT_ASSEMBLED = (
    "Not assembled: this plan renders the conditioning pilot's questions. Its "
    "answers are analysed per collection and never imported into a setup."
)
"""What the pilot's study plan records for each of the three assembly
rationales of a study that ``setup import`` assembles."""

QUANTITIES = ("direct_count", "direct_eal")
"""The answers the analysis reads, each as the log of its epistemic mean: the
direct question's expected annual count and its expected annual societal
harm. Any other quantile answer of a case is one more name here and one more
branch in ``_observation``: the chain count at its stipulated base
(``case.chain_count``), the loss of one qualified unit or a large-loss count.
The implied EAL also needs ``catmodel.setup.implied_eal.implied_eal``."""

EXCLUSIONS = ("unavailable", "exhausted", "no_log")
"""Why an answer does not enter a quantity: the respondent declared it
unavailable; its question exhausted its validation attempts; or its epistemic
mean is zero or not finite and has no log."""

RESOLVED_CHANGE = 1.25
"""The change a ``flat`` verdict must have been able to see: a standard error
of at most ``log(RESOLVED_CHANGE) / NOISE_Z``, at which a change by this
factor reaches ``NOISE_Z``."""

_NEGLIGIBLE = 1e-9
"""Log units below which an estimate or a standard error counts as zero."""

RULES = (
    "Every statistic is computed on the log of an answer's epistemic mean "
    "and shown with its standard error and its z, the statistic over its "
    "standard error; a percentage restates a log change, with its standard "
    "error to first order. A standard error treats the answers of one arm, "
    "trajectory node and respondent model as independent draws from that "
    "model's answers there, and the groups as independent, as the "
    "respondent-sampling error of a result does (NUM-RESPONDENT-SAMPLING). "
    "A slope is the mean over respondent models of each model's "
    "least-squares slope of its node means on frontier progress, the public "
    "frontier's gain over its as-of level by the middle of the study year, "
    "across every node of the plan, the no-progress node included. `Per 10 "
    "ECI points` restates the reference arm's slope as the percentage change "
    "in an answer per 10 ECI points of progress, `Scales fixed` the fixed-scale "
    "arm's, and `Stated scale` their difference, the part that follows the "
    "stated deployment scales. `Fastest over slowest` is the reference arm's "
    "log ratio between its fastest and its slowest node, to read beside "
    "`Noise floor`, the standard deviation of the log answer between one "
    "model's samples on one node. `Never falls` is the share of respondents, "
    "each a model's sample index with an answer on every node, whose "
    "reference answers never fall as progress rises. `Placebo` is the "
    "placebo arm's mean log ratio to the reference arm over the nodes and "
    "models both answered; zero is a ratio of one.\n\n"
    "The verdicts are screens chosen as reading aids, not "
    f"tests of validity. `responds`: the reference slope has z of at least "
    f"{NOISE_Z:g}, so answers rise with capability, the expected sign. "
    f"`responds against the expected sign`: z of at most -{NOISE_Z:g}. "
    f"`flat`: |z| below {NOISE_Z:g} and the fitted change over the plan's "
    "range of progress has a standard error of at most "
    f"log({RESOLVED_CHANGE:g}) / {NOISE_Z:g}, so a change by a factor of "
    f"{RESOLVED_CHANGE:g} would have reached z = {NOISE_Z:g}. `noisy`: |z| "
    f"below {NOISE_Z:g} with a larger standard error, so the pilot cannot "
    "tell. `not estimable`: a group holds fewer than two answers, or fewer "
    "than two nodes have answers. `with scales fixed` gives the fixed-scale "
    "arm the same screen. `moves with stated scale`: the reference slope "
    f"less the fixed-scale slope has |z| of at least {NOISE_Z:g}. "
    "`placebo-sensitive`: the placebo log ratio has |z| of at least "
    f"{NOISE_Z:g}; `placebo noisy`: its standard error is too large to tell, "
    "by the rule for `flat`. The screens are not adjusted for the number of "
    "scenarios and quantities screened; a few samples give an imprecise "
    "standard deviation, so z is not normal; and a response that is not "
    "monotone in progress can leave a slope near zero, which the node means "
    "in pilot.csv show."
)
"""How to read ``pilot.md`` and the stated rule behind every verdict."""

FIELDS = (
    *("scenario", "source_status", "quantity", "arm", "statistic", "node"),
    *("value", "standard_error", "z", "answers", "note"),
)
"""The columns of ``pilot.csv``, one statistic per row. ``value`` and
``standard_error`` are in log units except for a share or a count, and
``answers`` is the number of answers behind the row."""

type Key = tuple[str, str, str]
"""A scenario, source status and quantity."""
type Group = tuple[str, str, str]
"""A sampling group: an arm, trajectory node and respondent model."""


# Planning


def _instrument(path: Path) -> ContextualRiskInstrument:
    return ContextualRiskInstrument.from_saved(strict_json_loads(path.read_bytes()))


def _changed(
    instrument: ContextualRiskInstrument, arm: str, **changes: object
) -> ContextualRiskInstrument:
    """``instrument`` with ``changes``, validated again under the arm's own
    context identifier, which no question shows."""
    return ContextualRiskInstrument.model_validate(
        {
            **instrument.model_dump(mode="json"),
            **changes,
            "context_id": f"{arm}-{instrument.context_id}",
        }
    )


def fixed_scale_instrument(
    instrument: ContextualRiskInstrument,
) -> ContextualRiskInstrument | None:
    """``instrument`` with every path-linked scale rendered on the median path
    (progress ratio 1), whatever its node's progress; None when its scales
    are not path-linked."""
    exposure = instrument.path_linked_exposure
    if exposure is None or exposure.use_population_template is None:
        return None
    median = exposure.model_copy(update={"progress_ratio": 1.0})
    return _changed(
        instrument,
        "fixed_scale",
        path_linked_exposure=median.model_dump(mode="json"),
        deployment_binding={
            **instrument.deployment_binding.model_dump(mode="json"),
            "use_population": render_exposure(exposure.use_population_template, median),
        },
    )


def placebo_instrument(
    instrument: ContextualRiskInstrument,
) -> ContextualRiskInstrument | None:
    """``instrument`` with its trajectory's lab columns in reverse order: the
    labs, their paths and their observed scores permuted together; None when
    the trajectory names one lab."""
    condition = instrument.public_trajectory
    if condition is None or len(condition.coordinates) < 2:
        return None
    reordered = condition.model_copy(
        update={
            "coordinates": condition.coordinates[::-1],
            "values": tuple(row[::-1] for row in condition.values),
            "observed_values": condition.observed_values[::-1],
        }
    )
    return _changed(
        instrument, "placebo", public_trajectory=reordered.model_dump(mode="json")
    )


_CHANGED_ARMS = {
    "fixed_scale": (
        fixed_scale_instrument,
        "the context's deployment scales are not path-linked",
    ),
    "placebo": (placebo_instrument, "the trajectory names one lab"),
}
"""How each changed arm's instrument is made, and why a study has none."""


def plan_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Write the new pilot directory ``args.output`` and return its
    ``pilot.json``.

    ``templates/`` holds the context's two source-status templates for the
    named scenarios, ``study/`` the public study planned from them,
    ``arms/<arm>/instruments/`` the changed arms' instruments (the reference
    arm's are the study's own) and ``pilot.json`` every (arm, node, source
    status) with its instrument and its collection directory,
    ``arms/<arm>/collections/<node>-<status>``. The respondents are every
    model of the profile, at ``--samples`` or the profile's samples per
    question. An arm that would change nothing in this study is left out and
    named under ``skipped_arms``.
    """
    output = args.output.expanduser()
    require_new_evidence_output(output)
    context = strict_json_loads(args.context.read_bytes())
    if not isinstance(context, dict):
        raise ValueError("context must contain a JSON object")
    profile = load_model_profile(profile=args.profile, path=args.profiles)
    samples = args.samples or profile.settings.samples_per_question
    respondents = tuple(
        Respondent(
            respondent_id=f"{spec.model_id}:{sample}",
            model_id=spec.model_id,
            sample_index=sample,
        )
        for spec in profile.models
        for sample in range(samples)
    )
    bundle = args.source_bundle.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".conditioning-pilot-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "pilot"
        (stage / "templates").mkdir(parents=True)
        forecast = args.forecast
        if forecast is None:
            forecast = stage / "public-forecast.json"
            prepare_public_forecast_file(
                source_bundle=bundle,
                template=data_path(
                    "capability", "public-frontier-forecast-template.json"
                ),
                output=forecast,
            )
        templates = {}
        for status, template in public_study_templates(
            context,
            forecast=PublicFrontierForecast.model_validate(
                strict_json_loads(forecast.read_bytes())
            ),
            source_bundle=bundle,
            respondents=respondents,
            scenario_ids=tuple(args.scenarios),
        ).items():
            if template.parents:
                raise ValueError(
                    "name the pilot's scenarios; the complete catalog adds every "
                    "taxonomy-parent question"
                )
            templates[status] = stage / "templates" / f"{status}.json"
            freeze_json(templates[status], template.model_dump(mode="json"))
        rationales = (
            "exposure_rationale",
            "residual_independence_rationale",
            "parameter_coupling_rationale",
        )
        freeze_json(
            stage / "assembly-assumptions.json",
            dict.fromkeys(rationales, NOT_ASSEMBLED),
        )
        write_public_frontier_study(
            forecast_path=forecast,
            public_instrument=templates["public"],
            not_yet_public_instrument=templates["not_yet_public"],
            assumptions_path=stage / "assembly-assumptions.json",
            output=stage / "study",
            nodes=args.nodes,
        )
        study = strict_json_loads((stage / "study" / "public-study.json").read_bytes())
        quadrature = PublicFrontierQuadrature.from_saved(
            strict_json_loads((stage / "study" / study["quadrature"]).read_bytes())
        )
        contexts: list[dict[str, str]] = []
        skipped: dict[str, str] = {}
        calls = 0
        for row in study["contexts"]:
            identifier = f"{row['draw_id']}-{row['source_model_status']}"
            reference = _instrument(stage / "study" / row["instrument"])
            for arm in ARMS:
                instrument, location = reference, f"study/{row['instrument']}"
                if arm != "reference":
                    change, reason = _CHANGED_ARMS[arm]
                    changed = change(reference)
                    if changed is None:
                        skipped[arm] = reason
                        continue
                    instrument = changed
                    location = f"arms/{arm}/instruments/{identifier}.json"
                    (stage / location).parent.mkdir(parents=True, exist_ok=True)
                    freeze_json(stage / location, instrument.model_dump(mode="json"))
                calls += len(contextual_requests(instrument))
                contexts.append(
                    {
                        "arm": arm,
                        "draw_id": row["draw_id"],
                        "source_model_status": row["source_model_status"],
                        "instrument": location,
                        "capture": f"arms/{arm}/collections/{identifier}",
                    }
                )
        pilot = {
            "schema_version": 1,
            "source_bundle": str(bundle),
            "profile": args.profile,
            "profiles": args.profiles and str(args.profiles.expanduser().resolve()),
            "models": [spec.model_id for spec in profile.models],
            "samples_per_model": samples,
            "scenarios": list(args.scenarios),
            "nodes": {node.draw_id: node.progress for node in quadrature.nodes},
            "arms": [arm for arm in ARMS if arm not in skipped],
            "skipped_arms": skipped,
            "contexts": contexts,
            "initial_calls": calls,
        }
        freeze_json(stage / "pilot.json", pilot)
        require_new_evidence_output(output)
        stage.rename(output)
    return pilot


# Collection


def read_pilot(directory: Path) -> dict[str, Any]:
    """A pilot directory's ``pilot.json``."""
    path = directory / "pilot.json"
    if not path.is_file():
        raise ValueError(f"{directory} holds no pilot.json; plan the pilot first")
    pilot = strict_json_loads(path.read_bytes())
    if not isinstance(pilot, dict) or pilot.get("schema_version") != 1:
        raise ValueError(f"{path} is not a conditioning pilot plan")
    return pilot


def collect_pilot(args: argparse.Namespace) -> int:
    """Freeze every question of the pilot's contexts, or of ``--arm`` only,
    and unless ``--dry-run`` ask the respondent models, resuming what an
    earlier call collected. Return the number of contexts."""
    directory = args.pilot.expanduser()
    pilot = read_pilot(directory)
    arms = args.arms or pilot["arms"]
    if set(arms) - set(pilot["arms"]):
        raise ValueError(f"the pilot's arms are {pilot['arms']}")
    contexts = [
        (_instrument(directory / row["instrument"]), directory / row["capture"])
        for row in pilot["contexts"]
        if row["arm"] in arms
    ]
    run_collections(
        contexts,
        source_bundle=Path(pilot["source_bundle"]),
        profile=pilot["profile"],
        profiles=pilot["profiles"] and Path(pilot["profiles"]),
        dry_run=args.dry_run,
        answers=args.answers,
    )
    return len(contexts)


# Analysis


@dataclass(frozen=True)
class Estimate:
    """A weighted sum of sampling groups' mean log answers with its
    respondent-sampling standard error, None when a group holds one answer."""

    value: float
    error: float | None
    answers: int

    @property
    def z(self) -> float | None:
        """The estimate over its standard error; at a zero standard error,
        zero for a zero estimate and otherwise infinite."""
        if self.error is None:
            return None
        if self.error > _NEGLIGIBLE:
            return self.value / self.error
        if abs(self.value) <= _NEGLIGIBLE:
            return 0.0
        return math.copysign(math.inf, self.value)


def combine(
    weights: Mapping[Group, float], groups: Mapping[Group, Sequence[float]]
) -> Estimate | None:
    """``sum(weight * mean)`` over independent sampling groups. Its variance
    is ``sum(weight^2 * s^2 / n)`` with each group's unbiased sample variance
    (NUM-RESPONDENT-SAMPLING); None without a weight."""
    if not weights:
        return None
    terms = [(weight, list(groups[group])) for group, weight in weights.items()]
    value = math.fsum(
        weight * math.fsum(values) / len(values) for weight, values in terms
    )
    error = None
    if all(len(values) > 1 for _, values in terms):
        error = math.sqrt(
            math.fsum(
                weight**2 * sample_variance(values) / len(values)
                for weight, values in terms
            )
        )
    return Estimate(value, error, sum(len(values) for _, values in terms))


def slope_weights(
    groups: Mapping[Group, Sequence[float]], arm: str, progress: Mapping[str, float]
) -> dict[Group, float]:
    """The mean over respondent models of each model's least-squares slope of
    its node means on ``progress``, as weights on the arm's sampling groups.
    A model with answers on fewer than two nodes has no slope."""
    slopes = []
    for model in sorted({model for a, _, model in groups if a == arm}):
        nodes = [node for node in progress if (arm, node, model) in groups]
        if len(nodes) < 2:
            continue
        centre = math.fsum(progress[node] for node in nodes) / len(nodes)
        spread = math.fsum((progress[node] - centre) ** 2 for node in nodes)
        slopes.append(
            {(arm, node, model): (progress[node] - centre) / spread for node in nodes}
        )
    return {
        group: weight / len(slopes)
        for slope in slopes
        for group, weight in slope.items()
    }


def paired_weights(
    groups: Mapping[Group, Sequence[float]], pairs: Iterable[tuple[Group, Group]]
) -> dict[Group, float]:
    """The mean, over the pairs whose two groups both hold answers, of the
    first group's mean less the second's."""
    held = [pair for pair in pairs if pair[0] in groups and pair[1] in groups]
    return {
        group: sign / len(held)
        for pair in held
        for sign, group in zip((1.0, -1.0), pair)
    }


def _less(
    weights: Mapping[Group, float], control: Mapping[Group, float]
) -> dict[Group, float]:
    """``weights`` less ``control``, on other groups; nothing unless both
    exist."""
    if not weights or not control:
        return {}
    return {**weights, **{group: -weight for group, weight in control.items()}}


def noise_floor(groups: Iterable[Sequence[float]]) -> float | None:
    """The pooled standard deviation of log answers within a sampling group:
    how far one sample of an answer lies from another."""
    repeated = [list(values) for values in groups if len(values) > 1]
    freedom = sum(len(values) - 1 for values in repeated)
    if not freedom:
        return None
    return math.sqrt(
        math.fsum((len(values) - 1) * sample_variance(values) for values in repeated)
        / freedom
    )


def never_falls(
    answers: Mapping[Group, Mapping[int, float]], arm: str, nodes: Sequence[str]
) -> tuple[float, int] | None:
    """The share of the arm's respondents, each a model's sample index with
    an answer on every node, whose answers never fall as progress rises, and
    their number. Samples are separate calls, so a sample index pairs the
    nodes only as a study's assembly pairs them."""
    profiles: dict[tuple[str, int], dict[str, float]] = {}
    for (group_arm, node, model), samples in answers.items():
        if group_arm == arm:
            for sample, value in samples.items():
                profiles.setdefault((model, sample), {})[node] = value
    complete = [
        [profile[node] for node in nodes]
        for profile in profiles.values()
        if len(profile) == len(nodes)
    ]
    if not complete:
        return None
    rising = sum(all(b >= a for a, b in pairwise(values)) for values in complete)
    return rising / len(complete), len(complete)


def _observation(case: ContextualRiskCase, quantity: str) -> float | str:
    """The log of a case's epistemic mean of ``quantity``, or why it has none
    (``EXCLUSIONS``). The answer is the one first given, before any retry or
    cross-check showed the respondent its own figures."""
    answer = case.answer
    if answer.status != "available":
        return "unavailable"
    law = answer.rate if quantity == "direct_count" else answer.direct_eal
    assert law is not None
    try:
        mean = epistemic_mean(law)
    except ValueError:
        return "no_log"
    return math.log(mean) if mean > 0 else "no_log"


def read_answers(
    directory: Path, pilot: Mapping[str, Any]
) -> tuple[
    dict[Key, dict[Group, dict[int, float]]],
    dict[tuple[Key, str], Counter[str]],
    list[str],
]:
    """Every collected arm's log answers by key, sampling group and sample
    index; the number of answers left out by key, arm and reason
    (``EXCLUSIONS``); and the arms not collected. Each capture is replayed
    and checked against the instrument the pilot planned for it."""
    answers: dict[Key, dict[Group, dict[int, float]]] = {}
    excluded: dict[tuple[Key, str], Counter[str]] = {}
    missing: list[str] = []
    for arm in pilot["arms"]:
        rows = [row for row in pilot["contexts"] if row["arm"] == arm]
        collected = [
            (directory / row["capture"] / "contextual-risk.json").is_file()
            for row in rows
        ]
        if not all(collected):
            if any(collected):
                raise ValueError(
                    f"arm {arm}: {collected.count(False)} of {len(rows)} "
                    "collections are incomplete; finish collecting the arm first"
                )
            missing.append(arm)
            continue
        for row in rows:
            capture = directory / row["capture"]
            instrument, artifact, _ = load_contextual_collection(capture)
            planned = _instrument(directory / row["instrument"])
            if instrument.digest() != planned.digest():
                raise ValueError(f"{capture} holds another instrument's collection")
            # A direct question that exhausted its attempts leaves no case.
            for case in (*artifact.cases, *artifact.unanswered_cases):
                group = (arm, row["draw_id"], case.respondent.model_id)
                for quantity in QUANTITIES:
                    key = (case.node_id, row["source_model_status"], quantity)
                    observation = (
                        _observation(case, quantity)
                        if isinstance(case, ContextualRiskCase)
                        else "exhausted"
                    )
                    if isinstance(observation, str):
                        excluded.setdefault((key, arm), Counter())[observation] += 1
                    else:
                        answers.setdefault(key, {}).setdefault(group, {})[
                            case.respondent.sample_index
                        ] = observation
    return answers, excluded, missing


def _screen(slope: Estimate | None, span: float) -> str:
    """One arm's dose-response screen from its slope and the plan's range of
    progress, ``span`` (``RULES``)."""
    z = None if slope is None else slope.z
    if slope is None or slope.error is None or z is None:
        return "not estimable"
    if z >= NOISE_Z:
        return "responds"
    if z <= -NOISE_Z:
        return "responds against the expected sign"
    resolved = slope.error * span <= math.log(RESOLVED_CHANGE) / NOISE_Z
    return "flat" if resolved else "noisy"


def verdict(
    reference: Estimate | None,
    fixed: Estimate | None,
    scale: Estimate | None,
    placebo: Estimate | None,
    span: float,
) -> str:
    """A key's verdict from its two slopes, their difference and its placebo
    log ratio (``RULES``)."""
    parts = [_screen(reference, span)]
    if fixed is not None:
        parts.append(f"with scales fixed: {_screen(fixed, span)}")
    z = None if scale is None else scale.z
    if z is not None and abs(z) >= NOISE_Z:
        parts.append("moves with stated scale")
    z = None if placebo is None else placebo.z
    if placebo is not None and placebo.error is not None and z is not None:
        if abs(z) >= NOISE_Z:
            parts.append("placebo-sensitive")
        elif placebo.error > math.log(RESOLVED_CHANGE) / NOISE_Z:
            parts.append("placebo noisy")
    return "; ".join(parts)


def analyse_pilot(directory: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The pilot's statistics, one row of ``FIELDS`` each, and the arms not
    collected."""
    pilot = read_pilot(directory)
    # Slopes are per 10 ECI points of frontier progress.
    progress = {node: value / 10 for node, value in pilot["nodes"].items()}
    nodes = sorted(progress, key=lambda node: progress[node])
    slowest, fastest = nodes[0], nodes[-1]
    span = progress[fastest] - progress[slowest]
    answers, excluded, missing = read_answers(directory, pilot)
    rows: list[dict[str, Any]] = []
    for key in sorted({*answers, *(key for key, _ in excluded)}):
        samples = answers.get(key, {})
        groups = {group: list(values.values()) for group, values in samples.items()}
        models = sorted({model for _, _, model in groups})

        def add(arm, statistic, estimate=None, key=key, **fields) -> None:
            if estimate is None and not fields:
                return
            row: dict[str, Any] = dict.fromkeys(FIELDS)
            row.update(zip(FIELDS, key), arm=arm, statistic=statistic)
            if estimate is not None:
                row.update(
                    value=estimate.value,
                    standard_error=estimate.error,
                    z=estimate.z,
                    answers=estimate.answers,
                )
            rows.append({**row, **fields})

        slopes, ranges = {}, {}
        for arm in pilot["arms"]:
            own = [values for group, values in groups.items() if group[0] == arm]
            left_out = excluded.get((key, arm), Counter())
            if not own and not left_out:
                continue
            slopes[arm] = slope_weights(groups, arm, progress)
            ranges[arm] = paired_weights(
                groups, (((arm, fastest, m), (arm, slowest, m)) for m in models)
            )
            add(arm, "log_slope_per_10_eci", combine(slopes[arm], groups))
            add(arm, "fastest_to_slowest_log_ratio", combine(ranges[arm], groups))
            floor = noise_floor(own)
            if floor is not None:
                add(arm, "noise_floor_log_sd", value=floor)
            share = never_falls(samples, arm, nodes)
            if share is not None:
                add(arm, "never_falls_share", value=share[0], answers=share[1])
            for node in nodes:
                held = [group for group in groups if group[:2] == (arm, node)]
                if held:
                    mean = combine(dict.fromkeys(held, 1 / len(held)), groups)
                    add(arm, "node_mean_log", mean, node=node, z=None)
            add(arm, "answers_used", value=sum(map(len, own)))
            for reason in EXCLUSIONS:
                if left_out[reason]:
                    add(arm, f"excluded_{reason}", value=left_out[reason])
        scale = {
            statistic: combine(
                _less(weights.get("reference", {}), weights.get("fixed_scale", {})),
                groups,
            )
            for statistic, weights in (
                ("log_slope_per_10_eci", slopes),
                ("fastest_to_slowest_log_ratio", ranges),
            )
        }
        for statistic, estimate in scale.items():
            add("reference_minus_fixed_scale", statistic, estimate)
        placebo = combine(
            paired_weights(
                groups,
                (
                    (("placebo", node, model), ("reference", node, model))
                    for node in nodes
                    for model in models
                ),
            ),
            groups,
        )
        add("placebo_minus_reference", "mean_log_ratio", placebo)
        if "reference" in slopes:
            note = verdict(
                combine(slopes["reference"], groups),
                combine(slopes.get("fixed_scale", {}), groups),
                scale["log_slope_per_10_eci"],
                placebo,
                span,
            )
            add("", "verdict", note=note)
    return rows, missing


# Output


def _figure(row: Mapping[str, Any] | None, *, percent: bool = False) -> str:
    """A row's log change, or with ``percent`` its percentage change, with
    its standard error and its z."""
    if row is None:
        return "n/a"
    value, error, z = row["value"], row["standard_error"], row["z"]
    digits = 1 if percent else 2
    # A percentage change and, to first order in the log's, its standard error.
    change = 100 * math.expm1(min(value, 700)) if percent else value
    unit = 100 * math.exp(min(value, 700)) if percent else 1.0
    sign = "%" if percent else ""
    # Adding zero turns a rounded negative zero into zero.
    figure = f"{round(change, digits) + 0.0:+.{digits}f}{sign}"
    if error is None or z is None:
        return f"{figure} (no standard error)"
    z_text = f"{round(z, 1) + 0.0:.1f}" if math.isfinite(z) else "∞" if z > 0 else "-∞"
    return f"{figure} ± {unit * error:.{digits}f}{sign} (z {z_text})"


def render_markdown(
    pilot: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], missing: Sequence[str]
) -> str:
    """``pilot.md``: what was asked, how to read it, one line per scenario,
    source status and quantity, and the answers used and left out."""
    index = {
        (*(row[name] for name in FIELDS[:5]), row["node"] or ""): row for row in rows
    }
    arms = [arm for arm in pilot["arms"] if arm not in missing]
    left_out = {
        **pilot["skipped_arms"],
        **dict.fromkeys(missing, "not collected"),
    }
    lines = [
        "# Conditioning pilot",
        "",
        f"Respondents: {len(pilot['models'])} models "
        f"({', '.join(pilot['models'])}) at {pilot['samples_per_model']} samples "
        f"each, profile `{pilot['profile']}`. Scenarios: "
        f"{', '.join(pilot['scenarios'])}. Trajectory nodes, with their "
        "frontier progress in ECI points: "
        + ", ".join(
            f"{node} {value:.1f}"
            for node, value in sorted(pilot["nodes"].items(), key=lambda row: row[1])
        )
        + f". Arms analysed: {', '.join(arms)}."
        + "".join(f" Arm {arm} is left out: {why}." for arm, why in left_out.items()),
        "",
        *(f"- `{arm}`: {ARMS[arm]}" for arm in arms),
        "",
        "## How to read it",
        "",
        RULES,
        "",
        "## Results",
        "",
        "| Scenario | Source status | Quantity | Per 10 ECI points | Scales fixed | Stated scale | Fastest over slowest | Noise floor | Never falls | Placebo | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted({tuple(row[name] for name in FIELDS[:3]) for row in rows}):

        def at(arm: str, statistic: str, key=key) -> Mapping[str, Any] | None:
            return index.get((*key, arm, statistic, ""))

        slope = "log_slope_per_10_eci"
        floor = at("reference", "noise_floor_log_sd")
        share = at("reference", "never_falls_share")
        note = at("", "verdict")
        cells = (
            *key,
            _figure(at("reference", slope), percent=True),
            _figure(at("fixed_scale", slope), percent=True),
            _figure(at("reference_minus_fixed_scale", slope), percent=True),
            _figure(at("reference", "fastest_to_slowest_log_ratio")),
            "n/a" if floor is None else f"{floor['value']:.2f}",
            "n/a" if share is None else f"{share['value']:.0%} of {share['answers']}",
            _figure(at("placebo_minus_reference", "mean_log_ratio")),
            "n/a" if note is None else note["note"],
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Answers",
        "",
        "Answers that entered each quantity, over every scenario, source "
        "status and node, and those left out: declared unavailable by the "
        "respondent; exhausted, a question whose every validation attempt "
        "failed; and without a log, a zero or non-finite mean. Each is left "
        "out of its own node and model only.",
        "",
        "| Quantity | Arm | Used | Unavailable | Exhausted | No log |",
        "|---|---|---|---|---|---|",
    ]
    counts: dict[tuple[str, str], Counter[str]] = {}
    for row in rows:
        if row["statistic"].startswith(("answers_used", "excluded_")):
            counts.setdefault((row["quantity"], row["arm"]), Counter())[
                row["statistic"]
            ] += int(row["value"])
    lines += [
        f"| {quantity} | {arm} | {counts[quantity, arm]['answers_used']} | "
        + " | ".join(
            str(counts[quantity, arm][f"excluded_{reason}"]) for reason in EXCLUSIONS
        )
        + " |"
        for quantity in QUANTITIES
        for arm in arms
        if (quantity, arm) in counts
    ]
    return "\n".join(lines) + "\n"


def _cell(value: object) -> object:
    if value is None:
        return ""
    return f"{value:.10g}" if isinstance(value, float) else value


def write_analysis(directory: Path, output: Path) -> list[dict[str, Any]]:
    """Analyse the pilot and write ``pilot.md`` and ``pilot.csv`` into the
    new directory ``output``."""
    require_new_evidence_output(output)
    rows, missing = analyse_pilot(directory)
    if not rows:
        raise ValueError("no arm of the pilot has been collected")
    output.mkdir(parents=True)
    with (output / "pilot.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows([_cell(value) for value in row.values()] for row in rows)
    (output / "pilot.md").write_text(
        render_markdown(read_pilot(directory), rows, missing), encoding="utf-8"
    )
    return rows


# Command line


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="build every arm's instruments")
    plan.add_argument("output", type=Path, help="new pilot directory")
    plan.add_argument(
        "--source-bundle",
        type=Path,
        required=True,
        help="source bundle whose manifest every instrument binds",
    )
    plan.add_argument(
        "--context",
        type=Path,
        required=True,
        help="context JSON with source_statuses, as build-instrument reads it",
    )
    plan.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        required=True,
        help="scenario to ask about (repeatable); a handful, not the catalog",
    )
    plan.add_argument(
        "--nodes",
        type=int,
        default=3,
        help="interior trajectory nodes, an odd number of at least 3; a no-progress node is added (default: 3)",
    )
    plan.add_argument(
        "--forecast",
        type=Path,
        help="a public-frontier forecast with its releases, such as the forecast.json of a planned study (default: the committed template data/capability/public-frontier-forecast-template.json, filled from the bundle's Epoch ECI table)",
    )
    plan.add_argument(
        "--profile",
        default="example",
        help="profile name in the model profiles file (default: example)",
    )
    plan.add_argument(
        "--profiles",
        type=Path,
        help="model profiles file to read --profile from (default: the file CATMODEL_PROFILES names, else AIUC's example data/example_model_profiles.yaml)",
    )
    plan.add_argument(
        "--samples",
        type=int,
        help="samples per model (default: the profile's samples_per_question)",
    )
    collect = commands.add_parser("collect", help="freeze or ask every question")
    collect.add_argument("pilot", type=Path, help="pilot directory that plan wrote")
    collect.add_argument(
        "--dry-run",
        action="store_true",
        help="freeze every question without calling a provider",
    )
    collect.add_argument(
        "--live",
        action="store_true",
        help="call the respondent models; required unless --dry-run is given",
    )
    collect.add_argument(
        "--arm",
        dest="arms",
        action="append",
        choices=tuple(ARMS),
        help="collect only this arm (repeatable; default: every arm)",
    )
    collect.add_argument(
        "--answers",
        type=Path,
        help="answer table directory: a request the table already answers is not sent, and every response is added to it (a new path starts a table)",
    )
    analyse = commands.add_parser("analyse", help="write pilot.md and pilot.csv")
    analyse.add_argument("pilot", type=Path, help="pilot directory that plan wrote")
    analyse.add_argument(
        "--output", type=Path, required=True, help="new directory for the two files"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            pilot = plan_pilot(args)
            print(
                f"{args.output}: {len(pilot['contexts'])} collections in arms "
                f"{', '.join(pilot['arms'])} on {len(pilot['nodes'])} trajectory "
                f"nodes; {pilot['initial_calls']} initial calls before retries "
                "and cross-checks"
            )
            for arm, reason in pilot["skipped_arms"].items():
                print(f"arm {arm} is left out: {reason}")
        elif args.command == "collect":
            if args.dry_run == args.live:
                print(
                    "Error: pass exactly one of --dry-run, which freezes every "
                    "question without calling a provider, or --live, which "
                    "collects from the respondent models of the pilot's "
                    "profile; every call is billed.",
                    file=sys.stderr,
                )
                return 2
            contexts = collect_pilot(args)
            print(
                f"{args.pilot}: {'froze' if args.dry_run else 'collected'} "
                f"{contexts} collections"
            )
        else:
            rows = write_analysis(args.pilot.expanduser(), args.output.expanduser())
            for row in rows:
                if row["statistic"] == "verdict":
                    print(*(row[name] for name in FIELDS[:3]), row["note"], sep="  ")
            print(f"wrote {args.output / 'pilot.md'} and {args.output / 'pilot.csv'}")
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
