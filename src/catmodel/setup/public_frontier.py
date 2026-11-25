"""Mix annual risk kernels conditional on complete shared public trajectories."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, get_args

from catmodel.contracts.contextual_risk import (
    SAFEGUARD_LEVEL_FIELDS,
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskInstrument,
    ContextualRiskNode,
    scale_text,
)
from catmodel.contracts.incident import (
    DEFAULT_WORK_ALLOWANCES,
    DEFAULT_YEARS_PER_STATE,
    DeploymentBinding,
    Entity,
    EntityPath,
    ExecutionPlan,
    IncidentLaw,
    JointState,
    LocalResponse,
    PathKnot,
    PopulationAbsence,
    ResponsePoint,
)
from catmodel.contracts.law_overrides import OverrideWriter
from catmodel.contracts.public_frontier import NO_PROGRESS_NODE_ID
from catmodel.contracts.public_frontier_condition import (
    PublicTrajectoryCondition,
    SourceModelStatus,
)
from catmodel.contracts.run_setup import (
    AssumptionSet,
    EvidenceSet,
)
from catmodel.elicitation.contextual_frequency import (
    frequency_collection_summary,
)
from catmodel.elicitation.eal_crosscheck import (
    eal_crosscheck_limitation,
    eal_crosscheck_summary,
    follow_up_counts,
)
from catmodel.setup.contextual_compilation import (
    ContextualRespondentSelection,
    contextual_nodes,
    frequency_contribution,
    frequency_estimate_evidence,
    frequency_estimate_record,
    frequency_estimate_views,
    lab_loss_cap_limitation,
    large_loss_contribution,
    large_loss_levels_of,
    large_loss_limitation,
    large_loss_record,
)
from catmodel.setup.contextual_hierarchy import (
    HierarchyOutcome,
    hierarchy_factor_limitation,
    hierarchy_summary,
)
from catmodel.setup.contextual_node import (
    LAB_LOSS_DEPENDENCE,
    LAB_LOSS_TAIL_LIMITATION,
    CompileOptions,
    ContextualLawUnavailable,
    admit_node,
    compile_contextual_node,
    compile_once,
    contextual_scope,
    hypothetical_entity_limitations,
    threat_model_limitations,
)
from catmodel.setup.contextual_setup import (
    contextual_respondent_selections,
    selection_cases,
)
from catmodel.setup.excluded_samples import (
    MAX_EXCLUDED_SAMPLE_SHARE,
    Donors,
    Headline,
    RespondentKey,
    SampleExclusions,
    excluded_samples_evidence,
    excluded_samples_limitation,
    excluded_samples_record,
    lab_loss_state_weights,
    sample_exclusions,
    unanswered_scenarios_limitation,
)
from catmodel.setup.public_frontier_comparisons import (
    CONTEXT_VARIANT_FAMILIES,
    NO_MAPPING,
    AuthoredCapStresses,
    PublicForecastAlternative,
    PublicFrontierComparisons,
    StateCell,
    public_frontier_comparisons,
)
from catmodel.setup.sensitivity_catalog import (
    EXPOSURE_SCALE_VARIANTS,
)
from catmodel.simulation.tail_gate import tail_gate_controls

_STATUSES = get_args(SourceModelStatus)


type MapTasks = Callable[[Callable[[Any], Any], Sequence[Any]], Iterable[Any]]
"""Applies a function to every task and returns the results in task order."""


def map_in_order(function: Callable[[Any], Any], tasks: Sequence[Any]) -> list[Any]:
    """``MapTasks`` in this process, one task after another."""
    return [function(task) for task in tasks]


@dataclass(frozen=True)
class PublicFrontierRiskInput:
    """One captured context and its trajectory node's weight; ``files``, the
    capture's, give each exhausted call's cause.

    ``quadrature_sha256`` is the digest of the study's whole trajectory rule,
    weights included. Questions carry only their node's trajectory identity, so
    that nested rules can share collections; this digest, outside the prompts,
    keeps one study's inputs on one rule.

    ``outcomes`` are the collection's taxonomy-parent outcomes where whoever
    read the collection checked it (``checked_hierarchy_outcomes``); the
    draw's selections are then made from them, and without them making the
    selections checks the collection.
    """

    instrument: ContextualRiskInstrument
    artifact: ContextualRiskArtifact
    path_weight: float
    quadrature_sha256: str
    files: Mapping[str, bytes] = NO_MAPPING
    outcomes: Mapping[tuple[float, RespondentKey], HierarchyOutcome] | None = None


def _path(instrument: ContextualRiskInstrument) -> EntityPath:
    condition = instrument.public_trajectory
    assert condition is not None
    times, values = condition.target_path()
    duration = (condition.times[-1] - condition.times[0]).total_seconds()
    return EntityPath(
        entity_id=instrument.model.entity_id,
        scale_id=condition.coordinates[0].scale_id,
        history_status="stipulated",
        history_rationale=(
            f"Public general ECI context for {condition.target_lab}, not a forecast of this source model's "
            "internal capability. The conditional annual kernel has multiplier one; "
            "all joint lab trajectories are retained in its conditioning context."
        ),
        interpolation=condition.interpolation,
        knots=tuple(
            PathKnot(
                time=(at - condition.times[0]).total_seconds() / duration,
                value=value,
                model_revision=instrument.model.revision,
            )
            for at, value in zip(times, values, strict=True)
        ),
    )


def _held_terms(
    instrument: ContextualRiskInstrument, varied: Sequence[str] = ()
) -> tuple[object, ...]:
    """The lab and exposure terms every trajectory of one source status shares.

    Path-linked scales are rendered per trajectory, so the rendered
    ``use_population`` and progress ratio may differ. The scales, growth
    ceiling and ``use_population`` template may not, and the contract requires
    the rendered text to be that template at the progress ratio. ``varied``
    names the fields, as ``SAFEGUARD_LEVEL_FIELDS`` does, left out of the
    comparison.
    """

    def kept(name: str, record: DeploymentBinding | Entity) -> object:
        omitted = {
            path.split(".", 1)[1] for path in varied if path.startswith(f"{name}.")
        }
        return record.model_dump(mode="json", exclude=omitted) if omitted else record

    exposure = instrument.path_linked_exposure
    return (
        instrument.model,
        kept("deployment", instrument.deployment),
        kept("deployment_binding", _template_binding(instrument)),
        instrument.exposure,
        None if "fixed_controls" in varied else instrument.fixed_controls,
        instrument.fixed_adoption,
        None if exposure is None else (exposure.scales, exposure.max_growth_multiple),
    )


def _template_binding(instrument: ContextualRiskInstrument) -> DeploymentBinding:
    """The deployment binding with path-linked scales left as placeholders."""
    exposure = instrument.path_linked_exposure
    if exposure is None or exposure.use_population_template is None:
        return instrument.deployment_binding
    return instrument.deployment_binding.model_copy(
        update={"use_population": exposure.use_population_template}
    )


def context_variant(instrument: ContextualRiskInstrument) -> tuple[str, str] | None:
    """The context variant an instrument carries, as its family and variant ID."""
    exposure = instrument.path_linked_exposure
    if exposure is not None and exposure.variant is not None:
        return "exposure_scale", exposure.variant.variant_id
    if instrument.safeguard_level is not None:
        return "safeguard_level", instrument.safeguard_level
    return None


def _variant_record(instrument: ContextualRiskInstrument) -> object:
    """Everything a context variant records, the exposure factor included."""
    exposure = instrument.path_linked_exposure
    return (
        None if exposure is None else exposure.variant,
        instrument.safeguard_level,
    )


def path_linked_exposure_limitation(
    instrument: ContextualRiskInstrument,
) -> tuple[str, ...]:
    """State the path-linked deployment scales and their rule, if any."""
    exposure = instrument.path_linked_exposure
    if exposure is None:
        return ()
    return (
        "Deployment scales are path-linked (POP-HYPOTHETICAL-LAB): on each "
        "trajectory node a scale is its as-of level plus (its study-year median-path "
        "level minus the as-of level) times g(x), to two significant figures, "
        "where x is the node's frontier progress over the base forecast's median "
        "progress, both at the middle of the policy period, "
        "g(x) = x up to the median path and 1 + (M - 1) tanh((x - 1)/(M - 1)) "
        f"above it, and M = {exposure.max_growth_multiple:g} is the ceiling on "
        "growth as a multiple of the median-path growth; the no-progress node "
        "keeps the as-of levels. "
        + "; ".join(
            f"{row.scale_id}: {scale_text(row.as_of)} as of, {scale_text(row.study_year_median)} on the median path"
            for row in exposure.scales
        )
        + ". Each trajectory's laws carry their own rendered scales; the scope's "
        "deployment bindings show the template with its placeholders.",
    )


def _samples_per_model(instrument: ContextualRiskInstrument) -> str:
    counts = Counter(row.model_id for row in instrument.respondents)
    return " to ".join(str(value) for value in sorted(set(counts.values())))


def context_variant_limitations(
    first: ContextualRiskInstrument,
    variants: Mapping[str, Mapping[str, Sequence[PublicFrontierRiskInput]]],
) -> tuple[str, ...]:
    """Disclose the context variants a setup carries, if any, by family."""
    shared = (
        "Each variant is a separately collected study on the embedded trajectory rule's "
        "nodes; {shows} and its states carry zero weight outside its own arm. Its "
        "no-progress node, the headline's current-frontier trajectory, carries zero "
        "weight in its arm too and serves only capability-conditioned reporting. Each "
        "variant is measured against the headline captures reweighted to the same "
        "embedded rule, so its effect includes the difference between independently "
        "collected answers as well as the {change}. Variant and control states also draw "
        "independent simulation streams, so their tail statistics differ by "
        "simulation noise; an EAL contrast from native analytic means has none."
    )
    rows = []
    for family, by_variant in variants.items():
        if not by_variant:
            continue
        parts = []
        for variant_id, inputs in by_variant.items():
            instrument = inputs[0].instrument
            size = (
                f"({len(inputs) // 2 - 1} nodes and the no-progress node, "
                f"{_samples_per_model(instrument)} samples per model)"
            )
            if family == "exposure_scale":
                exposure = instrument.path_linked_exposure
                assert exposure is not None and exposure.variant is not None
                parts.append(
                    f"{variant_id} multiplies every path-linked scale by "
                    f"{exposure.variant.factor:g} {size}"
                )
            else:
                parts.append(
                    f"{variant_id} states the study config's {variant_id} safeguards {size}"
                )
        if family == "exposure_scale":
            rows.append(
                "Exposure-scale variants (POP-HYPOTHETICAL-LAB): "
                + "; ".join(parts)
                + f". The headline has {_samples_per_model(first)} samples per model. "
                + shared.format(
                    shows="its questions show the scaled numbers as the study's own, "
                    "without the factor,",
                    change="scale change",
                )
            )
        else:
            rows.append(
                "Safeguard-level variants (POP-SAFEGUARD-LEVEL): "
                + "; ".join(parts)
                + f". The headline has {_samples_per_model(first)} samples per model. "
                "Each level replaces the shared controls, each source status's tools "
                "and permissions and human oversight, and the internal deployment's "
                "definition and use population with one stipulated posture. "
                + shared.format(
                    shows="its questions state that level's safeguards as the "
                    "study's own, with no mention of another level,",
                    change="safeguard change",
                )
                + " The effect measures the respondents' judgment of how the stated "
                "safeguards change frequency and severity, unvalidated against data, "
                "and may over- or under-react to wording."
            )
    return tuple(rows)


def public_state_id(draw_id: str, selection_id: str) -> str:
    """A joint state's ID: its trajectory node, then its respondent selection.

    The setup ID stays out, so imports that differ only in it draw identical
    streams. Node IDs contain no colon, so the node is the prefix.
    """
    if ":" in draw_id:
        raise ValueError("public trajectory node IDs cannot contain a colon")
    return f"{draw_id}:{selection_id}"


def public_state_node(state_id: str) -> str:
    """The trajectory node of a public-frontier joint state."""
    return state_id.split(":", 1)[0]


def _compile_conditioned(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    path: EntityPath,
    status: str,
    options: CompileOptions,
    answers: Mapping[float, ContextualRiskAnswer],
    severity_factors: Mapping[float, float] | None,
    **changes: Any,
) -> IncidentLaw | PopulationAbsence:
    """One node's trajectory kernel for one source status, admitted for the run;
    ``changes`` replace some of ``options`` (``compile_once``).

    Its count reads the rank of its scenario at its source status in its
    joint state's rank streams: each simulated year one rank for that rank
    group, independent of every other state's unless states declare one rank
    stream (``JointState.rank_stream_id``)."""
    law = compile_contextual_node(
        instrument,
        node,
        answers,
        0,
        rank_group=f"{status}::{node.population.scenario_id}",
        severity_factors=severity_factors,
        options=replace(options, **changes),
    )
    if isinstance(law, PopulationAbsence):
        return law
    return admit_node(instrument, node, path, _conditioned(law, instrument, status))


def _conditioned(
    law: IncidentLaw, instrument: ContextualRiskInstrument, status: str
) -> IncidentLaw:
    """Mark a fixed-score node law as the annual kernel of one public trajectory.

    The result is an unvalidated copy. Admission validates it, once, with the
    setup it admits it in (``admit_node``)."""
    condition = instrument.public_trajectory
    assert condition is not None
    return law.model_copy(
        update={
            "deployment_id": instrument.deployment.entity_id,
            "response": LocalResponse(
                domain="conditional_annual_kernel",
                interpolation="linear",
                extrapolation="flat",
                points=(ResponsePoint(value=0, multiplier=1),),
                held_fixed="Annual law conditional on the complete public trajectory; no capability response is applied again.",
            ),
            "segments": tuple(
                segment.model_copy(
                    update={
                        "source_public_status": status,
                        "source_model_id": instrument.model.entity_id,
                        "source_model_revision": instrument.model.revision
                        or "hypothetical-source-population",
                    }
                )
                for segment in law.segments
            ),
            "conditioning_context": {
                **{
                    key: value
                    for key, value in law.conditioning_context.items()
                    if key
                    not in {"score", "parameter_interpolation", "coordinate_kind"}
                },
                "coordinate_kind": "public_trajectory",
                "public_trajectory": condition.model_dump(mode="json"),
                "source_status_basis": "originating_incident",
                "forecast_integration": {
                    "scope": "trajectory_quadrature_node",
                    "continuous_forecast_loss_moments": "not_established",
                },
            },
        }
    )


def _compile_cells(
    grouped: Mapping[str, Mapping[str, PublicFrontierRiskInput]],
    populations,
    exclusions: SampleExclusions,
    options: CompileOptions,
    *,
    edits: Mapping[
        str, Callable[[ContextualRiskNode], ContextualRiskNode]
    ] = NO_MAPPING,
    contributions: list | None = None,
    large_loss_rows: list | None = None,
    map_tasks: MapTasks = map_in_order,
    selections: dict[str, _DrawSelections] | None = None,
) -> list[StateCell]:
    """Compile every trajectory draw and kept respondent selection, node by node.

    ``grouped`` holds each draw's inputs as its selections read them
    (``_Study.selection_inputs``), and ``exclusions`` gives each draw's kept
    respondents, pooled weights and donors.
    ``options`` are the reference law's or a comparison arm's
    (``CompileOptions``), and ``edits`` replaces the named scenarios' node
    definitions for an arm; the reference passes none. Selections do not
    depend on the arm: a caller that compiles the same draws for several arms
    passes one ``selections`` dictionary to every call, which the first fills
    by draw and the others read, so each draw's are made once. Making them
    reads the draw's collections; where an input carries its collection's
    outcomes they are made from those, and otherwise the collection is
    checked and its taxonomy parents pooled first.
    ``contributions`` collects each node's ``frequency_estimate`` exhibit rows
    and ``large_loss_rows`` its ``large_loss_anchor`` exhibit rows. Draws
    compile independently; ``map_tasks`` runs them, in worker processes where
    the caller supplies them, and returns them in order, so the cells and
    rows do not depend on it.
    """
    made = {} if selections is None else selections
    tasks = [
        _DrawTask(
            draw_id=draw_id,
            pair={
                status: _DrawInput(
                    item.instrument,
                    # Only making the selections reads the collection.
                    None if draw_id in made else item.artifact,
                    item.path_weight,
                    None if draw_id in made else item.outcomes,
                )
                for status, item in pair.items()
            },
            weights=exclusions.weights[draw_id],
            donors=exclusions.donors[draw_id],
            populations=tuple(populations),
            options=options,
            edits=dict(edits),
            contributions=contributions is not None,
            large_loss_rows=large_loss_rows is not None,
            selections=made.get(draw_id),
        )
        for draw_id, pair in sorted(grouped.items())
    ]
    cells: list[StateCell] = []
    for task, (drawn, frequency_rows, large_rows, chosen) in zip(
        tasks, map_tasks(_compile_draw, tasks), strict=True
    ):
        cells.extend(drawn)
        if chosen is not None:
            made[task.draw_id] = chosen
        if contributions is not None:
            contributions.extend(frequency_rows)
        if large_loss_rows is not None:
            large_loss_rows.extend(large_rows)
    return cells


type _DrawSelections = Mapping[str, tuple[ContextualRespondentSelection, ...]]
"""One draw's respondent selections, by source status."""


@dataclass(frozen=True)
class _DrawInput:
    """One source status of a draw. The collection is carried only where the
    draw's selections are still to be made, with its outcomes where whoever
    read it checked it (``PublicFrontierRiskInput.outcomes``)."""

    instrument: ContextualRiskInstrument
    artifact: ContextualRiskArtifact | None
    path_weight: float
    outcomes: Mapping[tuple[float, RespondentKey], HierarchyOutcome] | None = None


@dataclass(frozen=True)
class _DrawTask:
    """One trajectory draw's compilation, without its captured files, so it
    can be sent to a worker process; the options are ``_compile_cells``'s.
    ``donors`` are the draw's (``excluded_samples.Donors``), and
    ``selections`` its selections where an earlier compilation made them."""

    draw_id: str
    pair: Mapping[str, _DrawInput]
    weights: Mapping[RespondentKey, float]
    donors: Donors
    populations: tuple
    edits: Mapping[str, Callable[[ContextualRiskNode], ContextualRiskNode]]
    options: CompileOptions
    contributions: bool
    large_loss_rows: bool
    selections: _DrawSelections | None = None


def _compile_draw(
    task: _DrawTask,
) -> tuple[list[StateCell], list, list, _DrawSelections | None]:
    """One draw's cells, in respondent and node order, with its exhibit rows,
    and its selections where this compilation made them."""
    draw_id, pair, populations, edits = (
        task.draw_id,
        task.pair,
        task.populations,
        task.edits,
    )
    cells: list[StateCell] = []
    contributions: list = []
    large_loss_rows: list = []
    selections, paths, nodes = {}, {}, {}
    made: dict[str, tuple[ContextualRespondentSelection, ...]] = {}
    for status, item in pair.items():
        if task.selections is not None:
            chosen = task.selections[status]
        else:
            if item.artifact is None:
                raise ValueError("a draw without selections needs its collection")
            chosen = made[status] = contextual_respondent_selections(
                item.instrument,
                item.artifact,
                task.weights,
                outcomes=item.outcomes,
                donors=task.donors,
            )
        selections[status] = {row.respondent.key(): row for row in chosen}
        paths[status] = _path(item.instrument)
        nodes[status] = tuple(
            edits[node.population.scenario_id](node)
            if node.population.scenario_id in edits
            else node
            for node in contextual_nodes(item.instrument)
        )
    for key, reference in selections["public"].items():
        outcomes: dict[str, tuple[IncidentLaw, ...] | PopulationAbsence] = {}
        for node_index, population in enumerate(populations):
            statuses = []
            for status in _STATUSES:
                instrument = pair[status].instrument
                selection = selections[status][key]
                node = nodes[status][node_index]
                compile_law = compile_once(
                    partial(
                        _compile_conditioned,
                        instrument,
                        node,
                        paths[status],
                        status,
                        task.options,
                    )
                )
                law = compile_law(
                    selection.answers_by_node[node.node_id],
                    selection.severity_factors_by_node[node.node_id],
                )
                if task.contributions:
                    contributions.append(
                        (
                            public_state_id(draw_id, reference.selection_id),
                            pair["public"].path_weight * reference.weight,
                            population.scenario_id,
                            frequency_contribution(
                                instrument,
                                node,
                                selection,
                                0,
                                law,
                                compile_law=compile_law,
                                options=task.options,
                            ),
                        )
                    )
                if task.large_loss_rows:
                    large_loss_rows.append(
                        (
                            public_state_id(draw_id, reference.selection_id),
                            pair["public"].path_weight * reference.weight,
                            population.scenario_id,
                            large_loss_contribution(
                                instrument,
                                node,
                                selection,
                                0,
                                law,
                                compile_law=compile_law,
                                options=task.options,
                            ),
                        )
                    )
                statuses.append(law)
            missing = [
                law
                for law in statuses
                if isinstance(law, PopulationAbsence)
                and law.disposition == "unavailable"
            ]
            existing = tuple(law for law in statuses if isinstance(law, IncidentLaw))
            outcomes[population.scenario_id] = (
                PopulationAbsence(
                    scenario_id=population.scenario_id,
                    disposition="unavailable",
                    reason="; ".join(law.reason for law in missing),
                )
                if missing
                else existing
                or PopulationAbsence(
                    scenario_id=population.scenario_id,
                    disposition="structural_zero",
                    reason="Both source-status contributions have zero count.",
                )
            )
        cells.append(
            StateCell(
                state_id=public_state_id(draw_id, reference.selection_id),
                draw_id=draw_id,
                path_weight=pair["public"].path_weight,
                selection=reference,
                paths=tuple(paths.values()),
                outcomes=outcomes,
            )
        )
    return cells, contributions, large_loss_rows, made or None


@dataclass(frozen=True)
class _Study:
    grouped: dict[str, dict[str, PublicFrontierRiskInput]]
    representatives: dict[str, ContextualRiskInstrument]

    def exclusions(
        self, max_share: float, headline: Headline | None = None
    ) -> SampleExclusions:
        """The study's excluded samples and pooled weights by trajectory node,
        and the leaves a model does not answer with their donors; a variant
        study passes what it follows of its ``headline``."""
        return sample_exclusions(
            {
                draw: [(item.instrument, item.artifact) for item in pair.values()]
                for draw, pair in self.grouped.items()
            },
            {draw: pair["public"].path_weight for draw, pair in self.grouped.items()},
            max_share=max_share,
            headline=headline,
        )

    def selection_inputs(
        self, exclusions: SampleExclusions
    ) -> dict[str, dict[str, PublicFrontierRiskInput]]:
        """Each draw's inputs as its selections read them: the cases and
        outcomes of ``contextual_setup.selection_cases``, which are an input's
        own unless a sample of its draw borrows a leaf."""
        selected: dict[str, dict[str, PublicFrontierRiskInput]] = {}
        for draw, pair in self.grouped.items():
            selected[draw] = {}
            for status, item in pair.items():
                cases, outcomes = selection_cases(
                    item.instrument, item.artifact, exclusions, draw, item.outcomes
                )
                selected[draw][status] = replace(
                    item, artifact=cases, outcomes=outcomes
                )
        return selected

    def captures(self) -> list[tuple[ContextualRiskArtifact, Mapping[str, bytes]]]:
        return [
            (item.artifact, item.files)
            for pair in self.grouped.values()
            for item in pair.values()
        ]


def _definitions(instrument: ContextualRiskInstrument) -> tuple[dict, ...]:
    """Each node's incident definition, without its chain's exposure driver,
    which the scenario chooses per capability access."""
    return tuple(
        {
            **node.model_dump(mode="json", exclude={"frequency_chain": {"exposure"}}),
            "population": node.population.model_dump(
                mode="json", exclude={"deployment_ids"}
            ),
        }
        for node in contextual_nodes(instrument)
    )


def _group_inputs(
    inputs: Sequence[PublicFrontierRiskInput], first: ContextualRiskInstrument
) -> _Study:
    """Check one study's paired inputs against ``first`` and group them by node.

    Every input shares ``first``'s lab, loss scope, period, sources, scale and
    incident definitions, and the trajectory quadrature, target and respondents
    of the study's own first input.
    """
    if not inputs:
        raise ValueError("public-frontier setup requires captured inputs")
    roster = sorted(r.key() for r in inputs[0].instrument.respondents)
    definitions = _definitions(first)
    grouped: dict[str, dict[str, PublicFrontierRiskInput]] = {}
    identities = set()
    quadrature = None
    representatives: dict[str, ContextualRiskInstrument] = {}
    for item in inputs:
        instrument, condition = item.instrument, item.instrument.public_trajectory
        if condition is None:
            raise ValueError("every input must have a public trajectory condition")
        if instrument.context_id in identities:
            raise ValueError("public trajectory instruments require unique context IDs")
        identities.add(instrument.context_id)
        if not math.isfinite(item.path_weight) or not 0 <= item.path_weight <= 1:
            raise ValueError("path weights must be finite probabilities")
        if quadrature is None:
            quadrature = item.quadrature_sha256
        if item.quadrature_sha256 != quadrature:
            raise ValueError("all inputs must use the same trajectory quadrature")
        if (
            instrument.period != first.period
            or instrument.source_bundle_manifest_sha256
            != first.source_bundle_manifest_sha256
            or instrument.lab.model_dump(exclude={"related_entity_ids"})
            != first.lab.model_dump(exclude={"related_entity_ids"})
            or instrument.capability_scale != first.capability_scale
            or any(
                getattr(instrument, name) != getattr(first, name)
                for name in (
                    "lab_compensation_definition",
                    "lab_annual_business_loss_definition",
                    "lab_loss_scope",
                    "lab_annual_business_loss_dependence",
                )
            )
            or sorted(r.key() for r in instrument.respondents) != roster
        ):
            raise ValueError(
                "all public conditions require the same lab, loss scope, period, sources, scale and respondents"
            )
        if _definitions(instrument) != definitions:
            raise ValueError(
                "public conditions must preserve the same incident definitions"
            )
        status = condition.source_model_status
        by_status = grouped.setdefault(condition.draw_id, {})
        if status in by_status:
            raise ValueError(
                "a public draw requires exactly one input per source status"
            )
        by_status[status] = item
        previous = representatives.setdefault(status, instrument)
        if _held_terms(previous) != _held_terms(instrument) or _variant_record(
            previous
        ) != _variant_record(instrument):
            raise ValueError(
                "source-status lab and exposure assumptions cannot change across public paths"
            )
    weights = []
    for pair in grouped.values():
        if set(pair) != {"public", "not_yet_public"}:
            raise ValueError(
                "every public draw needs public and not_yet_public source inputs"
            )
        public, private = pair["public"], pair["not_yet_public"]
        a, b = public.instrument.public_trajectory, private.instrument.public_trajectory
        assert a is not None and b is not None
        if public.path_weight != private.path_weight or a.model_dump(
            exclude={"source_model_status", "rationale"}
        ) != b.model_dump(exclude={"source_model_status", "rationale"}):
            raise ValueError(
                "source statuses must share identical public paths and weights"
            )
        weights.append(public.path_weight)
    if not math.isclose(math.fsum(weights), 1, abs_tol=1e-12):
        raise ValueError("distinct public path weights must sum to one")
    if (
        len({i.model.entity_id for i in representatives.values()}) != 2
        or len({i.deployment.entity_id for i in representatives.values()}) != 2
    ):
        raise ValueError(
            "source-status populations need distinct model and deployment identities"
        )
    return _Study(grouped, representatives)


def _same_path(a: PublicTrajectoryCondition, b: PublicTrajectoryCondition) -> bool:
    """Whether two conditions hold the same joint trajectory, up to roundoff."""
    return (
        a.times == b.times
        and a.coordinates == b.coordinates
        and all(
            math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-7)
            for left, right in zip(a.values, b.values, strict=True)
            for x, y in zip(left, right, strict=True)
        )
    )


def _check_variant_nodes(
    variant: _Study,
    grouped: Mapping[str, Mapping[str, PublicFrontierRiskInput]],
    embedded: Mapping[str, float],
    noun: str,
) -> dict[str, str]:
    """A variant's nodes are the embedded rule's nodes, at its weights, and the
    headline's no-progress node at weight zero, on the headline's trajectory
    for each; the no-progress node enters no arm. Returns each variant node's
    headline node."""
    if NO_PROGRESS_NODE_ID not in grouped:
        raise ValueError(f"the headline has no no-progress node {NO_PROGRESS_NODE_ID}")
    targets = {
        draw: pair["public"].instrument.public_trajectory
        for draw, pair in grouped.items()
        if embedded.get(draw, 0) > 0 or draw == NO_PROGRESS_NODE_ID
    }
    matched = {}
    for node, pair in variant.grouped.items():
        condition = pair["public"].instrument.public_trajectory
        assert condition is not None
        hits = [
            draw
            for draw, target in targets.items()
            if target is not None and _same_path(condition, target)
        ]
        if len(hits) != 1 or not math.isclose(
            pair["public"].path_weight, embedded[hits[0]], rel_tol=1e-8, abs_tol=1e-12
        ):
            raise ValueError(
                f"{noun} variant node must be an embedded-rule node at its weight, or "
                "the headline's no-progress node at weight zero, on the headline's "
                "trajectory"
            )
        matched[node] = hits[0]
    if sorted(matched.values()) != sorted(targets):
        raise ValueError(
            f"{noun} variant must cover exactly the embedded rule's nodes and the "
            f"no-progress node {NO_PROGRESS_NODE_ID}"
        )
    return matched


def variant_no_progress_requirements(
    variant_cells: Mapping[str, Mapping[str, Sequence[StateCell]]],
) -> dict[str, tuple[str, ...]]:
    """Each variant arm's no-progress states, which it simulates at weight zero.

    They enter no average; simulating them under the variant's own law gives
    each variant current-frontier draws for capability-conditioned reporting.
    """
    return {
        f"{family}::{variant_id}": tuple(
            cell.state_id for cell in cells if cell.draw_id == NO_PROGRESS_NODE_ID
        )
        for family, by_variant in variant_cells.items()
        for variant_id, cells in by_variant.items()
    }


def _joint_state(
    cell: StateCell,
    *,
    weight: float,
    epistemic_group_id: str,
    prefix: str = "",
    rank_stream_id: str | None = None,
    lab_loss_left_out: bool = False,
) -> JointState:
    """``prefix`` marks a context variant's own groups, and ``rank_stream_id``
    names the rank streams a variant state reads instead of its own.
    ``lab_loss_left_out`` says that the cell's sample is left out of the
    views that count the lab's loss in its study
    (``SampleExclusions.left_out``)."""
    laws, absent = [], []
    for scenario_id, outcome in cell.outcomes.items():
        if isinstance(outcome, tuple):
            laws.extend(outcome)
        elif outcome.disposition == "unavailable":
            raise ContextualLawUnavailable(
                f"Incomplete source-status population {scenario_id!r}, draw {cell.draw_id!r}: "
                + outcome.reason
            )
        else:
            absent.append(outcome)
    return JointState(
        state_id=cell.state_id,
        epistemic_group_id=epistemic_group_id,
        # One trajectory node and one respondent model (NUM-RESPONDENT-SAMPLING).
        sampling_group_id=f"{prefix}{cell.draw_id}:{cell.selection.respondent.model_id}",
        # Both source statuses' selections read one donor map
        # (``_compile_draw``).
        borrowed_leaves=tuple(sorted(cell.selection.donors_by_node)),
        lab_loss_left_out=lab_loss_left_out,
        rank_stream_id=rank_stream_id,
        weight=weight,
        entity_paths=cell.paths,
        laws=tuple(laws),
        absent_populations=tuple(absent),
    )


def public_frontier_setup_fields(
    inputs: tuple[PublicFrontierRiskInput, ...],
    *,
    setup_id: str,
    exposure_rationale: str,
    residual_independence_rationale: str,
    parameter_coupling_rationale: str,
    years_per_state: int = DEFAULT_YEARS_PER_STATE,
    max_reference_state_years: int | None = None,
    max_comparison_state_years: int | None = None,
    work_allowances: Mapping[str, int] = DEFAULT_WORK_ALLOWANCES,
    epistemic_tail_scale: float = 1.0,
    seed: int = 42,
    comparison_families: tuple[str, ...] | None = None,
    forecast_alternatives: Mapping[str, PublicForecastAlternative] = NO_MAPPING,
    cap_stresses: Mapping[str, AuthoredCapStresses | str] = NO_MAPPING,
    context_variants: Mapping[
        str, Mapping[str, tuple[PublicFrontierRiskInput, ...]]
    ] = NO_MAPPING,
    embedded_path_weights: Mapping[str, float] = NO_MAPPING,
    max_excluded_sample_share: float = MAX_EXCLUDED_SAMPLE_SHARE,
    map_tasks: MapTasks = map_in_order,
    override_directory: Path | None = None,
) -> dict[str, Any]:
    """The fields of a ``RunSetup``, unvalidated: one state per public draw
    and kept respondent; both source statuses coexist. A caller that adds
    evidence or limitations adds them before constructing the setup, so it is
    validated once.

    ``comparison_families=None`` selects ``DEFAULT_CONTEXT_BOUND_FAMILIES``.
    Every catalog family becomes a frozen comparison requirement: required arms
    for a selected family this route generates, and otherwise unavailable or
    inapplicable rows with their reasons. ``forecast_alternatives`` holds the
    exact trajectory weights of each ``public_forecast`` arm, by label, and
    ``cap_stresses`` the authored cap stresses of each scenario, or the reason
    none apply to its frozen node. ``context_variants`` holds each context variant's
    inputs, by family (``exposure_scale`` or ``safeguard_level``) and variant ID,
    with path weights from the embedded rule (``embedded_path_weights``, by
    headline node), which also weights the variants' matched control. Variant states
    join the setup at zero reference weight. ``work_allowances`` sets the
    plan's work allowances per simulated year. The reference reads every
    count law as stated (``POOL-SPREAD``). In each study,
    the headline and every variant, a model whose samples lacking one leaf
    exceed ``max_excluded_sample_share`` does not answer that leaf, and its
    states read other models' answers to it; any other respondent with an
    unanswered leaf in either source status's capture of a draw has no state
    there, within the same share (``excluded_samples``). A variant also leaves
    a model out of each leaf it does not answer in the headline, with the
    headline's donors (``_compile_context_variants``).

    The headline study compiles first (``_compile_cells``), then each
    context variant (``_compile_context_variants``) and the comparison arms
    (``public_frontier_comparisons``); the exhibits (``_setup_evidence``),
    the plan (``_execution_plan``) and the limitations
    (``_setup_limitations``) describe them. With ``override_directory``, each
    simulated arm's law overrides are written there as the arm compiles and
    released, and the returned ``overrides`` read them from there, so the
    directory must outlive the setup built from these fields.
    """
    _check_setup_arguments(
        inputs,
        rationales={
            "exposure_rationale": exposure_rationale,
            "residual_independence_rationale": residual_independence_rationale,
            "parameter_coupling_rationale": parameter_coupling_rationale,
        },
        years_per_state=years_per_state,
        seed=seed,
    )
    tail_gate = tail_gate_controls(
        max_reference_state_years, max_comparison_state_years
    )
    reference = CompileOptions(epistemic_tail_scale=epistemic_tail_scale)
    first = inputs[0].instrument
    study = _group_inputs(inputs, first)
    representatives = study.representatives
    exclusions = study.exclusions(max_excluded_sample_share)
    selected = study.selection_inputs(exclusions)

    nodes = contextual_nodes(first)
    populations = tuple(
        node.population.model_copy(
            update={
                "deployment_ids": tuple(
                    representatives[status].deployment.entity_id
                    for status in ("public", "not_yet_public")
                )
            }
        )
        for node in nodes
    )
    contributions: list = []
    large_loss_rows: list = []
    # The reference makes each draw's selections; every arm reuses them.
    selections: dict[str, _DrawSelections] = {}
    cells = _compile_cells(
        selected,
        populations,
        exclusions,
        reference,
        contributions=contributions,
        large_loss_rows=large_loss_rows,
        map_tasks=map_tasks,
        selections=selections,
    )
    states = [
        _joint_state(
            cell,
            weight=cell.path_weight * cell.selection.weight,
            epistemic_group_id=cell.selection.selection_id,
            lab_loss_left_out=cell.selection.respondent.key()
            in exclusions.left_out(cell.draw_id),
        )
        for cell in cells
    ]
    # The exhibits weigh the headline's states in each view as the run does.
    lab_weights, gross_withheld = lab_loss_state_weights(states)
    variants = _compile_context_variants(
        context_variants,
        study,
        exclusions,
        first,
        populations,
        reference,
        embedded_path_weights=embedded_path_weights,
        max_excluded_sample_share=max_excluded_sample_share,
        map_tasks=map_tasks,
    )
    states.extend(variants.states)
    comparisons = public_frontier_comparisons(
        cells,
        states,
        nodes,
        partial(
            _compile_cells,
            selected,
            populations,
            exclusions,
            map_tasks=map_tasks,
            selections=selections,
        ),
        reference=reference,
        comparison_families=comparison_families,
        forecast_alternatives=forecast_alternatives,
        cap_stresses=cap_stresses,
        path_linked=first.path_linked_exposure is not None,
        context_variants=variants.cells,
        embedded_path_weights=embedded_path_weights,
        writer=OverrideWriter(override_directory),
    )
    # The exhibits describe the states the run keeps.
    kept = [
        _KeptInput(
            item,
            exclusions.kept_view(_draw_id(item), item.artifact),
            exclusions.run_view(_draw_id(item), item.artifact),
            frozenset(node.node_id for node in contextual_nodes(item.instrument)),
        )
        for item in inputs
    ]
    exhibits, excluded = _setup_evidence(
        inputs,
        first,
        study,
        exclusions,
        variants,
        kept,
        contributions=contributions,
        large_loss_rows=large_loss_rows,
        lab_loss_weights=lab_weights,
        gross_withheld=gross_withheld,
        reference=reference,
        max_excluded_sample_share=max_excluded_sample_share,
    )
    plan = _execution_plan(
        setup_id,
        first,
        populations,
        states,
        comparisons,
        variants,
        years_per_state=years_per_state,
        seed=seed,
        work_allowances=work_allowances,
        tail_gate=tail_gate,
    )
    entities = (
        first.lab.model_copy(
            update={
                "related_entity_ids": tuple(
                    i.model.entity_id for i in representatives.values()
                )
            }
        ),
        *(
            entity
            for i in representatives.values()
            for entity in (i.model, i.deployment)
        ),
    )
    return dict(
        setup_id=setup_id,
        scope=contextual_scope(
            first,
            scope_id=setup_id,
            title="Risk conditional on public frontier trajectories",
            populations=populations,
            entities=entities,
            deployments=tuple(_template_binding(i) for i in representatives.values()),
        ),
        evidence=EvidenceSet(
            records=(*exhibits.records, *excluded.records),
            files={**exhibits.files, **excluded.files},
        ),
        joint_states=tuple(states),
        alternatives=comparisons.alternatives,
        overrides=comparisons.overrides,
        assumptions=AssumptionSet(
            independence=(
                exposure_rationale,
                residual_independence_rationale,
                parameter_coupling_rationale,
                "Respondents are matched across conditions. Source-status/node parameter ranks are independent, then reused across public trajectories.",
                LAB_LOSS_DEPENDENCE,
            ),
        ),
        execution=plan,
        limitations=_setup_limitations(
            inputs,
            first,
            entities,
            context_variants,
            comparisons,
            kept,
            study,
            exclusions,
            variants,
            max_excluded_sample_share=max_excluded_sample_share,
        ),
    )


def _check_setup_arguments(
    inputs: tuple[PublicFrontierRiskInput, ...],
    *,
    rationales: Mapping[str, object],
    years_per_state: object,
    seed: object,
) -> None:
    """Refuse a blank aggregation rationale, no inputs, or years per state or a
    seed that is not an integer at its minimum."""
    for name, value in rationales.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must state the aggregation assumption")
    if not inputs:
        raise ValueError("public-frontier setup requires captured inputs")
    for name, value, minimum in (
        ("years_per_state", years_per_state, 1),
        ("seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")


def _draw_id(item: PublicFrontierRiskInput) -> str:
    """The trajectory node of an input."""
    condition = item.instrument.public_trajectory
    assert condition is not None
    return condition.draw_id


class _KeptInput(NamedTuple):
    """One input as the exhibits read it: ``kept`` holds the cases the run
    keeps of its collection, each once, for what was asked and answered, and
    ``read`` the cases its draw's states read, a borrowed case in each state
    that reads it, for the run's laws (``SampleExclusions.kept_view`` and
    ``run_view``). ``node_ids`` are its instrument's nodes."""

    item: PublicFrontierRiskInput
    kept: ContextualRiskArtifact
    read: ContextualRiskArtifact
    node_ids: frozenset[str]


@dataclass(frozen=True)
class _ContextVariants:
    """Every context variant's compiled study (``_compile_context_variants``).

    ``cells`` holds each variant's state cells, by family and variant ID;
    ``exclusions`` each variant's excluded samples with its grouped study; and
    ``states`` the variants' zero-weight joint states, in variant order.
    """

    cells: dict[str, dict[str, tuple[StateCell, ...]]]
    exclusions: dict[tuple[str, str], tuple[SampleExclusions, _Study]]
    states: list[JointState]


def _compile_context_variants(
    context_variants: Mapping[str, Mapping[str, tuple[PublicFrontierRiskInput, ...]]],
    study: _Study,
    exclusions: SampleExclusions,
    first: ContextualRiskInstrument,
    populations: tuple,
    reference: CompileOptions,
    *,
    embedded_path_weights: Mapping[str, float],
    max_excluded_sample_share: float,
    map_tasks: MapTasks,
) -> _ContextVariants:
    """Check and compile each context variant's study at the reference options.

    A variant's nodes are the embedded rule's (``_check_variant_nodes``), its
    instruments carry its own variant and differ from the headline's only in
    it (``_check_variant_study``), and its states are renamed into the
    family's prefix at zero reference weight; each reads the rank streams of
    the headline state at its node and respondent selection. A variant's
    unanswered leaves are those of its own captures and those of the
    headline, whose ``exclusions`` these are, and a sample that borrows a
    leaf in both studies takes its headline donor at the same node
    (``sample_exclusions``), so that the variant and its matched control
    read a leaf from the same answers.
    """
    grouped, representatives = study.grouped, study.representatives
    variant_cells: dict[str, dict[str, tuple[StateCell, ...]]] = {}
    variant_exclusions: dict[tuple[str, str], tuple[SampleExclusions, _Study]] = {}
    states: list[JointState] = []
    models = {row.model_id for row in first.respondents}
    if any(context_variants.values()) and (
        set(embedded_path_weights) != set(grouped)
        or not math.isclose(math.fsum(embedded_path_weights.values()), 1, abs_tol=1e-12)
    ):
        raise ValueError(
            "context variants need the embedded rule's weights on every headline node"
        )
    for family, by_variant in context_variants.items():
        kind = CONTEXT_VARIANT_FAMILIES[family]
        for variant_id, variant_inputs in by_variant.items():
            variant = _group_inputs(variant_inputs, first)
            headline_nodes = _check_variant_nodes(
                variant, grouped, embedded_path_weights, kind.noun
            )
            _check_variant_study(
                variant, representatives, models, family=family, variant_id=variant_id
            )
            excluded = variant.exclusions(
                max_excluded_sample_share, exclusions.followed_at(headline_nodes)
            )
            variant_exclusions[family, variant_id] = (excluded, variant)
            compiled = _compile_cells(
                variant.selection_inputs(excluded),
                populations,
                excluded,
                reference,
                map_tasks=map_tasks,
            )
            cells_for_variant = tuple(
                replace(cell, state_id=kind.state_id(variant_id, cell.state_id))
                for cell in compiled
            )
            variant_cells.setdefault(family, {})[variant_id] = cells_for_variant
            # A variant state reads the parameter ranks of the headline state
            # at its node and respondent selection, whose ID its own had before
            # the prefix, so the variant arm stays paired with its control.
            states.extend(
                _joint_state(
                    cell,
                    weight=0.0,
                    epistemic_group_id=kind.state_id(
                        variant_id, cell.selection.selection_id
                    ),
                    prefix=kind.state_id(variant_id, ""),
                    rank_stream_id=headline.state_id,
                    lab_loss_left_out=cell.selection.respondent.key()
                    in excluded.left_out(cell.draw_id),
                )
                for headline, cell in zip(compiled, cells_for_variant, strict=True)
            )
    return _ContextVariants(variant_cells, variant_exclusions, states)


def _check_variant_study(
    variant: _Study,
    representatives: Mapping[str, ContextualRiskInstrument],
    models: set[str],
    *,
    family: str,
    variant_id: str,
) -> None:
    """Refuse a variant study whose instruments do not carry its own variant
    (and the headline's none), or that differs from the headline in anything
    but that variant and its respondent samples."""
    kind = CONTEXT_VARIANT_FAMILIES[family]
    for status in _STATUSES:
        instrument = variant.representatives[status]
        if context_variant(instrument) != (
            family,
            variant_id,
        ) or context_variant(representatives[status]):
            raise ValueError(
                f"{kind.noun.split(' ', 1)[1]} variants carry their own variant ID; "
                "the headline carries none"
            )
        exposure = instrument.path_linked_exposure
        if family == "exposure_scale" and (
            exposure is None
            or exposure.variant is None
            or exposure.variant.factor != EXPOSURE_SCALE_VARIANTS.get(variant_id)
        ):
            raise ValueError(
                f"exposure variant {variant_id!r} must use the catalog scale factor"
            )
        varied = SAFEGUARD_LEVEL_FIELDS[status] if family == "safeguard_level" else ()
        if (
            _held_terms(instrument, varied)
            != _held_terms(representatives[status], varied)
            or {row.model_id for row in instrument.respondents} != models
        ):
            raise ValueError(
                f"{kind.noun} variant must differ from the headline only in its "
                + ("scale factor" if family == "exposure_scale" else "safeguard texts")
                + " and respondent samples"
            )


def _setup_evidence(
    inputs: tuple[PublicFrontierRiskInput, ...],
    first: ContextualRiskInstrument,
    study: _Study,
    exclusions: SampleExclusions,
    variants: _ContextVariants,
    kept: Sequence[_KeptInput],
    *,
    contributions: list,
    large_loss_rows: list,
    lab_loss_weights: Mapping[str, float] | None,
    gross_withheld: str | None,
    reference: CompileOptions,
    max_excluded_sample_share: float,
) -> tuple[EvidenceSet, EvidenceSet]:
    """The setup's exhibits (``frequency_estimate`` and ``large_loss_anchor``)
    and its excluded-sample disclosure, the headline's and each variant's.

    What reports the answers themselves, the dual frequency collection, the
    EAL cross-check's cases and the follow-up counts, reads each kept case
    once; what describes the run's laws, the taxonomy parents' pools and each
    leaf's weighted EALs, reads the cases as the states do (``_KeptInput``).
    ``lab_loss_weights`` are the headline states' weights in the views that
    count the lab's loss where a state is left out of them, and
    ``gross_withheld`` why the gross views have no value, where they have
    none (``excluded_samples.lab_loss_state_weights``)."""
    disclosure = excluded_samples_record(
        exclusions, study.captures(), max_excluded_sample_share
    )
    variant_records: dict[str, dict[str, dict[str, object]]] = {}
    for (family, variant_id), (variant_excluded, variant) in sorted(
        variants.exclusions.items()
    ):
        variant_records.setdefault(f"{family}_variants", {})[variant_id] = (
            excluded_samples_record(
                variant_excluded, variant.captures(), max_excluded_sample_share
            )
        )
    exhibits = frequency_estimate_evidence(
        frequency_estimate_record(
            contributions,
            frequency_collection=frequency_collection_summary(
                (row.kept, row.node_ids) for row in kept
            ),
            eal_crosscheck=eal_crosscheck_summary(
                (
                    row.kept,
                    {
                        node.node_id: node.population.scenario_id
                        for node in contextual_nodes(row.item.instrument)
                    },
                    # Each context's cases sit at its one anchor, weighted as
                    # the run weights the states of their draw that read them.
                    {
                        (leaf, row.item.instrument.reference_eci, key): (
                            row.item.path_weight * weight
                        )
                        for (leaf, key), weight in exclusions.case_weights(
                            _draw_id(row.item)
                        ).items()
                    },
                )
                for row in kept
            ),
            follow_ups=follow_up_counts((row.kept, row.node_ids) for row in kept),
            hierarchy=hierarchy_summary(
                (
                    (
                        row.read,
                        row.item.instrument,
                        {
                            key: row.item.path_weight * weight
                            for key, weight in exclusions.weights[
                                _draw_id(row.item)
                            ].items()
                        },
                    )
                    for row in kept
                ),
                options=reference,
            ),
            lab_loss_weights=lab_loss_weights,
            gross_withheld=gross_withheld,
        ),
        large_loss_record(
            large_loss_rows,
            large_loss_levels_of(
                [node for item in inputs for node in contextual_nodes(item.instrument)]
            ),
            frequency_estimate_views(first),
            lab_loss_weights=lab_loss_weights,
            gross_withheld=gross_withheld,
        ),
    )
    return exhibits, excluded_samples_evidence(disclosure, variant_records)


def _execution_plan(
    setup_id: str,
    first: ContextualRiskInstrument,
    populations: tuple,
    states: list[JointState],
    comparisons: PublicFrontierComparisons,
    variants: _ContextVariants,
    *,
    years_per_state: int,
    seed: int,
    work_allowances: Mapping[str, int],
    tail_gate: Mapping[str, object],
) -> ExecutionPlan:
    """The setup's plan: the reference and every comparison arm, each
    simulating only its positive states (and a variant arm its variant's
    no-progress states too), the study's views and its controls."""
    reference_weights = {state.state_id: state.weight for state in states}
    variant_no_progress_states = variant_no_progress_requirements(variants.cells)
    return ExecutionPlan(
        plan_id=f"{setup_id}:execution",
        required_arms=(
            "reference",
            *(arm.alternative_id for arm in comparisons.alternatives),
        ),
        simulations=years_per_state,
        seed=seed,
        requested_views=(
            "gross",
            "welfare",
            f"lab_loss::{first.lab.entity_id}",
            *(
                f"source_status::{status}::{view}"
                for status in ("public", "not_yet_public")
                for view in ("welfare", f"lab_loss::{first.lab.entity_id}")
            ),
            *(
                f"scenario::{population.scenario_id}::{view}"
                for population in populations
                for view in ("gross", "welfare")
            ),
        ),
        comparison_requirements=comparisons.requirements,
        requested_comparison_families=tuple(
            sorted({row.family_id for row in comparisons.requirements})
        ),
        # Each arm simulates only its positive states, so a state another arm
        # alone weights costs, and narrows, only that arm. A variant arm also
        # simulates its variant's no-progress states at weight zero, so they have
        # draws for capability-conditioned reporting without entering any
        # average.
        required_state_ids_by_arm={
            arm: tuple(state_id for state_id, weight in weights.items() if weight > 0)
            + variant_no_progress_states.get(arm, ())
            for arm, weights in (
                ("reference", reference_weights),
                *(
                    (
                        row.alternative_id,
                        reference_weights
                        if row.state_weights is None
                        else row.state_weights,
                    )
                    for row in comparisons.alternatives
                ),
            )
        },
        numerical_controls={
            # Work allowances per simulated year (NUM-BUDGETS).
            **work_allowances,
            "max_members_per_campaign": 2_000_000,
            "tail_gate": tail_gate,
        },
    )


def _setup_limitations(
    inputs: tuple[PublicFrontierRiskInput, ...],
    first: ContextualRiskInstrument,
    entities,
    context_variants: Mapping[str, Mapping[str, tuple[PublicFrontierRiskInput, ...]]],
    comparisons: PublicFrontierComparisons,
    kept: Sequence[_KeptInput],
    study: _Study,
    exclusions: SampleExclusions,
    variants: _ContextVariants,
    *,
    max_excluded_sample_share: float,
) -> tuple[str, ...]:
    """The setup's limitations: its hypothetical entities, exposure and
    variants, the lab-loss rule, the exhibits' disclosures, excluded samples,
    unanswered scenarios and the public-trajectory conditioning."""
    return (
        *hypothetical_entity_limitations(entities),
        *threat_model_limitations(first),
        *path_linked_exposure_limitation(first),
        *context_variant_limitations(first, context_variants),
        first.lab_compensation_definition,
        first.lab_annual_business_loss_definition,
        LAB_LOSS_TAIL_LIMITATION,
        lab_loss_cap_limitation(
            alternatives_executed=any(
                row.family_id == "lab_loss_cap" and row.status == "required"
                for row in comparisons.requirements
            )
        ),
        large_loss_limitation(tuple(row.kept for row in kept)),
        eal_crosscheck_limitation(tuple(row.kept for row in kept)),
        *hierarchy_factor_limitation(
            (row.read, row.item.instrument, exclusions.weights[_draw_id(row.item)])
            for row in kept
        ),
        excluded_samples_limitation(
            exclusions, study.captures(), max_excluded_sample_share
        ),
        *unanswered_scenarios_limitation(exclusions, max_excluded_sample_share),
        *(
            f"{CONTEXT_VARIANT_FAMILIES[family].noun.split(' ', 1)[1].capitalize()} variant {variant_id}: "
            + excluded_samples_limitation(
                variant_excluded, variant.captures(), max_excluded_sample_share
            )
            + " The variant and its matched control, the headline captures "
            "reweighted to the embedded rule, can exclude different samples, "
            "so the variant's effect also carries that difference."
            for (family, variant_id), (variant_excluded, variant) in sorted(
                variants.exclusions.items()
            )
        ),
        *(
            f"{CONTEXT_VARIANT_FAMILIES[family].noun.split(' ', 1)[1].capitalize()} variant {variant_id}: "
            + text
            for (family, variant_id), (variant_excluded, _) in sorted(
                variants.exclusions.items()
            )
            for text in unanswered_scenarios_limitation(
                variant_excluded, max_excluded_sample_share
            )
        ),
        "Native incident simulation retains the separately elicited lab business-loss law. Frozen work allowances per simulated year refuse a run whose individually drawn events or campaign members exceed them.",
        "Annual laws are elicited conditional on complete shared public trajectories. Internal capability uncertainty is implicit in those judgments.",
        "Shared public paths preserve that source of dependence only; residual incident independence and parameter coupling remain explicit assumptions.",
        "Source status is defined at incident origin, separately from where a model is used; a public model used internally remains public.",
        "The entity path stores the selected lab's public general ECI, or the derived maximum over the forecast lab roster; it does not represent source-model internal capability. All joint lab trajectories are retained in each law's condition, and the annual-kernel multiplier is one.",
        "Cyber and bio risks are elicited conditional on general ECI with their benchmark results retained as contextual evidence. Domain strengths do not independently vary between futures with the same general ECI paths; no domain-specific response curve is converted or reused.",
        "The public forecast is integrated over its trajectory nodes by a fixed quadrature rule. Native loss precision intervals exclude that quadrature error; the sensitivity summary compares the full rule with its embedded lower-order rule on the same nodes as an indicator. No expected capability or mean conditional quantile replaces the mixture of loss distributions.",
        "Reported loss moments describe the integrated node mixture. Their existence does not establish finite expected loss or variance under the continuous public forecast; that requires a joint tail assessment of capability, risk response and parameter uncertainty. Tail paths between and beyond the nodes are represented only through the rule's node weights.",
        *sorted(
            {
                limitation
                for item in inputs
                if item.instrument.public_trajectory is not None
                for limitation in item.instrument.public_trajectory.limitations
            }
        ),
    )
