"""Compile a captured contextual study's respondents and its constant-path setup.

Each kept respondent's selection holds its pooled answers after the EAL
cross-checks and the taxonomy parents' pools
(``contextual_respondent_selections``), which both setup compilers read;
``compile_contextual_setup`` compiles a constant-path study into its run setup.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from types import MappingProxyType
from typing import Any

from catmodel.contracts.conditional import (
    ConditionalComponent,
    ConditionalContext,
    ConditionalKernel,
)
from catmodel.contracts.contextual_risk import (
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskCase,
    ContextualRiskInstrument,
    ContextualRiskNode,
)
from catmodel.contracts.elicitation import (
    Quantiles,
)
from catmodel.contracts.incident import (
    DEFAULT_WORK_ALLOWANCES,
    DEFAULT_YEARS_PER_STATE,
    EntityPath,
    ExecutionPlan,
    IncidentLaw,
    JointState,
    PopulationAbsence,
)
from catmodel.contracts.run_setup import (
    AssumptionSet,
    EvidenceSet,
    RunSetup,
)
from catmodel.elicitation.contextual_frequency import (
    chain_count_law_for_answer,
    chain_guardrail_replaces,
    frequency_collection_summary,
    severity_retry_replaces,
    validate_contextual_artifact,
)
from catmodel.elicitation.eal_crosscheck import (
    eal_crosscheck_limitation,
    eal_crosscheck_summary,
    final_answer,
    follow_up_counts,
    pooled_answer,
    pooled_answer_before_chain_guardrail,
    pooled_answer_before_crosscheck,
    pooled_answer_before_severity_retry,
    stated_large_loss,
)
from catmodel.setup.contextual_compilation import (
    FREQUENCY_ESTIMATES,
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
    hierarchy_outcomes,
    hierarchy_summary,
    validate_hierarchy_cases,
)
from catmodel.setup.contextual_node import (
    CompileOptions,
    ContextualLawUnavailable,
    admit_node,
    compile_contextual_node,
    compile_once,
    constant_path,
    contextual_independence,
    contextual_limitations,
    contextual_scope,
)
from catmodel.setup.excluded_samples import (
    MAX_EXCLUDED_SAMPLE_SHARE,
    Donors,
    SampleExclusions,
    excluded_samples_evidence,
    excluded_samples_limitation,
    excluded_samples_record,
    lab_loss_state_weights,
    sample_exclusions,
    unanswered_scenarios_limitation,
)
from catmodel.setup.implied_eal import validate_eal_crosschecks
from catmodel.setup.sensitivity_catalog import (
    FIXED_ANCHOR_UNAVAILABLE_CHALLENGES,
    unavailable_challenge_requirements,
)
from catmodel.simulation.tail_gate import tail_gate_controls


def _frequency_alternatives(
    case: ContextualRiskCase,
    parents_only: float | None,
    original_chain_count: Quantiles | None,
) -> tuple[
    dict[str, Quantiles | None],
    dict[str, ContextualRiskAnswer | None],
]:
    """A case's count laws and answers for each frequency exhibit arm.

    ``parents_only`` is the leaf's cumulative factor with weight one at every
    taxonomy parent (``HierarchyOutcome.parents_only``), None where the
    instrument asks no parents. ``original_chain_count`` is the count law of
    the chain answer as first given where a guardrail retry replaced it,
    otherwise None. A case whose retry replaced nothing keeps the pooled
    answer itself in that retry's arm, so the arm compiles no law of its own
    for it (``compile_once``).
    """
    direct = final_answer(case)
    direct_rate = direct.rate if direct.status == "available" else None
    chain = pooled_answer(case, weight=1.0)
    unchecked = pooled_answer_before_crosscheck(case)
    pooled = pooled_answer(case)
    unretried = (
        pooled_answer_before_severity_retry(case)
        if severity_retry_replaces(case)
        else pooled
    )
    unguarded = (
        pooled
        if original_chain_count is None
        else pooled_answer_before_chain_guardrail(case, original_chain_count)
    )
    category_answers = (
        None
        if parents_only is None or pooled.rate is None
        else _scale_answer(pooled, parents_only)
    )
    answers = {
        "direct": direct,
        "chain": chain,
        "before_severity_retry": unretried,
        "before_chain_guardrail": unguarded,
        "before_eal_crosscheck": unchecked,
        "before_hierarchy": pooled,
        "hierarchy_parents_only": category_answers,
    }
    if direct_rate is None:
        return dict.fromkeys(FREQUENCY_ESTIMATES), answers
    counts = {
        arm: None if answer is None or answer.status != "available" else answer.rate
        for arm, answer in answers.items()
    }
    return counts, answers


def _scaled(rate: Quantiles, factor: float) -> Quantiles:
    return Quantiles(
        **{key: value * factor for key, value in rate.model_dump().items()}
    )


def _scale_answer(answer: ContextualRiskAnswer, factor: float) -> ContextualRiskAnswer:
    """Scale an answer's count and large-loss counts by one frequency factor."""
    if factor == 1 or answer.rate is None:
        return answer
    return answer.model_copy(
        update={
            key: _scaled(value, factor)
            for key in ("rate", "large_loss_count", "very_large_loss_count")
            if (value := getattr(answer, key)) is not None
        }
    )


def checked_hierarchy_outcomes(
    instrument: ContextualRiskInstrument, artifact: ContextualRiskArtifact
) -> dict[tuple[float, tuple[str, str, int]], HierarchyOutcome]:
    """A collection's taxonomy-parent outcomes, after checking its answers,
    EAL cross-checks and parent cases against its instrument.

    The checks and the outcomes compile every case's law at unit count, the
    larger part of reading a collection, and a process keeps those laws only
    while it lives (``contextual_node.unit_law``). A caller that reads a
    collection in one process and selects from it in another carries these
    outcomes across instead of checking the collection again.
    """
    validate_contextual_artifact(artifact, instrument)
    validate_eal_crosschecks(artifact, instrument)
    outcomes = hierarchy_outcomes(instrument, artifact)
    validate_hierarchy_cases(artifact, instrument, outcomes)
    return outcomes


def selection_cases(
    instrument: ContextualRiskInstrument,
    artifact: ContextualRiskArtifact,
    exclusions: SampleExclusions,
    node: str,
    outcomes: Mapping[tuple[float, tuple[str, str, int]], HierarchyOutcome]
    | None = None,
) -> tuple[
    ContextualRiskArtifact,
    Mapping[tuple[float, tuple[str, str, int]], HierarchyOutcome] | None,
]:
    """The cases and taxonomy-parent outcomes from which the selections of one
    capture of ``node`` are made (``contextual_respondent_selections``).

    Where no sample of the node borrows a leaf, these are the collection
    itself and ``outcomes`` as given. Where one does
    (``POOL-UNANSWERED-SCENARIOS``), they are the cases the node's states
    read (``SampleExclusions.run_view``) and the collection's outcomes,
    checked here unless given, with every borrowing respondent's groups
    replayed from those cases: its pools then read each borrowed leaf beside
    its own leaves and its own parent answers. The view is never checked as
    a collection; the stored collection is.
    """
    borrowers = exclusions.borrowers(node)
    if not borrowers:
        return artifact, outcomes
    if outcomes is None:
        outcomes = checked_hierarchy_outcomes(instrument, artifact)
    view = exclusions.run_view(node, artifact)
    return view, {**outcomes, **hierarchy_outcomes(instrument, view, borrowers)}


def contextual_respondent_selections(
    instrument: ContextualRiskInstrument,
    artifact: ContextualRiskArtifact,
    weights: Mapping[tuple[str, str, int], float],
    *,
    outcomes: Mapping[tuple[float, tuple[str, str, int]], HierarchyOutcome]
    | None = None,
    donors: Donors = MappingProxyType({}),
) -> tuple[ContextualRespondentSelection, ...]:
    """One selection per kept respondent, reusable across paths and scores.

    ``weights`` holds each kept respondent's pooled weight
    (``excluded_samples.pooled_weights``); a respondent it omits, excluded for
    an unanswered leaf, has no selection. A selection keeps its place in the
    complete roster as its ID, so a state's ID and random streams do not
    depend on which other samples are excluded.

    ``outcomes`` are the collection's own (``checked_hierarchy_outcomes``),
    from a caller that has checked it; without them the collection is checked
    here. ``artifact`` and ``outcomes`` are ``selection_cases``'s where a
    sample of the node borrows a leaf, and ``donors`` then names, by leaf and
    borrowing respondent, the respondent whose cases a selection's borrowed
    answers are.
    """
    if outcomes is None:
        outcomes = checked_hierarchy_outcomes(instrument, artifact)
    chains = {node.node_id: node.frequency_chain for node in instrument.nodes}
    by_case, severity, alternatives, alternative_answers, stated = {}, {}, {}, {}, {}
    for case in artifact.cases:
        if case.respondent.key() not in weights:
            continue
        outcome = outcomes.get((case.anchor_score, case.respondent.key()))
        answer = pooled_answer(case)
        factors = None if outcome is None else outcome.factors.get(case.node_id)
        factor = 1.0 if factors is None else factors.frequency
        # The pools' factor scales the count and both large-loss counts
        # alike, so the large-loss share, and the law, are unchanged.
        by_case[case.key()] = _scale_answer(answer, factor)
        severity[case.key()] = 1.0 if factors is None else factors.severity
        stated[case.key()] = stated_large_loss(case)
        original_chain_count = None
        if chain_guardrail_replaces(case):
            chain = chains[case.node_id]
            assert chain is not None
            original_chain_count = chain_count_law_for_answer(
                case.chain_answer, instrument.chain_exposure(chain).count
            )
        alternatives[case.key()], alternative_answers[case.key()] = (
            _frequency_alternatives(
                case,
                None if outcome is None else outcome.parents_only.get(case.node_id),
                original_chain_count,
            )
        )
    nodes = sorted(instrument.nodes, key=lambda node: node.node_id)
    respondents = sorted(
        instrument.respondents, key=lambda respondent: respondent.key()
    )
    if not set(weights) <= {respondent.key() for respondent in respondents}:
        raise ValueError("selection weights name respondents outside the roster")
    return tuple(
        ContextualRespondentSelection(
            selection_id=f"respondent-{respondent_index}",
            respondent=respondent,
            weight=weights[respondent.key()],
            answers_by_node=MappingProxyType(
                {
                    node.node_id: MappingProxyType(
                        {
                            anchor: by_case[node.node_id, anchor, respondent.key()]
                            for anchor in instrument.anchor_scores
                        }
                        if node.unavailable_reason is None
                        else {}
                    )
                    for node in nodes
                }
            ),
            frequency_answers_by_node=MappingProxyType(
                {
                    node.node_id: MappingProxyType(
                        {
                            anchor: MappingProxyType(
                                alternative_answers[
                                    node.node_id, anchor, respondent.key()
                                ]
                            )
                            for anchor in instrument.anchor_scores
                        }
                        if node.unavailable_reason is None
                        else {}
                    )
                    for node in nodes
                }
            ),
            severity_factors_by_node=MappingProxyType(
                {
                    node.node_id: MappingProxyType(
                        {
                            anchor: severity[node.node_id, anchor, respondent.key()]
                            for anchor in instrument.anchor_scores
                        }
                        if node.unavailable_reason is None
                        else {}
                    )
                    for node in nodes
                }
            ),
            frequency_alternatives_by_node=MappingProxyType(
                {
                    node.node_id: MappingProxyType(
                        {
                            anchor: MappingProxyType(
                                alternatives[node.node_id, anchor, respondent.key()]
                            )
                            for anchor in instrument.anchor_scores
                        }
                        if node.unavailable_reason is None
                        else {}
                    )
                    for node in nodes
                }
            ),
            large_loss_by_node=MappingProxyType(
                {
                    node.node_id: MappingProxyType(
                        {
                            anchor: stated[node.node_id, anchor, respondent.key()]
                            for anchor in instrument.anchor_scores
                        }
                        if node.unavailable_reason is None
                        else {}
                    )
                    for node in nodes
                }
            ),
            donors_by_node=MappingProxyType(
                {
                    leaf: donor
                    for (leaf, borrower), donor in donors.items()
                    if borrower == respondent.key()
                }
            ),
        )
        for respondent_index, respondent in enumerate(respondents)
        if respondent.key() in weights
    )


def _compile_admitted(
    instrument: ContextualRiskInstrument,
    node: ContextualRiskNode,
    path: EntityPath,
    score: float,
    options: CompileOptions,
    answers: Mapping[float, ContextualRiskAnswer],
    severity_factors: Mapping[float, float] | None,
    **changes: Any,
) -> IncidentLaw | PopulationAbsence:
    """One node's law at one state, admitted like every other run law;
    ``changes`` replace some of ``options`` (``compile_once``)."""
    return admit_node(
        instrument,
        node,
        path,
        compile_contextual_node(
            instrument,
            node,
            answers,
            score,
            severity_factors=severity_factors,
            options=replace(options, **changes),
        ),
    )


def compile_contextual_setup(
    instrument: ContextualRiskInstrument,
    artifact: ContextualRiskArtifact,
    *,
    setup_id: str,
    years_per_state: int = DEFAULT_YEARS_PER_STATE,
    max_reference_state_years: int | None = None,
    work_allowances: Mapping[str, int] = DEFAULT_WORK_ALLOWANCES,
    epistemic_tail_scale: float = 1.0,
    seed: int = 42,
    max_excluded_sample_share: float = MAX_EXCLUDED_SAMPLE_SHARE,
    files: Mapping[str, bytes] = MappingProxyType({}),
) -> RunSetup:
    """Generate a run setup and dense point kernels from validated answers.

    The only positive marginal mass is on the exact authored reference score.
    Every modeled integer is executed even when its marginal mass is zero.
    The setup runs only its reference arm, so its tail gate caps only the
    reference arm's simulated state-years.
    ``work_allowances`` sets the plan's work allowances per simulated year.
    A model whose respondents lacking one leaf, at any anchor, exceed
    ``max_excluded_sample_share`` does not answer that leaf, and its
    respondents read other models' answers to it; any other respondent with
    an unanswered leaf is excluded at every coordinate, within the same
    share (``excluded_samples``). ``files``, the capture's, give each
    exhausted call's cause.
    """
    if instrument.public_trajectory is not None:
        raise ValueError(
            "public trajectory conditions require the joint public-frontier compiler"
        )
    if (
        isinstance(years_per_state, bool)
        or not isinstance(years_per_state, int)
        or years_per_state < 1
    ):
        raise ValueError("years_per_state must be an integer at least 1")
    tail_gate = tail_gate_controls(max_reference_state_years, None)
    options = CompileOptions(epistemic_tail_scale=epistemic_tail_scale)
    # A fixed-anchor setup is one node of samples: its context.
    study_node = instrument.context_id
    exclusions = sample_exclusions(
        {study_node: ((instrument, artifact),)},
        {study_node: 1.0},
        max_share=max_excluded_sample_share,
    )
    weights = exclusions.weights[study_node]
    # The exhibits count the kept cases, each once, and describe the laws of
    # the cases the states read, a borrowed case in each state that reads it.
    kept = exclusions.kept_view(study_node, artifact)
    read = exclusions.run_view(study_node, artifact)
    cases, outcomes = selection_cases(instrument, artifact, exclusions, study_node)
    selections = contextual_respondent_selections(
        instrument,
        cases,
        weights,
        outcomes=outcomes,
        donors=exclusions.donors[study_node],
    )
    nodes = contextual_nodes(instrument)
    coordinates = sorted(
        {
            *range(instrument.integer_min, instrument.integer_max + 1),
            instrument.reference_eci,
        }
    )
    # A left-out sample is left out at every coordinate.
    left_out = exclusions.left_out(study_node)
    states = []
    contributions = []
    large_loss_rows = []
    components: dict[int, list[ConditionalComponent]] = {
        score: [] for score in range(instrument.integer_min, instrument.integer_max + 1)
    }
    for selection in selections:
        group = selection.selection_id
        for coordinate_index, score in enumerate(coordinates):
            identifier = f"{instrument.context_id}:score-{coordinate_index}:{group}"
            path = constant_path(instrument, score)
            laws, absent = [], []
            weight = selection.weight if score == instrument.reference_eci else 0
            for node in nodes:
                compile_law = compile_once(
                    partial(
                        _compile_admitted,
                        instrument,
                        node,
                        path,
                        score,
                        options,
                    )
                )
                selected = compile_law(
                    selection.answers_by_node[node.node_id],
                    selection.severity_factors_by_node[node.node_id],
                )
                if weight > 0:
                    contributions.append(
                        (
                            identifier,
                            weight,
                            node.population.scenario_id,
                            frequency_contribution(
                                instrument,
                                node,
                                selection,
                                score,
                                selected,
                                compile_law=compile_law,
                                options=options,
                            ),
                        )
                    )
                    large_loss_rows.append(
                        (
                            identifier,
                            weight,
                            node.population.scenario_id,
                            large_loss_contribution(
                                instrument,
                                node,
                                selection,
                                score,
                                selected,
                                compile_law=compile_law,
                            ),
                        )
                    )
                if isinstance(selected, PopulationAbsence):
                    absent.append(selected)
                else:
                    laws.append(selected)
            if not laws and not any(
                row.disposition == "structural_zero" for row in absent
            ):
                raise ContextualLawUnavailable(
                    f"No requested observable can be captured at ECI {score:g} for "
                    f"respondent {selection.respondent.key()}: "
                    + "; ".join(f"{row.scenario_id}: {row.reason}" for row in absent)
                )
            states.append(
                JointState(
                    state_id=identifier,
                    epistemic_group_id=group,
                    # One coordinate and one respondent model
                    # (NUM-RESPONDENT-SAMPLING).
                    sampling_group_id=f"{instrument.context_id}:score-{coordinate_index}:{selection.respondent.model_id}",
                    borrowed_leaves=tuple(sorted(selection.donors_by_node)),
                    lab_loss_left_out=selection.respondent.key() in left_out,
                    weight=weight,
                    entity_paths=(path,),
                    laws=tuple(laws),
                    absent_populations=tuple(absent),
                )
            )
            if score in components:
                components[int(score)].append(
                    ConditionalComponent(state_id=identifier, weight=selection.weight)
                )
    context = ConditionalContext(
        context_id=instrument.context_id,
        lab_id=instrument.lab.entity_id,
        deployment_id=instrument.deployment.entity_id,
        entity_id=instrument.model.entity_id,
        scale_id=instrument.capability_scale.scale_id,
        coordinate_kind="constant_path",
        integer_min=instrument.integer_min,
        integer_max=instrument.integer_max,
        required_arms=("reference",),
        kernels=tuple(
            ConditionalKernel(eci=score, arm_id="reference", components=tuple(rows))
            for score, rows in components.items()
        ),
    )
    plan = ExecutionPlan(
        plan_id=f"{setup_id}:execution",
        required_arms=("reference",),
        simulations=years_per_state,
        seed=seed,
        requested_views=(
            "gross",
            "welfare",
            f"lab_loss::{instrument.lab.entity_id}",
            *(
                f"scenario::{node.population.scenario_id}::{measure}"
                for node in nodes
                for measure in ("gross", "welfare")
            ),
        ),
        # Every state, those at zero mass included: the conditional kernels
        # at each modeled integer read them, so each costs and counts.
        required_state_ids_by_arm={
            "reference": tuple(state.state_id for state in states)
        },
        numerical_controls={
            # Work allowances per simulated year (NUM-BUDGETS).
            **work_allowances,
            "max_members_per_campaign": 2_000_000,
            "tail_gate": tail_gate,
        },
        comparison_requirements=unavailable_challenge_requirements(
            FIXED_ANCHOR_UNAVAILABLE_CHALLENGES,
            tuple(node.population.scenario_id for node in nodes),
        ),
        requested_comparison_families=FIXED_ANCHOR_UNAVAILABLE_CHALLENGES,
    )
    # The exhibits weigh the states in each view as the run does.
    lab_weights, gross_withheld = lab_loss_state_weights(states)
    exhibits = frequency_estimate_evidence(
        frequency_estimate_record(
            contributions,
            frequency_collection=frequency_collection_summary(
                ((kept, {node.node_id for node in nodes}),)
            ),
            eal_crosscheck=eal_crosscheck_summary(
                (
                    (
                        kept,
                        {node.node_id: node.population.scenario_id for node in nodes},
                        # The run weights the states at the reference score,
                        # and a case as the states that read it.
                        {
                            (leaf, instrument.reference_eci, key): weight
                            for (leaf, key), weight in exclusions.case_weights(
                                study_node
                            ).items()
                        },
                    ),
                )
            ),
            follow_ups=follow_up_counts(((kept, {node.node_id for node in nodes}),)),
            hierarchy=hierarchy_summary(((read, instrument, weights),)),
            lab_loss_weights=lab_weights,
            gross_withheld=gross_withheld,
        ),
        large_loss_record(
            large_loss_rows,
            large_loss_levels_of(nodes),
            frequency_estimate_views(instrument),
            lab_loss_weights=lab_weights,
            gross_withheld=gross_withheld,
        ),
    )
    disclosure = excluded_samples_evidence(
        excluded_samples_record(
            exclusions, ((artifact, files),), max_excluded_sample_share
        )
    )
    return RunSetup(
        setup_id=setup_id,
        scope=contextual_scope(
            instrument,
            scope_id=setup_id,
            title=f"Captured contextual risk: {instrument.context_id}",
            populations=tuple(node.population for node in nodes),
        ),
        evidence=EvidenceSet(
            records=(*exhibits.records, *disclosure.records),
            files={**exhibits.files, **disclosure.files},
        ),
        joint_states=tuple(states),
        conditional_contexts=(context,),
        assumptions=AssumptionSet(independence=contextual_independence(instrument)),
        execution=plan,
        limitations=(
            *contextual_limitations(instrument),
            lab_loss_cap_limitation(alternatives_executed=False),
            large_loss_limitation((kept,)),
            eal_crosscheck_limitation((kept,)),
            *hierarchy_factor_limitation(((read, instrument, weights),)),
            excluded_samples_limitation(
                exclusions, ((artifact, files),), max_excluded_sample_share
            ),
            *unanswered_scenarios_limitation(exclusions, max_excluded_sample_share),
        ),
    )
