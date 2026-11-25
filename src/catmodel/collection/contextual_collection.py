"""Collect lab/deployment-conditioned answers through the frozen-call transport."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Generator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

from catmodel.artifacts import freeze_bytes, freeze_json, remove_temporary_files
from catmodel.collection.answer_table import AnswerTable
from catmodel.collection.collection_scheduler import (
    CollectionScheduler,
    watch_event_loop,
)
from catmodel.collection.collection_workers import (
    collect_in_processes,
    collector_count,
)
from catmodel.collection.frozen_call_collection import (
    ProviderFailed,
    ValidationAttemptsExhausted,
    collect_frozen_call,
    failed_calls,
    provider_failure_report,
)
from catmodel.contracts.contextual_risk import (
    ContextualParentCase,
    ContextualRiskAnswer,
    ContextualRiskArtifact,
    ContextualRiskCase,
    ContextualRiskInstrument,
    EalCrosscheck,
    ParentAnswer,
    UnansweredCase,
)
from catmodel.elicitation.contextual_frequency import (
    ContextualChainPrompt,
    chain_count_law_for_answer,
    validate_contextual_artifact,
)
from catmodel.elicitation.contextual_hierarchy import (
    ContextualParentPrompt,
    parent_prompts,
)
from catmodel.elicitation.contextual_risk import (
    ContextualPrompt,
    contextual_prompts,
)
from catmodel.elicitation.eal_crosscheck import crosscheck_prompt, pooled_answer
from catmodel.elicitation.question_flow import (
    DIRECT,
    DIRECT_ANSWERED,
    PARENT,
    Ask,
    chain_path,
    crosscheck_path,
    direct_path,
)
from catmodel.evidence.source_bundle import (
    PROVENANCE_PATH,
    capture_source_bundle_snapshot,
)
from catmodel.frozen_calls import (
    PARSE_ATTEMPTS,
    PROVIDER_FAILURES_PER_CALL,
    frozen_call_key,
)
from catmodel.json_io import strict_json_loads
from catmodel.setup.contextual_hierarchy import (
    RecordedParent,
    hierarchy_outcome,
    recorded_parents,
    validate_hierarchy_cases,
)
from catmodel.setup.contextual_inputs import (
    contextual_request,
    contextual_requests,
    load_contextual_collection,
    validate_contextual_sources,
)
from catmodel.setup.implied_eal import (
    crosscheck_reconciliation,
    recorded_crosscheck,
    validate_eal_crosschecks,
)
from catmodel.worker_pool import WorkerPool, sigterm_exits

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from catmodel.config import ExecutionSettings, ModelProfile
    from catmodel.llm.models import ModelSpec
    from catmodel.llm.transport import LLMProvider


def _providers_by_model(
    instrument: ContextualRiskInstrument, providers: Sequence[LLMProvider]
) -> dict[str, LLMProvider]:
    by_model = {provider.model_id: provider for provider in providers}
    if len(by_model) != len(providers) or set(by_model) != {
        r.model_id for r in instrument.respondents
    }:
        raise ValueError("providers must match the exact instrument model roster")
    return by_model


def _first(error: BaseException) -> BaseException:
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error


async def _together(*awaitables: Awaitable[Any]) -> list[Any]:
    """Await all at once. The first failure cancels the rest, so a failed
    case, group or collection sends nothing more, and is raised as itself.
    A transient provider failure (``ProviderFailed``) cancels nothing: the
    rest finish first, so no call in flight is lost, and the first such
    failure is raised then."""
    held: list[ProviderFailed] = []

    async def awaited(awaitable: Awaitable[Any]) -> Any:
        try:
            return await awaitable
        except ProviderFailed as failure:
            if not failure.transient:
                raise
            held.append(failure)
            return None

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(awaited(item)) for item in awaitables]
    except BaseExceptionGroup as failure:
        raise _first(failure) from None
    if held:
        raise held[0]
    return [task.result() for task in tasks]


def request_rank(remaining: int, collection: int, group: int) -> tuple[int, int, int]:
    """A request's place in its model's lane; lower goes first.

    Critical path first: the request with the most turns left (``remaining``;
    ``QuestionKind.turns``) goes first; among equal lengths, earlier
    collections and groups go first.
    """
    return (-remaining, collection, group)


async def collect_contextual_risk(
    *,
    instrument: ContextualRiskInstrument,
    run_directory: Path,
    providers: Sequence[LLMProvider],
    config: ExecutionSettings,
    scheduler: CollectionScheduler | None = None,
    collection: int = 0,
    answers: AnswerTable | None = None,
) -> ContextualRiskArtifact:
    """Collect numerical judgments on the instrument's authored definitions.

    Each node's direct and chain questions start independent paths: the direct
    path can add a large-loss coherence retry and an EAL cross-check, the chain
    path one guardrail retry. ``catmodel.elicitation.question_flow`` holds each
    path's sequence, and ``run`` asks what it yields. A resumed run reuses the
    stored attempts.

    A call whose every validation attempt fails is exhausted: it is recorded
    with answer None and collection goes on (``POOL-EXCLUDED-SAMPLES``). An
    exhausted direct question leaves its leaf unanswered (``UnansweredCase``):
    nothing more is asked about the leaf, and its (anchor, respondent) group
    is incomplete, so the group's parents are answered but not pooled here;
    import excludes the states the group belongs to, or gives them another
    model's answers to a leaf their own model does not answer
    (``POOL-UNANSWERED-SCENARIOS``) and then pools the parents itself. An
    exhausted chain question leaves no chain count; an exhausted guardrail
    retry or cross-check leaves the answers it would have revised standing,
    and an exhausted parent question leaves what is beneath the parent
    standing.

    A direct answer whose large-loss count contradicts the loss quantiles
    (``SEV-LARGE-LOSS-ANCHOR``) gets one coherence retry, its own frozen call;
    a resolving retry replaces only the counts and loss quantiles, and one that
    still contradicts or is exhausted leaves the original answers standing as
    ``severity_unresolved``.

    Each case then records its leaf EAL cross-check (``FREQ-EAL-CROSSCHECK``):
    the direct EAL's mean beside the EAL its direct count and severity imply.
    Only a case whose EALs differ by more than the cross-check factor gets the
    cross-check call; a revision of its count or loss judgments gets one
    coherence retry. The cross-checked direct count is pooled with the chain
    count law before the taxonomy parents are pooled with the leaves.

    Each taxonomy parent is asked once, beside its group's leaves, and no call
    follows its answer: once the group's leaves are final and its parents
    answered, each parent's answer is pooled with what is beneath it by a
    fixed rule (``HIER-CATEGORY-POOL``), whatever other groups are doing, and
    each parent case records the pool. A call that fails
    transiently at the provider, once the transport's own retries ran out
    (``ProviderFailed``), fails its group only: the group asks nothing that
    depends on the failed call, every other call goes on, and the collection
    then raises ``ProviderFailed`` naming every failed call. A resumed
    collection sends each failed call again as a new attempt and replays the
    rest; a collection with a call that already stored
    ``PROVIDER_FAILURES_PER_CALL`` failures fails before sending anything and
    needs a new run directory. Any other failure, a systematic provider
    failure above all, cancels every other call of the collection.

    Requests go to providers through ``scheduler``'s lanes, shared with every
    other collection given the same scheduler and ranked by ``request_rank``
    with ``collection`` as this collection's place; without one, this
    collection has lanes of its own. Every call reads and feeds the answer
    table ``answers``, when given (``collect_frozen_call``).
    """
    by_model = _providers_by_model(instrument, providers)
    failed = failed_calls(run_directory)
    if failed:
        raise ValueError(
            f"{len(failed)} stored calls failed at the provider "
            f"{PROVIDER_FAILURES_PER_CALL} times, the most one call may store; "
            f"start a new run: {', '.join(failed)}"
        )
    provider_failed: list[str] = []
    lanes = scheduler or CollectionScheduler(
        {model: config.concurrency_by_model[model] for model in by_model}
    )
    groups = {
        (score, respondent.key()): index
        for index, (score, respondent) in enumerate(
            (score, respondent)
            for score in instrument.anchor_scores
            for respondent in instrument.respondents
        )
    }

    async def call(question: Ask, group: int):
        """One question's answer from its frozen call; None when the call is
        exhausted. Any other failure, a provider failure above all, raises; a
        failed call is listed in ``provider_failed``."""
        prompt, kind = question.prompt, question.kind
        rank = request_rank(kind.turns, collection, group)
        # Rendering, replaying and freezing a request hold the event loop:
        # one request at a time, with the loop served in between.
        await lanes.turn()
        request = contextual_request(prompt)
        try:
            return await collect_frozen_call(
                run_directory=run_directory,
                request=request,
                provider=lanes.provider(by_model[prompt.respondent.model_id], rank),
                config=config,
                parse_answer=partial(kind.parser, prompt=prompt),
                response_model=kind.response_model,
                answers=answers,
            )
        except ValidationAttemptsExhausted:
            logger.warning(
                "call exhausted its validation attempts; collection continues",
                run_directory=str(run_directory),
                call=frozen_call_key(request),
                question=type(prompt).__name__,
                respondent=prompt.respondent.key(),
            )
            return None
        except ProviderFailed as failure:
            logger.warning(
                "call failed at the provider; its group is incomplete and "
                "resume sends it again",
                run_directory=str(run_directory),
                call=frozen_call_key(request),
                question=type(prompt).__name__,
                respondent=prompt.respondent.key(),
                transient=failure.transient,
                error=str(failure),
            )
            provider_failed.append(frozen_call_key(request))
            raise

    async def run(
        path: Generator[Any, Any, Any],
        group: int,
        direct: asyncio.Future[Any] | None = None,
    ) -> Any:
        """Drive one path of ``question_flow``: ask each question it yields,
        send it the answer and return what the path returns.

        ``direct`` is the leaf's direct answer, set as soon as its direct
        question has one: the leaf's chain path waits for it."""
        reply = None
        while True:
            try:
                step = path.send(reply)
            except StopIteration as finished:
                return finished.value
            if step is DIRECT_ANSWERED:
                assert direct is not None
                reply = (await direct) is not None
            elif step.kind is DIRECT:
                assert direct is not None
                try:
                    reply = await call(step, group)
                except BaseException:
                    # A failed direct question asks no guardrail retry; its
                    # case fails with it.
                    direct.set_result(None)
                    raise
                direct.set_result(reply)
            else:
                reply = await call(step, group)

    async def collect(prompt):
        group = groups[prompt.anchor_score, prompt.respondent.key()]
        chain_prompt = ContextualChainPrompt(
            instrument, prompt.node, prompt.respondent, prompt.anchor_score
        )
        direct_ready = asyncio.get_running_loop().create_future()

        async def direct_side():
            direct = await run(direct_path(prompt), group, direct_ready)
            if direct is None:
                return None, None
            return direct, await crosscheck(prompt, direct.stated)

        direct_result, chain_result = await _together(
            direct_side(), run(chain_path(chain_prompt), group, direct_ready)
        )
        direct, eal_crosscheck = direct_result
        chain, guardrail = chain_result
        if direct is None:
            return UnansweredCase(
                node_id=prompt.node.node_id,
                anchor_score=prompt.anchor_score,
                respondent=prompt.respondent,
                prompt_sha256=prompt.prompt_sha256(),
                chain_answer=chain,
                chain_prompt_sha256=chain_prompt.prompt_sha256(),
            )
        driving_chain = chain
        if (
            guardrail is not None
            and guardrail.answer is not None
            and guardrail.answer.status == "available"
        ):
            driving_chain = guardrail.answer
        return ContextualRiskCase(
            node_id=prompt.node.node_id,
            anchor_score=prompt.anchor_score,
            respondent=prompt.respondent,
            answer=direct.answer,
            prompt_sha256=prompt.prompt_sha256(),
            chain_answer=chain,
            chain_prompt_sha256=chain_prompt.prompt_sha256(),
            chain_guardrail=guardrail,
            chain_count=chain_count_law_for_answer(
                driving_chain, chain_prompt.exposure.count
            ),
            large_loss_share=direct.large_loss_share,
            severity_retry=direct.severity_retry,
            eal_crosscheck=eal_crosscheck,
        )

    async def crosscheck(
        prompt: ContextualPrompt, answer: ContextualRiskAnswer
    ) -> EalCrosscheck | None:
        """Check the direct answer's EAL, independent of the chain answer."""
        group = groups[prompt.anchor_score, prompt.respondent.key()]
        recorded = recorded_crosscheck(
            instrument, prompt.node, answer, prompt.anchor_score
        )
        if recorded is None:
            return None
        question = crosscheck_prompt(
            instrument,
            prompt.node,
            prompt.respondent,
            prompt.anchor_score,
            answer,
            recorded,
        )
        if question is None:
            return recorded
        reply, retry = await run(crosscheck_path(question), group)
        return recorded.model_copy(
            update={"reconciliation": crosscheck_reconciliation(question, reply, retry)}
        )

    def pooled(anchor_score, leaf_cases, mine) -> dict[str, RecordedParent]:
        """What each parent case of a complete group stores: the child EALs
        going into its pool and the pool, each parent after every parent
        beneath it."""
        leaves = {case.node_id: pooled_answer(case) for case in leaf_cases}
        return recorded_parents(
            hierarchy_outcome(instrument, leaves, mine, anchor_score)
        )

    def parent_cases(
        anchor_score, respondent, mine, stored: dict[str, RecordedParent] | None
    ) -> list[ContextualParentCase]:
        """A group's parent cases; without ``stored`` values, those of an
        incomplete group, never pooled."""
        return [
            ContextualParentCase(
                parent_id=parent_id,
                anchor_score=anchor_score,
                respondent=respondent,
                answer=answer,
                prompt_sha256=ContextualParentPrompt(
                    instrument,
                    next(
                        row for row in instrument.parents if row.parent_id == parent_id
                    ),
                    respondent,
                    anchor_score,
                ).prompt_sha256(),
                children_median_eal=None
                if stored is None
                else stored[parent_id].children_median_eal,
                children_mean_eal=None
                if stored is None
                else stored[parent_id].children_mean_eal,
                pool=None if stored is None else stored[parent_id].pool,
            )
            for parent_id, answer in mine.items()
        ]

    leaf_prompts = contextual_prompts(instrument)
    top_prompts = parent_prompts(instrument)

    async def collect_group(anchor_score, respondent):
        """One group's leaves and parent answers, then its parents' pools;
        an incomplete group, one with an unanswered leaf, is not pooled."""
        group = groups[anchor_score, respondent.key()]
        own = [
            prompt
            for prompt in leaf_prompts
            if prompt.anchor_score == anchor_score and prompt.respondent == respondent
        ]
        tops = [
            prompt
            for prompt in top_prompts
            if prompt.anchor_score == anchor_score and prompt.respondent == respondent
        ]
        collected = await _together(
            *(collect(prompt) for prompt in own),
            *(call(Ask(PARENT, prompt), group) for prompt in tops),
        )
        leaves = collected[: len(own)]
        leaf_cases = [row for row in leaves if isinstance(row, ContextualRiskCase)]
        unanswered = [row for row in leaves if isinstance(row, UnansweredCase)]
        answers = {
            prompt.parent.parent_id: cast(ParentAnswer | None, answer)
            for prompt, answer in zip(tops, collected[len(own) :], strict=True)
        }
        if not instrument.parents:
            return leaf_cases, unanswered, []
        if unanswered:
            # Every pool reads the leaves beneath its parent, so none can be
            # computed without the unanswered leaf. Import excludes the
            # group's states, or pools its parents itself once it knows whose
            # answers they read for that leaf (POOL-UNANSWERED-SCENARIOS).
            logger.warning(
                "group incomplete: a direct question exhausted its validation "
                "attempts, so its parents are not pooled",
                run_directory=str(run_directory),
                anchor_score=anchor_score,
                respondent=respondent.key(),
                unanswered=[row.node_id for row in unanswered],
            )
            return (
                leaf_cases,
                unanswered,
                parent_cases(anchor_score, respondent, answers, None),
            )
        return (
            leaf_cases,
            unanswered,
            parent_cases(
                anchor_score,
                respondent,
                answers,
                pooled(anchor_score, leaf_cases, answers),
            ),
        )

    try:
        finished = await _together(
            *(
                collect_group(score, respondent)
                for score in instrument.anchor_scores
                for respondent in instrument.respondents
            )
        )
    except ProviderFailed as failure:
        if not failure.transient:
            raise
        raise ProviderFailed(
            "groups incomplete: "
            + provider_failure_report(run_directory, provider_failed),
            transient=True,
        ) from failure
    by_key = {
        row.key(): row
        for leaf_cases, unanswered, _ in finished
        for row in (*leaf_cases, *unanswered)
    }
    ordered = [
        by_key[prompt.node.node_id, prompt.anchor_score, prompt.respondent.key()]
        for prompt in leaf_prompts
    ]
    artifact = ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        cases=tuple(row for row in ordered if isinstance(row, ContextualRiskCase)),
        unanswered_cases=tuple(
            row for row in ordered if isinstance(row, UnansweredCase)
        ),
        parent_cases=tuple(case for _, _, rows in finished for case in rows),
    )
    validate_contextual_artifact(artifact, instrument)
    validate_eal_crosschecks(artifact, instrument)
    validate_hierarchy_cases(artifact, instrument)
    return artifact


async def collect_study(
    contexts: Sequence[tuple[ContextualRiskInstrument, Path]],
    *,
    providers: Sequence[LLMProvider],
    config: ExecutionSettings,
    answers: AnswerTable | None = None,
) -> list[ContextualRiskArtifact | ValueError]:
    """Collect every context at once through one scheduler's lanes.

    No context holds provider slots: each model's lane sends the best-ranked
    ready request of any context. Each context freezes its artifact as soon as
    it completes. Meanwhile ``watch_event_loop`` reports a process that its
    CPU core bounds.

    A systematic provider failure stops the whole study: its attempt is
    frozen, every lane stops sending, every context is cancelled, which
    freezes nothing, and the error is raised. A cancelled request leaves its
    server-side run open at the provider; ``run_collections`` cancels every
    open run before its event loop closes. A transient provider failure fails
    only its group, and its context is returned as the ``ProviderFailed``
    naming its failed calls; a rerun resumes every context from its stored
    calls and sends each failed call again. An exhausted call stops nothing.
    Any other collection error of a context (a ``ValueError``, such as a call
    that stored its last allowed provider failure or a capture with symbolic
    links) stops that context only: its other calls are cancelled and it is
    returned as its error once the others finish.
    """
    scheduler = CollectionScheduler(config.concurrency_by_model)
    watch = asyncio.ensure_future(watch_event_loop())

    async def one(order: int, instrument, run_directory: Path):
        roster = {row.model_id for row in instrument.respondents}
        try:
            artifact = await collect_contextual_risk(
                instrument=instrument,
                run_directory=run_directory,
                providers=[row for row in providers if row.model_id in roster],
                config=config,
                scheduler=scheduler,
                collection=order,
                answers=answers,
            )
        except ValueError as error:
            if scheduler.failure is not None:
                logger.error(
                    "provider request failed; stopping the study",
                    run_directory=str(run_directory),
                    error=str(error),
                )
                raise
            logger.warning(
                "contextual collection incomplete; the study continues",
                run_directory=str(run_directory),
                error=str(error),
            )
            return error
        freeze_json(
            run_directory / "contextual-risk.json", artifact.model_dump(mode="json")
        )
        logger.info("contextual collection complete", run_directory=str(run_directory))
        return artifact

    try:
        return await _together(
            *(one(order, *row) for order, row in enumerate(contexts))
        )
    finally:
        watch.cancel()
        await asyncio.wait({watch})


def _prepare_collection(
    *,
    instrument: ContextualRiskInstrument,
    source_bundle: Path,
    run_directory: Path,
    profile: str,
    model_profile: ModelProfile,
) -> list[ModelSpec]:
    """Freeze one collection's sources, plan and initial requests, after
    deleting any temporary file an interrupted write left in it; return its
    models.

    The plan records the profile's name, the profiles file it was read from
    unless that was the packaged example, and its selected models and
    settings."""
    from catmodel.config import settings

    if run_directory.resolve().is_relative_to(source_bundle.resolve()):
        raise ValueError(
            "elicitation outputs cannot be written into the upstream source bundle"
        )
    if run_directory.is_symlink() or (
        run_directory.exists() and any(p.is_symlink() for p in run_directory.rglob("*"))
    ):
        raise ValueError("contextual capture cannot follow symbolic links")
    remove_temporary_files(run_directory)
    snapshot = capture_source_bundle_snapshot(source_bundle)
    validate_contextual_sources(instrument, snapshot)
    expected = {r.model_id for r in instrument.respondents}
    selected = [spec for spec in model_profile.models if spec.model_id in expected]
    if {spec.model_id for spec in selected} != expected:
        raise ValueError("instrument models must belong to the selected profile")
    execution = model_profile.execution_settings(
        selected, max_output_tokens=settings.ELICITATION_MAX_TOKENS
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    for name, data in snapshot.files.items():
        path = run_directory / "source-bundle" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Build provenance lies outside the bundle identity; a resumed run
        # keeps the record it first froze.
        if name != PROVENANCE_PATH or not path.exists():
            freeze_bytes(path, data)
    freeze_json(
        run_directory / "contextual-instrument.json", instrument.model_dump(mode="json")
    )
    requests = contextual_requests(instrument)
    plan: dict[str, object] = {
        "instrument_sha256": instrument.digest(),
        "profile": profile,
        "parse_attempts": PARSE_ATTEMPTS,
        "execution": {
            "models": [spec.model_dump(mode="json") for spec in selected],
            "settings": execution.model_dump(
                mode="json", exclude={"concurrency_by_model"}
            ),
        },
        "requests": [
            {
                "key": frozen_call_key(request),
                "prompt_sha256": request["prompt_sha256"],
            }
            for request in requests
        ],
    }
    if model_profile.profiles_file is not None:
        plan["profiles_file"] = model_profile.profiles_file.model_dump(mode="json")
    freeze_json(run_directory / "collection-plan.json", plan)
    for request in requests:
        path = run_directory / "calls" / frozen_call_key(request) / "request.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        freeze_json(path, request)
    return selected


@dataclass(frozen=True)
class _Preparation:
    """One context to prepare (``_prepare_collection``), so it can be sent to
    a worker process."""

    instrument: ContextualRiskInstrument
    source_bundle: Path
    run_directory: Path
    profile: str
    model_profile: ModelProfile


def _prepare(task: _Preparation) -> list[ModelSpec]:
    return _prepare_collection(
        instrument=task.instrument,
        source_bundle=task.source_bundle,
        run_directory=task.run_directory,
        profile=task.profile,
        model_profile=task.model_profile,
    )


def run_collections(
    contexts: Sequence[tuple[ContextualRiskInstrument, Path]],
    *,
    source_bundle: Path,
    profile: str,
    profiles: Path | None = None,
    dry_run: bool = False,
    answers: Path | None = None,
    workers: int = 1,
) -> tuple[ContextualRiskArtifact, ...] | None:
    """Freeze and collect one capture of each context's authored definitions.

    ``profile`` names a profile of the model profiles file ``profiles``, read
    once for every context (``load_model_profile`` gives the default file).
    Source validation checks the frozen source bundle and its captured
    capability evidence before any provider call. Every context is frozen
    first; collection then runs them all at once through shared lanes, one
    per model, whose ceilings are the profile's limits (see
    ``collect_study`` for how failures stop a group, a context or the
    study). A complete capture is reloaded rather than collected again, and
    an interrupted or failed one resumes from its stored calls; any
    incomplete context raises a ``ValueError`` naming it and its error, the
    failed calls among them, once every other context has finished.

    ``answers`` names an answer table (``AnswerTable``). Each request is then
    sent only if the table holds no response to its exact prompt, every
    response a provider returns is added to it, and so are the stored
    responses of a context that was already complete.

    One process collects on one CPU core, which bounds how many requests it
    can serve in flight. With ``workers`` above one, the contexts are frozen
    in up to that many worker processes, and those still to collect are
    divided among up to that many collector processes, each with its share
    of every model's ceiling (``catmodel.collection.collection_workers``);
    the files frozen are those one process freezes. An interruption (Ctrl-C,
    or SIGTERM in the main thread) ends the collection once it has cancelled
    its open server-side runs, in every collector.
    """
    from catmodel.config import load_model_profile, settings

    table = None if answers is None else AnswerTable(answers)
    model_profile = load_model_profile(profile=profile, path=profiles)
    preparations = [
        _Preparation(
            instrument=instrument,
            source_bundle=source_bundle,
            run_directory=run_directory,
            profile=profile,
            model_profile=model_profile,
        )
        for instrument, run_directory in contexts
    ]
    # Rendering a context's prompts to freeze them takes one core several
    # seconds; contexts prepare independently, each in its own directory.
    with WorkerPool(max(1, min(workers, len(preparations)))) as pool:
        prepared = pool.map(_prepare, preparations)
    models = {spec.model_id for selected in prepared for spec in selected}
    if dry_run:
        return None
    pending = [
        (instrument, run_directory)
        for instrument, run_directory in contexts
        if not (run_directory / "contextual-risk.json").exists()
    ]
    if table is not None:
        collecting = {run_directory for _, run_directory in pending}
        for _, run_directory in contexts:
            if run_directory not in collecting:
                table.add_collection(run_directory)
    if pending:
        selected = [spec for spec in model_profile.models if spec.model_id in models]
        execution = model_profile.execution_settings(
            selected, max_output_tokens=settings.ELICITATION_MAX_TOKENS
        )
        collectors = collector_count(
            workers, len(pending), execution.concurrency_by_model
        )
        logger.info(
            "collecting contexts",
            contexts=len(pending),
            collectors=collectors,
            concurrency_by_model=dict(execution.concurrency_by_model),
        )
        results: Sequence[object]
        if collectors > 1:
            results = collect_in_processes(
                [run_directory for _, run_directory in pending],
                models=selected,
                execution=execution,
                answers=answers,
                collectors=collectors,
            )
        else:
            from catmodel.llm.factory import build_providers
            from catmodel.llm.transport import with_open_runs_cancelled

            providers = build_providers(
                selected,
                api_keys=settings.provider_api_keys(),
                concurrency_limits=execution.concurrency_by_model,
            )
            # SIGTERM, like Ctrl-C, then cancels the collection, which
            # cancels its open server-side runs.
            with sigterm_exits():
                results = asyncio.run(
                    with_open_runs_cancelled(
                        collect_study(
                            pending,
                            providers=providers,
                            config=execution,
                            answers=table,
                        )
                    )
                )
        failed = [
            (run_directory, result)
            for (_, run_directory), result in zip(pending, results, strict=True)
            if isinstance(result, Exception)
        ]
        if failed:
            raise ValueError(
                f"{len(failed)} of {len(pending)} collections incomplete; "
                + "; ".join(f"{path}: {error}" for path, error in failed)
            ) from failed[0][1]
    return tuple(
        load_contextual_collection(run_directory, source_bundle)[1]
        for _, run_directory in contexts
    )


def run_contextual_collection(
    *,
    instrument: ContextualRiskInstrument,
    source_bundle: Path,
    run_directory: Path,
    profile: str,
    profiles: Path | None = None,
    dry_run: bool = False,
    answers: Path | None = None,
) -> ContextualRiskArtifact | None:
    """Freeze and collect one capture of the instrument's authored definitions."""
    artifacts = run_collections(
        [(instrument, run_directory)],
        source_bundle=source_bundle,
        profile=profile,
        profiles=profiles,
        dry_run=dry_run,
        answers=answers,
    )
    return None if artifacts is None else artifacts[0]


def run_study_collection(
    *,
    study: Path,
    source_bundle: Path,
    profile: str,
    profiles: Path | None = None,
    dry_run: bool = False,
    answers: Path | None = None,
    workers: int = 1,
) -> tuple[ContextualRiskArtifact, ...] | None:
    """Freeze and collect every context of a public study's plan at once, in
    up to ``workers`` collector processes (``run_collections``)."""
    from catmodel.setup.public_study import public_study_collections

    return run_collections(
        [
            (
                ContextualRiskInstrument.from_saved(
                    strict_json_loads(instrument.read_bytes())
                ),
                capture,
            )
            for instrument, capture in public_study_collections(study)
        ],
        source_bundle=source_bundle,
        profile=profile,
        profiles=profiles,
        dry_run=dry_run,
        answers=answers,
        workers=workers,
    )
