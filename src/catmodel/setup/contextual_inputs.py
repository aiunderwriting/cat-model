"""Source-bound intake and raw-response replay for contextual risk judgments."""

import hashlib
import re
from collections.abc import Generator, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, get_args

from catmodel.contracts.capability_evidence import local_capability_evidence
from catmodel.contracts.contextual_risk import (
    SAFEGUARD_LEVEL_FIELDS,
    ContextualRiskArtifact,
    ContextualRiskCase,
    ContextualRiskInstrument,
    ContextualRiskNode,
    UnansweredCase,
)
from catmodel.contracts.elicitation import Respondent
from catmodel.contracts.public_frontier_condition import SourceModelStatus
from catmodel.elicitation.contextual_frequency import (
    chain_prompts,
    validate_contextual_artifact,
)
from catmodel.elicitation.contextual_hierarchy import (
    hierarchy_parents,
    parent_prompts,
)
from catmodel.elicitation.contextual_risk import (
    contextual_node_from_scenario,
    contextual_prompts,
)
from catmodel.elicitation.eal_crosscheck import case_crosscheck_prompt
from catmodel.elicitation.question_flow import (
    CHAIN,
    CHAIN_GUARDRAIL_RETRY,
    DIRECT,
    DIRECT_ANSWERED,
    DIRECT_SEVERITY_RETRY,
    EAL_CROSSCHECK,
    EAL_CROSSCHECK_SEVERITY_RETRY,
    PARENT,
    Question,
    QuestionKind,
    chain_path,
    crosscheck_path,
    direct_path,
)
from catmodel.evidence.source_bundle import (
    SOURCE_CARDS_PATH,
    SourceBundleSnapshot,
    capture_source_bundle_snapshot,
    load_source_bundle_threat_models,
)
from catmodel.frozen_calls import (
    CaptureFiles,
    frozen_call_key,
    replay_call,
    validate_acquisition_closure,
)
from catmodel.json_io import strict_json_loads
from catmodel.resources import scenarios_path
from catmodel.scenarios.base_rates import SourceCardsFile
from catmodel.scenarios.loader import ScenarioDefinition, load_scenarios
from catmodel.scenarios.threat_models import ThreatModel
from catmodel.setup.sensitivity_catalog import (
    EXPOSURE_SCALE_VARIANTS,
    SAFEGUARD_LEVELS,
)
from catmodel.stored_files import DiskFile, StoredFiles
from catmodel.yaml_io import safe_load


def contextual_request(prompt: Question) -> dict[str, Any]:
    return {
        "call_id": prompt.call_id,
        "respondent": prompt.respondent.model_dump(mode="json"),
        "prompt_sha256": prompt.prompt_sha256(),
        "system_prompt": prompt.system_prompt,
        "user_prompt": prompt.render(),
    }


def contextual_requests(instrument: ContextualRiskInstrument) -> list[dict[str, Any]]:
    """The initial direct, chain and taxonomy-parent requests frozen for collection."""
    return [
        contextual_request(prompt)
        for prompt in (
            *contextual_prompts(instrument),
            *chain_prompts(instrument),
            *parent_prompts(instrument),
        )
    ]


def _scenario_identity(node: ContextualRiskNode) -> dict[str, object]:
    """What of a scenario's node a question shows or code computes from: its
    population, counting definition and the asked part of its chain."""
    chain = node.frequency_chain
    return {
        "population": node.population,
        "qualified_count_definition": node.qualified_count_definition,
        "frequency_chain": None if chain is None else chain.asked(),
    }


def require_scenario_nodes(
    instrument: ContextualRiskInstrument,
    scenarios: Mapping[str, ScenarioDefinition],
) -> None:
    """Refuse a node that departs from its repository scenario's definition.

    Every node whose population names a repository scenario is rebuilt from
    the current scenario YAML; its population, counting definition and the
    asked part of its frequency chain (``ContextualFrequencyChain.asked``) must
    be identical, so neither an explicit node nor a capture collected under an
    earlier definition can stand in for the authored one. Audit text that no
    question shows, such as an exposure base's rationale and source, may be
    revised without refusing earlier captures.
    """
    for node in instrument.nodes:
        scenario = scenarios.get(node.population.scenario_id)
        if scenario is None:
            continue
        try:
            expected = contextual_node_from_scenario(
                scenario,
                instrument.deployment.entity_id,
                instrument.capability_access,
            )
        except ValueError as exc:
            raise ValueError(
                f"contextual node for scenario {scenario.id!r} does not match its "
                f"scenario definition: {exc}"
            ) from exc
        actual, authored = _scenario_identity(node), _scenario_identity(expected)
        for field, value in actual.items():
            if value != authored[field]:
                raise ValueError(
                    f"contextual node for scenario {scenario.id!r} differs from its "
                    f"authored scenario definition in {field}"
                )


def _selected_scenarios(
    scenario_ids: Sequence[str], scenarios: Mapping[str, ScenarioDefinition]
) -> tuple[str, ...]:
    """The selected scenarios in order of their ids, whatever order they were
    named in: ``all``, which is used alone, is every scenario of the catalog,
    and any other selection names unique known scenarios."""
    if "all" in scenario_ids:
        if tuple(scenario_ids) != ("all",):
            raise ValueError("--scenario all must be used alone")
        return tuple(sorted(scenarios))
    if (
        len(set(scenario_ids)) != len(scenario_ids)
        or set(scenario_ids) - scenarios.keys()
    ):
        raise ValueError("scenario selection must be unique known identifiers")
    return tuple(sorted(scenario_ids))


def _covers_catalog(
    scenario_ids: Iterable[str], scenarios: Mapping[str, ScenarioDefinition]
) -> bool:
    """Whether the scenarios are the complete catalog, whose instrument also
    carries the taxonomy parents."""
    return set(scenario_ids) == set(scenarios)


def validate_contextual_sources(
    instrument: ContextualRiskInstrument, snapshot: SourceBundleSnapshot
) -> None:
    if instrument.source_bundle_manifest_sha256 != snapshot.manifest_sha256:
        raise ValueError("contextual instrument source bundle identity is stale")
    if instrument.public_trajectory is not None:
        for ref in instrument.public_trajectory.source_references:
            local = local_capability_evidence(ref)
            if local is not None:
                path, digest = local
                if (
                    not snapshot.exists(path)
                    or hashlib.sha256(snapshot.read_bytes(path)).hexdigest() != digest
                ):
                    raise ValueError(
                        "public trajectory evidence differs from its captured source"
                    )
    cards = SourceCardsFile.model_validate(
        safe_load(snapshot.read_text(SOURCE_CARDS_PATH))
    )
    available = {card.card_id: card for card in cards.cards}
    scenarios = {row.id: row for row in load_scenarios(scenarios_path())}
    require_scenario_nodes(instrument, scenarios)
    complete = _covers_catalog(
        (node.population.scenario_id for node in instrument.nodes), scenarios
    )
    if (complete or instrument.parents) and instrument.parents != hierarchy_parents(
        instrument.nodes, scenarios
    ):
        raise ValueError(
            "an instrument over the complete catalog carries taxonomy parents, and "
            "any taxonomy parents follow the current hierarchy contract and nodes"
        )
    threat_models = load_source_bundle_threat_models(snapshot)
    for node in instrument.nodes:
        for card in node.source_cards:
            if available.get(card.card_id) != card:
                raise ValueError(
                    f"contextual source card {card.card_id!r} differs from the captured source"
                )
        if node.threat_model is not None and node.threat_model != (
            threat_models.for_scenario(node.population.scenario_id)
        ):
            raise ValueError(
                f"threat model {node.threat_model.threat_model_id!r} differs from "
                "the captured source"
            )


def _eligible_threat_model(
    threat_model: ThreatModel | None, period: Mapping[str, Any]
) -> ThreatModel | None:
    """The scenario's threat model where its research ended by the study
    period's cutoffs, as source cards are selected."""
    if threat_model is None or not threat_model.eligible(period):
        return None
    return threat_model


_SOURCE_STATUSES = get_args(SourceModelStatus)
_SOURCE_STATUS_FIELDS = frozenset(
    {
        "model",
        "deployment",
        "deployment_binding",
        "capability_access",
        "exposure",
        "fixed_controls",
        "fixed_adoption",
    }
)
"""Context fields a ``source_statuses`` entry may author for one source status."""


def _select_source_status(
    data: dict[str, Any], source_status: str | None
) -> dict[str, Any]:
    """Merge one source status's fields into a context that describes both.

    A public study needs a public and a not-yet-public template at the same lab;
    ``source_statuses`` authors what differs between them once, beside the
    shared fields, so each template is built from the same context.
    """
    if "source_statuses" not in data:
        if source_status is not None:
            raise ValueError("a source status requires a context with source_statuses")
        return data
    statuses = data.pop("source_statuses")
    if (
        not isinstance(statuses, dict)
        or set(statuses) != set(_SOURCE_STATUSES)
        or any(not isinstance(block, dict) or not block for block in statuses.values())
    ):
        raise ValueError(
            "source_statuses must author fields for public and not_yet_public"
        )
    if source_status is None:
        raise ValueError("a context with source_statuses requires a source status")
    if source_status not in _SOURCE_STATUSES:
        raise ValueError(f"source status must be one of {_SOURCE_STATUSES}")
    fields = {frozenset(block) for block in statuses.values()}
    if len(fields) != 1 or not next(iter(fields)) <= _SOURCE_STATUS_FIELDS:
        raise ValueError(
            "both source statuses must author the same fields among "
            f"{sorted(_SOURCE_STATUS_FIELDS)}"
        )
    if next(iter(fields)) & data.keys():
        raise ValueError(
            "a field is authored once: shared, or in source_statuses, not both"
        )
    return {**data, **statuses[source_status]}


CAPABILITY_ANCHOR_FIELDS = (
    "capability_scale",
    "capability_domain_id",
    "reference_eci",
    "integer_min",
    "integer_max",
    "anchor_scores",
)
"""Context fields that state a fixed-anchor instrument's capability scale and
anchors. A public study's instruments take them from its forecast, so its
context may omit them."""

_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def _safeguard_levels(levels: Any) -> dict[str, dict[str, Any]]:
    """Check the study config's authored safeguard levels.

    Each level states its rationale, the shared controls and, for each source
    status, exactly the deployment and binding texts of
    ``SAFEGUARD_LEVEL_FIELDS``, so a level is one posture across scope,
    permissions and oversight.
    """
    if not isinstance(levels, dict) or set(levels) != set(SAFEGUARD_LEVELS):
        raise ValueError(f"safeguard_levels must author {list(SAFEGUARD_LEVELS)}")
    for level_id, block in levels.items():
        if (
            not isinstance(block, dict)
            or set(block) != {"rationale", "fixed_controls", "source_statuses"}
            or not isinstance(block["source_statuses"], dict)
            or set(block["source_statuses"]) != set(_SOURCE_STATUSES)
        ):
            raise ValueError(
                f"safeguard level {level_id!r} authors a rationale, fixed_controls "
                "and source_statuses for public and not_yet_public"
            )
        texts = {
            "rationale": block["rationale"],
            "fixed_controls": block["fixed_controls"],
        }
        statuses: dict[str, Any] = block["source_statuses"]
        for status, fields in statuses.items():
            if not isinstance(fields, dict) or any(
                not isinstance(value, dict) for value in fields.values()
            ):
                raise ValueError(
                    f"safeguard level {level_id!r} nests its texts by record"
                )
            paths = {
                f"{record}.{name}": value
                for record, values in fields.items()
                for name, value in values.items()
            }
            if {"fixed_controls", *paths} != set(SAFEGUARD_LEVEL_FIELDS[status]):
                raise ValueError(
                    f"safeguard level {level_id!r} must replace exactly "
                    f"{list(SAFEGUARD_LEVEL_FIELDS[status])} for {status}"
                )
            texts.update({f"{status}.{path}": value for path, value in paths.items()})
        if any(
            not isinstance(value, str) or not value.strip() for value in texts.values()
        ):
            raise ValueError(f"safeguard level {level_id!r} texts must be nonblank")
    return levels


def _apply_safeguard_level(
    data: dict[str, Any], level: Mapping[str, Any], source_status: str
) -> dict[str, Any]:
    """Replace one source status's safeguard texts with one authored level's.

    The replaced ``use_population`` keeps the study's scale placeholders, so
    the level changes what is used and how, not how much.
    """
    replaced = {**data, "fixed_controls": level["fixed_controls"]}
    for record, values in level["source_statuses"][source_status].items():
        if not isinstance(data.get(record), dict):
            raise ValueError(f"a safeguard level replaces text in {record}")
        replaced[record] = {**data[record], **values}
    before = data["deployment_binding"]["use_population"]
    after = replaced["deployment_binding"]["use_population"]
    if sorted(_PLACEHOLDER.findall(before)) != sorted(_PLACEHOLDER.findall(after)):
        raise ValueError(
            "a safeguard level's use_population keeps the study's scale placeholders"
        )
    return replaced


def build_contextual_instrument(
    payload: Mapping[str, Any],
    *,
    source_bundle: Path,
    respondents: Sequence[Respondent],
    scenario_ids: Sequence[str] = (),
    source_status: str | None = None,
    exposure_variant: str | None = None,
    safeguard_level: str | None = None,
) -> ContextualRiskInstrument:
    """Build one instrument from a context and the selected scenarios.

    The order that names ``scenario_ids`` does not matter: the nodes follow
    the scenarios' ids. A selection of the complete catalog, as ``all`` or as
    every id, also carries the taxonomy parents.

    A context with ``source_statuses`` describes both source statuses of a
    public study; ``source_status`` selects the template to build.
    ``exposure_variant`` names an ``exposure_scale`` variant whose factor multiplies
    every path-linked deployment scale; questions show only the scaled numbers.
    ``safeguard_level`` names a ``safeguard_level`` variant whose texts, authored
    in the context's ``safeguard_levels``, replace the study's stipulated
    safeguards; questions show them as the study's own.
    """
    snapshot = capture_source_bundle_snapshot(source_bundle)
    data = dict(payload)
    levels = (
        _safeguard_levels(data.pop("safeguard_levels"))
        if "safeguard_levels" in data
        else None
    )
    data = _select_source_status(data, source_status)
    missing = [name for name in CAPABILITY_ANCHOR_FIELDS if name not in data]
    if missing:
        raise ValueError(
            f"the context does not state {', '.join(missing)}, which a "
            "fixed-anchor instrument requires; if it describes a public study, "
            "whose instruments take them from its forecast, build the public "
            "study instead"
        )
    if exposure_variant is not None and safeguard_level is not None:
        raise ValueError("an instrument carries at most one context variant")
    if safeguard_level is not None:
        if safeguard_level not in SAFEGUARD_LEVELS:
            raise ValueError(f"safeguard level must be one of {list(SAFEGUARD_LEVELS)}")
        if levels is None or source_status is None:
            raise ValueError(
                "a safeguard level requires a context with safeguard_levels and "
                "source_statuses"
            )
        data = _apply_safeguard_level(data, levels[safeguard_level], source_status)
        data["safeguard_level"] = safeguard_level
    if exposure_variant is not None:
        if exposure_variant not in EXPOSURE_SCALE_VARIANTS:
            raise ValueError(
                f"exposure variant must be one of {sorted(EXPOSURE_SCALE_VARIANTS)}"
            )
        exposure = data.get("path_linked_exposure")
        if not isinstance(exposure, dict) or "variant" in exposure:
            raise ValueError(
                "an exposure variant requires path-linked exposure without an authored variant"
            )
        data["path_linked_exposure"] = {
            **exposure,
            "variant": {
                "variant_id": exposure_variant,
                "factor": EXPOSURE_SCALE_VARIANTS[exposure_variant],
            },
        }
    if "respondents" in data or "source_bundle_manifest_sha256" in data:
        raise ValueError(
            "context input does not set respondents or the source bundle digest"
        )
    if scenario_ids:
        if "nodes" in data:
            raise ValueError(
                "select authored nodes or scenario-derived nodes, not both"
            )
        scenarios = {row.id: row for row in load_scenarios(scenarios_path())}
        scenario_ids = _selected_scenarios(scenario_ids, scenarios)
        cards = SourceCardsFile.model_validate(
            safe_load(snapshot.read_text(SOURCE_CARDS_PATH))
        )
        threat_models = load_source_bundle_threat_models(snapshot)
        data["nodes"] = [
            contextual_node_from_scenario(
                scenarios[sid],
                data["deployment"]["entity_id"],
                data["capability_access"],
                source_cards=tuple(
                    card
                    for card in cards.scenario_cards(sid)
                    if card.eligible(data["period"])
                ),
                threat_model=_eligible_threat_model(
                    threat_models.for_scenario(sid), data["period"]
                ),
            ).model_dump(mode="json")
            for sid in scenario_ids
        ]
        if _covers_catalog(scenario_ids, scenarios):
            # The complete catalog also asks every taxonomy split that holds
            # an estimated scenario for its whole-category EAL.
            data["parents"] = [
                row.model_dump(mode="json")
                for row in hierarchy_parents(
                    [ContextualRiskNode.model_validate(row) for row in data["nodes"]],
                    scenarios,
                )
            ]
    instrument = ContextualRiskInstrument.model_validate(
        {
            **data,
            "respondents": respondents,
            "source_bundle_manifest_sha256": snapshot.manifest_sha256,
        }
    )
    validate_contextual_sources(instrument, snapshot)
    return instrument


REPLAY_ERRORS: Mapping[QuestionKind, str] = {
    DIRECT: "contextual artifact differs from its complete raw response capture",
    DIRECT_SEVERITY_RETRY: (
        "coherence retry answer differs from its raw response capture"
    ),
    EAL_CROSSCHECK: "EAL cross-check answer differs from its raw response capture",
    EAL_CROSSCHECK_SEVERITY_RETRY: (
        "EAL cross-check coherence retry differs from its raw response capture"
    ),
    CHAIN: "chain answer differs from its raw response capture",
    CHAIN_GUARDRAIL_RETRY: (
        "chain guardrail retry differs from its raw response capture"
    ),
    PARENT: "parent answer differs from its raw response capture",
}
"""What replay says when a question of each kind does not replay to the answer
its case records, or the case records none for a question its stored values
call for."""


def recorded_answers(
    row: ContextualRiskCase | UnansweredCase,
) -> dict[QuestionKind, Any]:
    """The answer a leaf's row records for each kind of question it was
    asked; None for an exhausted call."""
    recorded: dict[QuestionKind, Any] = {CHAIN: row.chain_answer}
    if isinstance(row, UnansweredCase):
        recorded[DIRECT] = None
        return recorded
    recorded[DIRECT] = row.answer
    if row.severity_retry is not None:
        recorded[DIRECT_SEVERITY_RETRY] = row.severity_retry.answer
    if row.chain_guardrail is not None:
        recorded[CHAIN_GUARDRAIL_RETRY] = row.chain_guardrail.answer
    check = row.eal_crosscheck
    if check is not None and check.reconciliation is not None:
        recorded[EAL_CROSSCHECK] = check.reconciliation.answer
        retry = check.reconciliation.severity_retry
        if retry is not None:
            recorded[EAL_CROSSCHECK_SEVERITY_RETRY] = retry.answer
    return recorded


def validate_contextual_capture(
    instrument: ContextualRiskInstrument,
    artifact: ContextualRiskArtifact,
    files: Mapping[str, bytes],
) -> None:
    """Replay every recorded answer from its raw responses.

    Each leaf's paths are those collection asked
    (``catmodel.elicitation.question_flow``), driven here by the answers its
    row records. A recorded answer must be its call's replay. A recorded None
    must be an exhausted call: every validation attempt answered and failed,
    never an unfinished call or one the provider failed
    (``POOL-EXCLUDED-SAMPLES``). Provider failures sent again on resume may
    come before any attempt that answered (``NUM-PROVIDER-FAILURES``). The
    leaves of an incomplete group, one with an unanswered leaf, and its parents'
    answers replay like any other; the capture stores no pool for its
    parents. Each complete group's stored child EALs and pools must be those
    its answers give (``validate_hierarchy_cases``).
    """
    validate_contextual_artifact(artifact, instrument)
    cases = {case.key(): case for case in artifact.cases}
    unanswered = {row.key(): row for row in artifact.unanswered_cases}
    recorded = {key: recorded_answers(row) for key, row in (cases | unanswered).items()}
    requests = []

    def replays_to(question: Question, kind: QuestionKind, answer: object) -> bool:
        """The question's call replays to ``answer``; a recorded None must be
        exhausted."""
        request = contextual_request(question)
        requests.append(request)
        replayed = replay_call(
            request=request,
            files=files,
            parse_answer=partial(kind.parser, prompt=question),
        )
        if replayed.answer != answer:
            return False
        return answer is not None or replayed.exhausted

    def replay(
        path: Generator[Any, Any, Any],
        answers: Mapping[QuestionKind, Any],
        direct_answered: bool = True,
    ) -> QuestionKind | None:
        """Drive one path of ``question_flow`` with the answers a row records.
        Returns the kind of the first question that does not replay to its
        recorded answer, or has none; None when every question replays."""
        reply = None
        while True:
            try:
                step = path.send(reply)
            except StopIteration:
                return None
            if step is DIRECT_ANSWERED:
                reply = direct_answered
            elif step.kind in answers and replays_to(
                step.prompt, step.kind, answers[step.kind]
            ):
                reply = answers[step.kind]
            else:
                return step.kind

    for prompt in contextual_prompts(instrument):
        key = (prompt.node.node_id, prompt.anchor_score, prompt.respondent.key())
        # The stored share, already checked against its answer, fixes whether
        # the case needed a coherence retry and the retry's question.
        differs = replay(direct_path(prompt, cases.get(key)), recorded[key])
        if differs is not None:
            raise ValueError(
                "an unanswered case's direct question is not exhausted in its "
                "raw response capture"
                if key in unanswered
                else REPLAY_ERRORS[differs]
            )
    for prompt in chain_prompts(instrument):
        key = (prompt.node.node_id, prompt.anchor_score, prompt.respondent.key())
        differs = replay(chain_path(prompt), recorded[key], key in cases)
        if differs is not None:
            raise ValueError(REPLAY_ERRORS[differs])
    from catmodel.setup.implied_eal import validate_eal_crosschecks

    validate_eal_crosschecks(artifact, instrument)
    for case in artifact.cases:
        # The stored cross-check values, already checked against the case's
        # answers, fix whether it needed the cross-check call and its question.
        crosscheck = case_crosscheck_prompt(instrument, case)
        check = case.eal_crosscheck
        if crosscheck is None or check is None or check.reconciliation is None:
            continue
        differs = replay(
            crosscheck_path(crosscheck, check.reconciliation), recorded[case.key()]
        )
        if differs is not None:
            raise ValueError(REPLAY_ERRORS[differs])
    if instrument.parents:
        from catmodel.setup.contextual_hierarchy import validate_hierarchy_cases

        validate_hierarchy_cases(artifact, instrument)
        parents = {case.key(): case for case in artifact.parent_cases}
        for prompt in parent_prompts(instrument):
            case = parents[
                prompt.parent.parent_id, prompt.anchor_score, prompt.respondent.key()
            ]
            if not replays_to(prompt, PARENT, case.answer):
                raise ValueError(REPLAY_ERRORS[PARENT])
    expected_calls = {
        f"calls/{frozen_call_key(request)}/request.json" for request in requests
    }
    if expected_calls != {
        name
        for name in files
        if name.startswith("calls/") and name.endswith("/request.json")
    }:
        raise ValueError("contextual request roster differs from the frozen instrument")
    expected_keys = {frozen_call_key(request) for request in requests}
    if any(
        name.startswith("calls/")
        and (len(parts := name.split("/", 2)) < 3 or parts[1] not in expected_keys)
        for name in files
    ):
        raise ValueError("contextual capture contains an unplanned call")
    validate_acquisition_closure(requests, files)


def read_contextual_capture_files(run_directory: Path) -> CaptureFiles:
    """The capture's files, left on disk: each is read when a caller asks for
    it, and must keep the bytes it had when first read."""
    root = run_directory.expanduser().resolve()
    if (
        run_directory.is_symlink()
        or not root.is_dir()
        or any(path.is_symlink() for path in root.rglob("*"))
    ):
        raise ValueError(
            "contextual capture requires a directory without symbolic links"
        )
    return CaptureFiles(
        (path.relative_to(root).as_posix(), DiskFile(path))
        for path in root.rglob("*")
        if path.is_file()
    )


def load_contextual_collection(
    run_directory: Path,
    source_bundle: Path | None = None,
) -> tuple[ContextualRiskInstrument, ContextualRiskArtifact, CaptureFiles]:
    root = run_directory.expanduser().resolve()
    files = read_contextual_capture_files(run_directory)
    required = {
        "contextual-instrument.json",
        "contextual-risk.json",
        "collection-plan.json",
    }
    if not required <= files.keys():
        raise ValueError(
            "contextual capture is missing its instrument, artifact or plan"
        )
    instrument = ContextualRiskInstrument.from_saved(
        strict_json_loads(files["contextual-instrument.json"])
    )
    artifact = ContextualRiskArtifact.from_saved(
        strict_json_loads(files["contextual-risk.json"])
    )
    plan = strict_json_loads(files["collection-plan.json"])
    snapshot = capture_source_bundle_snapshot(root / "source-bundle")
    validate_contextual_sources(instrument, snapshot)
    if source_bundle is not None:
        supplied = capture_source_bundle_snapshot(source_bundle)
        if supplied.manifest_sha256 != snapshot.manifest_sha256:
            raise ValueError(
                "supplied source bundle differs from the original contextual capture"
            )
    validate_contextual_capture(instrument, artifact, files)
    if plan.get("instrument_sha256") != instrument.digest() or plan.get("requests") != [
        {"key": frozen_call_key(request), "prompt_sha256": request["prompt_sha256"]}
        for request in contextual_requests(instrument)
    ]:
        raise ValueError("contextual plan differs from its frozen instrument")
    retained = CaptureFiles(
        StoredFiles.merged(
            files.selected(
                lambda name: (
                    name in required
                    or name.startswith(("acquisition/", "parser-protocols/"))
                    or (
                        name.startswith("calls/")
                        and (
                            name.endswith("/request.json")
                            or ("/attempt-" in name and name.endswith(".json"))
                        )
                    )
                )
            ),
            StoredFiles(snapshot.files).prefixed("source-bundle/"),
        )
    )
    return instrument, artifact, retained
