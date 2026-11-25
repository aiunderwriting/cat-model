"""Scenario-library catalog generation and cross-file validation.

``scripts/validate_scenario_library.py`` is the maintainer entry point: it runs
``validate_scenario_library`` and, with ``--fix``, regenerates every derived
artifact, including this module's catalog and documented counts.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catmodel.scenarios.base_rates import (
    SOURCE_CARDS_RELATIVE_PATH,
    source_card_coverage_errors,
)
from catmodel.scenarios.coverage_register import (
    COVERAGE_REGISTER_RELATIVE_PATH,
    coverage_register_errors,
    load_coverage_register,
)
from catmodel.scenarios.facets import FACET_ROLLUP_AXES, scenario_facet_projection
from catmodel.scenarios.hierarchy_contract import (
    HierarchyContract,
    children_by_id,
    load_hierarchy_contract,
    origin_role_of,
    validate_hierarchy_contract,
)
from catmodel.scenarios.loader import (
    ScenarioDefinition,
    ScenarioLibraryError,
    event_chain_terminal_stage,
    load_scenario_files,
)
from catmodel.scenarios.taxonomy_identity import (
    DEFAULT_SCENARIO_LINEAGE_REGISTER_RELATIVE_PATH,
    load_scenario_lineage_register,
    scenario_roster_sha256,
    validate_scenario_lineage,
)
from catmodel.scenarios.taxonomy_registry import DOMAIN_SORT_RANK, RISK_DOMAIN_ORDER
from catmodel.scenarios.threat_models import (
    THREAT_MODELS_RELATIVE_PATH,
    load_threat_models,
    threat_model_errors,
)

DEFAULT_CATALOG_RELATIVE_PATH = Path("scenarios/scenario_catalog.json")


@dataclass(frozen=True, slots=True)
class ScenarioLibrarySnapshot:
    scenarios_dir: Path
    scenarios: tuple[ScenarioDefinition, ...]
    scenario_paths: dict[str, Path]
    contract: HierarchyContract

    @property
    def cell_count(self) -> int:
        return len(self.scenarios)

    @property
    def normalisation_base_count(self) -> int:
        """Cells with an exposure base that is only a normalisation."""
        return sum(1 for scenario in self.scenarios if scenario.normalisation_bases)

    @property
    def risk_domain_count(self) -> int:
        return len({scenario.risk_domain for scenario in self.scenarios})

    @property
    def scenario_counts_by_domain(self) -> dict[str, int]:
        return {
            domain: sum(
                1 for scenario in self.scenarios if scenario.risk_domain == domain
            )
            for domain in RISK_DOMAIN_ORDER
        }

    @property
    def scenario_counts_by_origin(self) -> dict[str, int]:
        counts = {
            origin: 0
            for origin in self.contract.axes["origin_regime"].values
            if origin != "other"
        }
        for scenario in self.scenarios:
            origin = origin_role_of(self.contract, scenario.node_ids[0])[0]
            if origin in counts:
                counts[origin] += 1
        return counts

    @property
    def hierarchy_counts(self) -> dict[str, int]:
        """Nodes, splits, leaves by disposition and the parent questions: the
        splits whose subtree holds an estimated leaf."""
        children = children_by_id(self.contract)
        owned = {
            node_id for scenario in self.scenarios for node_id in scenario.node_ids
        }

        def estimated(node_id: str) -> bool:
            below = children[node_id]
            return node_id in owned if not below else any(map(estimated, below))

        splits = [node_id for node_id, below in children.items() if below]
        leaves = [node_id for node_id, below in children.items() if not below]
        dispositions = [
            node.unestimated_disposition.disposition
            for node in self.contract.nodes
            if node.unestimated_disposition is not None
        ]
        return {
            "nodes": len(self.contract.nodes),
            "splits": len(splits),
            "leaves": len(leaves),
            "leaves.estimated": sum(1 for node_id in leaves if node_id in owned),
            "leaves.unestimated": dispositions.count("unestimated"),
            "leaves.unsupported": dispositions.count("unsupported"),
            "parents": sum(1 for node_id in splits if estimated(node_id)),
        }


# Each document whose prose states library counts, with the marker keys it
# must carry; ``--fix`` rewrites their values.
DOCUMENT_COUNT_MARKERS: dict[Path, tuple[str, ...]] = {
    Path("README.md"): (
        "cells",
        "origin.deliberate_misuse",
        "origin.ai_failure",
        "origin.provider_conduct",
        "risk_domains",
        "leaves.unestimated",
        "leaves.unsupported",
        "normalisation_bases",
    ),
    Path("methodology.md"): (
        "splits",
        "leaves",
        "leaves.estimated",
        "leaves.unestimated",
    ),
    Path("docs/taxonomy.md"): (
        "nodes",
        "splits",
        "leaves",
        "cells",
        "leaves.estimated",
        "leaves.unestimated",
    ),
    Path("docs/limitations.md"): (
        "leaves.unestimated",
        "leaves.unsupported",
        "normalisation_bases",
    ),
    Path("runbook.md"): ("questions", "cells", "parents"),
}

_COUNT_MARKER_RE = re.compile(
    r"<!--\s*scenario-count:(?P<key>[a-z0-9_.-]+)\s*-->\s*"
    r"(?P<value>\d+)\s*<!--\s*/scenario-count\s*-->",
    flags=re.IGNORECASE,
)

# The source-bundle run-reference pattern (catmodel.evidence.source_bundle) plus
# prose citations of an elicitation run. Scenarios sit below packages in the
# import layers, so the pattern is repeated here.
_EVIDENCE_RUN_REF_RE = re.compile(
    r"\b(?:run|elicitation-run)://\S+|(?:^|/)runs/[^/\s#]+|\belicitation[- ]runs?\b",
    re.IGNORECASE,
)
_EVIDENCE_URL_RE = re.compile(r"https?://\S+(?: \([^()]+\))?")
_EVIDENCE_URL_TOKEN_RE = re.compile(r"https?://\S+")
_EVIDENCE_DOCUMENT_SUFFIX = r"\.(?:md|ya?ml|json|csv|py|txt|html|pdf)\b"
_EVIDENCE_PATH_LIKE_RE = re.compile(rf"[\w.-]*(?:/|{_EVIDENCE_DOCUMENT_SUFFIX})")
_EVIDENCE_DOCUMENT_TOKEN_RE = re.compile(rf"\w{_EVIDENCE_DOCUMENT_SUFFIX}")
_EVIDENCE_CITATION_TOKEN_RE = re.compile(r"[\w./-]+")
_EVIDENCE_PATH_REF_RE = re.compile(
    r"(?P<path>[\w./-]+)(?:#(?P<anchor>[\w-]+))?(?: \((?P<note>[^()]+)\))?"
)
_EVIDENCE_TEST_ROOTS = ("tests/", "test_support/")
# The operator's local reference cache is gitignored apart from its README, which
# documents source applicability; run directories fail the run-reference rule.
_EVIDENCE_LOCAL_DATA_ROOT = "data/external/"
_EVIDENCE_LOCAL_DATA_TRACKED = frozenset({"data/external/README.md"})


def scenario_catalog_json(catalog: dict[str, Any]) -> str:
    """Return stable JSON text for the generated scenario catalog."""
    return json.dumps(catalog, indent=2) + "\n"


def write_scenario_catalog(catalog: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(scenario_catalog_json(catalog), encoding="utf-8")


def load_scenario_library_snapshot(scenarios_dir: Path) -> ScenarioLibrarySnapshot:
    loaded = load_scenario_files(scenarios_dir)
    return ScenarioLibrarySnapshot(
        scenarios_dir=scenarios_dir,
        scenarios=tuple(
            sorted(
                loaded.values(),
                key=lambda s: (DOMAIN_SORT_RANK.get(s.risk_domain, 99), s.id),
            )
        ),
        scenario_paths={scenario.id: path for path, scenario in loaded.items()},
        contract=load_hierarchy_contract(),
    )


def build_scenario_catalog_from_snapshot(
    snapshot: ScenarioLibrarySnapshot,
    *,
    lineage_register_path: Path | None = None,
) -> dict[str, Any]:
    register = load_scenario_lineage_register(lineage_register_path)
    return {
        "meta": {
            "schema_version": 1,
            "source": "scenario YAML library",
            "scenario_taxonomy_version": register.current_taxonomy_version,
            "scenario_roster_sha256": scenario_roster_sha256(
                [scenario.id for scenario in snapshot.scenarios]
            ),
        },
        "taxonomy_rule": " ".join(snapshot.contract.rule.split()),
        "splits": _split_records(snapshot.contract),
        "scenario_cell_count": snapshot.cell_count,
        "normalisation_base_count": snapshot.normalisation_base_count,
        "risk_domain_count": snapshot.risk_domain_count,
        "risk_domains": {
            domain: {"scenario_cell_count": count}
            for domain, count in snapshot.scenario_counts_by_domain.items()
        },
        "facet_projections": _facet_projections(snapshot.scenarios),
        "scenarios": [
            _scenario_catalog_entry(
                scenario,
                snapshot.scenario_paths[scenario.id],
                scenarios_dir=snapshot.scenarios_dir,
            )
            for scenario in snapshot.scenarios
        ],
    }


def _split_records(contract: HierarchyContract) -> list[dict[str, Any]]:
    """Each split's question and the estimate that differs between its children."""
    children = children_by_id(contract)
    return [
        {
            "node_id": node.node_id,
            "question": node.classification_question.text,
            "dimension": node.classification_question.dimension,
            "children": list(children[node.node_id]),
            "split_rationale": node.split_rationale,
        }
        for node in contract.nodes
        if children[node.node_id] and node.classification_question is not None
    ]


def validate_scenario_library(
    scenarios_dir: Path,
    *,
    project_root: Path | None = None,
    catalog_path: Path | None = None,
    lineage_register_path: Path | None = None,
    coverage_register_path: Path | None = None,
    source_cards_path: Path | None = None,
    threat_models_path: Path | None = None,
    validate_document_counts: bool = True,
    fix: bool = False,
) -> list[str]:
    """Return scenario-library validation errors; empty list means success.

    Every scenario file is loaded and every error is reported with its path.
    Scenario evidence references resolve against ``project_root``, which also
    holds the registers, the source-cards file and the threat-model register
    unless their paths are given. The wheel does not carry the source cards,
    so validation runs from a repository checkout or an unpacked source
    distribution. A threat model written against a definition its
    scenario no longer has is an error. With ``fix``, a stale catalog and
    documented counts are rewritten instead of reported.
    """
    root = project_root or scenarios_dir.parent
    try:
        snapshot = load_scenario_library_snapshot(scenarios_dir)
    except ScenarioLibraryError as exc:
        return [f"ERROR {error}" for error in exc.errors]
    except Exception as exc:
        return [f"ERROR scenario load failed: {exc}"]

    errors = [
        f"ERROR {path}: the file is named {path.stem!r} but defines {scenario_id!r}; "
        "name each scenario file after its id"
        for scenario_id, path in snapshot.scenario_paths.items()
        if path.stem != scenario_id
    ]
    errors += _validate_evidence_refs(
        snapshot.scenarios, snapshot.scenario_paths, project_root=root
    )
    errors.extend(
        f"ERROR hierarchy contract invalid: {error}"
        for error in validate_hierarchy_contract(
            contract=snapshot.contract, scenarios=snapshot.scenarios
        ).errors
    )
    # Lineage validation is fail-closed: missing or invalid registers are errors.
    lineage_path = lineage_register_path or (
        root / DEFAULT_SCENARIO_LINEAGE_REGISTER_RELATIVE_PATH
    )
    try:
        lineage_register = load_scenario_lineage_register(lineage_path)
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"ERROR {exc}")
    else:
        errors.extend(validate_scenario_lineage(snapshot.scenarios, lineage_register))
    try:
        coverage_register = load_coverage_register(
            coverage_register_path or root / COVERAGE_REGISTER_RELATIVE_PATH
        )
    except ValueError as exc:
        errors.append(f"ERROR {exc}")
    else:
        errors.extend(
            f"ERROR {error}"
            for error in coverage_register_errors(coverage_register, snapshot.scenarios)
        )
    errors.extend(
        f"ERROR {error}"
        for error in source_card_coverage_errors(
            source_cards_path or root / SOURCE_CARDS_RELATIVE_PATH,
            [scenario.id for scenario in snapshot.scenarios],
        )
    )
    register_path = threat_models_path or root / THREAT_MODELS_RELATIVE_PATH
    try:
        threat_models = load_threat_models(register_path)
    except Exception as exc:
        errors.append(f"ERROR {register_path}: {exc}")
    else:
        errors.extend(
            f"ERROR {register_path}: {error}"
            for error in threat_model_errors(
                threat_models.threat_models, snapshot.scenarios
            )
        )
    if errors:
        # Derived artifacts follow a valid library only.
        return errors

    if catalog_path is not None:
        errors.extend(
            _validate_catalog_is_fresh(
                snapshot, catalog_path, lineage_register_path=lineage_path, fix=fix
            )
        )
    if validate_document_counts:
        errors.extend(_validate_documented_counts(root, snapshot, fix=fix))
    return errors


def _facet_projections(
    scenarios: Sequence[ScenarioDefinition],
) -> dict[str, dict[str, Any]]:
    projections: dict[str, dict[str, Any]] = {}
    for axis, additivity in FACET_ROLLUP_AXES.items():
        values: dict[str, list[str]] = {}
        for scenario in scenarios:
            axis_value = scenario_facet_projection(scenario)[axis]
            if isinstance(axis_value, list):
                axis_values = axis_value
            elif isinstance(axis_value, str):
                axis_values = [axis_value]
            else:
                raise TypeError(
                    f"facet projection {axis!r} must contain strings, "
                    f"got {type(axis_value).__name__}"
                )
            for value in axis_values:
                if not isinstance(value, str):
                    raise TypeError(
                        f"facet projection {axis!r} must contain strings, "
                        f"got {type(value).__name__}"
                    )
                values.setdefault(value, []).append(scenario.id)
        projections[axis] = {
            "additivity": additivity,
            "values": {
                value: sorted(scenario_ids)
                for value, scenario_ids in sorted(values.items())
            },
        }
    return projections


def _scenario_catalog_entry(
    scenario: ScenarioDefinition,
    source_path: Path,
    *,
    scenarios_dir: Path,
) -> dict[str, Any]:
    return {
        "id": scenario.id,
        "name": scenario.name,
        "node_ids": list(scenario.node_ids),
        "risk_domain": scenario.risk_domain,
        "lineage": scenario.lineage.model_dump(mode="json"),
        "source_path": _rel_path(source_path, scenarios_dir),
        "severity_occurrence_scope": scenario.severity_occurrence_scope,
        "facets": scenario_facet_projection(scenario),
        "occurrence_unit_id": scenario.occurrence_unit_id,
        "frequency_semantics": _frequency_semantics_entry(scenario),
        "composition_roster": scenario.estimand_composition.roster,
        "severity_governance": _severity_governance_catalog_entry(scenario),
        "empirical_grounding": scenario.empirical_grounding,
        "normalisation_bases": list(scenario.normalisation_bases),
    }


def _frequency_semantics_entry(scenario: ScenarioDefinition) -> dict[str, Any]:
    """The counted population and its event chain with the rendered terminal stage."""
    semantics = scenario.frequency_semantics
    return {
        "counting_identity": semantics.counting_identity,
        "terminal_event_unit": semantics.terminal_event_unit,
        "year_rule": semantics.year_rule,
        "attribution_nexus": scenario.attribution_nexus.model_dump(mode="json"),
        "event_chain": {
            **scenario.event_chain.model_dump(mode="json"),
            "terminal_stage": event_chain_terminal_stage(scenario).model_dump(
                mode="json"
            ),
        },
    }


def _severity_governance_catalog_entry(
    scenario: ScenarioDefinition,
) -> dict[str, Any]:
    """Project the authored tail posture that the assumption register cites."""
    governance = scenario.severity_governance
    return {
        "tail_class": governance.tail_class,
        "gpd_shape": governance.gpd_shape,
        "cap_posture": governance.cap_posture,
        "severity_cap": governance.severity_cap,
        "uncapped_endpoint_stress": governance.uncapped_endpoint_stress,
    }


def _validate_evidence_refs(
    scenarios: Sequence[ScenarioDefinition],
    scenario_paths: dict[str, Path],
    *,
    project_root: Path,
) -> list[str]:
    """Resolve every ``*evidence_refs`` entry of every loaded scenario.

    Each entry is exactly one of: an http(s) URL (not fetched), written
    ``<url>[ (note)]``; one repository file, written ``<path>[#anchor][ (note)]``
    and recognized by a first token containing ``/`` or a document suffix; or a
    nonblank bibliographic citation that names no repository path outside its
    URLs, so a repository file cannot escape resolution by being written as
    prose. A repository path must be a file under the project root; test
    fixtures, ignored local data and the citing scenario's own file are not
    evidence. An anchor must be a heading anchor of a Markdown document, and a
    note must not contain parentheses. No entry may cite elicitation output,
    because scenario definitions are upstream of elicitation.
    """
    errors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    for scenario in scenarios:
        scenario_path = scenario_paths.get(scenario.id)
        for json_path, value in _evidence_ref_entries(
            scenario.model_dump(mode="json"), ""
        ):
            reason = _evidence_ref_error(
                value,
                project_root=project_root,
                scenario_path=scenario_path,
                anchor_cache=anchor_cache,
            )
            if reason is not None:
                errors.append(
                    f"ERROR {scenario.id} {json_path}: evidence ref {value!r} {reason}"
                )
    return errors


def _evidence_ref_entries(node: Any, path: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key).endswith("evidence_refs") and isinstance(value, list):
                entries.extend(
                    (f"{child_path}[{index}]", str(item))
                    for index, item in enumerate(value)
                )
            else:
                entries.extend(_evidence_ref_entries(value, child_path))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            entries.extend(_evidence_ref_entries(item, f"{path}[{index}]"))
    return entries


def _evidence_ref_error(
    value: str,
    *,
    project_root: Path,
    scenario_path: Path | None,
    anchor_cache: dict[Path, set[str]],
) -> str | None:
    ref = " ".join(value.split())
    if not ref:
        return "must not be blank"
    if _EVIDENCE_RUN_REF_RE.search(ref):
        return "cites a run; scenario evidence must not cite elicitation output"
    if _EVIDENCE_URL_RE.fullmatch(ref):
        return None
    if not _EVIDENCE_PATH_LIKE_RE.match(ref.split(" ", 1)[0]):
        if _citation_names_repository_path(ref, project_root=project_root):
            return (
                "names a repository path inside a citation; cite a repository "
                "file as '<path>[#anchor][ (note)]'"
            )
        return None
    match = _EVIDENCE_PATH_REF_RE.fullmatch(ref)
    if match is None:
        return (
            "must be exactly one repository path written "
            "'<path>[#anchor][ (note)]', with no parentheses inside the note"
        )
    reference_root = project_root.resolve()
    document_path = (reference_root / match.group("path")).resolve()
    try:
        relative_path = document_path.relative_to(reference_root).as_posix()
    except ValueError:
        return "must stay under the project root"
    if relative_path.startswith(_EVIDENCE_TEST_ROOTS):
        return "cites a test fixture; fixtures are not evidence"
    if (
        relative_path.startswith(_EVIDENCE_LOCAL_DATA_ROOT)
        and relative_path not in _EVIDENCE_LOCAL_DATA_TRACKED
    ):
        return "cites ignored local data; capture it in a source bundle instead"
    if not document_path.is_file():
        return "file not found"
    if scenario_path is not None and document_path == scenario_path.resolve():
        return "cites the containing scenario file"
    anchor = match.group("anchor")
    if anchor is None:
        return None
    if document_path.suffix.lower() != ".md":
        return "anchor requires a Markdown document"
    if document_path not in anchor_cache:
        anchor_cache[document_path] = _markdown_anchors(document_path)
    if anchor not in anchor_cache[document_path]:
        return "anchor not found"
    return None


def _citation_names_repository_path(citation: str, *, project_root: Path) -> bool:
    """Whether a bibliographic citation names a repository path outside its URLs.

    A token names a repository path when it carries a document suffix, or when it
    contains ``/`` and its first segment is an entry at the project root. Prose
    such as ``war/hostile-act`` therefore remains part of a citation.
    """
    text = _EVIDENCE_URL_TOKEN_RE.sub(" ", citation)
    for token in _EVIDENCE_CITATION_TOKEN_RE.findall(text):
        if _EVIDENCE_DOCUMENT_TOKEN_RE.search(token):
            return True
        head, separator, _ = token.partition("/")
        if separator and head and (project_root / head).exists():
            return True
    return False


def _markdown_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    anchor_counts: dict[str, int] = {}
    in_fence = False
    fence_char = ""
    fence_length = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        leading_spaces = len(line) - len(line.lstrip(" "))
        stripped = line.lstrip()
        fence_match = re.match(r"(`{3,}|~{3,})", stripped)
        if leading_spaces <= 3 and fence_match is not None:
            marker = fence_match.group(1)
            if in_fence:
                rest = stripped[len(marker) :]
                if marker.startswith(fence_char * fence_length) and not rest.strip():
                    in_fence = False
                    fence_char = ""
                    fence_length = 0
            else:
                in_fence = True
                fence_char = marker[0]
                fence_length = len(marker)
            continue

        if in_fence or leading_spaces >= 4 or line.startswith("\t"):
            continue

        heading_match = re.match(r"#{1,6}(?:\s+|$)(?P<heading>.*)", stripped)
        if heading_match is None:
            continue
        heading = heading_match.group("heading").strip()
        base_anchor = _markdown_anchor(heading)
        if not base_anchor:
            continue
        prior_count = anchor_counts.get(base_anchor, 0)
        anchor_counts[base_anchor] = prior_count + 1
        anchor = base_anchor if prior_count == 0 else f"{base_anchor}-{prior_count}"
        anchors.add(anchor)
    return anchors


def _markdown_anchor(heading: str) -> str:
    normalized = heading.strip().lower()
    normalized = re.sub(r"`([^`]*)`", r"\1", normalized)
    normalized = re.sub(r"<[^>]+>", "", normalized)
    normalized = re.sub(r"[^\w\s-]", "", normalized)
    normalized = re.sub(r"\s+", "-", normalized)
    return normalized.strip("-")


def _validate_catalog_is_fresh(
    snapshot: ScenarioLibrarySnapshot,
    catalog_path: Path,
    *,
    lineage_register_path: Path | None = None,
    fix: bool = False,
) -> list[str]:
    expected = scenario_catalog_json(
        build_scenario_catalog_from_snapshot(
            snapshot, lineage_register_path=lineage_register_path
        )
    )
    actual = catalog_path.read_text(encoding="utf-8") if catalog_path.exists() else None
    if actual == expected:
        return []
    if fix:
        write_scenario_catalog(json.loads(expected), catalog_path)
        return []
    return [
        f"ERROR generated scenario catalog {catalog_path} is "
        f"{'missing' if actual is None else 'stale'}; run "
        "`uv run python scripts/validate_scenario_library.py --fix`"
    ]


def _validate_documented_counts(
    root: Path,
    snapshot: ScenarioLibrarySnapshot,
    *,
    fix: bool = False,
) -> list[str]:
    errors: list[str] = []
    for relative_path, expected_keys in DOCUMENT_COUNT_MARKERS.items():
        path = root / relative_path
        if not path.exists():
            errors.append(f"ERROR scenario count markers: document not found at {path}")
            continue
        text = path.read_text(encoding="utf-8")
        marker_matches = list(_COUNT_MARKER_RE.finditer(text))
        if not marker_matches:
            errors.append(
                f"ERROR scenario count markers: no markers found in {relative_path}"
            )
            continue
        found_keys = {match.group("key").lower() for match in marker_matches}
        for expected_key in expected_keys:
            if expected_key not in found_keys:
                errors.append(
                    f"ERROR {relative_path}: missing "
                    f"scenario-count:{expected_key} marker"
                )
        replacements: list[tuple[int, int, str]] = []
        for match in marker_matches:
            key = match.group("key").lower()
            actual = int(match.group("value"))
            try:
                expected = _expected_document_count(
                    key,
                    snapshot,
                )
            except ValueError as exc:
                errors.append(f"ERROR {relative_path}: {exc}")
                continue
            if expected is None:
                continue
            if actual != expected:
                if fix:
                    replacements.append(
                        (match.start("value"), match.end("value"), str(expected))
                    )
                else:
                    errors.append(
                        f"ERROR {relative_path} scenario-count:{key}: documented count "
                        f"{actual} != {expected}"
                    )
        if replacements:
            for start, end, replacement in reversed(replacements):
                text = text[:start] + replacement + text[end:]
            path.write_text(text, encoding="utf-8")
    return errors


def _expected_document_count(
    key: str,
    snapshot: ScenarioLibrarySnapshot,
) -> int | None:
    if key == "cells":
        return snapshot.cell_count
    if key == "normalisation_bases":
        return snapshot.normalisation_base_count
    if key == "risk_domains":
        return snapshot.risk_domain_count
    if key == "questions":
        # A complete catalog's initial questions per coordinate and respondent:
        # one direct and one chain question per scenario, one per parent.
        return 2 * snapshot.cell_count + snapshot.hierarchy_counts["parents"]
    if key in snapshot.hierarchy_counts:
        return snapshot.hierarchy_counts[key]
    if key.startswith("origin."):
        origin = key.removeprefix("origin.")
        if origin not in snapshot.scenario_counts_by_origin:
            raise ValueError(f"unknown scenario-count origin marker {key!r}")
        return snapshot.scenario_counts_by_origin[origin]
    if key.startswith("cells."):
        domain = key.removeprefix("cells.")
        if domain not in snapshot.scenario_counts_by_domain:
            raise ValueError(f"unknown scenario-count domain marker {key!r}")
        return snapshot.scenario_counts_by_domain[domain]
    raise ValueError(f"unknown scenario-count marker {key!r}")


def _rel_path(path: Path, scenarios_dir: Path) -> str:
    relative_path = path.relative_to(scenarios_dir)
    if scenarios_dir.name == "scenarios":
        return (Path(scenarios_dir.name) / relative_path).as_posix()
    return relative_path.as_posix()
