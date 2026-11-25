import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from catmodel.scenarios.base_rates import SOURCE_CARDS_RELATIVE_PATH
from catmodel.scenarios.catalog import (
    DOCUMENT_COUNT_MARKERS,
    build_scenario_catalog_from_snapshot,
    load_scenario_library_snapshot,
    validate_scenario_library,
)
from catmodel.scenarios.coverage_register import COVERAGE_REGISTER_RELATIVE_PATH
from catmodel.scenarios.loader import (
    effective_loss_population_contract,
    load_scenarios,
    rendered_severity_occurrence_scope,
)
from catmodel.scenarios.taxonomy_identity import (
    DEFAULT_SCENARIO_LINEAGE_REGISTER_RELATIVE_PATH,
)

_PROJECT_ROOT = Path(__file__).parent.parent


def test_renderer_owns_reserved_fine_prose(all_scenarios):
    for scenario in all_scenarios:
        assert (
            "consequential civil fines"
            not in " ".join(scenario.severity_occurrence_scope.split()).lower()
        )
        assert "{{loss_component:" not in rendered_severity_occurrence_scope(scenario)


@pytest.fixture(scope="session")
def scenarios_dir():
    return _PROJECT_ROOT / "scenarios"


@pytest.fixture(scope="session")
def all_scenarios(scenarios_dir):
    return load_scenarios(scenarios_dir)


def _cited_evidence_files(scenarios) -> set[Path]:
    """Repository files named by any scenario ``*evidence_refs`` entry."""

    def refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("evidence_refs") and isinstance(value, list):
                    yield from value
                else:
                    yield from refs(value)
        elif isinstance(node, list):
            for item in node:
                yield from refs(item)

    candidates = {
        Path(ref.split()[0].split("#")[0])
        for scenario in scenarios
        for ref in refs(scenario.model_dump(mode="json"))
    }
    return {path for path in candidates if (_PROJECT_ROOT / path).is_file()}


@pytest.fixture
def library_project(tmp_path, scenarios_dir, all_scenarios) -> Path:
    """A project root holding copies of every file library validation reads.

    Tests edit the copies; the cited evidence files are copied too, so evidence
    references resolve exactly as in the repository.
    """
    project_root = tmp_path / "project"
    shutil.copytree(scenarios_dir, project_root / "scenarios")
    for relative_path in {
        *DOCUMENT_COUNT_MARKERS,
        Path("assumptions.md"),
        DEFAULT_SCENARIO_LINEAGE_REGISTER_RELATIVE_PATH,
        COVERAGE_REGISTER_RELATIVE_PATH,
        SOURCE_CARDS_RELATIVE_PATH,
        *_cited_evidence_files(all_scenarios),
    }:
        target = project_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_PROJECT_ROOT / relative_path, target)
    return project_root


def _validate(project_root: Path, **options) -> list[str]:
    return validate_scenario_library(
        project_root / "scenarios",
        project_root=project_root,
        catalog_path=project_root / "scenarios" / "scenario_catalog.json",
        **options,
    )


def _edit_scenario(project_root: Path, relative: str, edit) -> dict:
    path = project_root / "scenarios" / relative
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def _edit_source_cards(project_root: Path, edit) -> Path:
    path = project_root / SOURCE_CARDS_RELATIVE_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


class TestScenarioCatalog:
    def test_catalog_meta_carries_taxonomy_identity(self, scenarios_dir):
        from catmodel.scenarios.taxonomy_identity import (
            load_scenario_lineage_register,
            scenario_roster_sha256,
        )

        catalog = build_scenario_catalog_from_snapshot(
            load_scenario_library_snapshot(scenarios_dir)
        )
        meta = catalog["meta"]
        assert meta["scenario_taxonomy_version"] == (
            load_scenario_lineage_register().current_taxonomy_version
        )
        assert meta["scenario_roster_sha256"] == scenario_roster_sha256(
            [entry["id"] for entry in catalog["scenarios"]]
        )

    def test_scenario_library_validation_passes(self, scenarios_dir):
        errors = validate_scenario_library(
            scenarios_dir,
            project_root=_PROJECT_ROOT,
            catalog_path=scenarios_dir / "scenario_catalog.json",
        )
        assert errors == []

    def test_maintainer_script_is_the_single_entry_point(self, library_project):
        catalog = library_project / "scenarios" / "scenario_catalog.json"
        catalog.unlink()
        explorer = library_project / "docs" / "taxonomy-explorer.html"
        explorer.parent.mkdir(parents=True, exist_ok=True)
        explorer.write_text("stale", encoding="utf-8")
        command = [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "validate_scenario_library.py"),
            "--project-root",
            str(library_project),
            "--scenarios-dir",
            str(library_project / "scenarios"),
        ]
        stale = subprocess.run(command, capture_output=True, text=True)
        assert stale.returncode == 1
        assert "scenario_catalog.json is missing" in stale.stderr
        fixed = subprocess.run([*command, "--fix"], capture_output=True, text=True)
        assert fixed.returncode == 0, fixed.stderr
        assert json.loads(catalog.read_text(encoding="utf-8"))["scenarios"]
        assert explorer.read_text(encoding="utf-8") == (
            _PROJECT_ROOT / "docs" / "taxonomy-explorer.html"
        ).read_text(encoding="utf-8")
        assert subprocess.run(command, capture_output=True).returncode == 0

    def test_stale_catalog_is_reported_and_fixed(self, library_project):
        catalog = library_project / "scenarios" / "scenario_catalog.json"
        catalog.write_text("{}\n", encoding="utf-8")
        assert any("is stale" in error for error in _validate(library_project))
        assert _validate(library_project, fix=True) == []
        assert catalog.read_text(encoding="utf-8") == (
            _PROJECT_ROOT / "scenarios" / "scenario_catalog.json"
        ).read_text(encoding="utf-8")

    def test_readme_domain_count_drift_rejected(self, library_project):
        readme = library_project / "README.md"
        readme.write_text(
            readme.read_text(encoding="utf-8").replace(
                "scenario-count:risk_domains -->8",
                "scenario-count:risk_domains -->7",
                1,
            ),
            encoding="utf-8",
        )
        assert _validate(library_project) == [
            "ERROR README.md scenario-count:risk_domains: documented count 7 != 8"
        ]

    def test_readme_non_count_updates_allowed(self, library_project):
        readme = library_project / "README.md"
        text = readme.read_text(encoding="utf-8")
        readme.write_text(
            text + "\nA sentence that states no count.\n", encoding="utf-8"
        )
        assert _validate(library_project) == []

    def test_document_count_fix_rewrites_stale_markers(
        self, library_project, all_scenarios
    ):
        cells = len(all_scenarios)
        stale = {
            Path("README.md"): "cells",
            Path("docs/taxonomy.md"): "leaves",
            Path("runbook.md"): "questions",
        }
        for relative_path, key in stale.items():
            path = library_project / relative_path
            path.write_text(
                re.sub(
                    rf"(scenario-count:{re.escape(key)} -->)\d+",
                    r"\g<1>1000",
                    path.read_text(encoding="utf-8"),
                ),
                encoding="utf-8",
            )
        assert len(_validate(library_project)) >= len(stale)
        assert _validate(library_project, fix=True) == []
        assert f"scenario-count:cells -->{cells}<" in (
            library_project / "README.md"
        ).read_text(encoding="utf-8")
        assert _validate(library_project) == []

    def test_missing_document_count_marker_rejected(self, library_project):
        readme = library_project / "README.md"
        readme.write_text(
            re.sub(
                r"<!-- scenario-count:normalisation_bases -->(\d+)<!-- /scenario-count -->",
                r"\1",
                readme.read_text(encoding="utf-8"),
            ),
            encoding="utf-8",
        )
        assert _validate(library_project) == [
            "ERROR README.md: missing scenario-count:normalisation_bases marker"
        ]

    def test_catalog_reports_counts_grounding_and_splits(
        self, scenarios_dir, all_scenarios
    ):
        catalog = build_scenario_catalog_from_snapshot(
            load_scenario_library_snapshot(scenarios_dir)
        )
        assert catalog["scenario_cell_count"] == len(all_scenarios)
        assert catalog["normalisation_base_count"] == sum(
            1 for scenario in all_scenarios if scenario.normalisation_bases
        )
        assert sum(
            record["scenario_cell_count"] for record in catalog["risk_domains"].values()
        ) == len(all_scenarios)
        assert catalog["taxonomy_rule"].startswith("The tree splits losses")
        splits = catalog["splits"]
        assert all(split["split_rationale"] for split in splits)
        assert {split["node_id"] for split in splits} >= {"omega", "n_misuse"}
        grounding = {
            scenario.id: scenario.empirical_grounding for scenario in all_scenarios
        }
        fields = set(catalog["scenarios"][0])
        for entry in catalog["scenarios"]:
            # Every entry states what evidence exists for its population.
            assert entry["empirical_grounding"] == grounding[entry["id"]]
            assert entry["empirical_grounding"].strip()
            assert set(entry) == fields
        # An entry reports the population, its placement and its governed
        # severity and frequency definitions; nothing about insurance.
        assert fields == {
            "id",
            "name",
            "node_ids",
            "risk_domain",
            "lineage",
            "source_path",
            "severity_occurrence_scope",
            "facets",
            "occurrence_unit_id",
            "frequency_semantics",
            "composition_roster",
            "severity_governance",
            "empirical_grounding",
            "normalisation_bases",
        }
        assert set(catalog["facet_projections"]) == {
            "causal_origin",
            "peril_mechanism",
            "lifecycle_stage",
            "deployment_control_mode",
            "harm_types",
            "bearers",
            "party_basis",
        }


class TestLibraryRules:
    def test_every_invalid_file_is_reported_with_its_path(self, library_project):
        def blank_name(payload):
            payload["name"] = ""

        def no_stages(payload):
            payload["event_chain"]["stages"] = []

        _edit_scenario(
            library_project, "misuse/compromise_organizations.yaml", blank_name
        )
        _edit_scenario(library_project, "failure/output_reliance_legal.yaml", no_stages)
        errors = _validate(library_project)
        assert any(
            "compromise_organizations.yaml: name: Value error, text must not be blank"
            in error
            for error in errors
        )
        assert any(
            "output_reliance_legal.yaml: event_chain.stages" in error
            for error in errors
        )

    def test_scenario_files_are_named_after_their_id(self, library_project):
        source = library_project / "scenarios" / "misuse" / "input_hijack.yaml"
        source.rename(source.with_name("hijack.yaml"))
        errors = _validate(library_project)
        assert len(errors) == 1
        assert "named 'hijack' but defines 'input_hijack'" in errors[0]

    def test_normalisation_bases_need_a_coverage_gap_row(self, library_project):
        register_path = library_project / COVERAGE_REGISTER_RELATIVE_PATH
        register = yaml.safe_load(register_path.read_text(encoding="utf-8"))
        register["gaps"] = [
            gap
            for gap in register["gaps"]
            if gap["gap_id"] != "model_asset_compromise_exposure_normalisation"
        ]
        register["gaps"][0]["scenario_ids"].append("retired_scenario")
        register_path.write_text(yaml.safe_dump(register), encoding="utf-8")
        errors = _validate(library_project)
        assert any(
            "names unknown scenarios ['retired_scenario']" in error for error in errors
        )
        assert any(
            error.startswith("ERROR model_asset_compromise: its public exposure base")
            for error in errors
        )

    def test_source_cards_list_every_scenario(self, library_project):
        cards = _edit_source_cards(
            library_project, lambda payload: payload["scenarios"].pop("input_hijack")
        )
        assert _validate(library_project) == [
            f"ERROR input_hijack: {cards} does not list it under scenarios; list "
            "every scenario there, with [] where its questions show no source cards"
        ]
        # The remedy the error names: an entry that shows no cards.
        _edit_source_cards(
            library_project,
            lambda payload: payload["scenarios"].update(input_hijack=[]),
        )
        assert _validate(library_project) == []

    def test_source_cards_list_only_library_scenarios(self, library_project):
        cards = _edit_source_cards(
            library_project,
            lambda payload: payload["scenarios"].update(retired_scenario=[]),
        )
        assert _validate(library_project) == [
            f"ERROR {cards}: scenarios lists 'retired_scenario', which is not a "
            "library scenario; remove the entry or correct its id"
        ]

    def test_maintainer_script_reports_a_scenario_the_source_cards_omit(
        self, library_project
    ):
        cards = _edit_source_cards(
            library_project, lambda payload: payload["scenarios"].pop("input_hijack")
        )
        result = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "validate_scenario_library.py"),
                "--project-root",
                str(library_project),
                "--scenarios-dir",
                str(library_project / "scenarios"),
                "--fix",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert (
            f"ERROR input_hijack: {cards.resolve()} does not list it under scenarios"
            in result.stderr
        )

    def test_estimated_leaf_cannot_be_marked_unestimated(self, all_scenarios):
        from catmodel.scenarios.hierarchy_contract import (
            HierarchyContract,
            cached_hierarchy_contract,
            validate_hierarchy_contract,
        )

        payload = cached_hierarchy_contract().model_dump(mode="json")
        owned = all_scenarios[0].node_ids[0]
        for node in payload["nodes"]:
            if node["node_id"] == owned:
                node["unestimated_disposition"] = {"disposition": "unestimated"}
        errors = validate_hierarchy_contract(
            contract=HierarchyContract.model_validate(payload),
            scenarios=all_scenarios,
        ).errors
        assert errors == (
            f"terminal {owned} is estimated by {all_scenarios[0].id} but is marked "
            "unestimated; remove its unestimated_disposition",
        )


def _scenario_evidence_ref_errors(
    scenarios_dir: Path,
    project_root: Path,
    ref: str,
    *,
    scenario_file: str = "misuse/compromise_organizations.yaml",
    ref_list_path: tuple[str | int, ...] = ("severity_governance", "evidence_refs"),
) -> list[str]:
    """Author ``ref`` as the only entry of one reference list; return its errors."""
    copied_scenarios = project_root / "scenarios"
    shutil.copytree(scenarios_dir, copied_scenarios)
    if not (project_root / "assumptions.md").exists():
        shutil.copy(_PROJECT_ROOT / "assumptions.md", project_root / "assumptions.md")
    scenario_path = copied_scenarios / scenario_file
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    container = payload
    for key in ref_list_path[:-1]:
        container = container[key]
    container[ref_list_path[-1]] = [ref]
    scenario_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    errors = validate_scenario_library(
        copied_scenarios,
        project_root=project_root,
        lineage_register_path=_PROJECT_ROOT
        / DEFAULT_SCENARIO_LINEAGE_REGISTER_RELATIVE_PATH,
        coverage_register_path=_PROJECT_ROOT / COVERAGE_REGISTER_RELATIVE_PATH,
        source_cards_path=_PROJECT_ROOT / SOURCE_CARDS_RELATIVE_PATH,
        validate_document_counts=False,
    )
    return [error for error in errors if error.startswith(f"ERROR {payload['id']} ")]


class TestScenarioEvidenceRefs:
    _GRAMMAR = (
        "must be exactly one repository path written '<path>[#anchor][ (note)]', "
        "with no parentheses inside the note"
    )
    _RUN = "cites a run; scenario evidence must not cite elicitation output"
    _FIXTURE = "cites a test fixture; fixtures are not evidence"
    _EMBEDDED = (
        "names a repository path inside a citation; cite a repository file as "
        "'<path>[#anchor][ (note)]'"
    )

    @pytest.mark.parametrize(
        ("ref", "present_file", "reason"),
        [
            ("data/gdpr_fines.csv", None, "file not found"),
            (
                "tests/fixtures/source_bundle/severity_calibration.yaml",
                "tests/fixtures/source_bundle/severity_calibration.yaml",
                _FIXTURE,
            ),
            ("test_support/fixture.yaml", "test_support/fixture.yaml", _FIXTURE),
            (
                "data/external/gdpr/enforcement_tracker.json",
                "data/external/gdpr/enforcement_tracker.json",
                "cites ignored local data; capture it in a source bundle instead",
            ),
            ("../outside.md", "../outside.md", "must stay under the project root"),
            ("assumptions.md#no-such-heading", None, "anchor not found"),
            (
                "scenarios/severity_governance_policies.yaml#policies",
                None,
                "anchor requires a Markdown document",
            ),
            ("elicitation run b967451267ad (canary campaign pool)", None, _RUN),
            ("data/runs/b967451267ad/responses.json", None, _RUN),
            ("run://b967451267ad", None, _RUN),
            (
                "assumptions.md#annual-frequency-and-source-pooling and "
                "assumptions.md#severity-and-tails (NPDB anchors)",
                None,
                _GRAMMAR,
            ),
            (
                "scenarios/misuse/compromise_organizations.yaml event_chain block",
                None,
                _GRAMMAR,
            ),
            ("assumptions.md#severity-and-tails (shape (xi 0.5))", None, _GRAMMAR),
            (
                "scenarios/misuse/compromise_organizations.yaml (event_chain block)",
                None,
                "cites the containing scenario file",
            ),
            # A repository path written as prose would otherwise escape resolution.
            ("GDPR fines (data/gdpr_fines.csv)", None, _EMBEDDED),
            ("See tests/fixtures/source_bundle/x.yaml", None, _EMBEDDED),
            ("Register (assumptions.md#nope)", None, _EMBEDDED),
            ("Internal notes under scenarios/cyber", None, _EMBEDDED),
        ],
    )
    def test_invalid_evidence_ref_rejected(
        self, scenarios_dir, tmp_path, ref, present_file, reason
    ):
        project_root = tmp_path / "project"
        if present_file is not None:
            present_path = project_root / present_file
            present_path.parent.mkdir(parents=True, exist_ok=True)
            present_path.write_text("## Present\n", encoding="utf-8")
        errors = _scenario_evidence_ref_errors(scenarios_dir, project_root, ref)
        assert errors == [
            f"ERROR compromise_organizations severity_governance.evidence_refs[0]: "
            f"evidence ref {ref!r} {reason}"
        ]

    @pytest.mark.parametrize(
        ("appended", "anchor", "found"),
        [
            ("\n```md\n## Fenced Only Heading\n```\n", "fenced-only-heading", False),
            (
                "\n```text\n```python\n## Still Fenced Heading\n```\n",
                "still-fenced-heading",
                False,
            ),
            ("\n    ## Indented Only Heading\n", "indented-only-heading", False),
            ("\n#Not A Heading Token\n", "not-a-heading-token", False),
            ("\n## Repeated Anchor\n\n## Repeated Anchor\n", "repeated-anchor-1", True),
        ],
    )
    def test_anchors_are_markdown_headings(
        self, scenarios_dir, tmp_path, appended, anchor, found
    ):
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "assumptions.md").write_text(
            (_PROJECT_ROOT / "assumptions.md").read_text(encoding="utf-8") + appended,
            encoding="utf-8",
        )
        errors = _scenario_evidence_ref_errors(
            scenarios_dir, project_root, f"assumptions.md#{anchor}"
        )
        assert (errors == []) == found, errors

    @pytest.mark.parametrize(
        "ref",
        [
            "https://www.enforcementtracker.com/",
            "https://www.enforcementtracker.com/ (CMS GDPR Enforcement Tracker)",
            "Merck & Co. v. ACE American Insurance (war/hostile-act exclusion, cyber)",
            "Mata v. Avianca, S.D.N.Y. 2023",
            "CNIL decision SAN-2022-019 (https://www.cnil.fr/decision.pdf)",
            "assumptions.md",
            "assumptions.md#severity-and-tails (governed tail shape and cap)",
            "data/external/README.md#source-inventory-and-applicability (tracked "
            "source inventory)",
            "scenarios/misuse/compromise_organizations.yaml",
        ],
    )
    def test_valid_evidence_ref_accepted(self, scenarios_dir, tmp_path, ref):
        project_root = tmp_path / "project"
        (project_root / "data" / "external").mkdir(parents=True)
        shutil.copy(
            _PROJECT_ROOT / "data" / "external" / "README.md",
            project_root / "data" / "external" / "README.md",
        )
        errors = _scenario_evidence_ref_errors(
            scenarios_dir,
            project_root,
            ref,
            scenario_file="misuse/model_asset_compromise.yaml",
        )
        assert errors == []

    def test_reference_lists_a_shared_policy_supplies_are_resolved(
        self, scenarios_dir, tmp_path
    ):
        """A scenario takes its sensitivity evidence from its severity policy, so
        a reference authored there is resolved for every scenario that names it."""
        project_root = tmp_path / "project"
        copied = project_root / "scenarios"
        shutil.copytree(scenarios_dir, copied)
        shutil.copy(_PROJECT_ROOT / "assumptions.md", project_root / "assumptions.md")
        policies_path = copied / "severity_governance_policies.yaml"
        policies = yaml.safe_load(policies_path.read_text(encoding="utf-8"))
        policies["policies"]["governed_tail_shape"]["sensitivity_evidence_refs"] = [
            "data/gdpr_fines.csv"
        ]
        policies_path.write_text(
            yaml.safe_dump(policies, sort_keys=False), encoding="utf-8"
        )
        errors = validate_scenario_library(
            copied,
            project_root=project_root,
            lineage_register_path=_PROJECT_ROOT
            / DEFAULT_SCENARIO_LINEAGE_REGISTER_RELATIVE_PATH,
            coverage_register_path=_PROJECT_ROOT / COVERAGE_REGISTER_RELATIVE_PATH,
            source_cards_path=_PROJECT_ROOT / SOURCE_CARDS_RELATIVE_PATH,
            validate_document_counts=False,
        )
        assert (
            "ERROR output_reliance_health "
            "severity_governance.sensitivity_evidence_refs[0]: "
            "evidence ref 'data/gdpr_fines.csv' file not found"
        ) in errors


def test_event_chains_name_the_lab_through_the_context_placeholder(all_scenarios):
    for scenario in all_scenarios:
        chain = scenario.event_chain
        assert "the named lab" in chain.opportunity.unit, scenario.id
        texts = (
            chain.opportunity.unit,
            chain.opportunity.counting_rule,
            chain.opportunity.cohort,
            *(stage.event for stage in chain.stages),
        )
        for text in texts:
            for other_name in ("named provider", "selected provider", "provider-year"):
                assert other_name not in text, (scenario.id, other_name)


def _sentences(text: str) -> list[str]:
    """Split prose at sentence ends; short fragments carry no restated rule."""
    return [
        sentence
        for sentence in re.split(r"(?<=[.?!])\s+", " ".join(text.split()))
        if len(sentence) >= 40
    ]


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def _population_contract_texts(scenario) -> list[str]:
    """Distinct population texts rendered to respondents.

    The loss-population contract echoes frequency fields by construction, so
    each distinct text counts once.
    """
    texts = _strings(
        [
            scenario.frequency_semantics.model_dump(mode="json"),
            effective_loss_population_contract(scenario),
            rendered_severity_occurrence_scope(scenario),
        ]
    )
    return list(dict.fromkeys(texts))


def test_description_does_not_restate_population_contract(all_scenarios):
    # The population contract is authored once; a contract sentence repeated in
    # the description invites reading the repeat as a second rule.
    for scenario in all_scenarios:
        summary = _sentences(scenario.description)
        contract = {
            sentence
            for text in _population_contract_texts(scenario)
            for sentence in _sentences(text)
        }
        assert len(summary) == len(set(summary)), scenario.id
        assert not contract.intersection(summary), scenario.id
