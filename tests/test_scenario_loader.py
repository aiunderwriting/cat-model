import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from catmodel.elicitation.contextual_risk import contextual_node_from_scenario
from catmodel.scenarios.facets import admissible_lifecycle_stages
from catmodel.scenarios.hierarchy_contract import (
    admission_clause,
    admission_threshold,
    cached_hierarchy_contract,
    origin_role_of,
    reporting_domain_of,
)
from catmodel.scenarios.loader import (
    COMPOSITION_ROSTERS_FILENAME,
    SEVERITY_GOVERNANCE_POLICIES_FILENAME,
    AttributionNexusDef,
    CompositionCellDef,
    CompositionRosters,
    EstimandCompositionDef,
    EventChainDef,
    EventChainStageDef,
    FrequencySemanticsDef,
    ScenarioDefinition,
    ScenarioLibraryError,
    SeverityGovernanceDef,
    SeverityGovernancePolicies,
    SharedLibrary,
    TailClassDef,
    TailClassEndpointDef,
    effective_loss_population_contract,
    event_chain_terminal_stage,
    expand_shared_content,
    load_scenarios,
    load_severity_governance_policies,
    load_shared_library,
    qualified_terminal_unit,
    rendered_severity_occurrence_scope,
    resolve_estimand_composition,
    resolve_severity_governance_block,
)
from catmodel.scenarios.sensitivity_coverage import (
    TAIL_SHAPE_SENSITIVITY_STEP,
    shifted_tail_shape,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS_DIR = _REPO_ROOT / "scenarios"

_TEST_NODE_BY_DOMAIN = {
    "cyber": "n_compromise_organizations",
    "eo": "n_reliance_other",
    "cbrn": "n_cbrn_chemical",
    "agent": "n_failure_autonomous_action",
    "ai_rd": "n_failure_developer_control",
    "societal": "n_influence",
    "svc": "n_service_outage",
    "regulatory": "n_claims_other",
}

_TERMINAL = "One governed terminal test event {{admission}}."
_CELLS = [
    {
        "cell_id": "victim_resources",
        "loss_form": "Victims' repair and response resources.",
        "nature": "resource_cost",
        "subject": "third_party",
    },
    {
        "cell_id": "provider_civil_fine",
        "loss_form": "consequential civil fines on the selected provider",
        "nature": "transfer",
        "subject": "provider",
        "loss_type": "civil_fines",
        "attachment_basis": "incident_consequence",
    },
]


def _facets(node_id: str) -> dict:
    try:
        hierarchy = cached_hierarchy_contract()
        domain = reporting_domain_of(hierarchy, node_id)
        origin_regime, ai_role = origin_role_of(hierarchy, node_id)
        allowed_lifecycle_stages = admissible_lifecycle_stages(origin_regime, ai_role)
    except (KeyError, ValueError):
        domain, ai_role = "cyber", None
        allowed_lifecycle_stages = frozenset({"external_deployment"})
    facets = {
        "peril_mechanism": ["technical_intrusion"]
        if domain in {"cyber", "cbrn", "societal"}
        else ["erroneous_output_reliance"]
        if domain == "eo"
        else ["autonomous_goal_pursuit"]
        if domain in {"agent", "ai_rd"}
        else ["service_availability_failure"]
        if domain == "svc"
        else ["enforcement_process"],
        "deployment_control_mode": "any_copy"
        if ai_role == "assisted_attacker"
        else "api",
        "harm_types": ["economic_loss"],
        "bearers": ["third_party_organizations"],
        "party_basis": "first_party" if domain == "regulatory" else "third_party",
    }
    if len(allowed_lifecycle_stages) > 1:
        facets["lifecycle_stage"] = sorted(allowed_lifecycle_stages)[0]
    return facets


def _minimal_scenario(**overrides) -> dict:
    base = {
        "id": "test_scenario",
        "name": "Test Scenario",
        "node_ids": [_TEST_NODE_BY_DOMAIN["cyber"]],
        "lineage": {"introduced_in": "r1"},
        "description": "A test scenario.",
        "estimand_composition": {
            "status": "OK",
            "rationale": "The minimal test severity has two classified cells.",
            "cells": [dict(cell) for cell in _CELLS],
        },
        "severity_occurrence_scope": "The complete causal gross economic loss for one governed test event, including {{loss_component:provider_civil_fine}}.",
        "severity_governance": {
            "gpd_shape": 0.5,
            "cap_posture": "uncapped",
            "source_classification": "internal_judgment",
            "status": "INITIAL_ESTIMATE",
            "materiality": "high",
            "rationale": "Minimal reviewed severity policy.",
            "empirical_grounding": "Judgmental test value.",
            "sensitivity_range": "Test gpd_shape plus or minus 0.1.",
            "evidence_refs": ["tests/test_scenario_loader.py"],
        },
        "frequency_semantics": _frequency_semantics(),
        "attribution_nexus": {
            "kind": "model_contribution",
            "terminal_qualification": "The assigned provider's model materially contributes.",
        },
        "occurrence_unit_id": "test_terminal_event",
        "empirical_grounding": "No governed dataset supports this test population.",
    }
    base.update(overrides)
    base.setdefault("event_chain", _event_chain())
    base.setdefault("facets", _facets(base["node_ids"][0]))
    return base


def _policy_reference_block(**overrides) -> dict:
    block = {
        "policy": "governed_tail_shape",
        "tail_class": "liability_enforcement",
        "tail_class_rationale": "One claim bounded by the claimant's interest.",
        "cap_posture": "uncapped",
        "evidence_refs": ["assumptions.md"],
    }
    block.update(overrides)
    return block


def _scenario_file(**overrides) -> dict:
    """A scenario as its file authors it: governance by named policy."""
    return _minimal_scenario(
        **{"severity_governance": _policy_reference_block(), **overrides}
    )


def _copy_shared_files(scenarios_dir: Path) -> None:
    for name in (SEVERITY_GOVERNANCE_POLICIES_FILENAME, COMPOSITION_ROSTERS_FILENAME):
        (scenarios_dir / name).write_text((_SCENARIOS_DIR / name).read_text())


def _event_chain(**overrides) -> dict:
    base = {
        "opportunity": {
            "unit": "One governed test opportunity.",
            "counting_rule": "One opportunity per governed test attempt.",
            "cohort": "Test attempts begun in the study year.",
        },
        "exposure": {
            "public": {
                "unit": "1 million weekly active users of the consumer assistant",
                "basis": "normalisation",
                "count": {
                    "kind": "derived",
                    "scale_id": "consumer_weekly_users",
                    "factor": 1e-6,
                    "rationale": "Governed test base.",
                    "source": "Test fixture.",
                },
                "derivation": "Weekly users divided by 1 million.",
            },
            "internal": {
                "unit": "pre-release evaluation organization",
                "basis": "judgment",
                "count": {
                    "kind": "fixed",
                    "count": 12,
                    "rationale": "Governed test base.",
                    "source": "Test fixture.",
                },
                "derivation": "Fixed by the scenario at twelve evaluation organizations.",
            },
        },
        "stages": [
            {"stage_id": "attempt_succeeds", "event": "The governed attempt succeeds."}
        ],
    }
    base.update(overrides)
    return base


def _frequency_semantics(**overrides) -> dict:
    base = {
        "counting_identity": "One independently resolvable governed exposure is one unit; repeated actions inside it do not create additional units.",
        "terminal_event_unit": _TERMINAL,
        "year_rule": "A test event is assigned to the year it occurs.",
    }
    base.update(overrides)
    return base


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def _expanded_terminal() -> str:
    return _TERMINAL.replace("{{admission}}", admission_clause())


class TestFrequencySemanticsDef:
    @pytest.mark.parametrize("field_name", ["counting_identity", "terminal_event_unit"])
    def test_rejects_blank_semantic_text(self, field_name: str):
        payload = _frequency_semantics()
        payload[field_name] = " \n\t "
        with pytest.raises(ValueError, match="must not be blank"):
            FrequencySemanticsDef(**payload)

    def test_rejects_blank_attribution_terminal_qualification(self):
        with pytest.raises(ValueError, match="must not be blank"):
            AttributionNexusDef(
                kind="model_contribution", terminal_qualification=" \n\t "
            )

    def test_terminal_unit_takes_the_admission_threshold_from_the_contract(self):
        semantics = FrequencySemanticsDef(**_frequency_semantics())
        assert semantics.terminal_event_unit == _expanded_terminal()
        amount = f"USD {admission_threshold():,.0f}"
        assert f"is at least {amount} (exactly {amount} is admitted)" in (
            semantics.terminal_event_unit
        )

    @pytest.mark.parametrize(
        "terminal",
        [
            "One governed terminal test event.",
            "One event {{admission}}, or {{admission}}.",
            "One event whose gross loss is at least USD 100,000.",
        ],
    )
    def test_terminal_unit_states_the_admission_marker_exactly_once(self, terminal):
        with pytest.raises(ValidationError, match=r"\{\{admission\}\}"):
            FrequencySemanticsDef(**_frequency_semantics(terminal_event_unit=terminal))


class TestScenarioDefinitionValidators:
    def test_severity_governance_and_its_shape_are_required(self):
        payload = _minimal_scenario()
        payload.pop("severity_governance")
        with pytest.raises(ValidationError, match=r"severity_governance\s+Field"):
            ScenarioDefinition.model_validate(payload)
        payload = _minimal_scenario()
        payload["severity_governance"].pop("gpd_shape")
        with pytest.raises(
            ValidationError, match=r"severity_governance\.gpd_shape\s+Field required"
        ):
            ScenarioDefinition.model_validate(payload)

    def test_finite_and_uncapped_severity_governance_are_distinct(self):
        payload = _minimal_scenario()
        payload["severity_governance"]["cap_posture"] = "finite"
        with pytest.raises(ValueError, match="positive finite severity_cap"):
            ScenarioDefinition.model_validate(payload)
        payload["severity_governance"]["severity_cap"] = 1000000.0
        payload["severity_governance"]["severity_cap_basis"] = "Authored basis."
        scenario = ScenarioDefinition.model_validate(payload)
        assert scenario.severity_governance.severity_cap == 1000000.0
        assert (
            scenario.severity_governance.effective_severity_cap_sensitivity_values
            == (
                750000.0,
                1250000.0,
            )
        )
        payload["severity_governance"]["cap_posture"] = "uncapped"
        with pytest.raises(ValueError, match="forbids severity_cap"):
            ScenarioDefinition.model_validate(payload)

    @pytest.mark.parametrize("domain", _TEST_NODE_BY_DOMAIN)
    def test_node_id_derives_reporting_domain(self, domain):
        scenario = ScenarioDefinition(
            **_minimal_scenario(node_ids=[_TEST_NODE_BY_DOMAIN[domain]])
        )
        assert scenario.risk_domain == domain

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("risk_domain", "cyber"),
            ("unknown_field", 1),
        ],
    )
    def test_derived_and_unknown_fields_are_rejected(self, field, value):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ScenarioDefinition(**_minimal_scenario(**{field: value}))

    @pytest.mark.parametrize(
        "required", ["frequency_semantics", "attribution_nexus", "event_chain"]
    )
    def test_population_blocks_are_required(self, required):
        data = _minimal_scenario()
        data.pop(required)
        with pytest.raises(ValidationError, match=required):
            ScenarioDefinition(**data)

    @pytest.mark.parametrize("value", ["", "   "])
    def test_frequency_semantics_rejects_blank_counting_identity(self, value):
        data = _minimal_scenario()
        data["frequency_semantics"]["counting_identity"] = value
        with pytest.raises(ValueError, match="must not be blank"):
            ScenarioDefinition(**data)

    @pytest.mark.parametrize(
        ("field", "text"),
        [
            ("counting_identity", "Count campaigns above USD 1 million once."),
            ("year_rule", "Assign a loss of 5 million dollars to its first year."),
            ("terminal_event_unit", "One event {{admission}}, or above $250,000."),
        ],
    )
    def test_no_other_amount_defines_an_incident(self, field, text):
        data = _minimal_scenario(
            frequency_semantics=_frequency_semantics(**{field: text})
        )
        with pytest.raises(ValidationError, match="monetary amount other than"):
            ScenarioDefinition(**data)

    def test_loss_population_contract_is_derived(self):
        scenario = ScenarioDefinition(**_minimal_scenario())
        contract = effective_loss_population_contract(scenario)
        assert set(contract) == {
            "scenario_id",
            "counting_identity",
            "terminal_event_unit",
            "year_rule",
            "loss_definition",
            "attribution_nexus",
            "severity_occurrence_scope",
            "severity_minimum",
            "materiality_predicate",
            "below_threshold_treatment",
        }
        assert contract["terminal_event_unit"] == _expanded_terminal()
        assert contract["severity_minimum"] == admission_threshold()
        assert contract["severity_occurrence_scope"] == (
            "The complete causal gross economic loss for one governed test event, "
            "including consequential civil fines on the selected provider."
        )
        assert "admission threshold" in contract["materiality_predicate"]

    def test_final_chain_stage_is_the_counted_terminal_unit(self):
        scenario = ScenarioDefinition(**_minimal_scenario())
        assert event_chain_terminal_stage(scenario) == EventChainStageDef(
            stage_id="qualified_terminal", event=_expanded_terminal()
        )
        assert qualified_terminal_unit(scenario) == _expanded_terminal()

    @pytest.mark.parametrize(
        ("stages", "message"),
        [
            ([], "at least 1 item"),
            (
                [
                    {"stage_id": f"stage_{index}", "event": f"Event {index}."}
                    for index in range(5)
                ],
                "at most 4 items",
            ),
            (
                [
                    {"stage_id": "repeat", "event": "First."},
                    {"stage_id": "repeat", "event": "Second."},
                ],
                "stage_id values repeat",
            ),
            (
                [{"stage_id": "qualified_terminal", "event": "Terminal."}],
                "reserved",
            ),
            (
                [
                    {"stage_id": "first", "event": "The same event."},
                    {"stage_id": "second", "event": "the  same event."},
                ],
                "distinct events",
            ),
            (
                [{"stage_id": "Bad-Id", "event": "An event."}],
                "String should match pattern",
            ),
        ],
    )
    def test_chain_stages_are_bounded_distinct_and_unreserved(self, stages, message):
        with pytest.raises(ValueError, match=message):
            EventChainDef.model_validate(_event_chain(stages=stages))

    @pytest.mark.parametrize(
        "event",
        [
            "Loss exceeds USD 1 million.",
            "Loss exceeds $250,000.",
            "Loss exceeds 10,000 USD.",
            "Loss exceeds 5 million dollars.",
        ],
    )
    def test_chain_stages_apply_no_monetary_screen(self, event):
        with pytest.raises(ValueError, match="name a monetary amount"):
            EventChainDef.model_validate(
                _event_chain(stages=[{"stage_id": "screened", "event": event}])
            )

    def test_chain_opportunity_applies_no_monetary_screen(self):
        # The admission threshold lives in the terminal definition only.
        chain = _event_chain(
            opportunity={
                "unit": "One governed test opportunity worth at least USD 100,000.",
                "counting_rule": "One opportunity per governed test attempt.",
                "cohort": "Test attempts begun in the study year.",
            }
        )
        with pytest.raises(ValueError, match=r"\['opportunity'\] name a monetary"):
            EventChainDef.model_validate(chain)

    @pytest.mark.parametrize(
        "event",
        [
            "One governed  terminal test event.",
            "One governed terminal test event occurs.",
            "the governed test event: one terminal",
        ],
    )
    def test_chain_stage_must_not_restate_the_terminal_unit(self, event):
        with pytest.raises(ValueError, match="restate the terminal unit"):
            ScenarioDefinition(
                **_minimal_scenario(
                    event_chain=_event_chain(
                        stages=[{"stage_id": "terminal_again", "event": event}]
                    ),
                )
            )

    def test_exposure_bases_read_declared_scales_and_state_a_basis(self):
        chain = _event_chain()
        chain["exposure"]["public"]["count"]["scale_id"] = "api_tokens_per_day"
        with pytest.raises(ValidationError, match="undeclared scales"):
            ScenarioDefinition(**_minimal_scenario(event_chain=chain))
        chain = _event_chain()
        chain["exposure"]["public"].pop("basis")
        with pytest.raises(ValidationError, match=r"exposure\.public\.basis"):
            ScenarioDefinition(**_minimal_scenario(event_chain=chain))
        chain["exposure"]["public"]["basis"] = "placeholder"
        with pytest.raises(ValidationError, match="basis"):
            ScenarioDefinition(**_minimal_scenario(event_chain=chain))

    def test_normalisation_bases_are_reported_by_access(self):
        scenario = ScenarioDefinition(**_minimal_scenario())
        assert scenario.normalisation_bases == ("public",)
        chain = _event_chain()
        chain["exposure"]["public"]["basis"] = "judgment"
        assert not ScenarioDefinition(
            **_minimal_scenario(event_chain=chain)
        ).normalisation_bases

    def test_every_library_chain_ends_at_its_counted_terminal_unit(self):
        for scenario in load_scenarios(_SCENARIOS_DIR):
            node = contextual_node_from_scenario(scenario, "deployment", "public")
            terminal = event_chain_terminal_stage(scenario)
            assert terminal.event == node.population.terminal_unit, scenario.id
            assert 2 <= len(scenario.event_chain.stages) + 1 <= 5, scenario.id

    def test_year_rule_is_authored_once_and_rendered(self):
        # The year rule bounds the counted population, so the direct question
        # renders it from the same field the chain question uses; no chain text
        # restates the admission amount.
        for scenario in load_scenarios(_SCENARIOS_DIR):
            semantics = scenario.frequency_semantics
            node = contextual_node_from_scenario(scenario, "deployment", "public")
            qualification = json.loads(node.population.causal_qualification)
            assert qualification["year_rule"] == semantics.year_rule, scenario.id
            chain = scenario.event_chain
            for text in (
                chain.opportunity.unit,
                chain.opportunity.cohort,
                *(stage.event for stage in chain.stages),
            ):
                assert not re.search(r"dated to|USD \d|100 individuals", text), (
                    scenario.id,
                    text,
                )

    def test_rendered_population_text_uses_no_modelling_terms(self):
        # The direct question renders the counted population in plain words; it
        # must not use the model's own frequency vocabulary.
        modelling = re.compile(
            r"exposure|\bgates?\b|ladder|P\(N|E\[N\]|at-least-one|"
            r"provider-year|Bernoulli|event tree|counting_identity|additive",
            re.IGNORECASE,
        )
        for scenario in load_scenarios(_SCENARIOS_DIR):
            node = contextual_node_from_scenario(scenario, "deployment", "public")
            population = node.population.model_dump(mode="json")
            texts = [
                node.qualified_count_definition.replace(
                    "No additional gate multiplication.", ""
                ),
                *(
                    str(value)
                    for key, value in population.items()
                    if key != "causal_qualification"
                ),
                *(
                    str(value)
                    for value in json.loads(population["causal_qualification"]).values()
                ),
                *(component.definition for component in node.components),
                scenario.event_chain.opportunity.unit,
                scenario.event_chain.opportunity.counting_rule,
                scenario.event_chain.opportunity.cohort,
                *(stage.event for stage in scenario.event_chain.stages),
            ]
            for text in texts:
                assert not modelling.search(text), (scenario.id, text[:200])


class TestComposition:
    def test_tracked_cells_render_into_the_severity_scope(self):
        scenario = ScenarioDefinition(**_minimal_scenario())
        assert [
            cell.cell_id for cell in scenario.estimand_composition.tracked_cells
        ] == ["provider_civil_fine"]
        assert "{{loss_component" not in rendered_severity_occurrence_scope(scenario)

    @pytest.mark.parametrize(
        ("scope", "message"),
        [
            ("The complete loss of one test event.", "appears once as a marker"),
            (
                "Loss with {{loss_component:provider_civil_fine}} and "
                "{{loss_component:provider_civil_fine}}.",
                "appears once as a marker",
            ),
            (
                "Loss with {{loss_component:provider_civil_fine}} and "
                "{{loss_component:victim_resources}}.",
                "not tracked",
            ),
            ("Loss including consequential civil fines.", "renderer-reserved"),
        ],
    )
    def test_severity_scope_marks_exactly_the_tracked_cells(self, scope, message):
        with pytest.raises(ValidationError, match=message):
            ScenarioDefinition(**_minimal_scenario(severity_occurrence_scope=scope))

    @pytest.mark.parametrize(
        ("cell", "message"),
        [
            ({"loss_type": "civil_fines"}, "names both"),
            ({"attachment_basis": "incident_consequence"}, "names both"),
            (
                {
                    "nature": "resource_cost",
                    "loss_type": "civil_fines",
                    "attachment_basis": "regulatory_scenario",
                },
                "civil fines must be transfers",
            ),
            (
                {
                    "subject": "third_party",
                    "loss_type": "remediation",
                    "attachment_basis": "incident_consequence",
                },
                "incident_consequence cells must be provider-borne",
            ),
            ({"loss_form": "{{accounting:unknown}}"}, "accounting"),
            ({"unknown_field": "value"}, "Extra inputs are not permitted"),
        ],
    )
    def test_cell_tracking_and_text_are_validated(self, cell, message):
        payload = {**_CELLS[0], **cell}
        with pytest.raises(ValidationError, match=message):
            CompositionCellDef.model_validate(payload)

    def test_composition_cells_are_a_unique_roster(self):
        with pytest.raises(ValidationError, match="at least 2"):
            EstimandCompositionDef(status="OK", rationale="r", cells=[_CELLS[0]])
        with pytest.raises(ValidationError, match="cell_id values repeat"):
            EstimandCompositionDef(
                status="OK",
                rationale="r",
                cells=[_CELLS[0], {**_CELLS[1], "cell_id": _CELLS[0]["cell_id"]}],
            )

    def test_named_roster_expands_with_restated_loss_forms(self):
        rosters = CompositionRosters.model_validate(
            {"rosters": {"pair": {"status": "OK", "rationale": "r", "cells": _CELLS}}}
        )
        expanded = resolve_estimand_composition(
            {"roster": "pair", "loss_forms": {"victim_resources": "Restated form."}},
            rosters,
        )
        composition = EstimandCompositionDef.model_validate(expanded)
        assert composition.roster == "pair"
        assert [cell.loss_form for cell in composition.cells] == [
            "Restated form.",
            _CELLS[1]["loss_form"],
        ]
        with pytest.raises(ValueError, match="unknown composition roster"):
            resolve_estimand_composition({"roster": "missing"}, rosters)
        with pytest.raises(ValueError, match="absent from roster"):
            resolve_estimand_composition(
                {"roster": "pair", "loss_forms": {"other": "Text."}}, rosters
            )
        with pytest.raises(ValueError, match="may add only loss_forms"):
            resolve_estimand_composition({"roster": "pair", "status": "GAP"}, rosters)
        with pytest.raises(ValueError, match="unknown composition roster"):
            resolve_estimand_composition({"roster": "pair"}, None)

    def test_library_rosters_are_shared(self):
        library = load_shared_library(_SCENARIOS_DIR)
        assert library.rosters is not None
        scenarios = load_scenarios(_SCENARIOS_DIR)
        used = {scenario.estimand_composition.roster for scenario in scenarios}
        # Every named roster is used, and most scenarios share one.
        assert set(library.rosters.rosters) <= used
        assert (
            sum(
                scenario.estimand_composition.roster is not None
                for scenario in scenarios
            )
            > len(scenarios) / 2
        )


class TestEmpiricalGrounding:
    def test_a_scenario_states_its_empirical_grounding(self):
        scenario = ScenarioDefinition(**_minimal_scenario())
        assert scenario.empirical_grounding == (
            "No governed dataset supports this test population."
        )
        missing = _minimal_scenario()
        del missing["empirical_grounding"]
        with pytest.raises(
            ValidationError, match=r"\nempirical_grounding\n\s+Field required"
        ):
            ScenarioDefinition(**missing)
        with pytest.raises(ValidationError, match="must not be blank"):
            ScenarioDefinition(**_minimal_scenario(empirical_grounding=" \n\t "))

    def test_library_scenarios_author_only_population_fields(self):
        """A scenario defines a loss population and states the evidence for it.
        The model forbids unknown fields, so a file authors nothing else, such
        as which insurance would respond."""
        fields = set(ScenarioDefinition.model_fields)
        for path in sorted(_SCENARIOS_DIR.glob("*/*.yaml")):
            authored = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert set(authored) <= fields, (path, sorted(set(authored) - fields))
            assert " ".join(authored["empirical_grounding"].split()), path
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ScenarioDefinition(
                **_minimal_scenario(responding_coverages=["cyber_liability"])
            )


class TestScenarioModelValidate:
    def test_model_validate_missing_required_field(self):
        data = _minimal_scenario()
        del data["name"]
        with pytest.raises(ValidationError, match=r"\nname\n\s+Field required"):
            ScenarioDefinition.model_validate(data)

    def test_model_validate_extra_field_rejected(self):
        data = _minimal_scenario(extra_field="should be ignored")
        with pytest.raises(Exception, match="Extra inputs are not permitted"):
            ScenarioDefinition.model_validate(data)


class TestLoadScenarios:
    def test_sorted_by_domain_then_id(self, tmp_path):
        _copy_shared_files(tmp_path)
        for scenario_id, domain in (
            ("reg_1", "regulatory"),
            ("cyber_2", "cyber"),
            ("cyber_1", "cyber"),
            ("cbrn_1", "cbrn"),
        ):
            _write_yaml(
                tmp_path / domain / f"{scenario_id}.yaml",
                _scenario_file(id=scenario_id, node_ids=[_TEST_NODE_BY_DOMAIN[domain]]),
            )
        assert [s.id for s in load_scenarios(tmp_path)] == [
            "cyber_1",
            "cyber_2",
            "cbrn_1",
            "reg_1",
        ]

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Scenarios directory not found"):
            load_scenarios(tmp_path / "nonexistent")

    def test_empty_directory_returns_empty(self, tmp_path):
        assert load_scenarios(tmp_path) == []

    def test_every_invalid_file_is_reported_with_its_path(self, tmp_path):
        _copy_shared_files(tmp_path)
        _write_yaml(
            tmp_path / "cyber" / "bad.yaml",
            {"id": "bad", "node_ids": [_TEST_NODE_BY_DOMAIN["cyber"]]},
        )
        malformed = _scenario_file(id="malformed", node_ids=["finance"])
        _write_yaml(tmp_path / "other" / "malformed.yaml", malformed)
        (tmp_path / "not_a_mapping.yaml").write_text("- one\n- two\n")
        with pytest.raises(ScenarioLibraryError) as raised:
            load_scenarios(tmp_path)
        errors = raised.value.errors
        assert any(
            str(tmp_path / "cyber" / "bad.yaml") in error and "name" in error
            for error in errors
        )
        assert any(
            str(tmp_path / "other" / "malformed.yaml") in error and "node_ids" in error
            for error in errors
        )
        assert any(
            str(tmp_path / "not_a_mapping.yaml") in error and "mapping" in error
            for error in errors
        )

    def test_a_scenario_file_repeating_a_key_is_refused(self, tmp_path):
        _copy_shared_files(tmp_path)
        path = tmp_path / "cyber" / "repeated.yaml"
        _write_yaml(path, _scenario_file(id="repeated"))
        assert [scenario.id for scenario in load_scenarios(tmp_path)] == ["repeated"]
        with open(path, "a") as f:
            f.write("name: A second name for the same scenario\n")
        with pytest.raises(ScenarioLibraryError) as raised:
            load_scenarios(tmp_path)
        assert len(raised.value.errors) == 1
        assert str(path) in raised.value.errors[0]
        assert "duplicate key 'name'" in raised.value.errors[0]

    def test_a_scenario_id_belongs_to_one_file(self, tmp_path):
        _copy_shared_files(tmp_path)
        _write_yaml(tmp_path / "cyber" / "a.yaml", _scenario_file(id="shared"))
        _write_yaml(tmp_path / "cyber" / "b.yaml", _scenario_file(id="shared"))
        with pytest.raises(ScenarioLibraryError, match="'shared' is defined by"):
            load_scenarios(tmp_path)

    def test_nested_subdirectories(self, tmp_path):
        _copy_shared_files(tmp_path)
        _write_yaml(
            tmp_path / "cyber" / "sub" / "deep.yaml",
            _scenario_file(id="nested_scenario"),
        )
        assert [s.id for s in load_scenarios(tmp_path)] == ["nested_scenario"]

    def test_non_yaml_and_shared_files_are_not_scenarios(self, tmp_path):
        _write_yaml(tmp_path / "cyber" / "s1.yaml", _scenario_file())
        (tmp_path / "cyber" / "README.md").write_text("# Readme")
        _copy_shared_files(tmp_path)
        assert len(load_scenarios(tmp_path)) == 1

    def test_all_reporting_domains_sort_order(self, tmp_path):
        _copy_shared_files(tmp_path)
        for domain, node_id in _TEST_NODE_BY_DOMAIN.items():
            _write_yaml(
                tmp_path / domain / "s.yaml",
                _scenario_file(id=f"{domain}_s", node_ids=[node_id]),
            )
        assert [s.risk_domain for s in load_scenarios(tmp_path)] == [
            "cyber",
            "eo",
            "cbrn",
            "agent",
            "ai_rd",
            "societal",
            "svc",
            "regulatory",
        ]

    def test_load_scenarios_expands_shared_references(self, tmp_path):
        _copy_shared_files(tmp_path)
        payload = _scenario_file(
            estimand_composition={"roster": "standard"},
            severity_occurrence_scope="The loss including {{loss_component:provider_civil_fine}}.",
        )
        _write_yaml(tmp_path / "cyber" / "test.yaml", payload)
        [scenario] = load_scenarios(tmp_path)
        policies = load_severity_governance_policies(tmp_path)
        assert policies is not None
        assert scenario.severity_governance.rationale == (
            policies.policies["governed_tail_shape"].rationale
        )
        assert scenario.severity_governance.gpd_shape == (
            policies.tail_classes["liability_enforcement"].gpd_shape
        )
        assert scenario.estimand_composition.roster == "standard"
        assert len(scenario.estimand_composition.cells) > 2


class TestSeverityGovernancePolicyExpansion:
    @staticmethod
    def _known_policy_block(**overrides):
        block = {
            "policy": "known",
            "gpd_shape": 0.5,
            "cap_posture": "uncapped",
            "evidence_refs": ["assumptions.md"],
        }
        block.update(overrides)
        return block

    def test_policy_reference_equals_fully_authored_equivalent(self):
        policies = load_severity_governance_policies(_SCENARIOS_DIR)
        assert policies is not None
        reference = _scenario_file()
        expanded = expand_shared_content(reference, SharedLibrary(policies, None))
        policy = policies.policies["governed_tail_shape"]
        fully_authored = _minimal_scenario(
            severity_governance={
                "gpd_shape": policies.tail_classes["liability_enforcement"].gpd_shape,
                "tail_class": "liability_enforcement",
                "tail_class_rationale": "One claim bounded by the claimant's interest.",
                "cap_posture": "uncapped",
                "source_classification": "internal_judgment",
                "status": "INITIAL_ESTIMATE",
                "materiality": "high",
                "rationale": policy.rationale,
                "empirical_grounding": policy.empirical_grounding,
                "sensitivity_range": policy.sensitivity_range_by_cap_posture.uncapped,
                "evidence_refs": ["assumptions.md"],
                "sensitivity_evidence_refs": [],
            }
        )
        assert ScenarioDefinition.model_validate(expanded).model_dump(
            mode="json"
        ) == ScenarioDefinition.model_validate(fully_authored).model_dump(mode="json")

    @staticmethod
    def _validate(**overrides):
        return ScenarioDefinition.model_validate(_minimal_scenario(**overrides))

    def test_uncapped_stress_alternative_requires_finite_posture(self):
        with pytest.raises(
            ValueError, match="severity_cap_uncapped_stress_alternative"
        ):
            self._validate(
                severity_governance={
                    **_GOVERNANCE_PROSE,
                    "gpd_shape": 0.5,
                    "cap_posture": "uncapped",
                    "severity_cap_uncapped_stress_alternative": True,
                }
            )

    def test_severity_scope_exclusions_are_unique_and_sorted(self):
        block = {
            **_GOVERNANCE_PROSE,
            "gpd_shape": 0.5,
            "cap_posture": "finite",
            "severity_cap": 500000000000.0,
            "severity_cap_basis": "Authored basis.",
            "severity_cap_uncapped_stress_alternative": True,
            "severity_scope_exclusions": ["existential_value"],
        }
        scenario = self._validate(severity_governance=block)
        assert scenario.severity_governance.severity_scope_exclusions == (
            "existential_value",
        )
        block["severity_scope_exclusions"] = ["existential_value", "existential_value"]
        with pytest.raises(ValueError, match="unique"):
            self._validate(severity_governance=block)

    def test_a_scenario_file_must_name_its_severity_policy(self, tmp_path):
        _copy_shared_files(tmp_path)
        path = tmp_path / "cyber" / "inline.yaml"
        fully_authored = _minimal_scenario()
        ScenarioDefinition.model_validate(fully_authored)
        _write_yaml(path, fully_authored)
        with pytest.raises(ScenarioLibraryError) as raised:
            load_scenarios(tmp_path)
        [error] = raised.value.errors
        assert str(path) in error
        assert "policy: governed_tail_shape" in error
        assert "tail_class" in error

    @pytest.mark.parametrize("names_policy", [False, True])
    def test_a_scenario_file_cannot_author_a_shape_its_tail_class_fixes(
        self, tmp_path, names_policy
    ):
        _copy_shared_files(tmp_path)
        policies = load_severity_governance_policies(tmp_path)
        assert policies is not None
        class_shape = policies.tail_classes["liability_enforcement"].gpd_shape
        authored_shape = 0.3
        assert authored_shape != class_shape
        if names_policy:
            governance = _policy_reference_block(gpd_shape=authored_shape)
            refusal = "owns gpd_shape"
        else:
            governance = {
                **_minimal_scenario()["severity_governance"],
                "gpd_shape": authored_shape,
                "tail_class": "liability_enforcement",
                "tail_class_rationale": "One claim bounded by the claimant's interest.",
            }
            refusal = "policy: governed_tail_shape"
        payload = _minimal_scenario(severity_governance=governance)
        if not names_policy:
            expanded = ScenarioDefinition.model_validate(payload)
            assert expanded.severity_governance.gpd_shape == authored_shape
        path = tmp_path / "cyber" / "disagrees.yaml"
        _write_yaml(path, payload)
        with pytest.raises(ScenarioLibraryError) as raised:
            load_scenarios(tmp_path)
        [error] = raised.value.errors
        assert str(path) in error
        assert refusal in error

    def test_a_missing_governance_block_is_left_to_validation(self):
        without_governance = {"id": "test_scenario"}
        assert (
            expand_shared_content(without_governance, SharedLibrary(None, None))
            == without_governance
        )

    def test_policy_reference_errors(self):
        policies = self._synthetic_policies()
        with pytest.raises(ValueError, match=r"unknown.*missing"):
            resolve_severity_governance_block(
                self._known_policy_block(policy="missing"), policies
            )
        with pytest.raises(ValueError, match="rationale"):
            resolve_severity_governance_block(
                self._known_policy_block(rationale="owned"), policies
            )
        with pytest.raises(ValueError, match="was not found"):
            resolve_severity_governance_block(_policy_reference_block(), None)
        with pytest.raises(ValueError, match="cap_posture"):
            resolve_severity_governance_block(
                self._known_policy_block(cap_posture="invalid"), policies
            )

    def test_policy_file_shape_is_strict(self, tmp_path):
        source_text = (
            _SCENARIOS_DIR / SEVERITY_GOVERNANCE_POLICIES_FILENAME
        ).read_text()
        policy_path = tmp_path / SEVERITY_GOVERNANCE_POLICIES_FILENAME

        def refused(mutate, match):
            source = yaml.safe_load(source_text)
            mutate(source)
            _write_yaml(policy_path, source)
            with pytest.raises(ValueError, match=match):
                load_severity_governance_policies(tmp_path)

        refused(lambda source: source.update(unexpected=True), "unexpected")
        refused(
            lambda source: source["policies"]["governed_tail_shape"].update(
                unexpected=True
            ),
            "unexpected",
        )
        refused(
            lambda source: source["policies"]["governed_tail_shape"][
                "sensitivity_range_by_cap_posture"
            ].pop("uncapped"),
            "sensitivity_range_by_cap_posture",
        )
        refused(
            lambda source: source["tail_classes"]["cyber_operational"].update(
                unexpected=True
            ),
            "cyber_operational",
        )

    @staticmethod
    def _synthetic_policies():
        policy = {
            "rationale": "rationale",
            "empirical_grounding": "grounding",
            "source_classification": "internal_judgment",
            "status": "INITIAL_ESTIMATE",
            "materiality": "high",
            "sensitivity_evidence_refs": [],
            "sensitivity_range_by_cap_posture": {
                "finite": "Default finite range.",
                "uncapped": "Default uncapped range.",
            },
        }
        return SeverityGovernancePolicies.model_validate(
            {
                "policies": {"known": policy, "governed_tail_shape": policy},
                "tail_classes": {
                    "plain": TailClassDef(
                        gpd_shape=0.5, rationale="r", empirical_grounding="g"
                    ),
                    "capped": TailClassDef(
                        gpd_shape=0.8,
                        rationale="r",
                        empirical_grounding="g",
                        endpoint=TailClassEndpointDef(
                            severity_cap=1e14,
                            severity_cap_sensitivity_values=(1e13, 5e14),
                            basis="Physical bound.",
                        ),
                    ),
                },
            }
        )

    def test_sensitivity_range_follows_cap_alternatives_and_shape_authority(self):
        policies = self._synthetic_policies()

        def resolved_range(**block):
            if "severity_cap" in block:
                block["severity_cap_basis"] = "Authored basis."
            return resolve_severity_governance_block(
                {"policy": "known", **block}, policies
            )["sensitivity_range"]

        assert resolved_range(cap_posture="uncapped") == "Default uncapped range."
        assert (
            resolved_range(cap_posture="finite", severity_cap=5e11)
            == "Default finite range."
        )
        assert resolved_range(
            cap_posture="finite",
            severity_cap=5e11,
            severity_cap_sensitivity_values=[1e11, 2.5e12],
        ) == (
            "Test the scenario-authored absolute severity_cap endpoints 1e+11 and "
            "2.5e+12, and re-derive the GPD scale from the unchanged p95/p99 "
            "anchor at gpd_shape 0.20 lower and higher."
        )
        stress_sentence = " The uncapped stress alternative removes the authored cap"
        stress_block = {
            "cap_posture": "finite",
            "severity_cap": 5e11,
            "severity_cap_uncapped_stress_alternative": True,
        }
        assert stress_sentence in resolved_range(**stress_block)
        assert stress_sentence in resolved_range(
            **stress_block, severity_cap_sensitivity_values=[1e11, 2.5e12]
        )

    def test_tail_class_owns_shape_and_endpoint(self):
        policies = self._synthetic_policies()

        def resolve(**block):
            return resolve_severity_governance_block(
                {"policy": "governed_tail_shape", **block}, policies
            )

        plain = resolve(
            tail_class="plain", tail_class_rationale="why", cap_posture="uncapped"
        )
        assert plain["gpd_shape"] == 0.5
        capped = resolve(tail_class="capped", tail_class_rationale="why")
        assert capped["gpd_shape"] == 0.8
        assert capped["cap_posture"] == "finite"
        assert capped["severity_cap"] == 1e14
        assert capped["severity_cap_sensitivity_values"] == [1e13, 5e14]
        assert capped["severity_cap_uncapped_stress_alternative"] is True
        assert capped["severity_cap_basis"] == "Physical bound."
        with pytest.raises(ValueError, match="requires a tail_class"):
            resolve(cap_posture="uncapped")
        with pytest.raises(ValueError, match="requires a tail_class"):
            resolve(tail_class="unknown", tail_class_rationale="why")
        with pytest.raises(ValueError, match="owns gpd_shape"):
            resolve(
                tail_class="plain",
                tail_class_rationale="why",
                cap_posture="uncapped",
                gpd_shape=0.6,
            )
        with pytest.raises(ValueError, match="owns its endpoint convention"):
            resolve(
                tail_class="capped", tail_class_rationale="why", cap_posture="finite"
            )
        for block in (
            {"cap_posture": "finite", "severity_cap": 1e9},
            {"cap_posture": "uncapped", "uncapped_endpoint_stress": 1e9},
        ):
            with pytest.raises(ValueError, match="requires severity_cap_basis"):
                resolve(tail_class="plain", tail_class_rationale="why", **block)
        with pytest.raises(ValueError, match="takes no tail_class"):
            resolve_severity_governance_block(
                self._known_policy_block(tail_class="plain"), policies
            )
        with pytest.raises(ValidationError, match="authored together"):
            _severity_governance(
                gpd_shape=0.5, cap_posture="uncapped", tail_class="plain"
            )
        with pytest.raises(ValidationError, match=r"gpd_shape\s+Field required"):
            _severity_governance(
                cap_posture="uncapped", tail_class="plain", tail_class_rationale="why"
            )


_GOVERNANCE_PROSE = {
    "source_classification": "internal_judgment",
    "status": "INITIAL_ESTIMATE",
    "materiality": "high",
    "rationale": "r",
    "empirical_grounding": "g",
    "sensitivity_range": "s",
}


def _severity_governance(**fields) -> SeverityGovernanceDef:
    return SeverityGovernanceDef.model_validate({**_GOVERNANCE_PROSE, **fields})


class TestSeverityGovernanceTailShapeAlternatives:
    """Governed shapes must admit both 0.2 tail-shape alternatives inside the runtime
    bounds: 0.95 uncapped (finite mean), 2.5 capped (sanity), -0.95 below; the
    uncapped stress alternative needs a finite mean at the governed shape."""

    def test_uncapped_shape_admits_higher_alternative_below_finite_mean_bound(self):
        accepted = _severity_governance(gpd_shape=0.7, cap_posture="uncapped")
        assert accepted.gpd_shape == 0.7
        with pytest.raises(
            ValidationError,
            match=r"higher tail-shape sensitivity alternative .* operating bound 0\.95",
        ):
            _severity_governance(gpd_shape=0.8, cap_posture="uncapped")

    def test_uncapped_higher_alternative_exactly_at_the_bound_is_refused(self):
        """An uncapped law's shape is below 0.95, so the comparison arm cannot
        hold an alternative of exactly 0.95 and the loader refuses the class."""
        assert _severity_governance(gpd_shape=0.74, cap_posture="uncapped")
        with pytest.raises(
            ValidationError,
            match=r"\(gpd_shape \+ 0\.2 = 0\.95\) is not below .* operating bound 0\.95",
        ):
            _severity_governance(gpd_shape=0.75, cap_posture="uncapped")

    def test_finite_shape_admits_higher_alternative_below_capped_sanity_bound(self):
        capped = {
            "cap_posture": "finite",
            "severity_cap": 5e11,
            "severity_cap_basis": "b",
        }
        assert _severity_governance(gpd_shape=2.25, **capped).gpd_shape == 2.25
        with pytest.raises(
            ValidationError,
            match=r"higher tail-shape sensitivity alternative .* capped sanity bound 2\.5",
        ):
            _severity_governance(gpd_shape=2.35, **capped)

    def test_uncapped_stress_alternative_bounds_governed_shape_by_finite_mean_bound(
        self,
    ):
        capped = {
            "cap_posture": "finite",
            "severity_cap": 5e11,
            "severity_cap_basis": "b",
        }
        assert _severity_governance(gpd_shape=0.95, **capped).gpd_shape == 0.95
        assert _severity_governance(
            gpd_shape=0.9, severity_cap_uncapped_stress_alternative=True, **capped
        ).severity_cap_uncapped_stress_alternative
        with pytest.raises(
            ValidationError,
            match=r"uncapped stress alternative needs a governed shape below .* 0\.95",
        ):
            _severity_governance(
                gpd_shape=0.95, severity_cap_uncapped_stress_alternative=True, **capped
            )

    def test_lower_alternative_stays_above_negative_shape_bound(self):
        accepted = _severity_governance(gpd_shape=-0.75, cap_posture="uncapped")
        assert accepted.gpd_shape == -0.75
        with pytest.raises(
            ValidationError, match=r"lower tail-shape sensitivity alternative .* -0\.95"
        ):
            _severity_governance(gpd_shape=-0.8, cap_posture="uncapped")

    def test_step_moves_every_library_tail_class_to_a_decimal_shape(self):
        """Each class's two alternatives are the decimals the documents state.

        Binary addition misses three of the six by a unit in the last place:
        0.7 - 0.2 (0.49999999999999994, below the 0.5 from which an uncapped
        law has infinite variance), 0.7 + 0.2 and 0.8 - 0.2.
        """
        policies = load_severity_governance_policies(_SCENARIOS_DIR)
        assert policies is not None
        step = TAIL_SHAPE_SENSITIVITY_STEP
        assert {
            name: (
                tail_class.gpd_shape,
                shifted_tail_shape(tail_class.gpd_shape, -step),
                shifted_tail_shape(tail_class.gpd_shape, step),
            )
            for name, tail_class in policies.tail_classes.items()
        } == {
            "liability_enforcement": (0.5, 0.3, 0.7),
            "cyber_operational": (0.7, 0.5, 0.9),
            "mass_casualty_catastrophe": (0.8, 0.6, 1.0),
        }


class TestUncappedEndpointStress:
    def test_endpoint_requires_uncapped_posture(self):
        with pytest.raises(ValidationError, match="requires cap_posture='uncapped'"):
            _severity_governance(
                gpd_shape=0.5,
                cap_posture="finite",
                severity_cap=5e11,
                uncapped_endpoint_stress=1e12,
                severity_cap_basis="b",
            )

    @pytest.mark.parametrize("value", [0.0, -1e9, float("inf"), float("nan")])
    def test_endpoint_must_be_positive_and_finite(self, value):
        with pytest.raises(ValidationError, match="uncapped_endpoint_stress"):
            _severity_governance(
                gpd_shape=0.5,
                cap_posture="uncapped",
                uncapped_endpoint_stress=value,
                severity_cap_basis="b",
            )


@pytest.mark.parametrize("field", ["name", "description"])
@pytest.mark.parametrize("value", ["", " ", "\n\t"])
def test_scenario_rejects_blank_respondent_prose(field, value):
    with pytest.raises(ValidationError, match="text must not be blank"):
        ScenarioDefinition.model_validate(_minimal_scenario(**{field: value}))


def test_library_tail_classes_share_shapes_and_one_catastrophe_endpoint():
    policies = load_severity_governance_policies(_SCENARIOS_DIR)
    assert policies is not None
    assert set(policies.tail_classes) == {
        "liability_enforcement",
        "cyber_operational",
        "mass_casualty_catastrophe",
    }
    endpoint = policies.tail_classes["mass_casualty_catastrophe"].endpoint
    assert endpoint is not None
    scenarios = load_scenarios(_SCENARIOS_DIR)
    for scenario in scenarios:
        governance = scenario.severity_governance
        assert governance.tail_class is not None
        assert (
            governance.gpd_shape
            == policies.tail_classes[governance.tail_class].gpd_shape
        )
        catastrophe = scenario.risk_domain in {"cbrn", "ai_rd"} or scenario.id == (
            "compromise_critical_infrastructure"
        )
        assert (governance.tail_class == "mass_casualty_catastrophe") == catastrophe
        if catastrophe:
            # One endpoint whichever origin owns the same event.
            assert governance.severity_cap == endpoint.severity_cap
            assert (
                governance.severity_cap_sensitivity_values
                == endpoint.severity_cap_sensitivity_values
            )
            assert governance.severity_cap_uncapped_stress_alternative
        if governance.severity_cap is not None:
            assert governance.severity_cap_basis, scenario.id
        assert governance.endpoint_from_tail_class == catastrophe, scenario.id
    # The liability class is one matter or one designation: the provider-conduct
    # leaves. A leaf that merges every reliance on one defect into one incident
    # takes the cyber/operational shape.
    assert {
        scenario.id
        for scenario in scenarios
        if scenario.severity_governance.tail_class == "liability_enforcement"
    } == {
        "enforcement_matter",
        "government_designation",
        "private_claims_copyright_ip",
        "private_claims_other",
        "private_claims_privacy",
        "private_claims_product_consumer",
    }
    enforcement = next(s for s in scenarios if s.id == "enforcement_matter")
    # Statutory maxima shape the matter's severity; they are a stress, not a cap.
    assert enforcement.severity_governance.cap_posture == "uncapped"
    assert enforcement.severity_governance.uncapped_endpoint_stress == 5e10
    assert "Article 99(3)" in (enforcement.severity_governance.severity_cap_basis or "")


def test_normalisation_bases_are_the_pinned_populations():
    """An exposure base tagged as a normalisation has no governed denominator;
    the set is pinned, so a basis edit shows up here."""
    flagged = {
        scenario.id: scenario.normalisation_bases
        for scenario in load_scenarios(_SCENARIOS_DIR)
        if scenario.normalisation_bases
    }
    conduct = {
        "enforcement_matter": ("public", "internal"),
        "government_designation": ("public", "internal"),
        "private_claims_copyright_ip": ("public", "internal"),
        "private_claims_other": ("public", "internal"),
        "private_claims_privacy": ("internal",),
        "private_claims_product_consumer": ("internal",),
    }
    internal_staff = {
        scenario_id: ("internal",)
        for scenario_id in (
            "bounded_decision_errors",
            "engagement_harm",
            "output_reliance_financial",
            "output_reliance_health",
            "output_reliance_legal",
            "output_reliance_other",
            "physical_equipment_control",
            "component_tampering",
        )
    }
    assert flagged == {
        **conduct,
        **internal_staff,
        "model_asset_compromise": ("public",),
        "ai_service_attack": ("public", "internal"),
    }


def test_assisted_attacker_cells_count_any_copy_of_the_model():
    for scenario in load_scenarios(_SCENARIOS_DIR):
        role = origin_role_of(cached_hierarchy_contract(), scenario.node_ids[0])[1]
        assert (scenario.facets.deployment_control_mode == "any_copy") == (
            role == "assisted_attacker"
        ), scenario.id
    payload = _minimal_scenario()
    payload["facets"]["deployment_control_mode"] = "api"
    with pytest.raises(ValidationError, match="any_copy"):
        ScenarioDefinition(**payload)


_FINAL_FINE_RULE = (
    "final amount after any appeal",
    "a re-imposed fine replaces the fine it supersedes",
    "an annulled fine counts as zero",
    "original imposition date",
)


def test_enforcement_matter_dates_conduct_start_and_counts_the_whole_matter() -> None:
    scenario = next(
        s for s in load_scenarios(_SCENARIOS_DIR) if s.id == "enforcement_matter"
    )
    semantics = scenario.frequency_semantics
    assert "first occurs" in semantics.year_rule
    assert "over the matter's life" in semantics.terminal_event_unit
    assert "policy-start USD" in semantics.terminal_event_unit
    assert "split by course" in semantics.counting_identity
    assert "single unitemized fine" in semantics.counting_identity
    assert "earliest course's start" in semantics.counting_identity
    for phrase in _FINAL_FINE_RULE:
        assert phrase in scenario.severity_occurrence_scope, phrase
    assert {cell.cell_id for cell in scenario.estimand_composition.tracked_cells} == {
        "provider_civil_fine",
        "provider_resource_cost",
        "provider_noncivil_penalty",
    }
