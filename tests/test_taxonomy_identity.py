import re
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import pytest
import yaml

from catmodel import resources
from catmodel.scenarios.loader import ScenarioLineageDef, load_scenarios
from catmodel.scenarios.taxonomy_identity import (
    PopulationTransitionDef,
    RetiredScenarioDef,
    ScenarioLineageRegister,
    TaxonomyGenerationDef,
    load_scenario_lineage_register,
    scenario_roster_sha256,
    validate_scenario_lineage,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCENARIOS_DIR = _PROJECT_ROOT / "scenarios"


# The register names the current generation; tests use a synthetic one.
_GENERATION = "r1"


@pytest.fixture(scope="module")
def all_scenarios():
    return load_scenarios(_SCENARIOS_DIR)


class TestTaxonomyIdentity:
    def test_roster_digest_is_order_insensitive_and_id_sensitive(self):
        assert scenario_roster_sha256(["b", "a"]) == scenario_roster_sha256(["a", "b"])
        assert scenario_roster_sha256(["a", "b"]) != scenario_roster_sha256(["a", "c"])


def _register(
    *,
    current: str = _GENERATION,
    generations: list[TaxonomyGenerationDef] | None = None,
    retired: list[RetiredScenarioDef] | None = None,
    transitions: list[PopulationTransitionDef] | None = None,
) -> ScenarioLineageRegister:
    return ScenarioLineageRegister(
        schema_version=1,
        current_taxonomy_version=current,
        generations=generations
        or [
            TaxonomyGenerationDef(
                version=_GENERATION,
                established="2026-08-04",
                scenario_roster_sha256=scenario_roster_sha256(["cyber_a", "eo_b"]),
                description="test baseline",
            )
        ],
        transitions=transitions or [],
        retired_scenarios=retired or [],
    )


def _transition(**fields) -> PopulationTransitionDef:
    """A carve-out of ``eo_b`` from ``cyber_a``, with ``fields`` replaced."""
    return PopulationTransitionDef.model_validate(
        {
            "transition_id": "carve_out_eo_b",
            "kind": "carve_out",
            "executed_in": _GENERATION,
            "predecessor_ids": ["cyber_a"],
            "successor_ids": ["eo_b"],
            "evidence_staling": "all",
            "rationale": "test",
            **fields,
        }
    )


class TestLineageRegisterSchema:
    def test_withdrawn_rejects_successors(self):
        with pytest.raises(ValueError, match="withdrawn"):
            RetiredScenarioDef(
                id="cyber_old",
                retired_in="r2",
                disposition="withdrawn",
                successor_ids=["cyber_new"],
                rationale="test",
            )

    def test_non_withdrawn_requires_successors(self):
        with pytest.raises(ValueError, match="requires at least one successor"):
            RetiredScenarioDef(
                id="cyber_old",
                retired_in="r2",
                disposition="renamed",
                successor_ids=[],
                rationale="test",
            )

    def test_self_referencing_successor_rejected(self):
        with pytest.raises(ValueError, match="references itself"):
            RetiredScenarioDef(
                id="cyber_old",
                retired_in="r2",
                disposition="renamed",
                successor_ids=["cyber_old"],
                rationale="test",
            )

    def test_unregistered_current_version_rejected(self):
        with pytest.raises(ValueError, match="not a registered generation"):
            _register(current="r99")

    def test_duplicate_generation_versions_rejected(self):
        generation = TaxonomyGenerationDef(
            version=_GENERATION,
            established="2026-08-04",
            scenario_roster_sha256="a" * 64,
            description="test",
        )
        with pytest.raises(ValueError, match="duplicate generation versions"):
            _register(generations=[generation, generation])

    def test_duplicate_retired_ids_rejected(self):
        row = RetiredScenarioDef(
            id="cyber_old",
            retired_in=_GENERATION,
            disposition="withdrawn",
            rationale="test",
        )
        with pytest.raises(ValueError, match="duplicate retired scenario ids"):
            _register(retired=[row, row])

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"successor_ids": []}, "must name a successor"),
            ({"successor_ids": ["eo_b", "eo_b"]}, "duplicate IDs"),
            ({"predecessor_ids": []}, "carve_out transition requires a predecessor"),
            ({"kind": "scope_expansion"}, "cannot name a predecessor"),
        ],
    )
    def test_incoherent_transitions_rejected(self, fields, message):
        with pytest.raises(ValueError, match=message):
            _transition(**fields)

    def test_a_scope_expansion_names_no_predecessor(self):
        transition = _transition(kind="scope_expansion", predecessor_ids=[])
        assert transition.predecessor_ids == []

    def test_duplicate_transition_ids_rejected(self):
        with pytest.raises(ValueError, match="duplicate transition ids"):
            _register(transitions=[_transition(), _transition()])


@dataclass(frozen=True)
class _Cell:
    id: str
    lineage: ScenarioLineageDef


def _cell(scenario_id: str, **lineage) -> _Cell:
    lineage.setdefault("introduced_in", _GENERATION)
    return _Cell(id=scenario_id, lineage=ScenarioLineageDef(**lineage))


class TestLineageReferentialIntegrity:
    def test_clean_roster_passes(self):
        cells = [_cell("cyber_a"), _cell("eo_b")]
        assert validate_scenario_lineage(cells, _register()) == []

    def test_roster_drift_from_current_generation_fails(self):
        cells = [_cell("cyber_a"), _cell("eo_b"), _cell("agent_new")]
        errors = validate_scenario_lineage(cells, _register())
        assert any("scenario roster drifted" in error for error in errors)

    def test_the_register_alone_names_the_current_generation(self):
        generations = [
            TaxonomyGenerationDef(
                version="r0",
                established="2026-08-04",
                scenario_roster_sha256=scenario_roster_sha256(["cyber_a", "eo_b"]),
                description="an earlier label",
            )
        ]
        cells = [
            _cell("cyber_a", introduced_in="r0"),
            _cell("eo_b", introduced_in="r0"),
        ]
        assert (
            validate_scenario_lineage(
                cells, _register(current="r0", generations=generations)
            )
            == []
        )

    def test_unknown_introduced_in_fails(self):
        cells = [_cell("cyber_a", introduced_in="r99"), _cell("eo_b")]
        errors = validate_scenario_lineage(cells, _register())
        assert any(
            "introduced_in 'r99' is not a registered generation" in error
            for error in errors
        )

    def test_retired_id_still_live_fails(self):
        retired = [
            RetiredScenarioDef(
                id="eo_b",
                retired_in=_GENERATION,
                disposition="withdrawn",
                rationale="test",
            )
        ]
        errors = validate_scenario_lineage(
            [_cell("cyber_a"), _cell("eo_b")], _register(retired=retired)
        )
        assert any("still present in the live roster" in error for error in errors)

    def test_unknown_retirement_generation_fails(self):
        retired = [
            RetiredScenarioDef(
                id="cyber_old",
                retired_in="r99",
                disposition="withdrawn",
                rationale="test",
            )
        ]
        errors = validate_scenario_lineage(
            [_cell("cyber_a"), _cell("eo_b")], _register(retired=retired)
        )
        assert any(
            "retired_in 'r99' is not a registered generation" in error
            for error in errors
        )

    def test_unknown_successor_fails(self):
        retired = [
            RetiredScenarioDef(
                id="cyber_old",
                retired_in=_GENERATION,
                disposition="renamed",
                successor_ids=["cyber_ghost"],
                rationale="test",
            )
        ]
        errors = validate_scenario_lineage(
            [_cell("cyber_a"), _cell("eo_b")], _register(retired=retired)
        )
        assert any("neither live nor registered retired" in error for error in errors)

    def test_a_transition_names_live_or_retired_scenarios(self):
        cells = [_cell("cyber_a"), _cell("eo_b")]
        assert (
            validate_scenario_lineage(cells, _register(transitions=[_transition()]))
            == []
        )
        retired = [
            RetiredScenarioDef(
                id="cyber_old",
                retired_in=_GENERATION,
                disposition="split",
                successor_ids=["cyber_a", "eo_b"],
                rationale="test",
            )
        ]
        split = _transition(
            kind="split",
            predecessor_ids=["cyber_old"],
            successor_ids=["cyber_a", "eo_b"],
        )
        assert (
            validate_scenario_lineage(
                cells, _register(retired=retired, transitions=[split])
            )
            == []
        )

    def test_a_transition_with_an_unknown_generation_or_scenario_fails(self):
        stray = _transition(executed_in="r99", predecessor_ids=["cyber_ghost"])
        errors = validate_scenario_lineage(
            [_cell("cyber_a"), _cell("eo_b")], _register(transitions=[stray])
        )
        assert any(
            "executed_in 'r99' is not a registered generation" in error
            for error in errors
        )
        assert any(
            "scenario ids ['cyber_ghost'] are neither live nor registered retired"
            in error
            for error in errors
        )


class TestScenarioLineageDef:
    def test_blank_introduced_in_rejected(self):
        with pytest.raises(ValueError, match="must not be blank"):
            ScenarioLineageDef(introduced_in="  ")


def _write_register(path: Path, version: str) -> None:
    register = _register(
        current=version,
        generations=[
            TaxonomyGenerationDef(
                version=version,
                established="2026-08-04",
                scenario_roster_sha256=scenario_roster_sha256(["cyber_a"]),
                description="test register",
            )
        ],
    )
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(register.model_dump(mode="json")), encoding="utf-8")


def test_default_register_uses_packaged_resources_outside_a_checkout(
    tmp_path, monkeypatch
):
    bundled = tmp_path / "_resources" / "registries" / "scenario_lineage.yaml"
    _write_register(bundled, "packaged")
    # A register under the working directory is not the packaged one.
    working = tmp_path / "elsewhere"
    _write_register(working / "registries" / "scenario_lineage.yaml", "working")
    monkeypatch.setattr(resources, "source_checkout_root", lambda: None)
    monkeypatch.setattr(resources, "files", lambda _: tmp_path)
    monkeypatch.chdir(working)

    assert load_scenario_lineage_register().current_taxonomy_version == "packaged"

    bundled.unlink()
    with pytest.raises(FileNotFoundError):
        load_scenario_lineage_register()


def test_the_scenario_guide_shows_a_valid_roster_change():
    """docs/adding-a-scenario.md shows the generation and the transition that a
    new scenario registers. With the roster's digest in place of the example's
    placeholder, they load and pass lineage validation, so the example has
    every field the schema requires. The example's generation is a placeholder,
    not a label the register already uses."""
    guide = (_PROJECT_ROOT / "docs" / "adding-a-scenario.md").read_text(
        encoding="utf-8"
    )
    (block,) = (
        block
        for block in re.findall(r"```yaml\n(.*?)```", guide, flags=re.DOTALL)
        if "transitions:" in block
    )
    example = yaml.safe_load(block)
    (generation,) = example["generations"]
    (transition,) = example["transitions"]
    (scenario_id,) = transition["successor_ids"]
    generation["scenario_roster_sha256"] = scenario_roster_sha256([scenario_id])

    register = ScenarioLineageRegister.model_validate({"schema_version": 1, **example})
    assert register.current_taxonomy_version == generation["version"]
    assert transition["kind"] == "scope_expansion"
    cell = _cell(scenario_id, introduced_in=generation["version"])
    assert validate_scenario_lineage([cell], register) == []
    assert transition["executed_in"] == generation["version"]
    assert (
        generation["version"]
        not in load_scenario_lineage_register().generation_versions
    )

    # The guide lists the values the schema admits beside the two it shows.
    fields = PopulationTransitionDef.model_fields
    for value in (
        *get_args(fields["kind"].annotation),
        *get_args(fields["evidence_staling"].annotation),
    ):
        assert f"`{value}`" in guide, value
