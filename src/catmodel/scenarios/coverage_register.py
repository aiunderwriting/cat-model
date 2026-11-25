"""Scenario coverage-gap register.

``registries/coverage_gaps.yaml`` records the known gaps in what individual
scenarios estimate. A gap row names the scenarios it qualifies, and a result
that includes one of them lists the row among its limitations. Register entries
create no comparison requirement or gate. Scenario-library validation requires
every row to name live scenarios and every scenario with a normalisation
exposure base to have an ``ungoverned_exposure_denominator`` row.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from catmodel.resources import resource_path
from catmodel.scenarios.loader import ScenarioDefinition
from catmodel.scenarios.schema import FrozenModel, Identifier, Prose
from catmodel.yaml_io import safe_load

COVERAGE_REGISTER_RELATIVE_PATH = Path("registries/coverage_gaps.yaml")


class CoverageGap(FrozenModel):
    """One registered coverage gap shared by the scenarios it names."""

    gap_id: Identifier
    scenario_ids: tuple[Identifier, ...] = Field(min_length=1)
    gap_type: Literal[
        "ungoverned_exposure_denominator",
        "evidence_population_quality",
        "unestimated_loss_layer",
    ]
    materiality: Literal["low", "medium", "high"]
    bias_direction: Literal["understatement", "overstatement", "uncertain"]
    rationale: Prose
    sensitivity_treatment: Prose
    remediation_trigger: Prose

    @model_validator(mode="after")
    def unique_scenarios(self) -> CoverageGap:
        if len(self.scenario_ids) != len(set(self.scenario_ids)):
            raise ValueError(f"coverage gap {self.gap_id} repeats a scenario")
        return self


class CoverageRegister(FrozenModel):
    schema_version: Literal[1]
    gaps: tuple[CoverageGap, ...]

    @model_validator(mode="after")
    def unique_gaps(self) -> CoverageRegister:
        ids = [gap.gap_id for gap in self.gaps]
        if len(ids) != len(set(ids)):
            raise ValueError("coverage gaps must have unique IDs")
        return self

    def for_scenarios(self, scenario_ids: Collection[str]) -> CoverageRegister:
        """The same register with only the gap rows of these scenarios."""
        return self.model_copy(
            update={
                "gaps": tuple(
                    gap
                    for gap in self.gaps
                    if set(gap.scenario_ids) & set(scenario_ids)
                )
            }
        )


def load_coverage_register(path: Path | str | None = None) -> CoverageRegister:
    """Load the coverage register, by default the packaged repository one."""
    resolved = (
        Path(path)
        if path is not None
        else resource_path(*COVERAGE_REGISTER_RELATIVE_PATH.parts)
    )
    try:
        payload = safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("coverage register must be a YAML mapping")
        return CoverageRegister.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Error loading coverage register {resolved}: {exc}") from exc


@cache
def repository_coverage_register() -> CoverageRegister:
    return load_coverage_register()


def coverage_register_errors(
    register: CoverageRegister, scenarios: Sequence[ScenarioDefinition]
) -> list[str]:
    """Rows must name live scenarios, and every scenario whose exposure base is
    only a normalisation needs an ``ungoverned_exposure_denominator`` row."""
    live = {scenario.id for scenario in scenarios}
    errors = [
        f"coverage gap {gap.gap_id} names unknown scenarios "
        f"{sorted(set(gap.scenario_ids) - live)}"
        for gap in register.gaps
        if not set(gap.scenario_ids) <= live
    ]
    recorded = {
        scenario_id
        for gap in register.gaps
        if gap.gap_type == "ungoverned_exposure_denominator"
        for scenario_id in gap.scenario_ids
    }
    errors.extend(
        f"{scenario.id}: its {' and '.join(scenario.normalisation_bases)} exposure "
        "base is a normalisation; record the gap as an "
        f"ungoverned_exposure_denominator row in {COVERAGE_REGISTER_RELATIVE_PATH}"
        for scenario in scenarios
        if scenario.normalisation_bases and scenario.id not in recorded
    )
    return errors
