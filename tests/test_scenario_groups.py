"""Scenario groups: scenario EALs added up by reporting domain and by the
facets on which every scenario has one value, and the summary's table of
reporting domains by severity tier."""

import csv
import json
import math

import pytest
from click.testing import CliRunner

from catmodel.cli import cli
from catmodel.resources import scenarios_path
from catmodel.results.execution import run_setup
from catmodel.results.results_summary import DOMAIN_NAMES, _Summary
from catmodel.results.scenario_groups import library_classification, scenario_groups
from catmodel.scenarios.catalog import DEFAULT_CATALOG_RELATIVE_PATH
from catmodel.scenarios.facets import FACET_ROLLUP_AXES, SCENARIO_GROUP_AXES
from catmodel.scenarios.loader import load_scenarios
from catmodel.scenarios.taxonomy_registry import RISK_DOMAIN_ORDER
from catmodel.setup.contextual_setup import compile_contextual_setup
from catmodel.setup.store import build_setup
from test_support.contextual_risk import (
    fixture_artifact,
    governed_contextual_instrument,
    scenario_node_answer,
)

_LIBRARY = tuple(row.id for row in load_scenarios(scenarios_path()))


def _contributions(eals, *, total="sum"):
    """A ``scenario_contributions`` section holding ``eals`` by scenario; a
    scenario whose EAL is None is unavailable."""
    known = [value for value in eals.values() if value is not None]
    pooled = math.fsum(known) if total == "sum" else total
    return {
        "status": "available",
        "measure": "welfare",
        "expected_annual_loss": pooled,
        "scenarios": [
            {
                "scenario_id": key,
                "status": "available" if value is not None else "unavailable",
                "expected_annual_loss": value,
            }
            for key, value in eals.items()
        ],
    }


def _groups(result, axis):
    (found,) = [row["groups"] for row in result["axes"] if row["axis"] == axis]
    return {group["value"]: group for group in found}


def test_the_axes_are_the_reporting_domain_and_the_single_valued_facets():
    assert SCENARIO_GROUP_AXES == (
        "risk_domain",
        "causal_origin",
        "lifecycle_stage",
        "deployment_control_mode",
        "party_basis",
    )
    # An axis on which a scenario can have several values is not grouped.
    assert not {
        axis
        for axis, additivity in FACET_ROLLUP_AXES.items()
        if additivity != "partition"
    } & set(SCENARIO_GROUP_AXES)
    classification = library_classification(_LIBRARY)
    assert set(classification) == set(_LIBRARY)
    for values in classification.values():
        assert set(values) == set(SCENARIO_GROUP_AXES)
        assert all(isinstance(value, str) and value for value in values.values())


def test_every_axis_puts_each_scenario_in_one_group_and_the_groups_add_up():
    eals = {key: float(index + 1) for index, key in enumerate(_LIBRARY)}
    result = scenario_groups(_contributions(eals), library_classification(_LIBRARY))
    assert result["status"] == "available"
    assert result["measure"] == "welfare"
    assert result["unclassified_scenarios"] == []
    assert [row["axis"] for row in result["axes"]] == list(SCENARIO_GROUP_AXES)
    total = math.fsum(eals.values())
    assert result["expected_annual_loss"] == total
    for axis in result["axes"]:
        members = [key for group in axis["groups"] for key in group["scenario_ids"]]
        assert sorted(members) == sorted(_LIBRARY)
        assert math.fsum(
            group["expected_annual_loss"] for group in axis["groups"]
        ) == pytest.approx(total, rel=1e-12)
        assert math.fsum(
            group["share_of_total"] for group in axis["groups"]
        ) == pytest.approx(1, rel=1e-12)
        for group in axis["groups"]:
            assert group["status"] == "available" and group["reason"] is None
            assert group["expected_annual_loss"] == pytest.approx(
                math.fsum(eals[key] for key in group["scenario_ids"])
            )
        # Decreasing EAL.
        amounts = [group["expected_annual_loss"] for group in axis["groups"]]
        assert amounts == sorted(amounts, reverse=True)


def test_the_reporting_domains_are_the_catalog_s():
    catalog = json.loads(
        (scenarios_path().parent / DEFAULT_CATALOG_RELATIVE_PATH).read_text()
    )
    result = scenario_groups(
        _contributions(dict.fromkeys(_LIBRARY, 1.0)), library_classification(_LIBRARY)
    )
    domains = _groups(result, "risk_domain")
    assert {key: len(group["scenario_ids"]) for key, group in domains.items()} == {
        key: row["scenario_cell_count"] for key, row in catalog["risk_domains"].items()
    }
    for row in catalog["scenarios"]:
        assert row["id"] in domains[row["risk_domain"]]["scenario_ids"]


def test_a_group_missing_an_eal_is_a_lower_bound_and_one_with_none_is_unavailable():
    cyber = (
        "compromise_organizations",
        "compromise_individuals",
        "fraud_organizations",
    )
    eals = {cyber[0]: 3.0, cyber[1]: None, cyber[2]: 5.0, "enforcement_matter": None}
    result = scenario_groups(
        _contributions(eals, total=10.0), library_classification(eals)
    )
    domains = _groups(result, "risk_domain")
    assert domains["cyber"] == {
        "value": "cyber",
        "scenario_ids": sorted(cyber),
        "status": "lower_bound",
        "reason": "The sum leaves out the group's scenarios without an EAL.",
        "expected_annual_loss": 8.0,
        "share_of_total": 0.8,
    }
    assert domains["regulatory"] == {
        "value": "regulatory",
        "scenario_ids": ["enforcement_matter"],
        "status": "unavailable",
        "reason": "None of the group's scenarios has an EAL.",
        "expected_annual_loss": None,
        "share_of_total": None,
    }
    # The group without an EAL comes last.
    assert list(domains) == ["cyber", "regulatory"]


def test_no_group_has_a_share_without_the_pooled_eal():
    eals = {"compromise_organizations": 3.0, "enforcement_matter": 1.0}
    result = scenario_groups(
        _contributions(eals, total=None), library_classification(eals)
    )
    assert result["expected_annual_loss"] is None
    for axis in result["axes"]:
        for group in axis["groups"]:
            assert group["expected_annual_loss"] is not None
            assert group["share_of_total"] is None


def test_scenarios_outside_the_library_are_named_and_in_no_group():
    eals = {"compromise_organizations": 3.0, "stipulated": 1.0}
    result = scenario_groups(_contributions(eals), library_classification(eals))
    assert result["unclassified_scenarios"] == ["stipulated"]
    (group,) = _groups(result, "risk_domain").values()
    assert group["scenario_ids"] == ["compromise_organizations"]
    assert group["share_of_total"] == pytest.approx(0.75)
    # A setup with no library scenario has no groups.
    assert scenario_groups(
        _contributions({"stipulated": 1.0}), library_classification(["stipulated"])
    ) == {
        "status": "inapplicable",
        "reason": "No scenario of this setup is a scenario of the library, so "
        "none has a reporting domain or facets.",
    }


def test_unavailable_contributions_give_their_reason():
    assert scenario_groups(
        {"status": "unavailable", "reason": "No additive view was requested."},
        library_classification(_LIBRARY),
    ) == {"status": "unavailable", "reason": "No additive view was requested."}


def _library_run(tmp_path, scenario_ids):
    instrument = governed_contextual_instrument(scenario_ids)
    setup = compile_contextual_setup(
        instrument,
        fixture_artifact(instrument, lambda prompt: scenario_node_answer(prompt.node)),
        setup_id="groups",
        years_per_state=4,
    )
    package = build_setup(
        destination=tmp_path / "package", setup=setup, code_commit="synthetic-fixture"
    )
    return run_setup(
        setup_path=package.root,
        output=tmp_path / "run",
        code_commit="synthetic-fixture",
    )


def test_a_run_groups_its_library_scenarios_and_the_summary_tabulates_domains(
    tmp_path,
):
    scenario_ids = ("agent_actions_for_deployer", "enforcement_matter")
    report = _library_run(tmp_path, scenario_ids)
    contributions = report["scenario_contributions"]
    eals = {
        row["scenario_id"]: row["expected_annual_loss"]
        for row in contributions["scenarios"]
    }
    assert set(eals) == set(scenario_ids) and all(eals.values())
    groups = report["scenario_groups"]
    assert groups["status"] == "available"
    assert groups["measure"] == contributions["measure"]
    assert groups["expected_annual_loss"] == contributions["expected_annual_loss"]
    domains = _groups(groups, "risk_domain")
    assert (
        domains["agent"]["expected_annual_loss"] == eals["agent_actions_for_deployer"]
    )
    assert domains["regulatory"]["expected_annual_loss"] == eals["enforcement_matter"]
    origins = _groups(groups, "causal_origin")
    assert set(origins) == {"ai_failure", "provider_conduct"}
    assert math.fsum(
        group["share_of_total"] for group in domains.values()
    ) == pytest.approx(1)

    result = CliRunner().invoke(
        cli,
        [
            "results",
            "summary",
            "--results",
            str(tmp_path / "run" / "results.json"),
            "--output",
            str(tmp_path / "summary"),
        ],
    )
    assert result.exit_code == 0, result.output
    markdown = (tmp_path / "summary" / "summary.md").read_text()
    section = markdown.split("## Expected annual loss by reporting domain")[1].split(
        "\n## "
    )[0]
    lines = section.splitlines()
    assert "| Reporting domain | Scenarios | EAL | Share of EAL | Note |" in lines
    assert (
        sum(
            line.startswith(("| `agent` | 1 |", "| `regulatory` | 1 |"))
            for line in lines
        )
        == 2
    )
    assert "`causal_origin`" in section and "`party_basis`" in section
    rows = [
        row
        for row in csv.DictReader((tmp_path / "summary" / "summary.csv").open())
        if row["section"] == "scenario_group"
    ]
    assert {row["subject"].split(":")[0] for row in rows} == set(SCENARIO_GROUP_AXES)
    # A contextual result holds each scenario's large-incident counts, so the
    # summary also tabulates the domains by severity tier. Each domain here
    # holds one scenario, so its counts are that scenario's, and the domains'
    # counts add up to the exhibit's total.
    tiers = markdown.split("## Reporting domains by severity tier")[1].split("\n## ")[0]
    assert (
        sum(
            line.startswith(
                (
                    "| Agent and physical-control failures (`agent`) |",
                    "| Regulatory and legal action (`regulatory`) |",
                )
            )
            for line in tiers.splitlines()
        )
        == 2
    )
    anchor = report["sensitivity_summary"]["large_loss_anchor"]
    for field in ("modelled_large_count", "modelled_very_large_count"):
        counts = {
            row["subject"]: float(row["value"])
            for row in rows
            if row["statistic"] == field
        }
        assert counts == {
            "risk_domain:agent": pytest.approx(
                anchor["by_scenario"]["agent_actions_for_deployer"][field]
            ),
            "risk_domain:regulatory": pytest.approx(
                anchor["by_scenario"]["enforcement_matter"][field]
            ),
        }
        assert all(count > 0 for count in counts.values())
        assert math.fsum(counts.values()) == pytest.approx(anchor["total"][field])
    share = next(
        row
        for row in rows
        if row["subject"] == "risk_domain:agent"
        and row["statistic"] == "share_of_total"
    )
    assert (share["unit"], share["status"], share["label"]) == (
        "share",
        "available",
        "",
    )
    assert float(share["value"]) == pytest.approx(domains["agent"]["share_of_total"])


def test_the_summary_marks_a_lower_bound_and_names_scenarios_outside_the_library():
    eals = {
        "compromise_organizations": 3e9,
        "compromise_individuals": None,
        "enforcement_matter": None,
        "stipulated": 1e9,
    }
    section = scenario_groups(
        _contributions(eals, total=5e9), library_classification(eals)
    )
    report = {
        "analysis_spec": {"reference_arm": "reference"},
        "arms": {"reference": {"metrics": {}}},
        "scenario_groups": section,
    }
    summary = _Summary(report)
    summary.scenario_groups()
    text = "\n".join(summary.lines)
    assert "| `cyber` | 2 | at least 3B | at least 60.0% |" in text
    assert (
        "| `regulatory` | 1 | — | — | None of the group's scenarios has an EAL. |"
        in text
    )
    assert "Not in the scenario library, so in no domain: `stipulated`." in text
    rows = {
        (row["subject"], row["statistic"]): row
        for row in summary.rows
        if row["section"] == "scenario_group"
    }
    bound = rows["risk_domain:cyber", "expected_annual_loss"]
    assert (bound["value"], bound["label"], bound["status"]) == (
        3e9,
        "lower_bound",
        "available",
    )
    missing = rows["risk_domain:regulatory", "expected_annual_loss"]
    assert (missing["value"], missing["label"], missing["status"]) == (
        None,
        None,
        "unavailable",
    )
    assert missing["reason"] == "None of the group's scenarios has an EAL."

    # A results file without the section, or with it unavailable, says so.
    for held, reason in (
        (None, "The results file has no scenario groups."),
        ({"status": "unavailable", "reason": "No view."}, "No view."),
    ):
        summary = _Summary({**report, "scenario_groups": held})
        summary.scenario_groups()
        assert f"Unavailable: {reason}" in summary.lines
        assert summary.rows[-1]["status"] == "unavailable"
        assert summary.rows[-1]["reason"] == reason
    # A setup with no library scenario has no domains, so no section.
    summary = _Summary(
        {**report, "scenario_groups": {"status": "inapplicable", "reason": "None."}}
    )
    summary.scenario_groups()
    assert summary.lines == [] and summary.rows == []


def test_every_reporting_domain_has_one_display_name():
    assert set(DOMAIN_NAMES) == set(RISK_DOMAIN_ORDER)
    assert len(set(DOMAIN_NAMES.values())) == len(DOMAIN_NAMES)


def _anchor(counts, *, very_large_loss=1e11):
    """A ``large_loss_anchor`` exhibit with each scenario's expected annual
    numbers of incidents at or above the two levels under the run's laws. A
    scenario whose counts are None has an unavailable record."""
    return {
        "levels": {"large_loss": 1e9, "very_large_loss": very_large_loss},
        "by_scenario": {
            key: {"status": "unavailable", "reasons": ["No law."]}
            if value is None
            else {
                "status": "available",
                "modelled_large_count": value[0],
                "modelled_very_large_count": value[1],
            }
            for key, value in counts.items()
        },
        "total": {"status": "available"},
    }


def _tiers(groups, anchor):
    """The summary of a USD result holding ``groups`` as its scenario groups
    and ``anchor`` as its large-loss anchor exhibit, after its table of
    reporting domains by severity tier."""
    report = {
        "analysis_spec": {"reference_arm": "reference"},
        "arms": {"reference": {"metrics": {}}},
        "study": {"monetary_bases": [{"currency": "USD"}]},
        "scenario_groups": groups,
        "sensitivity_summary": {} if anchor is None else {"large_loss_anchor": anchor},
    }
    summary = _Summary(report)
    summary.domain_tiers()
    return summary


_TIER_EALS = {
    "compromise_organizations": 3e9,
    "compromise_individuals": None,
    "fraud_organizations": 5e9,
    "cbrn_chemical": 2e9,
    "cbrn_biological_novice": 4e9,
    "enforcement_matter": 1e9,
}
# Counts at USD 1 billion and USD 100 billion. One cyber scenario's record is
# unavailable, and the exhibit has no record of the regulatory scenario.
_TIER_COUNTS = {
    "compromise_organizations": (0.02, 0.001),
    "compromise_individuals": None,
    "fraud_organizations": (0.03, 0.002),
    "cbrn_chemical": (0.25, 0.05),
    "cbrn_biological_novice": (0.5, 0.1),
}


def test_the_summary_adds_up_each_domain_s_eal_and_large_incident_counts():
    """The table names each reporting domain and gives the sums over its
    scenarios: the EAL of the scenario groups and the counts of the large-loss
    anchor exhibit at both levels. A sum that leaves a scenario out is a
    lower bound and one with nothing to add is a dash, each with its note."""
    groups = scenario_groups(
        _contributions(_TIER_EALS, total=15e9), library_classification(_TIER_EALS)
    )
    summary = _tiers(groups, _anchor(_TIER_COUNTS))
    assert summary.lines[0] == "## Reporting domains by severity tier"
    table = [line for line in summary.lines if line.startswith("|")]
    # By hand: cyber 3B + 5B, 0.02 + 0.03 and 0.001 + 0.002 without the
    # scenario that has neither; CBRN 2B + 4B, 0.25 + 0.5 and 0.05 + 0.1.
    assert table == [
        "| Reporting domain | EAL | Incidents ≥ 1B USD a year | Incidents ≥ 100B USD a year | Note |",
        "| --- | --- | --- | --- | --- |",
        "| Cyber and fraud (`cyber`) | at least 8B | at least 0.05 | at least 0.003 | "
        "EAL: The sum leaves out the group's scenarios without an EAL. 1B USD and "
        "100B USD: The sum leaves out 1 of the domain's 3 scenarios, which has no "
        "such count. |",
        "| CBRN weapons (`cbrn`) | 6B | 0.75 | 0.15 |  |",
        "| Regulatory and legal action (`regulatory`) | 1B | — | — | 1B USD and "
        "100B USD: None of the domain's scenarios has such a count. |",
    ]
    rows = {(row["subject"], row["statistic"]): row for row in summary.rows}
    assert set(rows) == {
        (f"risk_domain:{domain}", statistic)
        for domain in ("cyber", "cbrn", "regulatory")
        for statistic in ("modelled_large_count", "modelled_very_large_count")
    }
    assert {(row["section"], row["unit"]) for row in summary.rows} == {
        ("scenario_group", "count")
    }
    bound = rows["risk_domain:cyber", "modelled_large_count"]
    assert bound["value"] == pytest.approx(0.05)
    assert (bound["label"], bound["status"]) == ("lower_bound", "available")
    whole = rows["risk_domain:cbrn", "modelled_very_large_count"]
    assert whole["value"] == pytest.approx(0.15)
    assert (whole["label"], whole["status"], whole["reason"]) == (
        None,
        "available",
        None,
    )
    missing = rows["risk_domain:regulatory", "modelled_very_large_count"]
    assert (missing["value"], missing["label"], missing["status"]) == (
        None,
        None,
        "unavailable",
    )
    assert missing["reason"] == "None of the domain's scenarios has such a count."


def test_the_tier_table_has_only_the_levels_and_names_the_results_hold():
    eals = {"cbrn_chemical": 2e9, "service_outage_degradation": 1e9}
    counts = {"cbrn_chemical": (0.25, 0.05), "service_outage_degradation": (0.5, 0.1)}
    groups = scenario_groups(_contributions(eals), library_classification(eals))
    # An exhibit without the higher level gives the lower tier alone.
    summary = _tiers(groups, _anchor(counts, very_large_loss=None))
    assert [line for line in summary.lines if line.startswith("|")] == [
        "| Reporting domain | EAL | Incidents ≥ 1B USD a year | Note |",
        "| --- | --- | --- | --- |",
        "| CBRN weapons (`cbrn`) | 2B | 0.25 |  |",
        "| Service failures (`svc`) | 1B | 0.5 |  |",
    ]
    assert "100B" not in "\n".join(summary.lines)
    assert {row["statistic"] for row in summary.rows} == {"modelled_large_count"}
    # A domain the summary has no name for is shown by its code.
    (domains,) = [row for row in groups["axes"] if row["axis"] == "risk_domain"]
    domains["groups"][0]["value"] = "unnamed"
    summary = _tiers(groups, _anchor(counts))
    assert "| `unnamed` | 2B | 0.25 | 0.05 |  |" in summary.lines


@pytest.mark.parametrize(
    ("groups", "anchor"),
    [
        # No large-loss anchor exhibit: not a contextual setup.
        ("available", None),
        # The exhibit withheld for a reweighted result, and one of a setup in
        # which no scenario asks the counts.
        ("available", {"status": "unavailable", "reason": "Reweighted."}),
        (
            "available",
            {
                "levels": {"large_loss": 1e9, "very_large_loss": 1e11},
                "by_scenario": {},
                "total": {"status": "inapplicable"},
            },
        ),
        ("available", {**_anchor(_TIER_COUNTS), "levels": None}),
        # No scenario groups, or none available.
        (None, _anchor(_TIER_COUNTS)),
        ({"status": "unavailable", "reason": "No view."}, _anchor(_TIER_COUNTS)),
        ({"status": "inapplicable", "reason": "None."}, _anchor(_TIER_COUNTS)),
    ],
)
def test_a_result_without_domains_or_counts_has_no_tier_table(groups, anchor):
    if groups == "available":
        groups = scenario_groups(
            _contributions(_TIER_EALS, total=15e9), library_classification(_TIER_EALS)
        )
    summary = _tiers(groups, anchor)
    assert summary.lines == [] and summary.rows == []
