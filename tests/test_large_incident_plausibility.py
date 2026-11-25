"""The summary's plausibility check on incidents at or above the large-loss level.

The results summary reads, from the ``large_loss_anchor`` exhibit, the expected
annual number of such incidents each scenario's respondents stated, the number
after pooling and the number under the run's laws, with the probability of at
least one in the year, and flags a scenario whose count under the run's laws is
at least 0.1 a year and a total of at least 1 a year (``NUM-PLAUSIBILITY``).
The flags are a reading aid: they change no result.
"""

import copy
import csv
import math

import pytest

from catmodel.contracts.contextual_risk import ContextualRiskArtifact
from catmodel.elicitation.contextual_risk import contextual_prompts
from catmodel.results.plausibility import (
    LARGE_INCIDENT_FLAGS,
    LARGE_INCIDENT_STATISTICS,
    large_incident_table,
)
from catmodel.results.results_summary import results_summary, write_results_summary
from test_support.contextual_risk import contextual_case, governed_contextual_instrument
from test_support.results import reported_exhibit
from tests.test_contextual_study import _capture
from tests.test_large_loss_reporting import LEAF, _answer
from tests.test_results_file import _report, _study

NOT_POISSON = "The node's negative_binomial count is not Poisson."
# The world-output exhibit of a study in another currency than its reference.
WORLD_OUTPUT = {
    "status": "unavailable",
    "reason": "The populations' currencies ['EUR'] differ from the reference currency USD.",
    "reference": {},
    "flags": {},
}
HEADER = [
    "| Scenario | Stated | After pooling | Run laws | P(at least one) | Flag |",
    "| --- | --- | --- | --- | --- | --- |",
]


def _counts(stated, pooled, run, probability, withheld=()):
    """An available exhibit record with the fields the table reads."""
    return {
        "status": "available",
        "stated_large_count": stated,
        "pooled_large_count": pooled,
        "modelled_large_count": run,
        "probability_at_least_one_large": probability,
        "probability_withheld_reasons": list(withheld),
    }


def _total(stated, pooled, run, probability, withheld=()):
    """An available total, with the fields the summary's large-loss anchor
    section also reads."""
    return {
        **_counts(stated, pooled, run, probability, withheld),
        "eal_alternatives": {},
        "two_anchor_tail": {"very_large_count": None},
        **dict.fromkeys(
            (
                "extrapolated_large_count",
                "stated_very_large_count",
                "pooled_very_large_count",
                "modelled_very_large_count",
            )
        ),
    }


def _summary(sensitivity_summary):
    """The summary of a report with the given exhibits: its Markdown, the
    lines of its plausibility section and its CSV rows."""
    report = copy.deepcopy(_report())
    report["study"] = _study()
    report["sensitivity_summary"] = sensitivity_summary
    markdown, rows = results_summary(report)
    section = markdown.split("## Plausibility checks")[1].split("\n## ")[0]
    return markdown, section.splitlines(), rows


def _plausibility_rows(rows):
    return {
        (row["subject"], row["statistic"]): row
        for row in rows
        if row["section"] == "plausibility"
    }


def _exhibit():
    """Four scenarios: one over its threshold, one exactly at it, one just
    below it whose probability is withheld, and one without a law."""
    return {
        # Not the contract's level, so a hard-coded USD 1 billion would show.
        "levels": {"large_loss": 5e8, "very_large_loss": 1e11},
        "by_scenario": {
            "ban": _counts(0.05, 0.04, 0.0999, None, (NOT_POISSON,)),
            "fraud": _counts(4.0, 3.0, 0.8, 0.55),
            "leak": _counts(0.2, 0.15, 0.1, 0.095),
            "outage": {
                "status": "unavailable",
                "reasons": ["No law at state a.", "No law at state b."],
            },
        },
        "total": _total(4.25, 3.19, 1.0, None, (NOT_POISSON,)),
    }


def test_the_summary_tabulates_the_implied_large_incidents_with_review_flags():
    exhibits = {"large_loss_anchor": _exhibit()}
    untouched = copy.deepcopy(exhibits)
    _, lines, rows = _summary(exhibits)
    # The level is the exhibit's, in the study's currency, and the thresholds
    # are stated with their status.
    assert lines[2].startswith(
        "Report only. Expected annual incidents of at least 500M EUR in each scenario"
    )
    assert lines[2].endswith(
        "A scenario at or above 0.1 a year or a total at or above 1 a year under "
        "the run's laws is flagged for review (thresholds `INITIAL_ESTIMATE`)."
    )
    # The total first, then the scenarios by decreasing count under the run's
    # laws and those without one. A total of exactly 1 and a scenario of exactly
    # 0.1 are flagged; 0.0999 is not, and 0.8 is flagged only as a scenario.
    assert lines[4:11] == [
        *HEADER,
        "| Total | 4.25 | 3.19 | 1 | — | flagged |",
        "| `fraud` | 4 | 3 | 0.8 | 0.55 | flagged |",
        "| `leak` | 0.2 | 0.15 | 0.1 | 0.095 | flagged |",
        "| `ban` | 0.05 | 0.04 | 0.0999 | — |  |",
        "| `outage` | — | — | — | — |  |",
    ]
    note = lines[12]
    assert note.startswith("How to read it: a count is the expected number of separate")
    assert "the public record attributes to any one developer's models" in note
    assert note.endswith("changes no result.")
    # Rendering reads the exhibit and leaves it as it was.
    assert exhibits == untouched
    found = _plausibility_rows(rows)
    assert list(found) == [
        (subject, statistic)
        for subject in ("total", "fraud", "leak", "ban", "outage")
        for statistic in LARGE_INCIDENT_STATISTICS
    ]
    assert {
        key: (row["value"], row["unit"], row["label"], row["status"], row["reason"])
        for key, row in found.items()
        if key[0] in ("total", "ban", "outage")
    } == {
        ("total", "stated_large_count"): (4.25, "count", None, "available", None),
        ("total", "pooled_large_count"): (3.19, "count", None, "available", None),
        ("total", "modelled_large_count"): (1.0, "count", "flagged", "available", None),
        ("total", "probability_at_least_one_large"): (
            None,
            "probability",
            None,
            "unavailable",
            NOT_POISSON,
        ),
        ("ban", "stated_large_count"): (0.05, "count", None, "available", None),
        ("ban", "pooled_large_count"): (0.04, "count", None, "available", None),
        ("ban", "modelled_large_count"): (0.0999, "count", None, "available", None),
        ("ban", "probability_at_least_one_large"): (
            None,
            "probability",
            None,
            "unavailable",
            NOT_POISSON,
        ),
        **{
            ("outage", statistic): (
                None,
                "probability" if statistic.startswith("probability") else "count",
                None,
                "unavailable",
                "No law at state a.; No law at state b.",
            )
            for statistic in LARGE_INCIDENT_STATISTICS
        },
    }
    # Only the count under the run's laws carries the flag.
    assert {key for key, row in found.items() if row["label"] == "flagged"} == {
        (subject, "modelled_large_count") for subject in ("total", "fraud", "leak")
    }
    assert {row["view"] for row in found.values()} == {None}


def test_every_dash_of_the_large_incident_table_has_its_reason_in_the_gaps():
    markdown, _, _ = _summary({"large_loss_anchor": _exhibit()})
    gaps = markdown.split("## Unavailable, withheld and not requested")[1]
    gaps = gaps.split("\n## ")[0].splitlines()
    assert (
        "- Large-incident statistic unavailable (2): "
        "`total probability_at_least_one_large`, "
        f"`ban probability_at_least_one_large` — {NOT_POISSON}"
    ) in gaps
    assert (
        "- Large-incident counts unavailable (1): `outage` — "
        "No law at state a.; No law at state b."
    ) in gaps


@pytest.mark.parametrize(
    "exhibit, level, reason",
    [
        # Withheld for a reweighted result.
        (
            {"status": "unavailable", "reason": "Reweighted."},
            "the large-loss level",
            "Reweighted.",
        ),
        # No scenario with a large-loss anchor.
        (
            {"levels": None, "by_scenario": {}, "total": {"status": "inapplicable"}},
            "the large-loss level",
            "No scenario has a large-loss anchor.",
        ),
        # A total some scenario's law leaves unavailable.
        (
            {
                "levels": {"large_loss": 1e9, "very_large_loss": None},
                "by_scenario": {},
                "total": {"status": "unavailable", "reasons": ["No law at state a."]},
            },
            "1B EUR",
            "No law at state a.",
        ),
    ],
)
def test_an_exhibit_without_counts_renders_dashes_and_its_reason(
    exhibit, level, reason
):
    markdown, lines, rows = _summary(
        {"plausibility": WORLD_OUTPUT, "large_loss_anchor": exhibit}
    )
    # The world-output table comes first.
    assert lines[2].startswith("Shares of world output unavailable: ")
    assert f"Expected annual incidents of at least {level} in each" in lines[4]
    assert lines[6:9] == [*HEADER, "| Total | — | — | — | — |  |"]
    assert not lines[9]
    assert f"- Large-incident counts unavailable (1): `total` — {reason}" in (
        markdown.splitlines()
    )
    assert [
        (row["subject"], row["statistic"], row["value"], row["label"], row["status"])
        for row in rows
        if row["section"] == "plausibility"
    ] == [
        ("total", statistic, None, None, "unavailable")
        for statistic in LARGE_INCIDENT_STATISTICS
    ]
    assert {row["reason"] for row in rows if row["section"] == "plausibility"} == {
        reason
    }


def test_an_available_record_without_a_value_is_a_dash_never_a_zero():
    """A count the exhibit leaves null is unavailable with a reason, and a
    scenario without a count under the run's laws is not flagged whatever its
    respondents stated. A reported zero stays a zero."""
    _, lines, rows = _summary(
        {
            "large_loss_anchor": {
                "levels": {"large_loss": 1e9, "very_large_loss": None},
                "by_scenario": {"leak": _counts(5.0, None, None, None)},
                "total": _total(0.0, 0.0, 0.0, 0.0),
            }
        }
    )
    assert lines[6:8] == [
        "| Total | 0 | 0 | 0 | 0 |  |",
        "| `leak` | 5 | — | — | — |  |",
    ]
    found = _plausibility_rows(rows)
    assert found["total", "modelled_large_count"]["value"] == 0.0
    assert found["total", "modelled_large_count"]["status"] == "available"
    for statistic in LARGE_INCIDENT_STATISTICS[1:]:
        row = found["leak", statistic]
        assert (row["value"], row["label"], row["status"], row["reason"]) == (
            None,
            None,
            "unavailable",
            "The large-loss anchor exhibit reports no value for it.",
        )


def test_results_without_the_exhibit_have_no_large_incident_table():
    """Only a contextual setup carries the exhibit. Another result's section
    holds the world-output table alone, and a report with neither exhibit has
    no section."""
    markdown, lines, rows = _summary({"plausibility": WORLD_OUTPUT})
    assert [line for line in lines if line] == [
        "Shares of world output unavailable: The populations' currencies ['EUR'] "
        "differ from the reference currency USD."
    ]
    assert HEADER[0] not in markdown and "Large-incident" not in markdown
    assert not [row for row in rows if row["section"] == "plausibility"]
    report = copy.deepcopy(_report())
    report["study"] = _study()
    markdown, rows = results_summary(report)
    assert "Plausibility" not in markdown and "Large-incident" not in markdown


def test_the_thresholds_are_the_registered_initial_estimates():
    assert LARGE_INCIDENT_FLAGS.model_dump() == {
        "leaf_annual_count": 0.1,
        "total_annual_count": 1.0,
        "status": "INITIAL_ESTIMATE",
    }


@pytest.mark.parametrize(
    "count, scenario_flagged, total_flagged",
    [
        (0.0, False, False),
        (0.0999, False, False),
        (0.1, True, False),
        (0.5, True, False),
        (0.999, True, False),
        (1.0, True, True),
        (30.5, True, True),
        # A computed count a unit in the last place below a threshold reaches it.
        (math.nextafter(0.1, 0.0), True, False),
        (math.nextafter(1.0, 0.0), True, True),
        (None, False, False),
    ],
)
def test_a_count_at_or_above_its_threshold_is_flagged(
    count, scenario_flagged, total_flagged
):
    table = large_incident_table(
        {
            "levels": {"large_loss": 1e9, "very_large_loss": None},
            # A stated or pooled count never sets a flag.
            "by_scenario": {"leak": _counts(50.0, 50.0, count, 1.0)},
            "total": _counts(50.0, 50.0, count, 1.0),
        }
    )
    (scenario,) = table.scenarios
    assert scenario.scenario_id == "leak" and table.total.scenario_id is None
    assert scenario.flagged is scenario_flagged
    assert table.total.flagged is total_flagged
    assert table.flags == LARGE_INCIDENT_FLAGS


def test_scenarios_are_ordered_by_their_count_under_the_run_laws():
    table = large_incident_table(
        {
            "levels": {"large_loss": 1e9, "very_large_loss": None},
            "by_scenario": {
                "a": _counts(9.0, 9.0, 0.02, 0.02),
                "b": {"status": "unavailable", "reasons": ["No law."]},
                "c": _counts(0.1, 0.1, 3.0, 0.9),
                "d": _counts(1.0, 1.0, None, None),
                "e": _counts(0.0, 0.0, 0.02, 0.02),
                "f": _counts(0.0, 0.0, 0.0, 0.0),
            },
            "total": _counts(10.2, 10.2, 3.04, 0.9),
        }
    )
    # Largest first; equal counts, and then scenarios without one, by ID.
    assert [row.scenario_id for row in table.scenarios] == [
        "c",
        "a",
        "e",
        "f",
        "b",
        "d",
    ]
    assert table.scenarios[4].unavailable == "No law."
    assert table.scenarios[5].unavailable is None
    assert set(table.scenarios[5].missing) == set(LARGE_INCIDENT_STATISTICS[2:])


def test_a_contextual_run_reports_the_counts_of_its_large_loss_anchor_exhibit(
    tmp_path,
):
    """A run of two leaves whose anchors shape their tails: the summary's rows
    are the exhibit's, the leaf at about 0.12 incidents a year is flagged and
    the leaf at about 0.03 and the total at about 0.15 are not."""
    other = "fraud_organizations"
    shares = {LEAF: 9e-3, other: 2e-3}
    instrument = governed_contextual_instrument((LEAF, other))
    artifact = ContextualRiskArtifact(
        instrument_sha256=instrument.digest(),
        evidence_status="fixture",
        cases=tuple(
            contextual_case(
                instrument, prompt, _answer(prompt.node, shares[prompt.node.node_id])
            )
            for prompt in contextual_prompts(instrument)
        ),
    )
    _, _, report, _ = _capture(tmp_path, instrument, artifact)
    exhibit = reported_exhibit(report["sensitivity_summary"]["large_loss_anchor"])
    records = {"total": exhibit["total"], **exhibit["by_scenario"]}
    assert set(records) == {"total", LEAF, other}
    counts = {name: row["modelled_large_count"] for name, row in records.items()}
    assert 0.1 < counts[LEAF] < 1 and counts[other] < 0.1 < counts["total"] < 1
    markdown, rows = results_summary(report)
    section = markdown.split("## Plausibility checks")[1].split("\n## ")[0]
    lines = section.splitlines()
    # After the world-output table, at the contract's level.
    assert lines.index(HEADER[0]) > lines.index(
        "| Scenario | Welfare EAL | Share of world output | Flag |"
    )
    assert "incidents of at least 1B USD in each scenario" in section
    start = lines.index(HEADER[0])
    assert [line.split(" | ")[0] for line in lines[start + 2 : start + 5]] == [
        "| Total",
        f"| `{LEAF}`",
        f"| `{other}`",
    ]
    assert [line.endswith("| flagged |") for line in lines[start + 2 : start + 5]] == [
        False,
        True,
        False,
    ]
    found = _plausibility_rows(rows)
    for name, record in records.items():
        for statistic in LARGE_INCIDENT_STATISTICS:
            row = found[name, statistic]
            assert row["value"] == record[statistic] and row["status"] == "available"
            assert row["label"] == (
                "flagged"
                if (name, statistic) == (LEAF, "modelled_large_count")
                else None
            )
    # Every value is reported, so the table adds no gap.
    assert not [
        line for line in markdown.splitlines() if line.startswith("- Large-incident")
    ]
    # summary.csv holds the same rows.
    write_results_summary(
        results_path=tmp_path / "run" / "results.json", output=tmp_path / "summary"
    )
    with (tmp_path / "summary" / "summary.csv").open(encoding="utf-8") as stream:
        written = [
            row
            for row in csv.DictReader(stream)
            if row["section"] == "plausibility"
            and row["statistic"] in LARGE_INCIDENT_STATISTICS
        ]
    assert [(row["subject"], row["statistic"]) for row in written] == [
        (name, statistic)
        for name in ("total", LEAF, other)
        for statistic in LARGE_INCIDENT_STATISTICS
    ]
    for row in written:
        assert float(row["value"]) == pytest.approx(
            records[row["subject"]][row["statistic"]], rel=1e-12
        )
        assert row["unit"] == (
            "probability" if row["statistic"].startswith("probability") else "count"
        )
        assert row["status"] == "available" and not row["reason"]
    assert [row["subject"] for row in written if row["label"] == "flagged"] == [LEAF]
