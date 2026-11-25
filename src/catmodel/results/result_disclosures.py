"""The study scope, scenario coverage and limitations list of a reduced result.

The reduction states which views and comparison families a result covers. This
module adds what needs the setup: the study the result describes, which
scenario populations it includes, and one ordered limitations list built from
the setup's disclosed limitations and the registered coverage gaps of its
scenarios, the run's own coverage and the model-wide items of
``docs/limitations.md``. ``applicable_limitations`` gives a reader of the
result the items of that list that apply to its study.
"""

from collections.abc import Collection, Mapping
from typing import Any

from catmodel.contracts.incident import parse_scenario_view
from catmodel.contracts.results import (
    NativeResults,
    ResultLimitation,
    ScenarioCoverage,
    StudyMonetaryBasis,
    StudyScope,
)
from catmodel.contracts.run_setup import RunSetup
from catmodel.scenarios.coverage_register import (
    COVERAGE_REGISTER_RELATIVE_PATH,
    repository_coverage_register,
)
from catmodel.simulation.small_events import SMALL_EVENT_LIMITATION

LIMITATIONS_DOCUMENT = "docs/limitations.md"

# Items of the ranked limitations list that every result lists. Each anchor
# must name a heading of docs/limitations.md (checked by tests/test_doc_links.py).
MODEL_LIMITATIONS = (
    (
        "1-the-risk-judgments-come-from-llms-and-cannot-be-backtested-in-general",
        "Any elicited risk judgment comes from LLM respondents, and the model's predictions cannot in general be backtested, because most scenarios have no matching incident data. The respondent calibration on known-answer count questions is the first check against outcomes; it is a diagnostic, and no result applies its fit.",
    ),
    (
        "3-tail-shapes-caps-and-spread-factors-are-judgments",
        "Tail shapes, caps, quantile extrapolation and spread factors are judgments. VaR, TVaR and the occurrence statistics depend on them far more than EAL does and are illustrative. The EAL parameter band shows how the count laws' and respondents' uncertainty moves the EAL, with the capability forecast integrated; it is not a range of annual losses and is neither a confidence nor a predictive interval.",
    ),
    (
        "5-scenarios-are-conditionally-independent",
        "Scenarios are independent given the capability trajectory and respondent, so the tail statistics may understate correlated catastrophes. EAL does not depend on this assumption.",
    ),
    (
        "6-the-capability-forecast-rests-on-few-releases",
        "Where a capability forecast conditions the result, it rests on few frontier releases and a judgmental pace shape.",
    ),
    (
        "7-the-model-quantifies-selected-populations-only",
        "Only the setup's populations are quantified; populations outside it are unquantified, not counted as zero.",
    ),
    (
        "8-event-definitions-are-authored-judgments",
        "Event definitions (counted units, year rules and event chains) are authored judgments without matched incident data.",
    ),
    (
        "9-numerical-precision-at-default-budgets-is-modest",
        "Numerical standard errors and intervals cover simulation error only; they exclude parameter, elicitation, forecast and model uncertainty.",
    ),
    (
        "10-results-are-research-estimates-not-actuarial-work-product",
        "Results are research estimates, not actuarial work product.",
    ),
)
# The model-wide items that describe a collected study: its respondents'
# judgments and the scenario library's event definitions, which its questions
# ask about. A setup without elicited evidence has neither
# (``applicable_limitations``).
COLLECTED_STUDY_ANCHORS = frozenset(
    {
        "1-the-risk-judgments-come-from-llms-and-cannot-be-backtested-in-general",
        "8-event-definitions-are-authored-judgments",
    }
)
# The model-wide item on the independence of scenarios, which a study of one
# scenario has nothing to apply to (``applicable_limitations``).
SCENARIO_INDEPENDENCE_ANCHOR = "5-scenarios-are-conditionally-independent"
BASE_CASE_ANCHOR = "4-the-main-pipeline-challenges-only-some-of-its-assumptions"
SMALL_EVENT_ANCHOR = "9-numerical-precision-at-default-budgets-is-modest"


def study_scope(package: RunSetup) -> StudyScope:
    """The setup, title, dates, scenarios, named entities and monetary bases
    of the study a result describes, from its setup."""
    scope = package.scope
    bases: list[StudyMonetaryBasis] = []
    for population in scope.populations:
        basis: StudyMonetaryBasis = {
            "currency": population.currency,
            "monetary_basis": population.monetary_basis,
        }
        if basis not in bases:
            bases.append(basis)
    # Dates as the setup writes them.
    dates = scope.model_dump(
        mode="json",
        include={
            "period_start",
            "period_end",
            "forecast_as_of",
            "data_cutoff",
            "availability_cutoff",
            "information_date",
        },
    )
    return {
        "setup_id": package.setup_id,
        "scope_id": scope.scope_id,
        "title": scope.title,
        "period_start": dates["period_start"],
        "period_end": dates["period_end"],
        "forecast_as_of": dates["forecast_as_of"],
        "data_cutoff": dates["data_cutoff"],
        "availability_cutoff": dates["availability_cutoff"],
        "information_date": dates["information_date"],
        "scenario_ids": [population.scenario_id for population in scope.populations],
        "monetary_bases": bases,
        "entities": [
            {
                "entity_id": entity.entity_id,
                "kind": entity.kind,
                "display_name": entity.display_name,
                "revision": entity.revision,
                "status": entity.status,
            }
            for entity in scope.entities
        ],
        "deployments": [
            {"deployment_id": row.deployment_id, "model_id": row.model_id}
            for row in scope.deployments
        ],
    }


def scenario_coverage(
    package: RunSetup, report: NativeResults, positive_state_ids: Collection[str]
) -> list[ScenarioCoverage]:
    """Each setup population's status in the reference arm, with reasons.

    A population is ``unavailable`` when any positive-weight state records it
    as unavailable, ``structural_zero`` when every positive-weight state proves
    it absent, and otherwise ``included``. Unavailable per-scenario views are
    listed even for included populations.
    """
    reference = report["arms"][package.execution.reference_arm]
    metrics = reference.get("metrics", {})
    states = [
        state for state in package.joint_states if state.state_id in positive_state_ids
    ]
    rows: list[ScenarioCoverage] = []
    for population in package.scope.populations:
        scenario = population.scenario_id
        absences = [
            absence
            for state in states
            for absence in state.absent_populations
            if absence.scenario_id == scenario
        ]
        unavailable = [
            absence.reason
            for absence in absences
            if absence.disposition == "unavailable"
        ]
        if unavailable:
            status, reason = "unavailable", "; ".join(dict.fromkeys(unavailable))
        elif states and len(absences) == len(states):
            status = "structural_zero"
            reason = "; ".join(dict.fromkeys(absence.reason for absence in absences))
        else:
            status, reason = "included", None
        rows.append(
            {
                "scenario_id": scenario,
                "status": status,
                "reason": reason,
                "unavailable_views": {
                    view: metric.get("reason") or "Unavailable."
                    for view, metric in metrics.items()
                    if (scoped := parse_scenario_view(view)) is not None
                    and scoped[0] == scenario
                    and metric.get("status") != "available"
                },
            }
        )
    return rows


def result_limitations(
    package: RunSetup, report: NativeResults
) -> list[ResultLimitation]:
    """The setup's limitations and its scenarios' registered coverage gaps, then
    the run's coverage gaps, then model-wide items."""
    limitations: list[ResultLimitation] = [
        {"source": "setup", "text": text} for text in package.limitations
    ]
    selected = {population.scenario_id for population in package.scope.populations}
    for gap in repository_coverage_register().for_scenarios(selected).gaps:
        names = ", ".join(sorted(selected.intersection(gap.scenario_ids)))
        limitations.append(
            {
                "source": "setup",
                "text": f"Registered coverage gap {gap.gap_id} of {names} "
                f"({gap.gap_type}; {gap.materiality} materiality; bias "
                f"{gap.bias_direction}): {gap.rationale} {gap.sensitivity_treatment}",
                "reference": COVERAGE_REGISTER_RELATIVE_PATH.as_posix(),
            }
        )
    coverage = report["coverage"]
    status = coverage["comparisons"]["status"]
    if status == "not_requested":
        limitations.append(
            {
                "source": "run",
                "text": "No sensitivity arm was requested, so this result is a base case only; report it beside at least one executed perturbation of each material assumption.",
                "reference": f"{LIMITATIONS_DOCUMENT}#{BASE_CASE_ANCHOR}",
            }
        )
    elif status == "partial":
        limitations.append(
            {
                "source": "run",
                "text": "Comparison coverage is partial; coverage.comparisons and coverage.sensitivity_families name what is missing.",
            }
        )
    unavailable = [
        row["scenario_id"]
        for row in coverage.get("scenarios", [])
        if row["status"] == "unavailable"
    ]
    if unavailable:
        limitations.append(
            {
                "source": "run",
                "text": f"These populations are unavailable, so every pooled view that would include them is unavailable and no total is reported; per-scenario views of the other populations remain: {', '.join(unavailable)}.",
            }
        )
    small = report.get("small_event_aggregation") or {}
    if (small.get("events") or {}).get("summed"):
        limitations.append(
            {
                "source": "run",
                "text": SMALL_EVENT_LIMITATION,
                "reference": f"{LIMITATIONS_DOCUMENT}#{SMALL_EVENT_ANCHOR}",
            }
        )
    for anchor, text in MODEL_LIMITATIONS:
        limitations.append(
            {
                "source": "model",
                "text": text,
                "reference": f"{LIMITATIONS_DOCUMENT}#{anchor}",
            }
        )
    return limitations


def applicable_limitations(report: Mapping[str, Any]) -> list[ResultLimitation]:
    """The limitations of ``report`` that apply to its study.

    Two facts the result records about its study each leave model-wide items
    out, and every other item stays, in order:

    - the evidence its setup holds
      (``analysis_spec.capture_provenance.current_assessment``): where none of
      it is an elicited judgment the setup has no respondents, and the items
      that describe a collected study (``COLLECTED_STUDY_ANCHORS``) are left
      out;
    - its scenarios (``study.scenario_ids``): a study of one scenario leaves
      out the item on the independence of scenarios
      (``SCENARIO_INDEPENDENCE_ANCHOR``), whatever its evidence.

    A report that does not record one of the two keeps the items that depend
    on it.
    """
    rows: list[ResultLimitation] = list(report.get("limitations") or [])
    left_out: set[str] = set()
    provenance = (report.get("analysis_spec") or {}).get("capture_provenance") or {}
    evidence = (provenance.get("current_assessment") or {}).get("evidence")
    if evidence is not None and not any(
        row.get("origin") == "elicitation" for row in evidence
    ):
        left_out |= COLLECTED_STUDY_ANCHORS
    scenarios = (report.get("study") or {}).get("scenario_ids")
    if scenarios is not None and len(set(scenarios)) == 1:
        left_out.add(SCENARIO_INDEPENDENCE_ANCHOR)
    references = {f"{LIMITATIONS_DOCUMENT}#{anchor}" for anchor in left_out}
    return [
        row
        for row in rows
        if row["source"] != "model" or row.get("reference") not in references
    ]
