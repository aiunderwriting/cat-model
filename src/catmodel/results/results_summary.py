"""A readable Markdown and CSV summary of one results file.

The summary only reads ``results.json`` and the diagnostics file beside it,
which holds the exhibits; it never recomputes a statistic, and a table that
adds reported values up says so. Every number keeps its status: an unavailable
or not-requested value is a dash with its reason, never a zero, and a withheld
value is shown in parentheses for reference only. The 1-in-N losses, the
occurrence statistics, VaR and TVaR are labelled illustrative.
"""

import csv
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catmodel.contracts.incident import counts_lab_loss
from catmodel.output_paths import require_new_evidence_output
from catmodel.results.plausibility import (
    LARGE_INCIDENT_STATISTICS,
    LargeIncidentTable,
    large_incident_table,
)
from catmodel.results.respondent_sampling import NOISE_Z
from catmodel.results.result_disclosures import applicable_limitations
from catmodel.results.results_file import load_results, require_exhibits
from catmodel.results.scenario_contributions import view_reason

# Return periods of the aggregate (AEP) and occurrence (OEP) rows.
RETURN_PERIODS = (10, 100, 200, 250)
# summary.csv: one row per reported value or label
# (docs/setup-run-results.md#readable-summary).
CSV_FIELDS = (
    "section",
    "subject",
    "view",
    "statistic",
    "level",
    "value",
    "unit",
    "label",
    "status",
    "reason",
)
# The units of values that are not amounts of money.
_UNITS = {
    "annual_probability_positive": "probability",
    "probability_at_least_one_large": "probability",
    "stated_large_count": "count",
    "pooled_large_count": "count",
    "modelled_large_count": "count",
    "modelled_very_large_count": "count",
    "share_of_total": "share",
    "gross_eal_share_at_or_above_large": "share",
    "excluded_share": "share",
    "largest_respondent_share": "share",
    "respondents": "count",
    "cases_asked": "count",
    "answers_replaced": "count",
    "cases_pooled": "count",
    "cases_not_pooled": "count",
    "share_of_world_output": "share",
    "relative_change": "ratio",
    "implied_to_direct_ratio": "ratio",
    "sampling_z": "z",
    "tvar_coefficient": "coefficient",
    "years_beyond_var": "count",
    "exhausted_calls": "count",
    "attempts": "count",
    "samples": "count",
    "excluded_samples": "count",
    "lacking_samples": "count",
    "lab_loss_unavailable_cases": "count",
    "lab_loss_left_out_samples": "count",
    "lacking_share": "share",
    "pool_share": "share",
    "scenarios_answered_by_one_model": "count",
    "expected_annual_count": "count",
}
# Statistics whose result is a label rather than a value.
_LABELS = frozenset(
    {
        "control_arm",
        "eal_tail_coherence",
        "indicator",
        "view_coverage",
        "sensitivity_family_coverage",
        "scenario_coverage",
    }
)
# Each coverage table's identifier field, CSV statistic and label in the gaps.
_COVERAGE = {
    "views": ("view", "view_coverage", "View"),
    "sensitivity_families": (
        "family_id",
        "sensitivity_family_coverage",
        "Sensitivity family",
    ),
    "scenarios": ("scenario_id", "scenario_coverage", "Scenario"),
}
# The collection's follow-up calls, as the summary names them.
_FOLLOW_UPS = {
    "severity_retry": "Large-loss coherence retry",
    "chain_guardrail": "Chain guardrail retry",
    "eal_crosscheck": "Leaf EAL cross-check",
}
# The frequency-estimate arms that show the taxonomy parents' pools: the pooled
# leaf counts before any pool, the run's own and every category answer at
# weight one, by the summary.csv statistic each fills.
_POOL_ARMS = {
    "expected_annual_loss_before": "before_hierarchy",
    "expected_annual_loss_after": "final",
    "expected_annual_loss_parents_only": "hierarchy_parents_only",
}
# The pooled annual-loss views that stand in for the headline's two peer
# measures where a plan requests neither, with the measure each gives.
_AGGREGATE_MEASURES = (
    ("Gross component loss", "gross"),
    ("All transfers cancelled", "net_resource_cost"),
    ("Insured payment", "financial"),
)
_DASH = "—"
_COEFFICIENT_BASES = {
    "regression": "regression",
    "infinite_variance": "infinite variance",
    "no_exact_mean": "no exact mean",
}
# Why the 1-in-N losses, the occurrence statistics, VaR and TVaR are labelled
# illustrative (docs/limitations.md, items 3 and 5).
_ILLUSTRATIVE = (
    "they depend on the governed tail shapes, the caps and the conditional "
    "independence of scenarios far more than the expected annual loss does"
)
# Each reporting domain's display name. The codes are those of
# ``catmodel.scenarios.taxonomy_registry``, and each name says what
# contracts/scenarios/hierarchy-contract.yaml puts in the domain.
DOMAIN_NAMES = {
    # System compromise, fraud, AI systems hijacked or tampered with, and
    # theft of or attacks on the model and its service.
    "cyber": "Cyber and fraud",
    # Erroneous automated decisions, erroneous output a person relies on and
    # harm to a person from engaging with the model.
    "eo": "Erroneous outputs and harm to users",
    "cbrn": "CBRN weapons",
    # An agent's actions for its deployer that depart from intent, and AI
    # controlling physical equipment.
    "agent": "Agent and physical-control failures",
    # A model that escapes or subverts its developer's control.
    "ai_rd": "Loss of developer control",
    # An attacker manipulating beliefs or decisions with the model's help.
    "societal": "Influence operations",
    # Outages and other operational failures of the provider's service.
    "svc": "Service failures",
    # Designation, enforcement and private claims over the provider's conduct.
    "regulatory": "Regulatory and legal action",
}


def _estimator(estimator: Mapping[str, Any] | None) -> str:
    """A TVaR's control-variate coefficient and its basis, or a dash."""
    if not estimator:
        return _DASH
    coefficient = _number(estimator.get("coefficient"))
    basis = _COEFFICIENT_BASES.get(str(estimator.get("coefficient_basis")))
    if coefficient is None or basis is None:
        return _DASH
    return f"β = {coefficient:.3g} ({basis})"


def _number(value: object) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


def _money(value: object, status: str = "available") -> str:
    number = _number(value)
    if number is None or status in {"unavailable", "not_requested"}:
        return _DASH
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= scale:
            text = f"{number / scale:,.3g}{suffix}"
            break
    else:
        text = f"{number:,.3g}"
    return f"({text})" if status == "withheld" else text


def _share(value: object) -> str:
    number = _number(value)
    return _DASH if number is None else f"{100 * number:.1f}%"


def _counted(count: int, one: str, many: str) -> str:
    """``count`` followed by the wording that agrees with it in number."""
    return f"{count} {one if count == 1 else many}"


def _at_least(text: str, bound: bool) -> str:
    """A cell's value, marked as a lower bound where the sum it shows leaves
    out some of its group's scenarios; a dash stays a dash."""
    return f"at least {text}" if bound and text != _DASH else text


def _large_incident_share(
    row: Mapping[str, Any] | None,
) -> tuple[float | None, str | None]:
    """A large-loss anchor row's share of expected gross loss from incidents
    at or above the large-loss level, or the reason it has none."""
    if row is None:
        return None, "The scenario has no large-loss anchor."
    if row["status"] == "inapplicable":
        return None, "No scenario has a large-loss anchor."
    if row["status"] != "available":
        return None, "; ".join(row["reasons"])
    share = _number(row["gross_eal_share_at_or_above_large"])
    if share is not None:
        return share, None
    return None, (
        "; ".join(row["gross_eal_share_withheld_reasons"])
        or "The expected gross loss is zero or unavailable."
    )


def _percent(value: object) -> str:
    number = _number(value)
    return _DASH if number is None else f"{100 * number:.3g}%"


def _signed_percent(value: object) -> str:
    number = _number(value)
    return _DASH if number is None else f"{100 * number:+.3g}%"


def _ratio(value: object) -> str:
    number = _number(value)
    return _DASH if number is None else f"{number:.3g}×"


@dataclass(frozen=True)
class _LeafEal:
    """A leaf's three EAL figures and its cross-check cases: the respondent's
    direct judgment, what its count and severity answers imply and the run's,
    with how many cases were asked and how many stayed apart."""

    node_id: str
    direct: float | None
    implied: float | None
    run: float | None
    cases: int
    asked: int
    apart: int


def _quotient(numerator: float | None, denominator: float | None) -> float | None:
    """One figure over another; None where either is unknown or the second
    is zero."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _distance(ratio: float | None) -> float:
    """How far a ratio lies from one, in either direction; an unknown or
    nonpositive ratio sorts last."""
    return abs(math.log(ratio)) if ratio is not None and ratio > 0 else -1.0


def _total(values: Sequence[float | None]) -> float | None:
    """The sum of the values; None where one is unknown."""
    known = [value for value in values if value is not None]
    return math.fsum(known) if len(known) == len(values) else None


def _count(value: object) -> str:
    """An expected annual count, which can be far below one."""
    number = _number(value)
    return _DASH if number is None else f"{number:.3g}"


# The large-loss anchor exhibit's analytic EAL alternatives and their headings.
_ANCHOR_ALTERNATIVES = (
    ("anchor_ignored", "Anchor ignored"),
    ("anchor_lower", "Anchor odds lower"),
    ("anchor_higher", "Anchor odds higher"),
    ("two_anchor", "Two-anchor tail"),
)


def _probability(value: object, status: str = "available") -> str:
    number = _number(value)
    if number is None or status in {"unavailable", "not_requested"}:
        return _DASH
    text = f"{number:.4g}"
    return f"({text})" if status == "withheld" else text


def _interval(interval: object) -> str:
    if not isinstance(interval, Sequence) or len(interval) != 2:
        return _DASH
    return f"{_money(interval[0])} to {_money(interval[1])}"


def _sampling_z(value: object, noise: float) -> str:
    """A change's sampling z, marked within noise below ``noise``."""
    number = _number(value)
    if number is None:
        return _DASH
    return f"{number:.2f}" + (" (within noise)" if abs(number) < noise else "")


def _level(return_period: int) -> str:
    return str(round(1 - 1 / return_period, 6))


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _date(instant: str) -> str:
    """An ISO instant, as its date when it is midnight UTC."""
    return instant.removesuffix("T00:00:00Z")


def _variant_unanswered(unanswered: Mapping[str, Any] | None) -> str:
    """What a variant study's line says of the scenarios a model does not
    answer in it: each pair with what makes it one, and whether the variant
    differs from its matched control in them."""
    if not unanswered:
        return ""

    def reason(pair: Mapping[str, Any]) -> str:
        lacking = f"{pair['lacking_samples']} of {pair['samples']} samples lack it ({_share(pair['lacking_share'])})"
        return {
            "own": f"in the variant alone, where {lacking}",
            "own_and_headline": f"in the variant, where {lacking}, and in the headline",
            "headline": f"following the headline; in the variant {lacking}",
        }[pair["basis"]]

    matched = unanswered["matched_control"]
    differences = []
    if matched["pairs_not_in_headline"]:
        differences.append(
            f"pairs that the headline answers: {matched['pairs_not_in_headline']} of {len(unanswered['pairs'])}"
        )
    if matched["donors_not_headline"]:
        differences.append(
            f"donors other than the headline's, for the pairs of both studies: {matched['donors_not_headline']} of {matched['donors']}"
        )
    return (
        " In the variant study "
        + "; ".join(
            f"`{pair['model_id']}` does not answer `{pair['scenario_id']}` ({reason(pair)})"
            for pair in unanswered["pairs"]
        )
        + ". A variant leaves a model out of every scenario it does not answer in the headline and gives a borrowing sample its headline donor at the same node where it can. "
        + (
            "The variant and its matched control differ in which models answer a scenario or in whose answers a sample borrows ("
            + "; ".join(differences)
            + "), so the variant's effect also carries that difference."
            if matched["differs"]
            else "Here every pair and every donor is the headline's, so the variant and its matched control take each scenario from the same models and each borrowed case from the same sample."
        )
    )


class _Summary:
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = report
        self.reference = report["analysis_spec"]["reference_arm"]
        arm = report["arms"][self.reference]
        self.metrics: Mapping[str, Any] = arm.get("metrics", {})
        self.arm_reason = arm.get("reason")
        # The header requires the study; a section rendered alone does not.
        self.study: Mapping[str, Any] = report.get("study") or {}
        currencies = {row["currency"] for row in self.study.get("monetary_bases", ())}
        self.currency = currencies.pop() if len(currencies) == 1 else None
        self.lines: list[str] = []
        self.rows: list[dict[str, object]] = []

    def unit(self, statistic: str, view: str | None) -> str | None:
        """A row's unit: its statistic's, or the currency of an amount."""
        if statistic in _LABELS:
            return None
        if statistic in _UNITS:
            return _UNITS[statistic]
        basis = (self.metrics.get(view) or {}).get("basis") if view else None
        return (basis or {}).get("currency") or self.currency

    def row(
        self,
        section: str,
        view: str | None,
        statistic: str,
        *,
        subject: str | None = None,
        level: object = None,
        value: object = None,
        label: str | None = None,
        status: str = "available",
        reason: str | None = None,
    ) -> None:
        """One summary.csv row: a value or a label, its status and the reason it
        is withheld, unavailable or qualified."""
        self.rows.append(
            {
                "section": section,
                "subject": subject,
                "view": view,
                "statistic": statistic,
                "level": level,
                "value": _number(value),
                "unit": self.unit(statistic, view),
                "label": label,
                "status": status,
                "reason": reason,
            }
        )

    def statistic(self, view: str, name: str) -> tuple[object, str, str | None]:
        """Value, status and reason of one reported statistic of a view."""
        metric = self.metrics.get(view)
        if metric is None or metric.get("status") != "available":
            status = "not_requested" if metric is None else "unavailable"
            return None, status, view_reason(metric, name)
        if "::" in name:
            functional, level = name.split("::")
            tail = metric.get("tail_metrics", {}).get(level)
            if tail is None:
                return None, "not_requested", f"Level {level} was not reduced."
            value = tail.get(functional)
        else:
            value = metric.get(name)
        status = metric.get("statistics", {}).get(name)
        if status is None:
            available = _number(value) is not None
            return value, "available" if available else "unavailable", None
        return value, status["status"], status["reason"]

    def header(self) -> None:
        """The study the result describes, its status and its identity."""
        study = self.report["study"]
        coverage = self.report.get("coverage", {})
        comparisons = coverage.get("comparisons", {}).get("status", "unknown")
        scenarios = study["scenario_ids"]
        lines = [
            ("Setup", f"`{study['setup_id']}` (scope `{study['scope_id']}`)"),
            (
                "Study period",
                f"{_date(study['period_start'])} to {_date(study['period_end'])} (forecast as of {_date(study['forecast_as_of'])}; data cutoff {_date(study['data_cutoff'])})",
            ),
            (
                f"Scenarios ({len(scenarios)})",
                ", ".join(f"`{scenario}`" for scenario in scenarios)
                if len(scenarios) <= 8
                else "listed under the EAL by scenario",
            ),
        ]
        # Each deployment binding's model; a binding without a deployment
        # entity is named by its identifier alone.
        runs = {row["deployment_id"]: row["model_id"] for row in study["deployments"]}
        for kind, title in (
            ("lab", "Lab"),
            ("model", "Model"),
            ("operator", "Operator"),
            ("deployment", "Deployment"),
        ):
            named = []
            for row in study["entities"]:
                if row["kind"] != kind:
                    continue
                details = [f"`{row['entity_id']}`"]
                if row["revision"]:
                    details.append(f"revision `{row['revision']}`")
                if kind == "deployment" and row["entity_id"] in runs:
                    details.append(f"runs `{runs.pop(row['entity_id'])}`")
                details.append(row["status"])
                named.append(f"{row['display_name']} ({', '.join(details)})")
            if kind == "deployment":
                named += [f"`{name}` (runs `{model}`)" for name, model in runs.items()]
            if named:
                lines.append(
                    (title + ("s" if len(named) > 1 else ""), "; ".join(named))
                )
        lines += [
            (
                "Currency and monetary basis",
                "; ".join(
                    f"{row['currency']} ({row['monetary_basis'].rstrip('.')})"
                    for row in study["monetary_bases"]
                ),
            ),
            ("Reference arm", f"`{self.reference}`"),
            (
                "Result status",
                f"`{self.report['status']}`; comparison coverage `{comparisons}`",
            ),
            ("Analysis", f"`{self.report['analysis_sha256'][:12]}`"),
        ]
        self.lines += [
            f"# Results summary: {study['title']}",
            "",
            *(f"- **{label}:** {text}" for label, text in lines),
            "",
        ]

    def headline(self) -> None:
        """Societal harm and each named lab's economic loss, the headline's two
        peer measures, with their statuses. Where the reference arm reports
        neither, the headline gives instead the pooled annual-loss views it
        does report (``_AGGREGATE_MEASURES``), and the two peer measures as
        not requested if it reports none of those."""
        labs = [row for row in self.report["study"]["entities"] if row["kind"] == "lab"]
        measures = [("Societal harm", "welfare", None)] + (
            [
                (
                    f"Lab economic loss, {row['display_name']}",
                    f"lab_loss::{row['entity_id']}",
                    None,
                )
                for row in labs
            ]
            or [
                (
                    "Lab economic loss",
                    "lab_loss",
                    "No named lab is authored in this study.",
                )
            ]
        )
        if not any(view in self.metrics for _, view, _ in measures):
            measures = [
                (title, view, None)
                for title, view in _AGGREGATE_MEASURES
                if view in self.metrics
            ] or measures
        table = []
        for title, view, absent in measures:
            if absent is not None:
                value, status, reason = None, "not_requested", absent
            else:
                value, status, reason = self.statistic(view, "expected_annual_loss")
            amount = _money(value, status)
            table.append(
                [
                    title,
                    f"`{view}`",
                    amount
                    if amount == _DASH
                    else f"{amount} {self.unit('expected_annual_loss', view)}",
                    status.replace("_", " ")
                    + ("" if status == "available" else f": {reason}"),
                ]
            )
        self.lines += [
            "## Headline",
            "",
            *_table(("Measure", "View", "EAL", "Status"), table),
            "",
            "How to read it: EAL is the expected annual loss. P(L > 0) is the probability of any loss in a year. A 1-in-N loss is the annual loss exceeded with probability 1/N in a single year of the predictive law, whose years each read fresh parameter ranks; it is not the loss exceeded once in N years under a persistent but unknown rate. The EAL parameter band shows how uncertain the EAL itself is because the elicited count laws are uncertain. A dash marks a value that is unavailable or not requested; a value in parentheses is withheld because the simulation cannot support it, and is shown only for reference. The last sections give every reason.",
            "",
        ]

    def scenario_contributions(self) -> None:
        """Each scenario's EAL and share of the pooled EAL and, where the
        setup has a large-loss anchor, the share of its expected gross loss
        from incidents at or above the large-loss level."""
        section = self.report.get("scenario_contributions")
        self.lines += ["## Expected annual loss by scenario", ""]
        if section is None or section.get("status") != "available":
            reason = (
                "The results file has no scenario contributions."
                if section is None
                else section["reason"]
            )
            self.lines += [f"Unavailable: {reason}", ""]
            self.row(
                "scenario",
                None,
                "expected_annual_loss",
                status="unavailable",
                reason=reason,
            )
            return
        measure = section["measure"]
        total = section["expected_annual_loss"]
        anchor = (self.report.get("sensitivity_summary") or {}).get("large_loss_anchor")
        sized = anchor is not None and "by_scenario" in anchor
        unanswered = self._unanswered_notes()
        self.lines += [
            f"Measure `{measure}`."
            + (
                f" The large-incident share is the share of a scenario's expected gross loss from incidents of at least {self._large_loss_level(anchor)}, under the run's laws; it is a share of gross loss whatever the measure."
                if sized
                else ""
            ),
            "",
        ]
        # Decreasing EAL, then the scenarios without one.
        scenarios = sorted(
            section["scenarios"],
            key=lambda row: (
                row["expected_annual_loss"] is None,
                -(row["expected_annual_loss"] or 0.0),
            ),
        )
        header = ["Scenario", "EAL", "Share of EAL"]
        if sized:
            header.append("Large-incident share")
        table = []
        for row in scenarios:
            for statistic in ("expected_annual_loss", "share_of_total"):
                self.row(
                    "scenario",
                    row["view"],
                    statistic,
                    subject=row["scenario_id"],
                    value=row[statistic],
                    status=row["status"],
                    reason=row["reason"],
                )
            notes = (
                []
                if row["status"] == "available"
                else [f"{row['status']}: {row['reason']}"]
            )
            if row["scenario_id"] in unanswered:
                notes.append(unanswered[row["scenario_id"]])
            cells = [
                f"`{row['scenario_id']}`",
                _money(row["expected_annual_loss"], row["status"]),
                _share(row["share_of_total"]),
            ]
            if sized:
                share, reason = _large_incident_share(
                    anchor["by_scenario"].get(row["scenario_id"])
                )
                self._large_incident_row(row["scenario_id"], share, reason)
                cells.append(_share(share))
                if reason:
                    notes.append(f"large-incident share: {reason}")
            table.append([*cells, " ".join(notes)])
        unallocated = section["unallocated"]
        available = unallocated["expected_annual_loss"] is not None
        self.row(
            "scenario",
            measure,
            "expected_annual_loss",
            subject="unallocated",
            value=unallocated["expected_annual_loss"],
            status="available" if available else "unavailable",
            reason=unallocated["reason"],
        )
        table.append(
            [
                "unallocated",
                _money(unallocated["expected_annual_loss"]),
                _share(unallocated.get("share_of_total")),
                *([_DASH] if sized else []),
                unallocated["reason"] or "",
            ]
        )
        cells = [
            "total (pooled view)",
            _money(total) if total is not None else _DASH,
            "100.0%" if total else _DASH,
        ]
        notes = [section["total_reason"]] if section["total_reason"] else []
        if sized:
            share, reason = _large_incident_share(anchor["total"])
            self._large_incident_row(None, share, reason)
            cells.append(_share(share))
            if reason:
                notes.append(f"large-incident share: {reason}")
        table.append([*cells, " ".join(notes)])
        self.lines += _table((*header, "Note"), table)
        self.lines += ["", section["limitations"], ""]
        if anchor is not None and not sized:
            # Withheld for a reweighted result.
            self.lines += [f"Large-incident shares unavailable: {anchor['reason']}", ""]
            self._large_incident_row(None, None, anchor["reason"])

    def scenario_groups(self) -> None:
        """The scenario EALs added up by reporting domain. summary.csv also
        holds the sums on the results' other grouping axes. A setup with no
        scenario of the library has no domains, and no section."""
        section = self.report.get("scenario_groups")
        if section is not None and section.get("status") == "inapplicable":
            return
        self.lines += ["## Expected annual loss by reporting domain", ""]
        if section is None or section.get("status") != "available":
            reason = (
                "The results file has no scenario groups."
                if section is None
                else section["reason"]
            )
            self.lines += [f"Unavailable: {reason}", ""]
            self.row(
                "scenario_group",
                None,
                "expected_annual_loss",
                status="unavailable",
                reason=reason,
            )
            return
        measure = section["measure"]
        for axis in section["axes"]:
            for group in axis["groups"]:
                for statistic in ("expected_annual_loss", "share_of_total"):
                    self.row(
                        "scenario_group",
                        measure,
                        statistic,
                        subject=f"{axis['axis']}:{group['value']}",
                        value=group[statistic],
                        label="lower_bound"
                        if group["status"] == "lower_bound"
                        else None,
                        status="unavailable"
                        if group[statistic] is None
                        else "available",
                        reason=group["reason"],
                    )
        domains = next(
            axis["groups"] for axis in section["axes"] if axis["axis"] == "risk_domain"
        )
        self.lines += [
            f"Measure `{measure}`. A domain's EAL adds up the EALs of its scenarios.",
            "",
            *_table(
                ("Reporting domain", "Scenarios", "EAL", "Share of EAL", "Note"),
                [
                    [
                        f"`{group['value']}`",
                        str(len(group["scenario_ids"])),
                        _at_least(
                            _money(group["expected_annual_loss"]),
                            group["status"] == "lower_bound",
                        ),
                        _at_least(
                            _share(group["share_of_total"]),
                            group["status"] == "lower_bound",
                        ),
                        group["reason"] or "",
                    ]
                    for group in domains
                ],
            ),
            "",
        ]
        if section["unclassified_scenarios"]:
            names = ", ".join(f"`{key}`" for key in section["unclassified_scenarios"])
            self.lines += [
                f"Not in the scenario library, so in no domain: {names}.",
                "",
            ]
        others = ", ".join(
            f"`{axis['axis']}`"
            for axis in section["axes"]
            if axis["axis"] != "risk_domain"
        )
        self.lines += [
            f"{section['limitations']} summary.csv holds the same sums by {others}.",
            "",
        ]

    def domain_tiers(self) -> None:
        """Each reporting domain's EAL beside the expected annual numbers of
        its incidents at or above the large-loss levels under the run's laws.

        Every figure is a sum over the domain's scenarios of values the
        results hold: ``scenario_groups`` gives the EAL, and the
        ``large_loss_anchor`` exhibit each scenario's counts. A count that
        some of the domain's scenarios lack is a lower bound, like the EAL. A
        result without both, as of a setup with no scenario of the library or
        with no large-loss counts, has no such table."""
        section = self.report.get("scenario_groups") or {}
        anchor = (self.report.get("sensitivity_summary") or {}).get(
            "large_loss_anchor"
        ) or {}
        records = anchor.get("by_scenario") or {}
        levels = anchor.get("levels") or {}
        large = self._amount(levels.get("large_loss"))
        if section.get("status") != "available" or not records or large is None:
            return
        # Each severity tier's level and the field of a scenario's record that
        # holds its count at or above the level under the run's laws.
        tiers = [(large, "modelled_large_count")]
        very = self._amount(levels.get("very_large_loss"))
        if very is not None:
            tiers.append((very, "modelled_very_large_count"))
        domains = next(
            axis["groups"] for axis in section["axes"] if axis["axis"] == "risk_domain"
        )
        table = []
        for group in domains:
            code, scenarios = group["value"], group["scenario_ids"]
            cells = [
                f"{DOMAIN_NAMES[code]} (`{code}`)"
                if code in DOMAIN_NAMES
                else f"`{code}`",
                _at_least(
                    _money(group["expected_annual_loss"]),
                    group["status"] == "lower_bound",
                ),
            ]
            gaps: dict[str, list[str]] = {}
            for amount, field in tiers:
                counts = [
                    _number((records.get(key) or {}).get(field)) for key in scenarios
                ]
                known = [count for count in counts if count is not None]
                missing = len(counts) - len(known)
                total = math.fsum(known) if known else None
                bound = total is not None and missing > 0
                reason = (
                    None
                    if not missing
                    else f"The sum leaves out {missing} of the domain's {len(counts)} scenarios, which {'has' if missing == 1 else 'have'} no such count."
                    if bound
                    else "None of the domain's scenarios has such a count."
                )
                cells.append(_at_least(_count(total), bound))
                if reason:
                    gaps.setdefault(reason, []).append(amount)
                self.row(
                    "scenario_group",
                    None,
                    field,
                    subject=f"risk_domain:{code}",
                    value=total,
                    label="lower_bound" if bound else None,
                    status="unavailable" if total is None else "available",
                    reason=reason,
                )
            notes = [f"EAL: {group['reason']}"] if group["reason"] else []
            notes += [
                f"{' and '.join(amounts)}: {reason}" for reason, amounts in gaps.items()
            ]
            table.append([*cells, " ".join(notes)])
        self.lines += [
            "## Reporting domains by severity tier",
            "",
            f"Measure `{section['measure']}`. Each row adds up one reporting domain's scenarios: their EALs, and their expected annual numbers of incidents whose gross loss is at least "
            + " and at least ".join(amount for amount, _ in tiers)
            + " under the run's laws, from the large-loss anchor exhibit."
            + (
                f" The count at {very} rests on the governed tail shape beyond {large}; the respondents' stated count at that level enters no run law."
                if very is not None
                else ""
            )
            + f" The plausibility checks give each scenario's counts at {large}.",
            "",
            *_table(
                (
                    "Reporting domain",
                    "EAL",
                    *(f"Incidents ≥ {amount} a year" for amount, _ in tiers),
                    "Note",
                ),
                table,
            ),
            "",
        ]

    def _amount(self, value: object) -> str | None:
        """An amount of gross loss with its currency, such as a large-loss
        level; None for no amount."""
        amount = _money(value)
        if amount == _DASH:
            return None
        currency = self.unit("expected_annual_loss", "gross")
        return f"{amount} {currency}" if currency else amount

    def _large_loss_level(self, anchor: Mapping[str, Any]) -> str:
        level = self._amount((anchor.get("levels") or {}).get("large_loss"))
        return level or "the large-loss level"

    def _large_incident_row(
        self, scenario: str | None, share: float | None, reason: str | None
    ) -> None:
        self.row(
            "scenario",
            "gross",
            "gross_eal_share_at_or_above_large",
            subject=scenario,
            value=share,
            status="unavailable" if share is None else "available",
            reason=reason,
        )

    def exceedance(self) -> None:
        """The annual aggregate and the largest-occurrence losses at each
        return period, both labelled illustrative (``_ILLUSTRATIVE``)."""
        views = sorted(
            view for view in self.metrics if not view.startswith("scenario::")
        )
        for title, gloss, chosen in (
            (
                "Annual aggregate loss (AEP)",
                f"The total loss in a year, exceeded with probability 1/N in a single predictive year. The 1-in-N losses are illustrative: {_ILLUSTRATIVE}.",
                [v for v in views if not v.startswith("oep_")],
            ),
            (
                "Largest single occurrence in a year (OEP)",
                f"The largest single loss in a year, exceeded with probability 1/N in a single predictive year. Occurrence statistics are illustrative: {_ILLUSTRATIVE}.",
                [v for v in views if v.startswith("oep_")],
            ),
        ):
            self.lines += [f"## {title}, illustrative", "", gloss, ""]
            if not chosen:
                self.lines += ["No view of this kind was requested.", ""]
                continue
            table = []
            for view in chosen:
                eal, eal_status, eal_reason = self.statistic(
                    view, "expected_annual_loss"
                )
                positive, positive_status, positive_reason = self.statistic(
                    view, "annual_probability_positive"
                )
                self.row(
                    "exceedance",
                    view,
                    "expected_annual_loss",
                    value=eal,
                    status=eal_status,
                    reason=eal_reason,
                )
                self.row(
                    "exceedance",
                    view,
                    "annual_probability_positive",
                    value=positive,
                    status=positive_status,
                    reason=positive_reason,
                )
                cells = [
                    f"`{view}`",
                    _money(eal, eal_status),
                    _probability(positive, positive_status),
                ]
                for period in RETURN_PERIODS:
                    value, status, reason = self.statistic(
                        view, f"var::{_level(period)}"
                    )
                    self.row(
                        "exceedance",
                        view,
                        "var",
                        level=_level(period),
                        value=value,
                        status=status,
                        reason=reason,
                    )
                    cells.append(_money(value, status))
                table.append(cells)
            self.lines += _table(
                ("View", "EAL", "P(L > 0)", *(f"1-in-{p}" for p in RETURN_PERIODS)),
                table,
            )
            self.lines.append("")
            if title.startswith("Annual aggregate"):
                self.eal_tail_coherence(chosen)

    def parameter_band(self) -> None:
        """The report-only EAL parameter band beside the headline EAL."""
        band = (self.report.get("sensitivity_summary") or {}).get("eal_parameter_band")
        if not band:
            return
        self.lines += ["## EAL parameter band", ""]
        if band["status"] != "available":
            self.lines += [f"Unavailable: {band['reason']}", ""]
            return
        independent = band["method"]["independent"]
        self.lines += [
            "Report only. Count-law and respondent uncertainty, with the capability forecast integrated: the EAL with each rank group's count law (one scenario at one source status) held at one rank for every year instead of integrated over it, each respondent's trajectory nodes integrated by their weights at those ranks, and the respondents mixed by weight. Its mean is the EAL. The base forecast's rate is integrated; the forecast arms show how the EAL moves under alternative forecast laws. *Independent* ranks are the model's assumption; *comonotone* ranks give every rank group the same rank, which gives the larger mean of the worst 1−p share (TVaR_p) and the smaller mean of the best. A quantile under an intermediate dependence need not lie between the two, and negative dependence between ranks is not covered. The band is not a range of annual losses and is neither a confidence nor a predictive interval; the 1-in-N losses above are single-year predictive statistics, whose ranks are redrawn every year.",
            "",
            f"Independent quantiles are Monte Carlo over ranks alone ({independent['samples_per_epistemic_group']:,} per respondent, seed {independent['seed']}); a value in parentheses has a 95% half-width above {_share(independent['tolerance'])} of it. Comonotone quantiles are exact within their reported error bounds.",
            "",
        ]
        if band.get("incomplete_epistemic_groups"):
            self.lines += [
                "Incomplete respondents: "
                + ", ".join(f"`{name}`" for name in band["incomplete_epistemic_groups"])
                + ". "
                + band["incomplete_epistemic_groups_note"],
                "",
            ]
        for view, row in band["views"].items():
            if "states" in row:
                incomplete = row.get("incomplete_epistemic_groups") or ()
                self.lines += [
                    f"`{view}` is computed at its own weights, from the {row['states']} states that are not left out of the views that count the lab's own loss (Excluded samples)"
                    + (
                        ". Respondents left out at some trajectory node hold fewer of them than the others, and each integrates only its own: "
                        + ", ".join(f"`{name}`" for name in incomplete)
                        if incomplete
                        else ""
                    )
                    + ".",
                    "",
                ]
        table = []
        for view, row in band["views"].items():
            if row["status"] != "available":
                self.row(
                    "eal_parameter_band",
                    view,
                    "eal_band_quantile",
                    status="unavailable",
                    reason=row["reason"],
                )
                table.append([f"`{view}`", _DASH, *(_DASH for _ in range(6))])
                continue
            cells = [f"`{view}`", _money(row["expected_annual_loss"])]
            for case in ("independent", "comonotone"):
                for level, value in row[case]["quantiles"].items():
                    if case == "independent":
                        convergence = row[case]["convergence"][level]
                        status = (
                            "available"
                            if convergence["status"] == "converged"
                            else "withheld"
                        )
                        precision = ("eal_band_half_width", convergence["half_width"])
                    else:
                        status = "available"
                        precision = (
                            "eal_band_error_bound",
                            row[case]["error_bounds"][level],
                        )
                    self.row(
                        "eal_parameter_band",
                        view,
                        "eal_band_quantile",
                        subject=case,
                        level=level,
                        value=value,
                        status=status,
                        reason=None
                        if status == "available"
                        else "Its Monte Carlo 95% half-width is above the band's tolerance.",
                    )
                    self.row(
                        "eal_parameter_band",
                        view,
                        precision[0],
                        subject=case,
                        level=level,
                        value=precision[1],
                    )
                    cells.append(_money(value, status))
            table.append(cells)
        self.lines += _table(
            (
                "View",
                "EAL",
                "Independent p5",
                "p50",
                "p95",
                "Comonotone p5",
                "p50",
                "p95",
            ),
            table,
        )
        self.lines.append("")
        view = next(
            (
                name
                for name in ("welfare", "gross")
                if band["views"].get(name, {}).get("status") == "available"
            ),
            None,
        )
        if view is None:
            return
        groups = sorted(
            band["views"][view]["by_rank_group"].items(),
            key=lambda item: -item[1]["quantiles"]["0.95"],
        )[:10]
        if not groups:
            return
        self.lines += [
            f"Rank groups with the largest p95 in `{view}`, each with one rank, exact within the stated error bounds:",
            "",
            *_table(
                ("Rank group", "Mean", "p5", "p50", "p95"),
                [
                    [
                        f"`{group}`",
                        _money(row["expected_annual_loss"]),
                        *(_money(value) for value in row["quantiles"].values()),
                    ]
                    for group, row in groups
                ],
            ),
            "",
        ]

    def respondent_dispersion(self) -> None:
        """How far the respondents' own EALs lie apart, and how much of each
        pooled EAL its largest respondent carries.

        Each epistemic group's EAL comes from the reference arm's
        ``conditional_eal_by_epistemic_group`` and its weight from the
        analysis's state weights, or for a view that counts the lab's loss
        from the arm's ``lab_loss_state_weights`` where it has them; nothing
        is recomputed.
        A view is shown only where every weighted group has an EAL.
        """
        spec = self.report["analysis_spec"]
        groups = spec.get("epistemic_groups") or {}
        arm = self.report["arms"][self.reference]
        by_view = arm.get("conditional_eal_by_epistemic_group") or {}

        def by_group(state_weights: Mapping[str, float]) -> dict[str, float]:
            found: dict[str, float] = {}
            for state, weight in state_weights.items():
                if weight > 0 and state in groups:
                    found[groups[state]] = found.get(groups[state], 0.0) + weight
            return found

        arm_weights = by_group(spec.get("state_weights") or {})
        lab_loss_weights = (
            by_group(arm["lab_loss_state_weights"])
            if "lab_loss_state_weights" in arm
            else arm_weights
        )
        table = []
        for view, values in by_view.items():
            if not (view in {"welfare", "gross"} or view.startswith("lab_loss::")):
                continue
            weights = lab_loss_weights if counts_lab_loss(view) else arm_weights
            terms = sorted(
                (value, weights[group], group)
                for group, raw in values.items()
                if group in weights and (value := _number(raw)) is not None
            )
            if len(terms) < 2 or len(terms) != len(weights):
                continue
            total = math.fsum(weight for _, weight, _ in terms)
            pooled = math.fsum(value * weight for value, weight, _ in terms)
            # The least EAL at or below which half the respondent weight lies;
            # the tolerance keeps a tie from turning on the sum's last bit.
            reached, median = 0.0, terms[-1][0]
            for value, weight, _ in terms:
                reached += weight
                if reached >= 0.5 * total * (1 - 1e-12):
                    median = value
                    break
            lowest, highest = terms[0][0], terms[-1][0]
            share, largest = max(
                (value * weight / pooled if pooled > 0 else 0.0, group)
                for value, weight, group in terms
            )
            self.row("respondent_dispersion", view, "respondents", value=len(terms))
            for statistic, value in (
                ("lowest_respondent_eal", lowest),
                ("weighted_median_respondent_eal", median),
                ("highest_respondent_eal", highest),
            ):
                self.row("respondent_dispersion", view, statistic, value=value)
            if pooled > 0:
                self.row(
                    "respondent_dispersion",
                    view,
                    "largest_respondent_share",
                    subject=largest,
                    value=share,
                )
            table.append(
                [
                    f"`{view}`",
                    str(len(terms)),
                    _money(pooled / total),
                    _money(lowest),
                    _money(median),
                    _money(highest),
                    _ratio(highest / lowest) if lowest > 0 else _DASH,
                    f"{_share(share)} (`{largest}`)" if pooled > 0 else _DASH,
                ]
            )
        if not table:
            return
        self.lines += [
            "## Respondent dispersion",
            "",
            "Report only. The pooled EAL is the weighted mean of each respondent's own EAL, one model sample with its trajectory nodes or coordinates integrated, and a mean follows its largest terms. This table shows how far the respondents lie apart and how much of the pooled EAL the largest one carries. The weighted median is the least respondent EAL at or below which half the respondent weight lies: a reading of how typical the pooled figure is, not an alternative estimate, and medians do not add across scenarios or views.",
            "",
        ]
        self.lines += _table(
            (
                "View",
                "Respondents",
                "Pooled EAL",
                "Lowest",
                "Weighted median",
                "Highest",
                "Highest / lowest",
                "Largest share of pooled EAL",
            ),
            table,
        )
        self.lines.append("")
        if self._unanswered() is not None:
            self.lines += [
                "A respondent whose model does not answer a scenario carries other models' answers to it, another sample's at each trajectory node (Excluded samples), scaled by its own category answers where the study asks category questions. For that scenario the respondents are therefore not independent, and they tend to lie closer together here than independent answers would.",
                "",
            ]

    def respondent_sampling(self) -> None:
        """Each EAL's respondent-sampling standard error and the trajectory
        quadrature indicator read against it."""
        summary = self.report.get("sensitivity_summary") or {}
        record = summary.get("respondent_sampling")
        if not record:
            return
        self.lines += ["## Respondent sampling", ""]
        if record["status"] != "available":
            self.lines += [f"Unavailable: {record['reason']}", ""]
            return
        self.lines += [
            f"Report only (`NUM-RESPONDENT-SAMPLING`). How far each EAL could move if the respondents were sampled again: the kept samples of each trajectory node (or coordinate) and respondent model, {record['samples']} samples in {record['groups']} groups of at least {record['smallest_group']}, are treated as independent draws, and the standard error follows from each group's weight and the unbiased variance of its samples' EALs. An effect with |z| below {record['noise_z']:g}, its change over that standard error, is within respondent-sampling noise. It excludes the choice and weighting of respondent models, a bias every sample of a model shares, and quadrature, simulation and integration error.",
            "",
        ]
        table = []
        for view, row in record["views"].items():
            available = row["status"] == "available"
            self.row(
                "respondent_sampling",
                view,
                "sampling_standard_error",
                value=row.get("sampling_standard_error"),
                status=row["status"],
                reason=row.get("reason"),
            )
            table.append(
                [
                    f"`{view}`",
                    _money(row.get("expected_annual_loss")),
                    _money(row.get("sampling_standard_error")),
                    _share(row.get("relative_sampling_standard_error")),
                    _DASH if available else row["reason"],
                ]
            )
        self.lines += _table(
            ("View", "EAL", "Sampling SE", "SE / EAL", "Reason"), table
        )
        self.lines.append("")
        quadrature = summary.get("trajectory_quadrature") or {}
        if quadrature.get("status") != "available":
            return
        noise = record["noise_z"]
        self.lines += [
            f"Trajectory quadrature indicator in `{quadrature['view']}`: each arm's EAL under the full rule ({quadrature['interior_nodes']} interior nodes) and the embedded rule ({quadrature['embedded_interior_nodes']}) on the same samples. Where |z| is below {noise:g}, the indicator does not distinguish integration error from respondent-sampling noise.",
            "",
        ]
        table = []
        for arm, row in quadrature["by_arm"].items():
            z = _number(row.get("sampling_z"))
            reading = (
                row["sampling_reason"]
                if row.get("sampling_reason")
                else "within sampling noise"
                if z is not None and abs(z) < noise
                else _DASH
            )
            self.row(
                "trajectory_quadrature",
                quadrature["view"],
                "indicator",
                subject=arm,
                label=row["indicator"],
                reason=row.get("indicator_reason"),
            )
            for statistic in (
                "absolute_difference",
                "sampling_standard_error",
                "sampling_z",
            ):
                available = _number(row.get(statistic)) is not None
                self.row(
                    "trajectory_quadrature",
                    quadrature["view"],
                    statistic,
                    subject=arm,
                    value=row.get(statistic),
                    status="available" if available else "unavailable",
                    reason=None if available else row.get("sampling_reason"),
                )
            table.append(
                [
                    f"`{arm}`",
                    row["indicator"],
                    _money(row["expected_annual_loss"]),
                    _money(row["embedded_rule_expected_annual_loss"]),
                    _money(row["absolute_difference"]),
                    _money(row.get("sampling_standard_error")),
                    _DASH if z is None else f"{z:.2f}",
                    reading,
                ]
            )
        self.lines += _table(
            (
                "Arm",
                "Indicator",
                "Full rule",
                "Embedded rule",
                "Difference",
                "Sampling SE",
                "z",
                "Reading",
            ),
            table,
        )
        self.lines.append("")

    def eal_tail_coherence(self, views: Sequence[str]) -> None:
        """Views whose analytic EAL exceeds the simulated tail beside it."""
        incoherent = [
            (view, coherence)
            for view in views
            if (coherence := self.metrics[view].get("eal_tail_coherence") or {}).get(
                "status"
            )
            == "incoherent"
        ]
        if not incoherent:
            return
        for view, coherence in incoherent:
            self.row(
                "exceedance",
                view,
                "eal_tail_coherence",
                label="incoherent",
                reason=coherence.get("reason"),
            )
        self.lines += _table(
            ("View", "EAL", "Coherence with the simulated tail"),
            [
                [
                    f"`{view}`",
                    _money(self.metrics[view].get("expected_annual_loss")),
                    coherence.get("reason") or _DASH,
                ]
                for view, coherence in incoherent
            ],
        )
        self.lines.append("")

    def tvar(self) -> None:
        """Each view's VaR and TVaR at every reduced level, labelled
        illustrative (``_ILLUSTRATIVE``)."""
        self.lines += [
            "## Tail value at risk, illustrative",
            "",
            f"VaR and TVaR are illustrative: {_ILLUSTRATIVE}.",
            "",
            "TVaR is the mean loss in the worst years beyond the VaR. It is estimated as the average of the worst simulated years plus β times the analytic EAL less the simulated years' mean, over one minus the level (a control variate), and is at least the VaR. β is 0 for a view without an analytic EAL (no exact mean), leaving the sample tail mean, which is biased low for heavy tails; 1 when the law has no finite variance, so only the part below the VaR comes from the years; and otherwise the regression coefficient that minimizes the estimate's variance. Years beyond VaR counts the simulated years with loss above the VaR. Intervals are 95% batch-means intervals for simulation error, widened for a numerically integrated EAL by β times its certified error bound over one minus the level, and a TVaR whose half-width exceeds the tail gate's tolerance is withheld.",
            "",
        ]
        table = []
        for view in sorted(v for v in self.metrics if not v.startswith("scenario::")):
            metric = self.metrics[view]
            if metric.get("status") != "available":
                continue
            for level, tail in metric.get("tail_metrics", {}).items():
                value, status, reason = self.statistic(view, f"tvar::{level}")
                estimator = tail.get("tvar_estimator")
                count = tail.get("exceedance_count")
                table.append(
                    [
                        f"`{view}`",
                        level,
                        _money(
                            tail.get("var"), self.statistic(view, f"var::{level}")[1]
                        ),
                        _interval(tail.get("var_numerical_interval_95")),
                        _money(value, status),
                        _interval(tail.get("tvar_numerical_interval_95")),
                        _estimator(estimator),
                        str(count) if count is not None else _DASH,
                    ]
                )
                self.row(
                    "tvar",
                    view,
                    "tvar",
                    level=level,
                    value=value,
                    status=status,
                    reason=reason,
                )
                coefficient = (estimator or {}).get("coefficient")
                self.row(
                    "tvar",
                    view,
                    "tvar_coefficient",
                    level=level,
                    value=coefficient,
                    status="available"
                    if _number(coefficient) is not None
                    else "unavailable",
                    reason=None if _number(coefficient) is not None else reason,
                )
                self.row(
                    "tvar",
                    view,
                    "years_beyond_var",
                    level=level,
                    value=count,
                    status="available" if count is not None else "unavailable",
                )
        self.lines += (
            _table(
                (
                    "View",
                    "Level",
                    "VaR",
                    "VaR interval",
                    "TVaR",
                    "TVaR interval",
                    "Estimator",
                    "Years beyond VaR",
                ),
                table,
            )
            if table
            else ["No available view reports TVaR."]
        )
        self.lines.append("")

    def sensitivity(self) -> None:
        summary = self.report.get("sensitivity_summary") or {}
        tornado = summary.get("tornado") or {}
        noise = (summary.get("respondent_sampling") or {}).get("noise_z", NOISE_Z)
        self.lines += ["## Sensitivity", ""]
        rows = tornado.get("rows") or []
        if rows:
            self.lines += [
                f"EAL effects in `{tornado.get('view')}`, each against its matched control or the reference:",
                "",
            ]
            self.lines += _table(
                (
                    "Arm",
                    "Compared with",
                    "Control EAL",
                    "Alternative EAL",
                    "Change",
                    "Relative change",
                    "Sampling z",
                ),
                [
                    [
                        f"`{row['arm']}`",
                        f"`{row['control_arm']}`",
                        _money(row["control_eal"]),
                        _money(row["alternative_eal"]),
                        _money(row["absolute_difference"]),
                        _share(row.get("relative_change")),
                        _sampling_z(row.get("sampling_z"), noise),
                    ]
                    for row in rows
                ],
            )
            self.lines += [
                "",
                f"Sampling z is each change over its respondent-sampling standard error (`NUM-RESPONDENT-SAMPLING`); a change with |z| below {noise:g} is within respondent-sampling noise. A dash marks a change whose standard error is zero or unavailable; summary.csv gives each row's reason.",
            ]
            if tornado.get("extrapolation_note"):
                self.lines += ["", tornado["extrapolation_note"]]
            view = tornado.get("view")
            for row in rows:
                self.row(
                    "tornado",
                    view,
                    "control_arm",
                    subject=row["arm"],
                    label=row["control_arm"],
                )
                for statistic, field in (
                    ("control_expected_annual_loss", "control_eal"),
                    ("alternative_expected_annual_loss", "alternative_eal"),
                    ("absolute_difference", "absolute_difference"),
                    ("relative_change", "relative_change"),
                    ("sampling_standard_error", "sampling_standard_error"),
                    ("sampling_z", "sampling_z"),
                ):
                    available = _number(row.get(field)) is not None
                    self.row(
                        "tornado",
                        view,
                        statistic,
                        subject=row["arm"],
                        value=row.get(field),
                        status="available" if available else "unavailable",
                        reason=None
                        if available
                        else row.get("sampling_reason")
                        if field.startswith("sampling")
                        else "Relative change is undefined at a zero control EAL."
                        if field == "relative_change"
                        else None,
                    )
            self.lines.append("")
        else:
            status = (
                self.report.get("coverage", {}).get("comparisons", {}).get("status")
            )
            self.lines += [
                f"No executed sensitivity is ranked (comparison coverage `{status}`).",
                "",
            ]
        bracket = summary.get("dependence_bracket") or {}
        if bracket.get("status") == "available":
            executed = bracket["references"]["executed"]
            comonotone = bracket["references"]["comonotone"]
            self.lines += [
                f"Dependence bracket in `{bracket['view']}`. *Executed* is the simulated coupling of the scenarios. *Comonotone* lines up every scenario's bad years within each simulated state, the worst case for TVaR given each scenario's own distribution; it is not a bound on VaR.",
                "",
            ]
            table = []
            for level, tail in (executed.get("tail") or {}).items():
                upper = (comonotone.get("tail") or {}).get(level, {})
                table.append(
                    [
                        level,
                        _money(tail.get("var")),
                        _money(tail.get("tvar")),
                        _money(upper.get("var")),
                        _money(upper.get("tvar")),
                    ]
                )
                for case, record, values in (
                    ("executed", executed, tail),
                    ("comonotone", comonotone, upper),
                ):
                    for statistic in ("var", "tvar"):
                        available = _number(values.get(statistic)) is not None
                        self.row(
                            "dependence_bracket",
                            bracket["view"],
                            statistic,
                            subject=case,
                            level=level,
                            value=values.get(statistic),
                            status="available" if available else "unavailable",
                            reason=None if available else record.get("reason"),
                        )
            self.lines += _table(
                (
                    "Level",
                    "Executed VaR",
                    "Executed TVaR",
                    "Comonotone VaR",
                    "Comonotone TVaR",
                ),
                table,
            )
            self.lines.append("")
        elif bracket:
            self.lines += [
                f"Dependence bracket unavailable: {bracket.get('reason')}",
                "",
            ]

    def _unanswered(self) -> Mapping[str, Any] | None:
        """The headline study's record of the scenarios a model does not
        answer (``POOL-UNANSWERED-SCENARIOS``), or None without one."""
        record = (self.report.get("sensitivity_summary") or {}).get("excluded_samples")
        return (record or {}).get("unanswered_scenarios")

    def _unanswered_notes(self) -> dict[str, str]:
        """The scenario table's note for each scenario that some model of the
        headline study does not answer."""
        unanswered = self._unanswered()
        if unanswered is None:
            return {}
        unanswering: dict[str, list[str]] = {}
        for pair in unanswered["pairs"]:
            unanswering.setdefault(pair["scenario_id"], []).append(pair["model_id"])
        answering = {
            leaf["scenario_id"]: list(leaf["answering_models"])
            for leaf in unanswered["by_leaf"].values()
        }
        return {
            scenario: "not answered by "
            + ", ".join(f"`{model}`" for model in models)
            + ", whose samples use the answers of "
            + ", ".join(f"`{model}`" for model in answering[scenario])
            + " (Excluded samples)."
            for scenario, models in unanswering.items()
        }

    def _unanswered_scenarios(self, unanswered: Mapping[str, Any]) -> None:
        """The scenarios a model of the headline study does not answer: each
        unanswered pair with the models that answer and their shares of the
        scenario's pool, and how many scenarios one model answers."""
        single = unanswered["leaves_on_one_model"]
        self.lines += [
            f"Unanswered scenarios (`POOL-UNANSWERED-SCENARIOS`). A model whose samples lacking a scenario's direct answer exceed the control's share of its samples, over every node, does not answer that scenario. Its answers to it are set aside at every node, and each of its samples uses instead the whole case of a sample of a model that answers, by a fixed rotation; such a scenario excludes no sample. Scenarios answered by one model: {single} of the {len(unanswered['by_leaf'])} listed.",
            "",
        ]
        table = []
        for pair in unanswered["pairs"]:
            calls = pair["exhausted_direct_calls"]
            pool = unanswered["by_leaf"][pair["node_id"]]["answering_models"]
            table.append(
                [
                    f"`{pair['scenario_id']}`",
                    f"`{pair['model_id']}`",
                    f"{pair['lacking_samples']} of {pair['samples']} ({_share(pair['lacking_share'])})",
                    f"{calls['calls']}: "
                    + ", ".join(
                        f"{count} {cause}"
                        for cause, count in calls["by_cause"].items()
                        if count
                    ),
                    "; ".join(
                        f"`{model}` {_share(row['realized_share'])} ({_share(row['equal_share'])})"
                        for model, row in pool.items()
                    ),
                ]
            )
            subject = f"{pair['scenario_id']}:{pair['model_id']}"
            for statistic in ("lacking_samples", "lacking_share"):
                self.row(
                    "excluded_samples",
                    None,
                    statistic,
                    subject=subject,
                    value=pair[statistic],
                )
        self.lines += _table(
            (
                "Scenario",
                "Model not answering",
                "Samples lacking it",
                "Exhausted direct calls",
                "Answered by: share of the scenario's pool (equal share)",
            ),
            table,
        )
        self.lines += [
            "",
            "How to read it: such a scenario's direct and chain answers are all the answering models'. Where the study asks category questions, each borrowing sample's own category answers, and its answers to the category's other scenarios, still scale the borrowed counts through its category pools, so the scenario's EAL is not free of the other model's judgment. Borrowed answers repeat a donor's, so the samples are not independent. Results therefore report no respondent-sampling standard error and no EAL parameter band for an EAL or a change that weights a borrowing sample, and no standard deviation of the samples' EALs at a trajectory node that holds one. The share in parentheses is one over the number of answering models; the rotation balances the answering samples and not the models, so a model that keeps more samples holds a larger share.",
            "",
        ]
        for leaf in unanswered["by_leaf"].values():
            for model, row in leaf["answering_models"].items():
                self.row(
                    "excluded_samples",
                    None,
                    "pool_share",
                    subject=f"{leaf['scenario_id']}:{model}",
                    value=row["realized_share"],
                )
        self.row(
            "excluded_samples", None, "scenarios_answered_by_one_model", value=single
        )

    def _lab_loss_unavailable(self, record: Mapping[str, Any]) -> None:
        """The direct answers that declare the lab's own loss unavailable,
        the samples left out of the views that count the lab's loss for them
        or the guard that withholds those views, and each excluded sample
        with the leaves it lacks (``POOL-EXCLUDED-SAMPLES``)."""
        declared = record["lab_loss_unavailable"]
        if declared["cases"]:
            parts = ", ".join(
                f"{count} `{part}`"
                for part, count in declared["by_part"].items()
                if count
            )
            self.lines += [
                f"{_counted(declared['cases'], 'direct answer', 'direct answers')} declared the lab's own loss unavailable ({parts}). Such a leaf has every law but the lab's. The answer excludes no sample and changes no weight, so the welfare views, which count no part of the lab's loss, are as if it had been given. The views that count the lab's own loss are computed on the samples that give it: the lab-loss views and the gross views, since gross adds the lab's annual business loss. "
                + (
                    _counted(
                        declared["left_out_samples"],
                        "sample whose states read such an answer, its own or that of a sample whose answers it uses, is",
                        "samples whose states read such an answer, their own or that of a sample whose answers they use, are",
                    )
                    + " left out of those views ("
                    + "; ".join(
                        f"`{model}` {count}"
                        for model, count in declared["left_out_by_model"].items()
                    )
                    + "): there each weighs zero and its model's other samples at the node share its weight. A respondent that declares the lab's loss unavailable may be one that would have judged it larger or less knowable."
                    if declared["disposition"] == "samples_left_out"
                    else f"No sample is left out of those views, because leaving out the samples whose states read such an answer would breach an import guard ({declared['withheld_because']}); their states hold every other law, and the lab-loss views and the gross views are unavailable."
                ),
                "",
            ]
            self.lines += _table(
                (
                    "Leaf",
                    "Cases declaring the lab's loss unavailable",
                    "Compensation share",
                    "Annual business loss",
                ),
                [
                    [
                        f"`{subject}`",
                        str(values["cases"]),
                        str(values["by_part"]["lab_compensation"]),
                        str(values["by_part"]["lab_annual_business_loss"]),
                    ]
                    for subject, values in declared["by_subject"].items()
                ],
            )
            self.lines.append("")
            self.row(
                "excluded_samples",
                None,
                "lab_loss_unavailable_cases",
                value=declared["cases"],
            )
            for subject, values in declared["by_subject"].items():
                self.row(
                    "excluded_samples",
                    None,
                    "lab_loss_unavailable_cases",
                    subject=subject,
                    value=values["cases"],
                )
        if declared["disposition"] == "samples_left_out":
            self.row(
                "excluded_samples",
                None,
                "lab_loss_left_out_samples",
                value=declared["left_out_samples"],
            )
            for model, count in declared["left_out_by_model"].items():
                self.row(
                    "excluded_samples",
                    None,
                    "lab_loss_left_out_samples",
                    subject=model,
                    value=count,
                )
            if declared["left_out"]:
                self.lines += _table(
                    (
                        "Node",
                        "Respondent left out of the lab-loss and gross views",
                        "Model",
                        "Context",
                        "Leaf",
                        "Answer read",
                    ),
                    [
                        [
                            f"`{row['node']}`",
                            f"`{row['respondent'][0]}`",
                            f"`{row['respondent'][1]}`",
                            f"`{read['context_id']}`",
                            f"`{read['node_id']}`",
                            "its own"
                            if read["respondent"] == row["respondent"]
                            else f"`{read['respondent'][0]}`'s",
                        ]
                        for row in declared["left_out"]
                        for read in row["reads"]
                    ],
                )
                self.lines.append("")
        excluded = [
            (row["node"], row["respondent"], leaf)
            for row in record["excluded"]
            for leaf in row["unanswered"]
        ]
        if excluded:
            self.lines += _table(
                ("Node", "Excluded respondent", "Model", "Context", "Unanswered leaf"),
                [
                    [
                        f"`{node}`",
                        f"`{respondent[0]}`",
                        f"`{respondent[1]}`",
                        f"`{leaf['context_id']}`",
                        f"`{leaf['node_id']}`",
                    ]
                    for node, respondent, leaf in excluded
                ],
            )
            self.lines.append("")

    def excluded_samples(self) -> None:
        record = (self.report.get("sensitivity_summary") or {}).get("excluded_samples")
        if not record:
            return
        calls = record["exhausted_calls"]
        causes = ", ".join(
            f"{count} {cause}" for cause, count in calls["by_cause"].items() if count
        )
        models = "; ".join(
            f"`{model}` {row['excluded']} of {row['samples']} ({_share(row['excluded_share'])})"
            for model, row in record["by_model"].items()
        )
        unanswered = self._unanswered()
        self.lines += [
            "## Excluded samples",
            "",
            f"A call is exhausted when all {calls['parse_attempts']} of its validation attempts returned a response that failed validation, a provider's refusal or output-limit stop included. {_counted(calls['calls'], 'call was', 'calls were')} exhausted"
            + (f" (last attempt by cause: {causes})" if causes else "")
            + ". A sample (a respondent at one trajectory node, or at every coordinate of a fixed-anchor setup) whose leaf's direct question was exhausted"
            + (
                ", other than for a scenario its model does not answer (below),"
                if unanswered
                else ""
            )
            + " has no law for that leaf and is excluded"
            + f": {record['excluded_samples']} of {record['samples']} samples ({_share(record['excluded_share'])}; {_share(record['excluded_share_weighted'])} of the reference weight), against the frozen control {_share(record['max_excluded_sample_share'])} for the study and for each model. By model: {models}. At each node a model keeps its equal share, split equally among its remaining samples. Other exhausted calls exclude nothing; the answers they would have revised stand. Excluded samples may differ systematically from the ones kept: refusals are content-linked and may concentrate on high-severity leaves, output-limit stops on the longest reasoning.",
            "",
        ]
        for statistic in ("samples", "excluded_samples", "excluded_share"):
            self.row("excluded_samples", None, statistic, value=record[statistic])
        self._lab_loss_unavailable(record)
        rows = [
            (subject, kind, values)
            for subject, kinds in calls["by_subject"].items()
            for kind, values in kinds.items()
        ]
        if rows:
            self.lines += _table(
                ("Leaf or parent", "Call", "Exhausted calls", "Attempts", "Causes"),
                [
                    [
                        f"`{subject}`",
                        f"`{kind}`",
                        str(values["calls"]),
                        str(values["attempts"]),
                        ", ".join(
                            f"{count} {cause}"
                            for cause, count in values["by_cause"].items()
                        ),
                    ]
                    for subject, kind, values in rows
                ],
            )
            self.lines.append("")
        for subject, kind, values in rows:
            for statistic, field in (
                ("exhausted_calls", "calls"),
                ("attempts", "attempts"),
            ):
                self.row(
                    "excluded_samples",
                    None,
                    statistic,
                    subject=f"{subject}:{kind}",
                    value=values[field],
                )
        renormalized = [
            (node, model, values)
            for node, models in record["renormalized_weights"].items()
            for model, values in models.items()
        ]
        if renormalized:
            self.lines += _table(
                (
                    "Node",
                    "Model",
                    "Samples kept",
                    "Weight per kept sample",
                    "Without exclusion",
                ),
                [
                    [
                        f"`{node}`",
                        f"`{model}`",
                        f"{values['kept']} of {values['samples']}",
                        f"{values['weight_per_kept_sample']:.4g}",
                        f"{values['weight_per_sample']:.4g}",
                    ]
                    for node, model, values in renormalized
                ],
            )
            self.lines.append("")
        if unanswered:
            self._unanswered_scenarios(unanswered)
        for family, label in (
            ("exposure_scale", "Exposure-scale"),
            ("safeguard_level", "Safeguard-level"),
        ):
            for variant_id, variant in (record.get(f"{family}_variants") or {}).items():
                self.lines += [
                    f"{label} variant `{variant_id}`: {variant['excluded_samples']} of {variant['samples']} samples excluded ({_share(variant['excluded_share'])}); {_counted(variant['exhausted_calls']['calls'], 'call', 'calls')} exhausted"
                    + (
                        f" and {_counted(variant['lab_loss_unavailable']['cases'], 'direct answer', 'direct answers')} declared the lab's loss unavailable ("
                        + (
                            f"{_counted(variant['lab_loss_unavailable']['left_out_samples'], 'sample', 'samples')} left out of the lab-loss and gross views"
                            if variant["lab_loss_unavailable"]["disposition"]
                            == "samples_left_out"
                            else "lab-loss and gross views withheld"
                        )
                        + ")"
                        if variant["lab_loss_unavailable"]["cases"]
                        else ""
                    )
                    + ". The variant and its matched control can exclude different samples"
                    + (
                        ", and leave different samples out of the lab-loss and gross views."
                        if variant["lab_loss_unavailable"]["cases"]
                        else "."
                    )
                    + _variant_unanswered(variant.get("unanswered_scenarios")),
                    "",
                ]
                for statistic in ("samples", "excluded_samples", "excluded_share"):
                    self.row(
                        "excluded_samples",
                        None,
                        statistic,
                        subject=f"{family}::{variant_id}",
                        value=variant[statistic],
                    )

    def large_loss_anchor(self) -> None:
        """The analytic alternatives of the large-loss anchor and the counts
        at both levels (``SEV-LARGE-LOSS-ANCHOR``)."""
        exhibit = (self.report.get("sensitivity_summary") or {}).get(
            "large_loss_anchor"
        )
        if not exhibit or (exhibit.get("total") or {}).get("status") == "inapplicable":
            return
        self.lines += ["## Large-loss anchor", ""]
        if exhibit.get("status") == "unavailable":
            self.lines += [f"Unavailable: {exhibit['reason']}", ""]
            return
        total = exhibit["total"]
        if total["status"] != "available":
            self.lines += ["Unavailable: " + "; ".join(total["reasons"]), ""]
            return
        levels = exhibit.get("levels") or {}
        currency = f"{self.currency} " if self.currency else ""
        large = f"{currency}{_money(levels.get('large_loss'))}"
        very = levels.get("very_large_loss")
        self.lines += [
            f"Report only. Each view's EAL, over the scenarios that ask the large-loss counts, under the run's laws and under analytic alternatives of the anchor at {large}: the anchor ignored (the p99-only tail, a design comparison, not a lower bound), every shaping share's conditional odds of reaching it scaled down and up, and the two-anchor tail"
            + (
                f", which also passes each law through its stated count at {currency}{_money(very)} wherever that count is coherent. No reference law reads the count at {currency}{_money(very)}, so the last column shows what the stated count would do to EAL."
                if very is not None
                else "."
            )
            + " EAL effects only: no tail statistic is recomputed.",
            "",
        ]
        alternatives = total["eal_alternatives"]
        self.lines += _table(
            ("View", "Run laws", *(heading for _, heading in _ANCHOR_ALTERNATIVES)),
            [
                [
                    f"`{view}`",
                    _money(row.get("expected_annual_loss")),
                    *(
                        f"{_money(row[key].get('expected_annual_loss'), row[key]['status'])} ({_signed_percent(row[key].get('relative_effect'))})"
                        if row[key]["status"] == "available"
                        else _DASH
                        for key, _ in _ANCHOR_ALTERNATIVES
                    ),
                ]
                for view, row in alternatives.items()
            ],
        )
        for view, row in alternatives.items():
            self.row(
                "large_loss_anchor",
                view,
                "expected_annual_loss",
                subject="run",
                value=row.get("expected_annual_loss"),
                status="available"
                if _number(row.get("expected_annual_loss")) is not None
                else "unavailable",
            )
            for key, _ in _ANCHOR_ALTERNATIVES:
                for statistic, field in (
                    ("expected_annual_loss", "expected_annual_loss"),
                    ("relative_change", "relative_effect"),
                ):
                    self.row(
                        "large_loss_anchor",
                        view,
                        statistic,
                        subject=key,
                        value=row[key].get(field),
                        status=row[key]["status"],
                    )
        counts = [
            ("large_loss::stated", total["stated_large_count"]),
            ("large_loss::pooled", total["pooled_large_count"]),
            ("large_loss::modelled", total["modelled_large_count"]),
            ("large_loss::anchor_ignored", total["extrapolated_large_count"]),
            ("very_large_loss::stated", total["stated_very_large_count"]),
            ("very_large_loss::pooled", total["pooled_very_large_count"]),
            ("very_large_loss::modelled", total["modelled_very_large_count"]),
            (
                "very_large_loss::two_anchor",
                total["two_anchor_tail"]["very_large_count"],
            ),
        ]
        values = {subject: _count(value) for subject, value in counts}
        self.lines += [
            "",
            f"Expected annual incidents of at least {large}: {values['large_loss::stated']} stated, {values['large_loss::pooled']} after pooling, {values['large_loss::modelled']} under the run's laws and {values['large_loss::anchor_ignored']} with the anchor ignored."
            + (
                f" Of at least {currency}{_money(very)}: {values['very_large_loss::stated']} stated, {values['very_large_loss::pooled']} after pooling, {values['very_large_loss::modelled']} under the run's laws and {values['very_large_loss::two_anchor']} under the two-anchor tail."
                if very is not None
                else ""
            ),
            "",
        ]
        roles = total.get("very_large_law_roles") or {}
        if roles:
            self.lines += [
                "How each case's higher count enters the two-anchor tail: "
                + ", ".join(f"{count} `{role}`" for role, count in roles.items())
                + ".",
                "",
            ]
        for subject, value in counts:
            self.row(
                "large_loss_anchor",
                None,
                "expected_annual_count",
                subject=subject,
                value=value,
                status="available" if _number(value) is not None else "unavailable",
            )

    def plausibility(self) -> None:
        """The report-only plausibility checks (``NUM-PLAUSIBILITY``): each
        welfare EAL as a share of world output and, where the results carry
        the large-loss anchor exhibit, the numbers of large incidents the run
        implies."""
        table = (self.report.get("sensitivity_summary") or {}).get("plausibility")
        incidents = self._large_incidents()
        if not table and incidents is None:
            return
        self.lines += ["## Plausibility checks", ""]
        if table:
            self._world_output_shares(table)
        if incidents is not None:
            self._large_incident_counts(incidents)

    def _world_output_shares(self, table: Mapping[str, Any]) -> None:
        """Each scenario's welfare EAL, and the total, as a share of world
        output, flagged above the exhibit's thresholds."""
        output = table["reference"]
        if table["status"] != "available":
            self.lines += [
                f"Shares of world output unavailable: {table['reason']}",
                "",
            ]
            return
        flags = table["flags"]
        self.lines += [
            f"Report only. Welfare EAL as a share of world output, {_money(output['value'])} {output['currency']} in {output['year']} (`{output['status']}`; {output['citation']}). A scenario above {_percent(flags['leaf_share'])} or a total above {_percent(flags['total_share'])} is flagged for review (thresholds `{flags['status']}`); a flag changes no result.",
            "",
        ]
        if table.get("reference_year_note"):
            self.lines += [table["reference_year_note"], ""]
        rows = [("Total", table["total"]), *table["by_scenario"].items()]
        self.lines += _table(
            ("Scenario", "Welfare EAL", "Share of world output", "Flag"),
            [
                [
                    name if name == "Total" else f"`{name}`",
                    _money(row.get("expected_annual_loss"), row["status"]),
                    _percent(row.get("share_of_world_output")),
                    "flagged" if row.get("flagged") else "",
                ]
                for name, row in rows
            ],
        )
        for name, row in rows:
            self.row(
                "plausibility",
                "welfare",
                "share_of_world_output",
                subject="total" if name == "Total" else name,
                value=row.get("share_of_world_output"),
                label="flagged" if row.get("flagged") else None,
                status=row["status"],
                reason=row.get("reason"),
            )
        self.lines.append("")

    def _large_incidents(self) -> LargeIncidentTable | None:
        """The plausibility checks' table of large incidents, read from the
        large-loss anchor exhibit; None for results without the exhibit,
        which only a contextual setup carries."""
        anchor = (self.report.get("sensitivity_summary") or {}).get("large_loss_anchor")
        return None if anchor is None else large_incident_table(anchor)

    def _large_incident_counts(self, table: LargeIncidentTable) -> None:
        """The expected annual numbers of incidents at or above the large-loss
        level that the respondents stated and the run implies, in total and
        by scenario, with the review flags. A value the exhibit does not
        report is a dash whose reason ``gaps`` lists."""
        anchor = self.report["sensitivity_summary"]["large_loss_anchor"]
        large = self._large_loss_level(anchor)
        flags = table.flags
        self.lines += [
            f"Report only. Expected annual incidents of at least {large} in each scenario that asks that count, and in total: as the respondents stated them, after pooling and under the run's laws, with the probability of at least one such incident in the year under the run's laws. A scenario at or above {flags.leaf_annual_count:g} a year or a total at or above {flags.total_annual_count:g} a year under the run's laws is flagged for review (thresholds `{flags.status}`).",
            "",
        ]
        rows = [
            ("Total", table.total),
            *((f"`{row.scenario_id}`", row) for row in table.scenarios),
        ]
        self.lines += _table(
            (
                "Scenario",
                "Stated",
                "After pooling",
                "Run laws",
                "P(at least one)",
                "Flag",
            ),
            [
                [
                    name,
                    _count(row.values["stated_large_count"]),
                    _count(row.values["pooled_large_count"]),
                    _count(row.values["modelled_large_count"]),
                    _probability(row.values["probability_at_least_one_large"]),
                    "flagged" if row.flagged else "",
                ]
                for name, row in rows
            ],
        )
        self.lines += [
            "",
            "How to read it: a count is the expected number of separate admitted incidents of at least that gross loss for the study's lab in the study year, each one root cause as its scenario defines it. The three counts can differ: the count after pooling is the stated count after the leaf EAL cross-check, scaled with the pool of the direct and chain counts, and it sets a run law's tail only where the level lies above the law's p99; elsewhere the stated loss quantiles set what the law gives at that level. The count under the run's laws also carries the factor of the taxonomy parents' pools, which scales every count of a leaf alike. Set the counts under the run's laws against how many incidents of that size the public record attributes to any one developer's models in a year. A flag prompts review of the judgments behind a count and changes no result.",
            "",
        ]
        for _, row in rows:
            for statistic in LARGE_INCIDENT_STATISTICS:
                value = row.values[statistic]
                self.row(
                    "plausibility",
                    None,
                    statistic,
                    subject=row.scenario_id or "total",
                    value=value,
                    label="flagged"
                    if row.flagged and statistic == "modelled_large_count"
                    else None,
                    status="unavailable" if value is None else "available",
                    reason=row.reason(statistic),
                )

    def _large_incident_gaps(self) -> list[tuple[str, str, str]]:
        """The kind, reason and name of each value the large-incident table
        shows as a dash: a row without any value by its scenario, a single
        missing value by its scenario and statistic."""
        table = self._large_incidents()
        if table is None:
            return []
        found: list[tuple[str, str, str]] = []
        for row in (table.total, *table.scenarios):
            subject = row.scenario_id or "total"
            if row.unavailable is not None:
                found.append(("Large-incident counts", row.unavailable, subject))
            found += [
                ("Large-incident statistic", reason, f"{subject} {statistic}")
                for statistic, reason in row.missing.items()
            ]
        return found

    def eal_crosscheck(self) -> None:
        """The leaf EAL cross-check: its status counts, then each leaf's
        direct EAL judgment beside the EAL its own count and severity answers
        imply and the leaf's EAL in the run. The cases are rows of
        ``summary.csv`` only."""
        frequency = (self.report.get("sensitivity_summary") or {}).get(
            "frequency_estimate"
        )
        if not frequency or "eal_crosscheck" not in frequency:
            return
        check = frequency["eal_crosscheck"]
        statuses = check["cases_by_status"]
        view = "welfare"
        run = (frequency.get("views") or {}).get(view, {}).get("by_scenario") or {}
        self.lines += [
            "## Leaf EAL cross-check",
            "",
            f"Each leaf's direct EAL mean is compared with the EAL implied by its count and severity means. A gap above {check['crosscheck_ratio']:g} times (`eal_gap`) triggers one cross-check call. Cases by trigger: "
            + ", ".join(
                f"{count} {trigger}"
                for trigger, count in check["cases_by_trigger"].items()
            )
            + ". Cases: "
            + ", ".join(f"{count} {status}" for status, count in statuses.items())
            + f". {check['eal_crosscheck_unresolved']} declined or remain apart after revision; {check['eal_crosscheck_exhausted']} cross-check calls exhausted their validation attempts.",
            "",
            f"The table sets three figures for each leaf side by side, in `{view}`. Direct EAL is the respondent's direct judgment and implied EAL is what its own count and severity answers give; both are means after any revision, weighted over the leaf's cases as the run weights its states. Run EAL is the leaf's expected annual loss in the run, whose count pools the direct count with the chain count and carries the factor of the taxonomy parents' pools. The run never simulates the direct EAL, and no figure here says which is right. Leaves are ordered by how far the direct EAL lies from the run's. Each case that was asked is a row of `summary.csv`.",
            "",
        ]
        leaves = [
            _LeafEal(
                node_id=node_id,
                direct=_number(leaf.get("weighted_direct_eal")),
                implied=_number(leaf.get("weighted_implied_eal")),
                run=_number(
                    (run.get(leaf.get("scenario_id", node_id)) or {}).get("final")
                ),
                cases=len(leaf["cases"]),
                asked=sum(bool(row["triggers"]) for row in leaf["cases"]),
                apart=sum(
                    bool(row["eal_crosscheck_unresolved"]) for row in leaf["cases"]
                ),
            )
            for node_id, leaf in check["by_leaf"].items()
        ]
        leaves.sort(
            key=lambda leaf: (
                -_distance(_quotient(leaf.direct, leaf.run)),
                leaf.node_id,
            )
        )
        total = _LeafEal(
            node_id="**All leaves**",
            direct=_total([leaf.direct for leaf in leaves]),
            implied=_total([leaf.implied for leaf in leaves]),
            run=_total([leaf.run for leaf in leaves]),
            cases=sum(leaf.cases for leaf in leaves),
            asked=sum(leaf.asked for leaf in leaves),
            apart=sum(leaf.apart for leaf in leaves),
        )
        self.lines += _table(
            (
                "Leaf",
                "Direct EAL",
                "Implied EAL",
                "Run EAL",
                "Direct ÷ implied",
                "Direct ÷ run",
                "Cases asked",
                "Still apart",
            ),
            [
                [
                    name,
                    _money(leaf.direct),
                    _money(leaf.implied),
                    _money(leaf.run),
                    _ratio(_quotient(leaf.direct, leaf.implied)),
                    _ratio(_quotient(leaf.direct, leaf.run)),
                    f"{leaf.asked} of {leaf.cases}",
                    str(leaf.apart),
                ]
                for name, leaf in (
                    *((f"`{leaf.node_id}`", leaf) for leaf in leaves),
                    (total.node_id, total),
                )
            ],
        )
        self.lines.append("")
        for leaf in leaves:
            for statistic, value in (
                ("weighted_direct_eal", leaf.direct),
                ("weighted_implied_eal", leaf.implied),
            ):
                self.row(
                    "eal_crosscheck",
                    view,
                    statistic,
                    subject=leaf.node_id,
                    value=value,
                    status="available" if value is not None else "unavailable",
                )
        for node_id, leaf in check["by_leaf"].items():
            for row in leaf["cases"]:
                if row["status"] == "agreed":
                    continue
                ratio = _number(row["implied_to_direct_after"])
                self.row(
                    "eal_crosscheck",
                    view,
                    "implied_to_direct_ratio",
                    subject=f"{node_id}:{row['anchor_score']:g}:{row['respondent'][0]}",
                    value=ratio,
                    label=row["status"],
                    status="available" if ratio is not None else "unavailable",
                )

    def follow_up_effects(self) -> None:
        """What each follow-up call of the collection changed: the cases it
        was asked of, the answers it replaced and the EAL with those answers
        put back, from the ``frequency_estimate`` exhibit."""
        frequency = (self.report.get("sensitivity_summary") or {}).get(
            "frequency_estimate"
        )
        if not frequency or "follow_ups" not in frequency:
            return
        view = "welfare"
        effects = (frequency.get("views") or {}).get(view, {}).get(
            "follow_up_effects"
        ) or {}
        self.lines += [
            "## Effect of follow-up calls",
            "",
            f"Each row puts back the answers one kind of follow-up call replaced and keeps every other stored answer. Later calls were asked about the replaced answers, so a row is a comparison within this capture, not the result of a collection without that follow-up. EAL in `{view}`, integrated over parameter ranks and before any taxonomy parent's pool. No tail statistic is recomputed.",
            "",
        ]
        table = []
        # In the order collection asks them; a results file sorts its keys.
        for kind, label in _FOLLOW_UPS.items():
            counts = frequency["follow_ups"][kind]
            effect = effects.get(kind) or {
                "status": "unavailable",
                "reason": f"The exhibit has no `{view}` view.",
            }
            available = effect["status"] == "available"
            before = effect.get("expected_annual_loss_before")
            after = effect.get("expected_annual_loss_after")
            change = effect.get("relative_change")
            table.append(
                [
                    label,
                    f"{counts['cases_asked']:,}",
                    f"{counts['answers_replaced']:,}",
                    _money(before),
                    _money(after),
                    _share(change),
                ]
            )
            for statistic in ("cases_asked", "answers_replaced"):
                self.row(
                    "follow_up_effects",
                    None,
                    statistic,
                    subject=kind,
                    value=counts[statistic],
                )
            for statistic, value in (
                ("expected_annual_loss_before", before),
                ("expected_annual_loss_after", after),
                ("relative_change", change),
            ):
                self.row(
                    "follow_up_effects",
                    view,
                    statistic,
                    subject=kind,
                    value=value,
                    status="available"
                    if available and _number(value) is not None
                    else "unavailable",
                    reason=effect.get("reason")
                    if not available
                    else None
                    if _number(value) is not None
                    else "The EAL before the follow-up is zero.",
                )
        self.lines += _table(
            (
                "Follow-up call",
                "Cases asked",
                "Answers replaced",
                "EAL before",
                "EAL after",
                "Change",
            ),
            table,
        )
        self.lines += [
            "",
            "A dash marks a follow-up that replaced no answer or whose EAL is unavailable; summary.csv gives each row's reason.",
            "",
        ]

    def hierarchy_pool(self) -> None:
        """What the taxonomy parents' pools changed: the parent cases pooled
        and not pooled, and the EAL before the pools, after them and with
        every category answer at weight one, from the ``frequency_estimate``
        exhibit. An instrument that asks no parents has no such section."""
        frequency = (self.report.get("sensitivity_summary") or {}).get(
            "frequency_estimate"
        ) or {}
        hierarchy = frequency.get("hierarchy")
        if not hierarchy:
            return
        view = "welfare"
        arms = (frequency.get("views") or {}).get(view, {}).get("by_arm") or {}
        pooled = sum(row["cases_pooled"] for row in hierarchy.values())
        not_pooled = sum(row["cases_not_pooled"] for row in hierarchy.values())
        eal: dict[str, float | None] = {}
        reasons: dict[str, str | None] = {}
        for statistic, name in _POOL_ARMS.items():
            arm = arms.get(name) or {
                "status": "unavailable",
                "reason": f"The exhibit has no `{view}` view.",
            }
            available = arm["status"] == "available"
            eal[statistic] = (
                _number(arm.get("expected_annual_loss")) if available else None
            )
            reasons[statistic] = None if available else arm.get("reason")
        before = eal["expected_annual_loss_before"]
        after = eal["expected_annual_loss_after"]
        ratio = _quotient(after, before)
        change = None if ratio is None else ratio - 1
        self.lines += [
            "## Effect of the taxonomy parents' pools",
            "",
            f"Each taxonomy parent's own answer is pooled with what is beneath it by a fixed rule, with no further call: the answer weighs one over the parent's height plus one, and the pool's factor scales the count of every leaf beneath the parent. A parent case is one parent at one coordinate for one kept respondent. Of {pooled + not_pooled:,} parent cases, {pooled:,} were pooled and {not_pooled:,} were not; a parent that is not pooled leaves what is beneath it unchanged. EAL in `{view}`, integrated over parameter ranks. No tail statistic is recomputed.",
            "",
        ]
        self.lines += _table(
            (
                "Parent cases pooled",
                "Parent cases not pooled",
                "EAL before the pools",
                "EAL after the pools",
                "Change",
            ),
            [
                [
                    f"{pooled:,}",
                    f"{not_pooled:,}",
                    _money(before),
                    _money(after),
                    _share(change),
                ]
            ],
        )
        self.lines += [
            "",
            f"With every category's own answer standing for its category, weight one at every parent, the EAL is {_money(eal['expected_annual_loss_parents_only'])}. Before the pools the category answers weigh zero, so these are the two ends of their weight; across levels the run's EAL need not lie between them. A dash marks an EAL that is unavailable; summary.csv gives its reason.",
            "",
        ]
        for statistic, count in (
            ("cases_pooled", pooled),
            ("cases_not_pooled", not_pooled),
        ):
            self.row("hierarchy_pool", None, statistic, value=count)
        for statistic, value in eal.items():
            self.row(
                "hierarchy_pool",
                view,
                statistic,
                value=value,
                status="available" if value is not None else "unavailable",
                reason=reasons[statistic],
            )
        self.row(
            "hierarchy_pool",
            view,
            "relative_change",
            value=change,
            status="available" if change is not None else "unavailable",
            reason=None
            if change is not None
            else reasons["expected_annual_loss_before"]
            or reasons["expected_annual_loss_after"]
            or "The EAL before the pools is zero.",
        )

    def gaps(self) -> None:
        self.lines += ["## Unavailable, withheld and not requested", ""]
        grouped: dict[tuple[str, str, str], list[str]] = {}
        if self.arm_reason:
            grouped.setdefault(
                ("Reference arm", "unavailable", self.arm_reason), []
            ).append(self.reference)
        for view, metric in sorted(self.metrics.items()):
            if metric.get("status") != "available":
                grouped.setdefault(
                    ("View", "unavailable", view_reason(metric, "")), []
                ).append(view)
                continue
            if view.startswith("scenario::"):
                # Scenario EAL gaps are in the scenario table.
                continue
            for name, status in sorted(metric.get("statistics", {}).items()):
                if status["status"] != "available":
                    grouped.setdefault(
                        ("Statistic", status["status"], status["reason"]), []
                    ).append(f"{view} {name}")
        coverage = self.report.get("coverage", {})
        for key, (field, statistic, kind) in _COVERAGE.items():
            for row in coverage.get(key, []):
                if row["status"] in {"available", "complete", "included"}:
                    continue
                grouped.setdefault(
                    (kind, row["status"], row.get("reason") or ""), []
                ).append(row[field])
                self.row(
                    "coverage",
                    None,
                    statistic,
                    subject=row[field],
                    label=row["status"],
                    reason=row.get("reason"),
                )
        for kind, reason, name in self._large_incident_gaps():
            grouped.setdefault((kind, "unavailable", reason), []).append(name)
        items = [
            f"{kind} {status.replace('_', ' ')} ({len(names)}): {', '.join(f'`{name}`' for name in names)} — {reason}"
            for (kind, status, reason), names in grouped.items()
        ]
        self.lines += [f"- {item}" for item in items] or [
            "Nothing is unavailable, withheld or not requested."
        ]
        self.lines.append("")

    def limitations(self) -> None:
        """The result's limitations that apply to its study
        (``applicable_limitations``)."""
        self.lines += ["## Limitations", ""]
        for index, row in enumerate(applicable_limitations(self.report), start=1):
            reference = f" See `{row['reference']}`." if row.get("reference") else ""
            self.lines.append(f"{index}. {row['text']}{reference} _({row['source']})_")
        if not self.report.get("limitations"):
            self.lines.append("The results file lists no limitations.")
        self.lines.append("")


def results_summary(report: Mapping[str, Any]) -> tuple[str, list[dict[str, object]]]:
    """Return the Markdown summary and its long-format CSV rows.

    ``report`` is a whole report, with its exhibits; a report read from
    ``results.json`` alone is refused, because its summary would leave the
    exhibits out without saying so."""
    require_exhibits(report)
    summary = _Summary(report)
    summary.header()
    summary.headline()
    summary.scenario_contributions()
    summary.scenario_groups()
    summary.domain_tiers()
    summary.exceedance()
    summary.parameter_band()
    summary.respondent_dispersion()
    summary.respondent_sampling()
    summary.tvar()
    summary.sensitivity()
    summary.large_loss_anchor()
    summary.eal_crosscheck()
    summary.follow_up_effects()
    summary.hierarchy_pool()
    summary.excluded_samples()
    summary.plausibility()
    summary.gaps()
    summary.limitations()
    return "\n".join(summary.lines).rstrip() + "\n", summary.rows


def write_results_summary(*, results_path: Path, output: Path) -> Path:
    """Write ``summary.md`` and ``summary.csv`` for one results file, which
    must be a current results.json whose content matches its analysis
    identity, with its diagnostics file beside it (``load_results``)."""
    require_new_evidence_output(output)
    markdown, rows = results_summary(load_results(results_path, diagnostics=True))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output.mkdir(parents=True)
    (output / "summary.md").write_text(markdown, encoding="utf-8")
    (output / "summary.csv").write_text(buffer.getvalue(), encoding="utf-8")
    return output / "summary.md"
