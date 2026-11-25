"""Threat models: the research each scenario's questions can show.

A threat model is written before any quantity is asked. A research stage
reads sources for one scenario and says how its incidents arise: who causes
them, the steps to an admitted incident and the steps that make one larger,
what the AI system changes at each step, what limits it, what defenders do
there, how outcomes differ in severity, which published analogues exist and
what remains unknown. It states no probability or loss estimate of its own,
so the respondents who read it supply every number, and every figure it does
state is one that a source it cites reports.

``research_brief`` is the task given to the research stage, ``ThreatModel``
the artifact it returns, and ``ThreatModelsFile`` the register of threat
models that a source bundle captures. A threat model reaches a question only
from that register. It records the digest of the scenario definition it was
written against, and library validation and bundle capture refuse one whose
scenario has changed since. ``coverage_against_reference`` is the arithmetic
of a blind comparison with a published expert threat model.
"""

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from catmodel.dateutil import parse_datetime_utc
from catmodel.hashing import canonical_json_sha256
from catmodel.scenarios.base_rates import (
    CARD_ID_PATTERN,
    DatedEvidenceText,
    Source,
    quotation_structure_error,
)
from catmodel.scenarios.loader import (
    ScenarioDefinition,
    effective_loss_population_contract,
    qualified_terminal_unit,
    rendered_severity_occurrence_scope,
)
from catmodel.yaml_io import safe_load

THREAT_MODELS_RELATIVE_PATH = Path("data/threat_models.yaml")
"""The repository's register of threat models, relative to the project root.
Like the source cards, it reaches questions only through a captured source
bundle."""

THREAT_MODEL_CLOSE = "</threat_model>"
"""The line that closes a quoted threat model."""

MAX_RENDERED_WORDS = 3000
"""The most words a threat model may take in a question. A longer one would
crowd out the scenario's definition and the excerpts beside it."""

_URL = re.compile(r"https?://", re.IGNORECASE)

_STATED_ODDS = re.compile(
    r"\bone\s+(?:in|out\s+of)\s+(?:two|three|four|five|six|seven|eight|nine|ten"
    r"|twenty|fifty|a\s+hundred|a\s+thousand|a\s+million)\b",
    re.IGNORECASE,
)
"""Odds stated in words, which the figure rule cannot see."""

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December)"
)
_NOT_FIGURES = re.compile(
    # A date's day, beside its month.
    rf"\b\d{{1,2}}\s+{_MONTH}\b|\b{_MONTH}\s+\d{{1,2}}\b"
    # A numbered provision or document, and a list marker.
    r"|\b(?:Article|Section|Annex|Chapter|Title|Rule|Part|Recital|Paragraph|No\.)"
    r"\s+\d+(?:\.\d+)*|\(\d+\)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.\-])\d[\d,]*(?:\.\d+)?(?![\w\-])")
_YEAR = re.compile(r"(?:19|20)\d{2}")


def figures(text: str) -> frozenset[str]:
    """The figures ``text`` states, each in one written form: ``1,200`` and
    ``1200.0`` are the same figure.

    A calendar year, a day beside its month, the number of an article or
    section, a list marker and a number inside an identifier date or name
    something and measure nothing, so they are not figures.
    """
    found = set()
    for match in _NUMBER.finditer(_NOT_FIGURES.sub(" ", text)):
        written = match.group(0).rstrip(",")
        if _YEAR.fullmatch(written):
            continue
        whole, _, fraction = written.replace(",", "").partition(".")
        fraction = fraction.rstrip("0")
        found.add(f"{int(whole)}.{fraction}" if fraction else str(int(whole)))
    return frozenset(found)


def _quoted(value: str) -> str:
    if not value.strip():
        raise ValueError("threat-model text must be nonblank")
    if (error := quotation_structure_error(value)) is not None:
        raise ValueError(
            f"threat-model text holds {error}, which could break out of its "
            "quotation in a question"
        )
    if _URL.search(value):
        raise ValueError("threat-model text holds a URL, which belongs in sources")
    return value


def _analysis(value: str) -> str:
    value = _quoted(value)
    if match := _STATED_ODDS.search(" ".join(value.split())):
        raise ValueError(
            f"threat-model text states odds ({match.group(0).strip()!r}); the "
            "analysis states none of its own"
        )
    return value


def _unnumbered(value: str) -> str:
    value = _analysis(value)
    if stated := sorted(figures(value)):
        raise ValueError(
            f"the summary and the key uncertainties cite no source, so they "
            f"state no figure; found {stated}"
        )
    return value


type Reported = Annotated[str, AfterValidator(_quoted)]
"""What a source reports."""

type Analysis = Annotated[str, AfterValidator(_analysis)]
"""Text of the analysis, which states no odds of its own."""

type Unnumbered = Annotated[str, AfterValidator(_unnumbered)]
"""Analysis that cites no source and so states no figure."""

type ItemId = Annotated[str, Field(pattern=CARD_ID_PATTERN)]


class _Item(BaseModel):
    """One cited part of a threat model.

    ``sources`` lists the IDs of the threat model's sources that support it.
    A part that cites none is the analysis's own judgment and is rendered as
    one. Every figure a part states is in the finding of a source it cites
    (``ThreatModel`` checks this), so a part without sources states none.
    """

    model_config = ConfigDict(extra="forbid")
    sources: tuple[ItemId, ...] = ()

    def texts(self) -> tuple[str, ...]:
        """The part's prose, without its own ID."""
        return tuple(
            value
            for name, value in self
            if isinstance(value, str) and not name.endswith("_id") and name != "kind"
        )

    def label(self) -> str:
        return next(value for name, value in self if name.endswith("_id"))

    @model_validator(mode="after")
    def distinct_sources(self) -> Self:
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("a threat-model part cites a source twice")
        return self

    def cited(self) -> str:
        return (
            "Sources: " + ", ".join(self.sources)
            if self.sources
            else "Sources: none; a judgment of the analysis."
        )


class Actor(_Item):
    """Who causes or takes part in an incident of the scenario: an attacker,
    an operator, a user, a claimant or a regulator."""

    actor_id: ItemId
    description: Analysis


class PathwayStep(_Item):
    """One step of the pathway.

    A ``necessary`` step is one without which no incident is counted; the
    first is the starting condition. An ``escalation`` step is not needed
    for a counted incident and makes one larger, such as reaching a system
    whose failure spreads the loss.
    """

    step_id: ItemId
    kind: Literal["necessary", "escalation"] = "necessary"
    description: Analysis
    """What has to happen at this step."""
    ai_contribution: Analysis
    """What the AI system contributes or changes here, and the capability
    that takes."""
    bottleneck: Analysis
    """What most limits the step today."""
    defender_response: Analysis
    """What defenders, operators or authorities do at this step."""


class SeverityMode(_Item):
    """One distinct way an incident ends, with what drives its loss."""

    mode_id: ItemId
    description: Analysis
    loss_drivers: Analysis


class Analogue(_Item):
    """A published incident or dataset of the closest kind."""

    analogue_id: ItemId
    description: Reported
    outcome: Reported
    """What the sources report happened: losses, casualties or counts, with
    their units and dates."""
    sources: tuple[ItemId, ...] = Field(min_length=1)


class LiabilityPathway(_Item):
    """How loss reaches the lab: the lab-loss view's counterpart of the pathway."""

    def label(self) -> str:
        return "liability"

    theories: Analysis
    """Legal theories and regulatory powers under which the lab would bear loss."""
    defences: Analysis
    contract_terms: Analysis
    """Terms that allocate loss between the lab and its customers."""


class ThreatModelSource(BaseModel):
    """One source and what the research stage read in it."""

    model_config = ConfigDict(extra="forbid")
    source_id: ItemId
    citation: Source
    published_utc: date
    """When the source was published or last revised, as it states."""
    accessed_utc: date
    """When the research stage opened it."""
    finding: Reported
    """What the source states that the threat model relies on, with every
    figure a part takes from it. A part is checked against the findings of
    the sources it cites; questions do not quote them."""

    @model_validator(mode="after")
    def read_after_publication(self) -> Self:
        if self.accessed_utc < self.published_utc:
            raise ValueError(
                f"source {self.source_id!r} was read before it was published"
            )
        return self


class Drafting(BaseModel):
    """A draft's working notes: how the research was done."""

    model_config = ConfigDict(extra="forbid")
    drafted_by: str = Field(min_length=1)
    """The research stage, for example an LLM agent with web retrieval, how
    it read its sources (in full, or through a tool that summarises pages)
    and whoever edited its draft."""
    brief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Digest of the research brief the draft answers (``research_brief``)."""
    left_out: tuple[str, ...] = ()
    """Sources the research wanted and left out, each with its reason: it
    could not be opened, it states no date, or the length limit excluded it."""


class Review(BaseModel):
    """A record an entry may carry beside ``drafting``. Nothing reads it. Its
    fields are declared so that a captured source bundle whose register holds
    it, and an instrument built from that bundle, load and are saved again
    unchanged."""

    model_config = ConfigDict(extra="forbid")
    reviewed_by: str = Field(min_length=1)
    reviewer_kind: Literal["person", "ai_agent"]
    reviewed_on: date
    accuracy_review: str = Field(min_length=1)
    hazard_review: str = Field(min_length=1)


class ThreatModel(BaseModel):
    """One scenario's threat model."""

    model_config = ConfigDict(extra="forbid")
    threat_model_id: ItemId
    scenario_id: str = Field(min_length=1)
    scenario_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Digest of the scenario definition the threat model was written against
    (``scenario_definition_sha256``)."""
    research_through_utc: date
    """The information cutoff: no source was published after it."""
    summary: Unnumbered
    actors: tuple[Actor, ...] = Field(min_length=1)
    pathway: tuple[PathwayStep, ...] = Field(min_length=2)
    severity_modes: tuple[SeverityMode, ...] = Field(min_length=1)
    analogues: tuple[Analogue, ...] = ()
    key_uncertainties: tuple[Unnumbered, ...] = Field(min_length=1)
    liability: LiabilityPathway | None = None
    sources: tuple[ThreatModelSource, ...] = Field(min_length=1)
    drafting: Drafting | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """A draft's working notes. A register entry needs none."""
    review: Review | None = Field(default=None, exclude_if=lambda value: value is None)
    """Not needed either. No question shows either record, so an entry
    renders the same text with them as without. A saved threat model holds
    the records its entry carries and no others, so a captured source bundle
    and the instruments built from it keep their digests."""

    def _items(self) -> tuple[_Item, ...]:
        liability = () if self.liability is None else (self.liability,)
        return (
            *self.actors,
            *self.pathway,
            *self.severity_modes,
            *self.analogues,
            *liability,
        )

    @model_validator(mode="after")
    def validate_threat_model(self) -> Self:
        for rows, field in (
            (self.actors, "actor_id"),
            (self.pathway, "step_id"),
            (self.severity_modes, "mode_id"),
            (self.analogues, "analogue_id"),
            (self.sources, "source_id"),
        ):
            ids = [getattr(row, field) for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"threat model {field} values must be unique")
        if self.pathway[0].kind != "necessary":
            raise ValueError(
                "a pathway begins with a necessary step, the starting condition"
            )
        findings = {row.source_id: figures(row.finding) for row in self.sources}
        cited = {source for item in self._items() for source in item.sources}
        if unknown := sorted(cited - findings.keys()):
            raise ValueError(f"threat model cites unknown sources {unknown}")
        if unused := sorted(findings.keys() - cited):
            raise ValueError(f"no part of the threat model cites sources {unused}")
        for item in self._items():
            reported = frozenset().union(*(findings[key] for key in item.sources))
            stated = frozenset().union(*(figures(text) for text in item.texts()))
            if untraced := sorted(stated - reported):
                raise ValueError(
                    f"{item.label()} states {untraced}, which no finding of the "
                    "sources it cites states; a figure is one a cited source "
                    "reports, recorded in that source's finding"
                )
        if late := sorted(
            row.source_id
            for row in self.sources
            if row.published_utc > self.research_through_utc
        ):
            raise ValueError(
                f"sources {late} were published after the research's cutoff"
            )
        words = len(self.render().split())
        if words > MAX_RENDERED_WORDS:
            raise ValueError(
                f"the threat model takes {words:,} words in a question, more "
                f"than the {MAX_RENDERED_WORDS:,} a question can give it"
            )
        return self

    def eligible(self, temporal_fields: Mapping[str, Any]) -> bool:
        """Whether the research's information cutoff is within every cutoff of
        a study period, as ``DatedEvidenceText.eligible`` asks of a source
        card."""
        for key in (
            "information_date_utc",
            "availability_cutoff_utc",
            "data_cutoff_utc",
            "as_of",
        ):
            cutoff = parse_datetime_utc(temporal_fields.get(key))
            if cutoff is not None and self.research_through_utc > cutoff.date():
                return False
        return True

    def render(self) -> str:
        """The threat model as a question quotes it, between delimiter lines.

        Each source is cited without its finding, which the parts that cite
        it restate. The ``drafting`` and ``review`` records are not shown."""
        lines = [
            f'<threat_model id="{self.threat_model_id}">',
            f"Sources published through {self.research_through_utc}.",
            f"Summary: {_flat(self.summary)}",
            "Actors:",
        ]
        for actor in self.actors:
            lines.append(
                f"- {actor.actor_id}: {_flat(actor.description)} {actor.cited()}"
            )
        for kind, heading in (
            (
                "necessary",
                "Steps to a counted incident, in order; without any one of them "
                "none is counted:",
            ),
            (
                "escalation",
                "Steps that make a counted incident larger; none is needed for "
                "one to be counted:",
            ),
        ):
            steps = [step for step in self.pathway if step.kind == kind]
            if not steps:
                continue
            lines.append(heading)
            for number, step in enumerate(steps, start=1):
                lines.extend(
                    (
                        f"{number}. {step.step_id}: {_flat(step.description)}",
                        f"   What the AI system changes: {_flat(step.ai_contribution)}",
                        f"   Bottleneck: {_flat(step.bottleneck)}",
                        f"   Defender response: {_flat(step.defender_response)}",
                        f"   {step.cited()}",
                    )
                )
        lines.append("Severity modes:")
        for mode in self.severity_modes:
            lines.append(
                f"- {mode.mode_id}: {_flat(mode.description)} Loss drivers: "
                f"{_flat(mode.loss_drivers)} {mode.cited()}"
            )
        if self.analogues:
            lines.append("Analogues:")
            for analogue in self.analogues:
                lines.append(
                    f"- {analogue.analogue_id}: {_flat(analogue.description)} "
                    f"Reported outcome: {_flat(analogue.outcome)} {analogue.cited()}"
                )
        else:
            lines.append("Analogues: the research found none.")
        if self.liability is not None:
            lines.extend(
                (
                    "How loss reaches the lab:",
                    f"- Theories: {_flat(self.liability.theories)}",
                    f"- Defences: {_flat(self.liability.defences)}",
                    f"- Contract terms: {_flat(self.liability.contract_terms)}",
                    f"- {self.liability.cited()}",
                )
            )
        lines.append("Key uncertainties:")
        lines.extend(f"- {_flat(row)}" for row in self.key_uncertainties)
        lines.append("Sources:")
        lines.extend(
            f"- {row.source_id}: {row.citation} (published {row.published_utc})"
            for row in self.sources
        )
        lines.append(THREAT_MODEL_CLOSE)
        return "\n".join(lines)


def _flat(text: str) -> str:
    return " ".join(text.split())


def scenario_definition_sha256(scenario: ScenarioDefinition) -> str:
    """Digest of what a threat model is written against: the scenario's
    description, its terminal-loss population as questions state it (the
    counted unit, the counting and year rules, the attribution rule and the
    loss scope) and the opportunity and stages of its event chain.

    Exposure bases, tail governance and reporting facets are left out: a
    change to them does not change what the threat model describes.
    """
    return canonical_json_sha256(
        {
            "description": scenario.description,
            "population": effective_loss_population_contract(scenario),
            "terminal_unit": qualified_terminal_unit(scenario),
            "opportunity": scenario.event_chain.opportunity.model_dump(mode="json"),
            "stages": [
                stage.model_dump(mode="json") for stage in scenario.event_chain.stages
            ],
        }
    )


class ThreatModelsFile(BaseModel):
    """The register of threat models, at most one for a scenario. A source
    that several of them cite carries one publication date in all."""

    model_config = ConfigDict(extra="forbid")
    threat_models: tuple[ThreatModel, ...] = ()

    @model_validator(mode="after")
    def validate_register(self) -> Self:
        for field in ("threat_model_id", "scenario_id"):
            values = [getattr(row, field) for row in self.threat_models]
            if len(values) != len(set(values)):
                raise ValueError(f"threat-model register repeats a {field}")
        dated: dict[str, dict[date, list[str]]] = {}
        for row in self.threat_models:
            for source in row.sources:
                dated.setdefault(str(source.citation).rstrip("/"), {}).setdefault(
                    source.published_utc, []
                ).append(row.threat_model_id)
        for citation, dates in dated.items():
            if len(dates) > 1:
                raise ValueError(
                    f"threat models date the source {citation} differently: "
                    + "; ".join(
                        f"{published} in {', '.join(sorted(ids))}"
                        for published, ids in sorted(dates.items())
                    )
                    + ". A source cited by several threat models carries one date"
                )
        return self

    def for_scenario(self, scenario_id: str) -> ThreatModel | None:
        return next(
            (row for row in self.threat_models if row.scenario_id == scenario_id),
            None,
        )


def threat_model_errors(
    threat_models: Iterable[ThreatModel], scenarios: Iterable[ScenarioDefinition]
) -> list[str]:
    """Where threat models and a scenario library disagree: a threat model for
    an ID that is no scenario, or one written against a definition the
    scenario no longer has."""
    library = {scenario.id: scenario for scenario in scenarios}
    errors = []
    for row in threat_models:
        scenario = library.get(row.scenario_id)
        if scenario is None:
            errors.append(
                f"threat model {row.threat_model_id}: {row.scenario_id!r} is not "
                "a library scenario"
            )
        elif row.scenario_definition_sha256 != scenario_definition_sha256(scenario):
            errors.append(
                f"threat model {row.threat_model_id}: scenario {scenario.id} has "
                "changed since it was written. Check it against the current "
                "definition, then record scenario_definition_sha256 "
                f"{scenario_definition_sha256(scenario)}"
            )
    return errors


def load_threat_models(path: Path) -> ThreatModelsFile:
    """The register at ``path``; a missing file is an empty register."""
    if not path.exists():
        return ThreatModelsFile()
    payload = safe_load(path.read_text(encoding="utf-8"))
    return ThreatModelsFile.model_validate({} if payload is None else payload)


_TEMPLATE = """\
threat_model_id: tm-{scenario_id_dashed}
scenario_id: {scenario_id}
scenario_definition_sha256: {definition}
research_through_utc: {research_through}
summary: >-
  Three to six sentences: how incidents of this scenario arise and what most
  limits them. No figures.
actors:
  - actor_id: short-lowercase-id
    description: >-
      Who they are, what they want and what they can already do.
    sources: [source-id]
pathway:
  - step_id: short-lowercase-id
    kind: necessary
    description: >-
      What has to happen at this step.
    ai_contribution: >-
      What the AI system contributes or changes here, and the capability that
      takes.
    bottleneck: >-
      What most limits the step today.
    defender_response: >-
      What defenders, operators or authorities do at this step.
    sources: [source-id]
severity_modes:
  - mode_id: short-lowercase-id
    description: >-
      One distinct way an incident ends.
    loss_drivers: >-
      What makes the loss larger or smaller in this mode.
    sources: [source-id]
analogues:
  - analogue_id: short-lowercase-id
    description: >-
      What happened, where and when.
    outcome: >-
      The reported losses, casualties or counts, with units and dates.
    sources: [source-id]
key_uncertainties:
  - >-
    One thing the panel has to judge that the sources do not settle. No
    figures.
liability:
  theories: >-
    Legal theories and regulatory powers under which the lab would bear loss.
  defences: >-
    The lab's defences.
  contract_terms: >-
    Terms that allocate loss between the lab and its customers.
  sources: [source-id]
sources:
  - source_id: short-lowercase-id
    citation: https://example.org/report
    published_utc: YYYY-MM-DD
    accessed_utc: YYYY-MM-DD
    finding: >-
      What this source states that the threat model relies on, with every
      figure a part takes from it.
drafting:
  drafted_by: >-
    The agent and tools that wrote the draft, how the sources were read (in
    full, or through a tool that summarises pages) and who edited it.
  brief_sha256: {brief_marker}
  left_out:
    - A source the research wanted and left out, and why.
"""

_BRIEF_MARKER = "the digest printed with this brief"


def research_brief(
    scenario: ScenarioDefinition,
    cards: Sequence[DatedEvidenceText],
    *,
    research_through: date,
) -> str:
    """The task for the research stage: write ``scenario``'s threat model from
    sources read with retrieval, none published after ``research_through``.

    The brief quotes the scenario as its questions define it and the source
    cards the questions already show. ``brief_sha256`` is its digest, which
    the draft records.
    """
    chain = scenario.event_chain
    stages = "\n".join(
        f"{number}. {stage.stage_id}: {_flat(stage.event)}"
        for number, stage in enumerate(chain.stages, start=1)
    )
    shown = (
        "\n\n".join(card.render() for card in cards)
        if cards
        else "None. No source excerpt is shown for this scenario."
    )
    template = _TEMPLATE.format(
        scenario_id=scenario.id,
        scenario_id_dashed=scenario.id.replace("_", "-"),
        definition=scenario_definition_sha256(scenario),
        research_through=research_through.isoformat(),
        brief_marker=_BRIEF_MARKER,
    )
    return "\n\n".join(
        (
            f"# Research brief: threat model for {scenario.name} ({scenario.id})",
            "A panel of forecasters will estimate how often incidents of the "
            "scenario below happen and what they cost. Before they do, write "
            "the threat model they will read. Find sources, read them and "
            "structure what they establish. The panel supplies every "
            "probability and every loss estimate; the threat model supplies "
            "none.",
            "## The scenario",
            _flat(scenario.description),
            f"One counted incident: {_flat(qualified_terminal_unit(scenario))}",
            f"How incidents are counted: "
            f"{_flat(scenario.frequency_semantics.counting_identity)}",
            f"Loss counted for one incident: "
            f"{_flat(rendered_severity_occurrence_scope(scenario))}",
            f"What ties an incident to the lab's model: "
            f"{_flat(scenario.attribution_nexus.terminal_qualification)}",
            "The event chain the panel is asked about today starts from this "
            f"opportunity: {_flat(chain.opportunity.unit)} Its stages before the "
            f"counted incident are:\n{stages}\n"
            "The chain question then asks one more probability as its last "
            "stage: that an opportunity which reached every stage above becomes "
            "a counted incident as defined above. The admission threshold and "
            "the tie to the lab's model enter there, so the chain has a stage "
            "for them, though not a separate one.",
            "What the scenario file says about existing evidence: "
            f"{_flat(scenario.empirical_grounding)}",
            "## Source excerpts the panel already sees",
            shown,
            "The panel reads these beside the threat model, so do not restate "
            "them. Cite one of their sources only if you opened it and it adds "
            "something the excerpt lacks.",
            "## What to research",
            "1. Actors: who causes or takes part in such an incident, such as "
            "attackers, operators, users, claimants or regulators, what they "
            "want and what they can already do.\n"
            "2. Pathway: the steps, in order, at least two. Begin with the "
            "starting condition. Mark a step necessary when no incident is "
            "counted without it, and escalation when it is not needed for a "
            "counted incident and makes one larger. For each step say what "
            "has to happen, what the AI system contributes or changes and the "
            "capability that takes, what most limits the step today, and what "
            "defenders, operators or authorities do there.\n"
            "3. Severity modes: the distinct ways an incident ends, such as "
            "interdicted, contained and propagating, and what drives loss in "
            "each.\n"
            "4. Analogues: published incidents or datasets of the closest kind, "
            "with what the sources report happened, in units and with dates.\n"
            "5. Key uncertainties: what the panel has to judge that the sources "
            "do not settle.\n"
            "6. How loss reaches the lab: legal theories, defences and contract "
            "terms, in the United States and the European Union unless the "
            "scenario names another jurisdiction.",
            "## Rules",
            "- Read each source yourself and record in its finding, in a "
            "sentence or two, the claims and figures your parts take from it. "
            "Findings are the record your parts are checked against; the "
            "panel does not see them. Cite nothing from memory or from a "
            "search result's summary. Where a tool gave you only a summary of "
            "a page, say so in drafting. List under left_out the sources you "
            "wanted and left out, with the reason: not opened, no date, or no "
            "room.\n"
            f"- Cite nothing published after {research_through}. Give each "
            "source the date of the version you read, its last revision where "
            "it states one, or for terms and policies the date they took "
            "effect, and the date you opened it. Where a source states only a "
            "month or a year, give its last day.\n"
            "- State no probability, rate or loss estimate of your own.\n"
            "- A part may state a figure only when a source it cites reports "
            "it, and that source's finding records the same figure. The "
            "summary and the key uncertainties cite nothing, so they state no "
            "figures. Dates, the numbers of articles and sections, and "
            "identifiers are not figures.\n"
            "- A part with no source is your judgment and is shown as one.\n"
            "- Say what each step is and why it is hard, never how it is done. "
            "Category words, such as phishing, ransomware or vulnerability "
            "discovery, and the names of published incidents and actors are "
            "fine. A specific technique, vulnerability, tool, configuration, "
            "material or amount of it, named target or way around a safeguard "
            "is not, even where a public source states it, and that holds for "
            "findings too. Where a step cannot be described without one, name "
            "the step and stop there.\n"
            "- Write plain prose. No headings, markup, links or chat-role "
            "labels inside the text fields: links go in sources, and a "
            "less-than sign before a letter reads as markup, so write "
            '"under". Do not address the reader.\n'
            f"- Be brief: the threat model may take at most "
            f"{MAX_RENDERED_WORDS:,} words as a question quotes it, which is "
            "everything but the findings and the drafting record.\n"
            "- Where one event is both, such as a protective shutdown that is "
            "itself the loss, describe it under the necessary step and give an "
            "escalation step only what goes beyond it.\n"
            "- Where the steps you find differ from the event chain above, say "
            "so in summary: the chain can be revised.",
            "## Output",
            "Return one YAML document in this form, with as many entries in each "
            "list as the research supports. Every source is cited by at least "
            "one part.",
            template.rstrip("\n"),
        )
    )


def brief_sha256(brief: str) -> str:
    return canonical_json_sha256(brief)


def answers_current_brief(
    draft: ThreatModel,
    scenarios: Iterable[ScenarioDefinition],
    cards: Mapping[str, Sequence[DatedEvidenceText]],
) -> bool:
    """Whether a draft answers the brief its scenario has today.

    False where the scenario's source cards or the brief's own wording have
    changed since the draft was written, where the draft records no brief
    digest, or where the scenario is unknown. None of these stops a draft:
    the contract and the scenario's definition digest decide what a draft may
    hold.
    """
    scenario = next((row for row in scenarios if row.id == draft.scenario_id), None)
    return (
        scenario is not None
        and draft.drafting is not None
        and draft.drafting.brief_sha256
        == brief_sha256(
            research_brief(
                scenario,
                cards.get(scenario.id, ()),
                research_through=draft.research_through_utc,
            )
        )
    )


class ReferenceMapping(BaseModel):
    """One reading of a blind comparison: for each part of the reference
    threat model, the candidate's part that states the same thing, or None
    where the candidate has none."""

    model_config = ConfigDict(extra="forbid")
    pathway: dict[str, str | None]
    severity_modes: dict[str, str | None]
    actors: dict[str, str | None]


def coverage_against_reference(
    reference: ThreatModel, candidate: ThreatModel, mapping: ReferenceMapping
) -> dict[str, Any]:
    """How much of a reference threat model a candidate covers, by a mapping
    of their parts.

    The reference is a published expert threat model written out in this
    contract; the candidate was produced blind to it. Whether two parts state
    the same thing is a judgment, recorded in ``mapping``. This function
    checks the mapping against both threat models and counts: for each kind
    of part, the reference's parts, how many the candidate covers, and the
    candidate's parts the reference does not have.
    """

    def count(
        kind: str,
        mapped: Mapping[str, str | None],
        expected: Collection[str],
        offered: Collection[str],
    ) -> dict[str, Any]:
        if set(mapped) != set(expected):
            raise ValueError(
                f"the mapping's {kind} must list every part of the reference: "
                f"missing={sorted(set(expected) - set(mapped))}, "
                f"extra={sorted(set(mapped) - set(expected))}"
            )
        matched = [value for value in mapped.values() if value is not None]
        if unknown := sorted(set(matched) - set(offered)):
            raise ValueError(f"the mapping's {kind} names unknown parts {unknown}")
        if len(matched) != len(set(matched)):
            raise ValueError(
                f"the mapping's {kind} matches one candidate part to two "
                "reference parts"
            )
        return {
            "reference": len(expected),
            "covered": len(matched),
            "missed": sorted(key for key, value in mapped.items() if value is None),
            "candidate_only": sorted(set(offered) - set(matched)),
        }

    return {
        "reference": reference.threat_model_id,
        "candidate": candidate.threat_model_id,
        "pathway": count(
            "pathway",
            mapping.pathway,
            [row.step_id for row in reference.pathway],
            [row.step_id for row in candidate.pathway],
        ),
        "severity_modes": count(
            "severity_modes",
            mapping.severity_modes,
            [row.mode_id for row in reference.severity_modes],
            [row.mode_id for row in candidate.severity_modes],
        ),
        "actors": count(
            "actors",
            mapping.actors,
            [row.actor_id for row in reference.actors],
            [row.actor_id for row in candidate.actors],
        ),
    }
