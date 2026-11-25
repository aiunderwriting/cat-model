"""Dated source cards: the evidence each scenario's questions quote.

A card states what its cited sources report, with their units and dates, and
cites at least one source. Questions quote a card between delimiter lines, so
its text holds nothing that could end the quotation or open a section of the
question, and nothing a respondent could read as authored input: no prior,
numerical default, answering instruction or term of the model's own.
``SourceCardsFile`` lists, for every scenario, the cards its questions show;
scenario-library validation checks the repository file's listing against the
library with ``source_card_coverage_errors``.
"""

import re
import unicodedata
from collections.abc import Collection, Mapping
from datetime import date
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Self
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from catmodel.dateutil import parse_datetime_utc
from catmodel.resources import scenarios_path
from catmodel.scenarios.loader import load_scenarios
from catmodel.yaml_io import safe_load

SOURCE_CARDS_RELATIVE_PATH = Path("data/elicitation_source_cards.yaml")
"""The repository's cards file, relative to the project root, for
scenario-library validation. The wheel does not carry it: cards reach
questions only through a captured source bundle."""

CARD_ID_PATTERN = r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$"
"""A card ID is lowercase words joined by hyphens, underscores or dots, so it
can stand inside the delimiter line that opens its quotation."""

EXCERPT_CLOSE = "</excerpt>"
"""The line that closes a quoted card."""

_STRUCTURE: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^[ \t]{0,3}#{1,6}(?:[ \t]|$)", re.MULTILINE), "a Markdown heading"),
    (
        re.compile(r"^[ \t]{0,3}(?:=+|-{3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE),
        "a heading underline or horizontal rule",
    ),
    (re.compile(r"`{3}|~{3}"), "a code fence"),
    (re.compile(r"<[ \t]*/?[ \t]*[A-Za-z!?]"), "a markup tag"),
    (
        re.compile(
            r"^[ \t]*(?:system|assistant|user|human)[ \t]*:",
            re.MULTILINE | re.IGNORECASE,
        ),
        "a chat-role label",
    ),
)
"""Structure that could end a quoted card early or start a section of the
question: questions use Markdown headings and close each card with a tag."""

_START = r"(?:^|[.;:!?]\s+|\s-\s+)"
"""The start of a sentence, clause or list item in whitespace-normalized text,
where wrapped lines no longer begin new matches."""

_AUTHORED: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"https?://", re.IGNORECASE), "a URL, which belongs in sources"),
    (
        re.compile(_START + r"sources?(?:\s*\([^)]*\))?\s*:", re.IGNORECASE),
        "a source line, which belongs in sources",
    ),
    (
        re.compile(
            r"\bauthored\s+(?:model\s+)?(?:judg\w*|priors?|defaults?|conventions?"
            r"|values?|assumptions?|estimates?)\b",
            re.IGNORECASE,
        ),
        "an authored value or convention",
    ),
    (
        re.compile(
            r"\bjudge?ment(?:al)?\s+(?:priors?|ranges?|defaults?|values?|estimates?)\b",
            re.IGNORECASE,
        ),
        "a judgment prior or range",
    ),
    (
        re.compile(
            r"\b(?:priors?|defaults?)\s*[:=]|\b(?:base|default)\s+(?:value\s+)?\d",
            re.IGNORECASE,
        ),
        "a numerical prior or default",
    ),
    (
        re.compile(r"\byou(?:rs?|rself)?\b", re.IGNORECASE),
        "words addressed to the respondent",
    ),
    (
        re.compile(
            _START + r"(?:do not|don't|estimate|assume|treat|ignore|answer|explain"
            r"|preserve|match|consider)\b",
            re.IGNORECASE,
        ),
        "an answering instruction",
    ),
    (
        re.compile(
            r"\bgoverned\b(?!\s+by\b)|\bleaf\b|\badmission threshold\b", re.IGNORECASE
        ),
        "a term of the model's own",
    ),
)
"""Phrases that mark text as authored input rather than reported evidence: an
inline citation, an authored prior, default or judgment range, an instruction
to the respondent, or a reference to the model. They are matched against the
whitespace-normalized text, and each is narrow, so that reported facts,
including statutes' duties ("must report"), pass."""

_YEAR = re.compile(r"\b(?:1[89]|20)\d{2}\b")


def _structure_error(text: str) -> str | None:
    if any(
        unicodedata.category(char) in {"Cc", "Cf", "Co", "Cs"} and char not in "\n\t"
        for char in text
    ):
        return "control or invisible formatting characters"
    for pattern, label in _STRUCTURE:
        if pattern.search(text):
            return label
    return None


def quotation_structure_error(text: str) -> str | None:
    """What in ``text`` could end its quotation in a question or open a
    section of the question, or None: a heading, a rule, a code fence, a
    markup tag, a chat-role label or an invisible character."""
    return _structure_error(text)


def _source(entry: str) -> str:
    """An https URL, or a precise citation on one line that names its year."""
    if (error := _structure_error(entry)) is not None:
        raise ValueError(f"a source holds {error}")
    if not entry or entry != entry.strip() or any(c in entry for c in "\n\t"):
        raise ValueError("a source is one nonblank line without surrounding whitespace")
    if "://" in entry:
        parts = urlsplit(entry)
        if (
            parts.scheme != "https"
            or not parts.netloc
            or any(char.isspace() or char in "\"'`<>" for char in entry)
        ):
            raise ValueError(
                f"source {entry!r} is not one https URL; list each URL as its own source"
            )
        return entry
    if _YEAR.search(entry) is None:
        raise ValueError(
            f"source {entry!r} is neither an https URL nor a citation naming its year"
        )
    return entry


type Source = Annotated[str, AfterValidator(_source)]


class DatedEvidenceText(BaseModel):
    """One source card; conservative curation bounds are not publication dates."""

    model_config = ConfigDict(extra="forbid")
    card_id: str = Field(pattern=CARD_ID_PATTERN)
    observation_through_utc: date
    availability_date_utc: date
    availability_basis: str = Field(min_length=1)
    text: str = Field(min_length=1)
    sources: tuple[Source, ...] = Field(min_length=1)
    """What the card cites: https URLs or precise citations naming their year."""

    @model_validator(mode="after")
    def validate_card(self) -> Self:
        if self.observation_through_utc > self.availability_date_utc:
            raise ValueError("source observation window cannot end after availability")
        if not self.availability_basis.strip() or not self.text.strip():
            raise ValueError("source availability basis and text must be nonblank")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(f"source card {self.card_id!r} cites a source twice")
        if (error := _structure_error(self.text)) is not None:
            raise ValueError(
                f"source card {self.card_id!r} text holds {error}, which could "
                "break out of its quotation in a question"
            )
        flat = " ".join(self.text.split())
        for pattern, label in _AUTHORED:
            if match := pattern.search(flat):
                raise ValueError(
                    f"source card {self.card_id!r} text holds {label} "
                    f"({match.group(0).strip()!r}); cards hold only what their "
                    "sources report"
                )
        return self

    def eligible(self, temporal_fields: Mapping[str, Any]) -> bool:
        for key, observed in (
            ("information_date_utc", self.availability_date_utc),
            ("availability_cutoff_utc", self.availability_date_utc),
            ("data_cutoff_utc", self.observation_through_utc),
            ("as_of", self.availability_date_utc),
        ):
            cutoff = parse_datetime_utc(temporal_fields.get(key))
            if cutoff is not None and observed > cutoff.date():
                return False
        return True

    def render(self) -> str:
        """The card as a question quotes it: between delimiter lines, with its
        dates and sources. The availability basis is curation metadata and is
        not shown."""
        return "\n".join(
            (
                f'<excerpt id="{self.card_id}">',
                f"Observations through {self.observation_through_utc}; "
                f"available by {self.availability_date_utc}.",
                self.text.strip(),
                "Sources: " + "; ".join(self.sources),
                EXCERPT_CLOSE,
            )
        )


@cache
def _library_scenario_ids() -> frozenset[str]:
    return frozenset(row.id for row in load_scenarios(scenarios_path()))


class SourceCardsFile(BaseModel):
    """Dated study evidence and, for every scenario, the cards its questions
    show; no methodological guidance or numerical defaults."""

    model_config = ConfigDict(extra="forbid")
    cards: tuple[DatedEvidenceText, ...]
    scenarios: dict[str, tuple[str, ...]]
    """Each library scenario's cards in the order its questions show them; an
    empty list shows none."""

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        ids = [card.card_id for card in self.cards]
        if len(ids) != len(set(ids)):
            raise ValueError("source card IDs must be unique")
        library = _library_scenario_ids()
        missing = sorted(library - set(self.scenarios))
        extra = sorted(set(self.scenarios) - library)
        if missing or extra:
            raise ValueError(
                "source-card scenarios must list every library scenario: "
                f"missing={missing}, extra={extra}"
            )
        known = set(ids)
        for scenario_id, shown in self.scenarios.items():
            if len(shown) != len(set(shown)):
                raise ValueError(f"scenario {scenario_id!r} lists a source card twice")
            if unknown := sorted(set(shown) - known):
                raise ValueError(
                    f"scenario {scenario_id!r} lists unknown source cards {unknown}"
                )
        if unused := sorted(known.difference(*self.scenarios.values())):
            raise ValueError(f"no scenario shows source cards {unused}")
        return self

    def scenario_cards(self, scenario_id: str) -> tuple[DatedEvidenceText, ...]:
        """The cards ``scenario_id``'s questions show, in their listed order."""
        by_id = {card.card_id: card for card in self.cards}
        return tuple(by_id[card_id] for card_id in self.scenarios[scenario_id])


def source_card_coverage_errors(path: Path, scenario_ids: Collection[str]) -> list[str]:
    """Where the ``scenarios`` map of the cards file at ``path`` and a scenario
    roster differ: a scenario the map does not list, or an entry for an id that
    is not a scenario. A missing or unreadable file is an error as well.

    ``SourceCardsFile`` applies the same rule against the runtime library
    (``scenarios_path()``), but only where cards are loaded: when a source
    bundle is captured or an instrument is built. Scenario-library validation
    applies it to the roster it loaded, so a scenario added without an entry is
    reported with the library's other errors. The cards themselves are not
    validated here.
    """
    try:
        payload = safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"source cards file not found: {path}"]
    except Exception as exc:
        return [f"failed to read source cards file {path}: {exc}"]
    listed = payload.get("scenarios") if isinstance(payload, Mapping) else None
    if not isinstance(listed, Mapping):
        return [
            f"{path}: scenarios must map every library scenario to the source "
            "cards its questions show"
        ]
    library = set(scenario_ids)
    errors = [
        f"{scenario_id}: {path} does not list it under scenarios; list every "
        "scenario there, with [] where its questions show no source cards"
        for scenario_id in sorted(library - set(listed))
    ]
    errors.extend(
        f"{path}: scenarios lists {entry!r}, which is not a library scenario; "
        "remove the entry or correct its id"
        for entry in sorted(set(listed) - library, key=str)
    )
    return errors
